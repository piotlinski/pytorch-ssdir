"""SSDIR encoder, decoder, model and guide declarations."""
import warnings
from argparse import ArgumentParser
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as PILImage
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as functional
import wandb
from pyro.infer import Trace_ELBO
from pytorch_ssd.data.datasets import datasets
from pytorch_ssd.data.transforms import DataTransform, TrainDataTransform
from pytorch_ssd.modeling.model import SSD
from pytorch_ssd.modeling.visualize import denormalize
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.dataloader import DataLoader

from pytorch_ssdir.modeling.depth import DepthEncoder
from pytorch_ssdir.modeling.present import PresentEncoder
from pytorch_ssdir.modeling.what import WhatDecoder, WhatEncoder
from pytorch_ssdir.modeling.where import WhereEncoder, WhereTransformer
from pytorch_ssdir.run.loss import per_site_loss
from pytorch_ssdir.run.transforms import corner_to_center_target_transform

warnings.filterwarnings(
    "ignore",
    message="^.* was not registered in the param store because requires_grad=False",
)


class Encoder(nn.Module):
    """Module encoding input image to latent representation.

    .. latent representation consists of:
       - $$z_{what} ~ N(\\mu^{what}, \\sigma^{what})$$
       - $$z_{where} in R^4$$
       - $$z_{present} ~ Bernoulli(p_{present})$$
       - $$z_{depth} ~ N(\\mu_{depth}, \\sigma_{depth})$$
    """

    def __init__(
        self,
        ssd: SSD,
        z_what_size: int = 64,
        z_what_scale_const: Optional[float] = None,
        z_depth_scale_const: Optional[float] = None,
        train_what: bool = True,
        train_where: bool = True,
        train_present: bool = True,
        train_depth: bool = True,
        train_backbone: bool = True,
    ):
        super().__init__()
        self.ssd_backbone = ssd.backbone.requires_grad_(train_backbone)
        self.what_enc = WhatEncoder(
            z_what_size=z_what_size,
            z_what_scale_const=z_what_scale_const,
            feature_channels=ssd.backbone.out_channels,
            feature_maps=ssd.backbone.feature_maps,
        ).requires_grad_(train_what)
        self.where_enc = WhereEncoder(
            ssd_box_predictor=ssd.predictor,
            ssd_anchors=ssd.anchors,
            ssd_center_variance=ssd.center_variance,
            ssd_size_variance=ssd.size_variance,
        ).requires_grad_(train_where)
        self.present_enc = PresentEncoder(
            ssd_box_predictor=ssd.predictor
        ).requires_grad_(train_present)
        self.depth_enc = DepthEncoder(
            feature_channels=ssd.backbone.out_channels,
            z_depth_scale_const=z_depth_scale_const,
        ).requires_grad_(train_depth)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], ...]:
        """Takes images tensors (batch_size x channels x image_size x image_size)
        .. and outputs latent representation tuple
        .. (z_what (loc & scale), z_where, z_present, z_depth (loc & scale))
        """
        features = self.ssd_backbone(images)
        z_where = self.where_enc(features)
        z_present = self.present_enc(features)
        z_what_loc, z_what_scale = self.what_enc(features)
        z_depth_loc, z_depth_scale = self.depth_enc(features)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )


class Decoder(nn.Module):
    """Module decoding latent representation.

    .. Pipeline:
       - sort z_depth ascending
       - sort $$z_{what}$$, $$z_{where}$$, $$z_{present}$$ accordingly
       - decode $$z_{what}$$ where $$z_{present} = 1$$
       - transform decoded objects according to $$z_{where}$$
       - merge transformed images based on $$z_{depth}$$
    """

    def __init__(
        self,
        ssd: SSD,
        z_what_size: int = 64,
        drop_empty: bool = True,
        weighted_merge: bool = False,
        train_what: bool = True,
    ):
        super().__init__()
        self.what_dec = WhatDecoder(z_what_size=z_what_size).requires_grad_(train_what)
        self.where_stn = WhereTransformer(image_size=ssd.image_size[0])
        self.indices = nn.Parameter(
            self.reconstruction_indices(
                feature_maps=ssd.backbone.feature_maps,
                boxes_per_loc=ssd.backbone.boxes_per_loc,
            ),
            requires_grad=False,
        )
        self.drop = drop_empty
        self.weighted = weighted_merge
        self.empty_obj_const = nn.Parameter(torch.tensor(-1000.0), requires_grad=False)
        self.pixel_means = ssd.backbone.PIXEL_MEANS
        self.pixel_stds = ssd.backbone.PIXEL_STDS
        self.bg_where = nn.Parameter(
            torch.tensor([0.5, 0.5, 1.0, 1.0]), requires_grad=False
        )
        self.bg_present = nn.Parameter(torch.ones(1), requires_grad=False)
        self.bg_depth = nn.Parameter(torch.tensor([-1000.0]), requires_grad=False)

    @staticmethod
    def reconstruction_indices(
        feature_maps: List[int], boxes_per_loc: List[int]
    ) -> torch.Tensor:
        """Get indices for reconstructing images.

        .. Caters for the difference between z_what, z_depth and z_where, z_present.
        """
        indices = []
        img_idx = last_img_idx = 0
        for feature_map, n_boxes in zip(feature_maps, boxes_per_loc):
            for feature_map_idx in range(feature_map ** 2):
                img_idx = last_img_idx + feature_map_idx
                indices.append(
                    torch.full(size=(n_boxes,), fill_value=img_idx, dtype=torch.float)
                )
            last_img_idx = img_idx + 1
        indices.append(
            torch.full(size=(1,), fill_value=last_img_idx, dtype=torch.float)
        )
        return torch.cat(indices, dim=0)

    @staticmethod
    def pad_indices(n_present: torch.Tensor) -> torch.Tensor:
        """Using number of objects in chunks create indices
        .. so that every chunk is padded to the same dimension.

        .. Assumes index 0 refers to "starter" (empty) object, added to every chunk.

        :param n_present: number of objects in each chunk
        :return: indices for padding tensors
        """
        end_idx = 1
        max_objects = torch.max(n_present)
        indices = []
        for chunk_objects in n_present:
            start_idx = end_idx
            end_idx = end_idx + chunk_objects
            idx_range = torch.arange(
                start=start_idx, end=end_idx, dtype=torch.long, device=n_present.device
            )
            indices.append(
                functional.pad(idx_range, pad=[0, max_objects - chunk_objects])
            )
        return torch.cat(indices)

    def _pad_reconstructions(
        self,
        transformed_images: torch.Tensor,
        z_depth: torch.Tensor,
        n_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad tensors to have identical 1. dim shape
        .. and reshape to (batch_size x n_objects x ...)
        """
        image_starter = transformed_images.new_zeros(
            (1, 3, self.where_stn.image_size, self.where_stn.image_size)
        )
        z_depth_starter = z_depth.new_full((1, 1), fill_value=-float("inf"))
        images = torch.cat((image_starter, transformed_images), dim=0)
        z_depth = torch.cat((z_depth_starter, z_depth), dim=0)
        max_present = torch.max(n_present)
        padded_shape = max_present.item()
        indices = self.pad_indices(n_present)
        images = images[indices].view(
            -1,
            padded_shape,
            3,
            self.where_stn.image_size,
            self.where_stn.image_size,
        )
        z_depth = z_depth[indices].view(-1, padded_shape)
        return images, z_depth

    @staticmethod
    def merge_reconstructions_masked(
        reconstructions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine decoded images into one by masked merging."""
        sorted_weights, sort_index = torch.sort(weights, dim=1, descending=True)
        sorted_reconstructions = (
            reconstructions.gather(
                dim=1,
                index=sort_index.view(*sort_index.shape, 1, 1, 1).expand_as(
                    reconstructions
                ),
            )
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        outputs = sorted_reconstructions[0]
        for instance_batch in sorted_reconstructions[1:]:
            mask = torch.where(outputs < 1e-3, 1.0, 0.0)
            outputs = outputs + instance_batch * mask
        return outputs.clamp_(0.0, 1.0)

    @staticmethod
    def merge_reconstructions_weighted(
        reconstructions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine decoder images into one by weighted sum."""
        weighted_images = reconstructions * functional.softmax(weights, dim=1).view(
            *weights.shape[:2], 1, 1, 1
        )
        return torch.sum(weighted_images, dim=1)

    def reconstruct_objects(
        self,
        z_what: torch.Tensor,
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render reconstructions and their depths from batch."""
        z_what_shape = z_what.shape
        z_where_shape = z_where.shape
        z_depth_shape = z_depth.shape
        if self.drop:
            present_mask = z_present == 1
            n_present = torch.sum(present_mask, dim=1).squeeze(-1)
            z_what = z_what[present_mask.expand_as(z_what)].view(-1, z_what_shape[-1])
            z_where = z_where[present_mask.expand_as(z_where)].view(
                -1, z_where_shape[-1]
            )
            z_depth = z_depth[present_mask.expand_as(z_depth)].view(
                -1, z_depth_shape[-1]
            )
        z_what_flat = z_what.view(-1, z_what.shape[-1])
        z_where_flat = z_where.view(-1, z_where.shape[-1])
        decoded_images = self.what_dec(z_what_flat)
        transformed_images = self.where_stn(decoded_images, z_where_flat)
        if self.drop:
            reconstructions, depths = self._pad_reconstructions(
                transformed_images=transformed_images,
                z_depth=z_depth,
                n_present=n_present,
            )
        else:
            reconstructions = transformed_images.view(
                -1,
                z_what_shape[-2],
                3,
                self.where_stn.image_size,
                self.where_stn.image_size,
            )
            depths = z_depth.where(z_present == 1.0, self.empty_obj_const)
        return reconstructions, depths

    def append_bg_latents(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add background latents to z_where, z_present and z_depth."""
        z_what, z_where, z_present, z_depth = latents
        batch_size = z_what.shape[0]
        where_bg = self.bg_where.expand(batch_size, 1, 4)
        present_bg = self.bg_present.expand(batch_size, 1, 1)
        depth_bg = self.bg_depth.expand(batch_size, 1, 1)
        return (
            z_what,
            torch.cat((z_where, where_bg), dim=1),
            torch.cat((z_present, present_bg), dim=1),
            torch.cat((z_depth, depth_bg), dim=1),
        )

    def pad_latents(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad latents according to Decoder's settings."""
        z_what, z_where, z_present, z_depth = latents
        # repeat rows to match z_where and z_present
        z_what = z_what.index_select(dim=1, index=self.indices.long())
        z_depth = z_depth.index_select(dim=1, index=self.indices.long())
        return z_what, z_where, z_present, z_depth

    def forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Takes latent variables tensors tuple (z_what, z_where, z_present, z_depth)
        .. and outputs reconstructed images batch
        .. (batch_size x channels x image_size x image_size)
        """
        latents_bg = self.append_bg_latents(latents)
        z_what, z_where, z_present, z_depth = self.pad_latents(latents_bg)
        # render reconstructions
        reconstructions, depths = self.reconstruct_objects(
            z_what, z_where, z_present, z_depth
        )
        # merge reconstructions
        if self.weighted:
            output = self.merge_reconstructions_weighted(
                reconstructions=reconstructions, weights=depths
            )
        else:
            output = self.merge_reconstructions_masked(
                reconstructions=reconstructions, weights=depths
            )
        return output


class SSDIR(pl.LightningModule):
    """Single-Shot Detect, Infer, Repeat."""

    def __init__(
        self,
        ssd_model: SSD,
        learning_rate: float = 1e-3,
        ssd_lr_multiplier: float = 1.0,
        warm_restart_epochs: float = 1 / 3,
        warm_restart_len_mult: int = 2,
        auto_lr_find: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        z_what_size: int = 64,
        z_present_p_prior: float = 0.01,
        z_where_loc_prior: float = 0.5,
        z_where_scale_prior: float = 0.25,
        z_where_scale: float = 0.05,
        drop: bool = True,
        z_what_scale_const: Optional[float] = None,
        z_depth_scale_const: Optional[float] = None,
        weighted_merge: bool = False,
        what_coef: float = 1.0,
        where_coef: float = 1.0,
        present_coef: float = 1.0,
        depth_coef: float = 1.0,
        rec_coef: float = 1.0,
        train_what: bool = True,
        train_where: bool = True,
        train_present: bool = True,
        train_depth: bool = True,
        train_backbone: bool = True,
        visualize_inference: bool = True,
        visualize_inference_freq: int = 500,
        n_visualize_objects: int = 10,
        visualize_latents: bool = True,
        visualize_latents_freq: int = 5,
        **_kwargs,
    ):
        """
        :param ssd_model: trained SSD to use as backbone
        :param learning_rate: learning rate
        :param ssd_lr_multiplier: ssd learning rate multiplier (learning rate * mult)
        :param warm_restart_epochs: number of epochs before resetting learning rate
        :param warm_restart_len_mult: coef to multiply number of epochs after each reset
        :param auto_lr_find: perform auto lr finding
        :param batch_size: mini-batch size for training
        :param num_workers: number of workers for dataloader
        :param pin_memory: pin memory for training
        :param z_what_size: latent what size
        :param z_present_p_prior: present prob prior
        :param z_where_loc_prior: prior z_where loc
        :param z_where_scale_prior: prior z_where scale
        :param z_where_scale: z_where scale used in inference
        :param drop: drop empty objects' latents
        :param z_what_scale_const: fixed z_what scale (if None - use NN to model)
        :param z_depth_scale_const: fixed z_depth scale (if None - use NN to model)
        :param weighted_merge: merge output images using weighted sum (else: masked)
        :param what_coef: z_what loss component coefficient
        :param where_coef: z_where loss component coefficient
        :param present_coef: z_present loss component coefficient
        :param depth_coef: z_depth loss component coefficient
        :param rec_coef: reconstruction error component coefficient
        :param train_what: train what encoder and decoder
        :param train_where: train where encoder
        :param train_present: train present encoder
        :param train_depth: train depth encoder
        :param train_backbone: train ssd backbone
        :param visualize_inference: visualize inference
        :param visualize_inference_freq: how often to visualize inference
        :param n_visualize_objects: number of objects to visualize
        :param visualize_latents: visualize model latents
        :param visualize_latents_freq: how often to visualize latents
        """
        super().__init__()

        self.encoder = Encoder(
            ssd=ssd_model,
            z_what_size=z_what_size,
            z_what_scale_const=z_what_scale_const,
            z_depth_scale_const=z_depth_scale_const,
            train_what=train_what,
            train_where=train_where,
            train_present=train_present,
            train_depth=train_depth,
            train_backbone=train_backbone,
        )
        self.decoder = Decoder(
            ssd=ssd_model,
            z_what_size=z_what_size,
            drop_empty=drop,
            weighted_merge=weighted_merge,
            train_what=train_what,
        )

        self.lr = learning_rate
        self.ssd_lr_multiplier = ssd_lr_multiplier
        self.warm_restart_epochs = warm_restart_epochs
        self.warm_restart_len_mult = warm_restart_len_mult
        self.auto_lr_find = auto_lr_find
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pixel_means = ssd_model.backbone.PIXEL_MEANS
        self.pixel_stds = ssd_model.backbone.PIXEL_STDS

        self.image_size = ssd_model.image_size
        self.flip_train = ssd_model.flip_train
        self.augment_colors_train = ssd_model.augment_colors_train
        self.dataset = ssd_model.dataset
        self.data_dir = ssd_model.data_dir

        self.z_what_size = z_what_size
        self.z_present_p_prior = z_present_p_prior
        self.z_where_loc_prior = z_where_loc_prior
        self.z_where_scale_prior = z_where_scale_prior
        self.z_where_scale = z_where_scale
        self.drop = drop

        self.what_coef = what_coef
        self.where_coef = where_coef
        self.present_coef = present_coef
        self.depth_coef = depth_coef
        self.rec_coef = rec_coef

        self.visualize_inference = visualize_inference
        self.visualize_inference_freq = visualize_inference_freq
        self.n_visualize_objects = n_visualize_objects
        self.visualize_latents = visualize_latents
        self.visualize_latents_freq = visualize_latents_freq

        self.n_objects = sum(
            features ** 2 for features in ssd_model.backbone.feature_maps
        )
        self.n_ssd_features = sum(
            boxes * features ** 2
            for features, boxes in zip(
                ssd_model.backbone.feature_maps, ssd_model.backbone.boxes_per_loc
            )
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add SSDIR args to parent argument parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="MNIST",
            help=f"Used dataset name. Available: {list(datasets.keys())}",
        )
        parser.add_argument(
            "--data-dir", type=str, default="data", help="Dataset files directory"
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=1e-3,
            help="Learning rate used for training the model",
        )
        parser.add_argument(
            "--ssd-lr-multiplier",
            type=float,
            default=1.0,
            help="Learning rate multiplier for training SSD backbone",
        )
        parser.add_argument(
            "--warm-restart-epochs",
            type=float,
            default=1 / 3,
            help="Number of epochs after which a warm restart is performed",
        )
        parser.add_argument(
            "--warm-restart-len-mult",
            type=int,
            default=2,
            help="Coef to multiply warm restart epochs after each restart",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Mini-batch size used for training the model",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of workers used to load the dataset",
        )
        parser.add_argument(
            "--pin-memory",
            default=True,
            action="store_true",
            help="Pin data in memory while training",
        )
        parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
        parser.add_argument(
            "--z-what-size", type=int, default=64, help="z_what latent size"
        )
        parser.add_argument(
            "--z-present-p-prior",
            type=float,
            default=0.01,
            help="z_present probability prior",
        )
        parser.add_argument(
            "--z-where-loc-prior", type=float, default=0.5, help="prior z_where loc"
        )
        parser.add_argument(
            "--z-where-scale-prior",
            type=float,
            default=0.05,
            help="prior z_where scale",
        )
        parser.add_argument(
            "--z-where-scale",
            type=float,
            default=0.05,
            help="z_where scale used in inference",
        )
        parser.add_argument(
            "--drop",
            default=True,
            action="store_true",
            help="Drop empty objects' latents",
        )
        parser.add_argument("--no-drop", dest="drop", action="store_false")
        parser.add_argument(
            "--z-what-scale-const",
            type=float,
            default=None,
            help="constant z_what scale",
        )
        parser.add_argument(
            "--z-depth-scale-const",
            type=float,
            default=None,
            help="constant z_depth scale",
        )
        parser.add_argument(
            "--weighted-merge",
            default=False,
            action="store_true",
            help="Use weighted output merging method",
        )
        parser.add_argument(
            "--masked-merge", dest="weighted_merge", action="store_false"
        )
        parser.add_argument(
            "--what-coef",
            type=float,
            default=1.0,
            help="z_what loss component coefficient",
        )
        parser.add_argument(
            "--where-coef",
            type=float,
            default=1.0,
            help="z_where loss component coefficient",
        )
        parser.add_argument(
            "--present-coef",
            type=float,
            default=1.0,
            help="z_present loss component coefficient",
        )
        parser.add_argument(
            "--depth-coef",
            type=float,
            default=1.0,
            help="z_depth loss component coefficient",
        )
        parser.add_argument(
            "--rec-coef",
            type=float,
            default=1.0,
            help="Reconstruction error component coefficient",
        )
        parser.add_argument(
            "--train-what",
            default=True,
            action="store_true",
            help="Train what encoder and decoder",
        )
        parser.add_argument("--no-train-what", dest="train_what", action="store_false")
        parser.add_argument(
            "--train-where",
            default=True,
            action="store_true",
            help="Train where encoder",
        )
        parser.add_argument(
            "--no-train-where", dest="train_where", action="store_false"
        )
        parser.add_argument(
            "--train-present",
            default=True,
            action="store_true",
            help="Train present encoder",
        )
        parser.add_argument(
            "--no-train-present", dest="train_present", action="store_false"
        )
        parser.add_argument(
            "--train-depth",
            default=True,
            action="store_true",
            help="Train depth encoder",
        )
        parser.add_argument(
            "--no-train-depth", dest="train_depth", action="store_false"
        )
        parser.add_argument(
            "--train-backbone",
            default=True,
            action="store_true",
            help="Train SSD backbone",
        )
        parser.add_argument(
            "--no-train-backbone", dest="train_backbone", action="store_false"
        )
        parser.add_argument(
            "--flip-train",
            default=False,
            action="store_true",
            help="Flip train images during training",
        )
        parser.add_argument("--no-flip-train", dest="flip_train", action="store_false")
        parser.add_argument(
            "--augment-colors-train",
            default=False,
            action="store_true",
            help="Perform random colors augmentation during training",
        )
        parser.add_argument(
            "--no-augment-colors-train",
            dest="augment_colors_train",
            action="store_false",
        )
        parser.add_argument(
            "--visualize-inference",
            default=True,
            action="store_true",
            help="Log visualizations of model predictions",
        )
        parser.add_argument(
            "--no-visualize-inference", dest="visualize_inference", action="store_false"
        )
        parser.add_argument(
            "--visualize-inference-freq",
            type=int,
            default=500,
            help="How often to perform inference visualization.",
        )
        parser.add_argument(
            "--n-visualize-objects",
            type=int,
            default=5,
            help="Number of objects to visualize",
        )
        parser.add_argument(
            "--visualize-latents",
            default=True,
            action="store_true",
            help="Log visualizations of model latents",
        )
        parser.add_argument(
            "--no-visualize-latents", dest="visualize_latents", action="store_false"
        )
        parser.add_argument(
            "--visualize-latents-freq",
            type=int,
            default=10,
            help="How often to perform latents visualization.",
        )
        return parser

    def encoder_forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass through encoder network."""
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = self.encoder(inputs)
        z_what = dist.Normal(z_what_loc, z_what_scale).sample()
        z_present = dist.Bernoulli(z_present).sample()
        z_depth = dist.Normal(z_depth_loc, z_depth_scale).sample()
        return z_what, z_where, z_present, z_depth

    def decoder_forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        outputs = self.decoder(latents)
        return outputs

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Pass data through the model."""
        latents = self.encoder_forward(images)
        return self.decoder_forward(latents)

    def model(self, x: torch.Tensor):
        """Pyro model; $$P(x|z)P(z)$$."""
        pyro.module("decoder", self.decoder)
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            z_what_loc = x.new_zeros(batch_size, self.n_objects + 1, self.z_what_size)
            z_what_scale = torch.ones_like(z_what_loc)

            z_where_loc = x.new_full(
                (batch_size, self.n_ssd_features, 4), fill_value=self.z_where_loc_prior
            )
            z_where_scale = torch.full_like(
                z_where_loc, fill_value=self.z_where_scale_prior
            )

            z_present_p = x.new_full(
                (batch_size, self.n_ssd_features, 1),
                fill_value=self.z_present_p_prior,
            )

            z_depth_loc = x.new_zeros((batch_size, self.n_objects, 1))
            z_depth_scale = torch.ones_like(z_depth_loc)

            with poutine.scale(scale=self.what_coef):
                z_what = pyro.sample(
                    "z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2)
                )

            with poutine.scale(scale=self.where_coef):
                z_where = pyro.sample(
                    "z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2)
                )

            with poutine.scale(scale=self.present_coef):
                z_present = pyro.sample(
                    "z_present", dist.Bernoulli(z_present_p).to_event(2)
                )

            with poutine.scale(scale=self.depth_coef):
                z_depth = pyro.sample(
                    "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
                )

            output = self.decoder((z_what, z_where, z_present, z_depth))

            with poutine.scale(scale=self.rec_coef):
                pyro.sample(
                    "obs",
                    dist.Bernoulli(output.permute(0, 2, 3, 1)).to_event(3),
                    obs=denormalize(
                        x.permute(0, 2, 3, 1),
                        pixel_mean=self.pixel_means,
                        pixel_std=self.pixel_stds,
                    ),
                )

    def guide(self, x: torch.Tensor):
        """Pyro guide; $$q(z|x)$$."""
        pyro.module("encoder", self.encoder)
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            (
                (z_what_loc, z_what_scale),
                z_where_loc,
                z_present_p,
                (z_depth_loc, z_depth_scale),
            ) = self.encoder(x)
            z_where_scale = torch.full_like(z_where_loc, fill_value=self.z_where_scale)

            with poutine.scale(scale=self.what_coef):
                pyro.sample("z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2))

            with poutine.scale(scale=self.where_coef):
                pyro.sample(
                    "z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2)
                )

            with poutine.scale(scale=self.present_coef):
                pyro.sample("z_present", dist.Bernoulli(z_present_p).to_event(2))

            with poutine.scale(scale=self.depth_coef):
                pyro.sample(
                    "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
                )

    def filtered_parameters(
        self,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        recurse: bool = True,
    ) -> Iterator[nn.Parameter]:
        """Get filtered SSDIR parameters for the optimizer.
        :param include: parameter name part to include
        :param exclude: parameter name part to exclude
        :param recurse: iterate recursively through model parameters
        :return: iterator of filtered parameters
        """
        for name, param in self.named_parameters(recurse=recurse):
            if include is not None and include not in name:
                continue
            if exclude is not None and exclude in name:
                continue
            yield param

    def get_inference_visualization(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        reconstruction: torch.Tensor,
        z_where: torch.Tensor,
        objects: torch.Tensor,
    ) -> Tuple[PILImage.Image, Dict[str, Any]]:
        """Create model inference visualization."""
        denormalized_image = denormalize(
            image.permute(1, 2, 0),
            pixel_mean=self.pixel_means,
            pixel_std=self.pixel_stds,
        )
        vis_image = PILImage.fromarray(
            (denormalized_image.cpu().numpy() * 255).astype(np.uint8)
        )
        vis_reconstruction = PILImage.fromarray(
            (reconstruction.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        vis_objects = objects[: self.n_visualize_objects].squeeze(1)
        inference_image = PILImage.new(
            "RGB",
            (
                vis_image.width
                + vis_objects.shape[0] * vis_objects.shape[-1]
                + vis_reconstruction.width,
                max(vis_image.height, vis_reconstruction.height, vis_objects.shape[-2]),
            ),
            "white",
        )
        inference_image.paste(vis_image, (0, 0))

        output = vis_objects.new_zeros(vis_objects.shape[1:])
        for idx, obj in enumerate(vis_objects):
            filtered_obj = obj * torch.where(output == 0, 1.0, 0.3)
            output += filtered_obj
            obj_image = PILImage.fromarray(
                (filtered_obj.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            inference_image.paste(
                obj_image, (vis_image.width + idx * obj_image.width, 0)
            )

        inference_image.paste(
            vis_reconstruction,
            (
                vis_image.width + vis_objects.shape[0] * vis_objects.shape[-1],
                0,
            ),
        )
        wandb_inference_boxes = {
            "gt": {
                "box_data": [
                    {
                        "position": {
                            "middle": (
                                box[0].int().item(),
                                box[1].int().item(),
                            ),
                            "width": box[2].int().item(),
                            "height": box[3].int().item(),
                        },
                        "domain": "pixel",
                        "box_caption": "gt_object",
                        "class_id": 1,
                    }
                    for box in boxes * self.image_size[-1]
                ],
                "class_labels": {1: "object"},
            },
            "where": {
                "box_data": [
                    {
                        "position": {
                            "middle": (
                                box[0].int().item()
                                + vis_image.width
                                + vis_objects.shape[0] * vis_objects.shape[-1],
                                box[1].int().item(),
                            ),
                            "width": box[2].int().item(),
                            "height": box[3].int().item(),
                        },
                        "domain": "pixel",
                        "box_caption": "object",
                        "class_id": 1,
                    }
                    for box in z_where * self.image_size[-1]
                ],
                "class_labels": {1: "object"},
            },
        }
        return inference_image, wandb_inference_boxes

    def common_run_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_nb: int,
        stage: str,
    ):
        """Common model running step for training and validation."""
        criterion = Trace_ELBO().differentiable_loss

        images, boxes, _ = batch
        loss = criterion(self.model, self.guide, images)

        self.log(f"{stage}_loss", loss, prog_bar=False, logger=True)

        for site, site_loss in per_site_loss(self.model, self.guide, images).items():
            self.log(f"{stage}_loss_{site}", site_loss, prog_bar=False, logger=True)

        vis_images = images.detach()
        vis_boxes = boxes.detach()
        if self.visualize_latents and (
            self.global_step % self.visualize_latents_freq == 0 or batch_nb == 0
        ):
            with torch.no_grad():
                (
                    (z_what_loc, z_what_scale),
                    z_where_loc,
                    z_present_p,
                    (z_depth_loc, z_depth_scale),
                ) = self.encoder(vis_images)
            latents_dict = {
                "z_what_loc": z_what_loc,
                "z_what_scale": z_what_scale,
                "z_where_loc": z_where_loc,
                "z_present_p": z_present_p,
                "z_depth_loc": z_depth_loc,
                "z_depth_scale": z_depth_scale,
            }
            for latent_name, latent in latents_dict.items():
                self.logger.experiment.log(
                    {f"{stage}_{latent_name}": wandb.Histogram(latent.cpu())},
                    step=self.global_step,
                )
        if self.visualize_inference and (
            self.global_step % self.visualize_inference_freq == 0 or batch_nb == 0
        ):
            with torch.no_grad():
                latents = self.encoder_forward(vis_images)
                latents_bg = self.decoder.append_bg_latents(latents)
                z_what, z_where, z_present, z_depth = self.decoder.pad_latents(
                    latents_bg
                )
                reconstructions = self.decoder_forward(latents)
                objects, depths = self.decoder.reconstruct_objects(
                    z_what[0].unsqueeze(0),
                    z_where[0].unsqueeze(0),
                    z_present[0].unsqueeze(0),
                    z_depth[0].unsqueeze(0),
                )
                _, sort_index = torch.sort(depths, dim=1, descending=True)
                sorted_objects = objects.gather(
                    dim=1,
                    index=sort_index.view(1, -1, 1, 1, 1).expand_as(objects),
                )
                filtered_z_where = z_where[0][
                    (z_present[0] == 1).expand_as(z_where[0])
                ].view(-1, z_where.shape[-1])

            (
                inference_image,
                wandb_inference_boxes,
            ) = self.get_inference_visualization(
                image=vis_images[0],
                boxes=vis_boxes[0],
                reconstruction=reconstructions[0],
                z_where=filtered_z_where,
                objects=sorted_objects[0],
            )
            self.logger.experiment.log(
                {
                    f"{stage}_inference_image": wandb.Image(
                        inference_image,
                        boxes=wandb_inference_boxes,
                        caption="model inference",
                    )
                },
                step=self.global_step,
            )

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_nb: int
    ):
        """Step for training."""
        return self.common_run_step(batch, batch_nb, stage="train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_nb: int
    ):
        """Step for validation."""
        return self.common_run_step(batch, batch_nb, stage="val")

    def configure_optimizers(self):
        """Configure training optimizer."""
        warm_restart_steps = int(
            len(self.train_dataloader()) * self.warm_restart_epochs
        )

        if self.ssd_lr_multiplier != 1:
            optimizer_params = [
                {"params": self.filtered_parameters(exclude="ssd")},
                {
                    "params": self.filtered_parameters(include="ssd"),
                    "lr": self.lr * self.ssd_lr_multiplier,
                },
            ]
        else:
            optimizer_params = self.parameters()

        optimizer = torch.optim.Adam(optimizer_params, lr=self.lr)
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=warm_restart_steps,
            T_mult=self.warm_restart_len_mult,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def train_dataloader(self) -> DataLoader:
        """Prepare train dataloader."""
        data_transform = TrainDataTransform(
            image_size=self.image_size,
            pixel_mean=self.pixel_means,
            pixel_std=self.pixel_stds,
            flip=self.flip_train,
            augment_colors=self.augment_colors_train,
        )
        dataset = self.dataset(
            self.data_dir,
            data_transform=data_transform,
            target_transform=corner_to_center_target_transform,
            subset="train",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Prepare validation dataloader."""
        data_transform = DataTransform(
            image_size=self.image_size,
            pixel_mean=self.pixel_means,
            pixel_std=self.pixel_stds,
        )
        dataset = self.dataset(
            self.data_dir,
            data_transform=data_transform,
            target_transform=corner_to_center_target_transform,
            subset="test",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
