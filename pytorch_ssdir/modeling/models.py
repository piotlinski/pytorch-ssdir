"""SSDIR encoder, decoder, model and guide declarations."""
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
from pytorch_ssd.args import str2bool
from pytorch_ssd.data.datasets import datasets
from pytorch_ssd.data.transforms import DataTransform, TrainDataTransform
from pytorch_ssd.modeling.model import SSD
from pytorch_ssd.modeling.visualize import denormalize
from torch.utils.data.dataloader import DataLoader

from pytorch_ssdir.args import parse_kwargs
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

optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
lr_schedulers = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


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
        z_what_hidden: int = 2,
        z_what_scale_const: Optional[float] = None,
        z_depth_scale_const: Optional[float] = None,
        z_present_eps: float = 1e-3,
        train_what: bool = True,
        train_where: bool = True,
        train_present: bool = True,
        train_depth: bool = True,
        train_backbone: bool = True,
        train_backbone_layers: int = -1,
        clone_backbone: bool = False,
    ):
        super().__init__()
        self.ssd_backbone = ssd.backbone.requires_grad_(train_backbone)
        self.clone_backbone = clone_backbone
        if self.clone_backbone:
            self.ssd_backbone_cloned = deepcopy(self.ssd_backbone).requires_grad_(True)
        if train_backbone_layers >= 0 and train_backbone:
            for module in list(self.ssd_backbone.children())[train_backbone_layers:][
                ::-1
            ]:
                module.requires_grad_(False)
        self.z_present_eps = z_present_eps
        self.what_enc = WhatEncoder(
            z_what_size=z_what_size,
            n_hidden=z_what_hidden,
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

        self.register_buffer(
            "indices",
            self.latents_indices(
                feature_maps=ssd.backbone.feature_maps,
                boxes_per_loc=ssd.backbone.boxes_per_loc,
            ),
        )
        self.register_buffer("empty_loc", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("empty_scale", torch.tensor(1.0, dtype=torch.float))

    @staticmethod
    def latents_indices(
        feature_maps: List[int], boxes_per_loc: List[int]
    ) -> torch.Tensor:
        """Get indices for reconstructing images.

        .. Caters for the difference between z_what, z_depth and z_where, z_present.
        """
        indices = []
        idx = 0
        for feature_map, n_boxes in zip(feature_maps, boxes_per_loc):
            for feature_map_idx in range(feature_map ** 2):
                indices.append(
                    torch.full(size=(n_boxes,), fill_value=idx, dtype=torch.float)
                )
                idx += 1
        return torch.cat(indices, dim=0)

    def pad_latents(
        self,
        latents: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Pad latents according to Encoder's settings."""
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = latents
        # repeat rows to match z_where and z_present
        indices = self.indices.long()
        what_indices = torch.hstack((indices, indices.max() + 1))  # consider background
        z_what_loc = z_what_loc.index_select(dim=1, index=what_indices)
        z_what_scale = z_what_scale.index_select(dim=1, index=what_indices)
        z_depth_loc = z_depth_loc.index_select(dim=1, index=indices)
        z_depth_scale = z_depth_scale.index_select(dim=1, index=indices)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )

    def reset_non_present(
        self,
        latents: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Reset latents, whose z_present is 0.

        .. note: this will set all "non-present" locs to 0. and scales to 1.
        """
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = latents
        present_mask = torch.gt(z_present, self.z_present_eps)
        what_present_mask = torch.hstack(  # consider background
            (
                present_mask,
                present_mask.new_full((1,), fill_value=True).expand(
                    present_mask.shape[0], 1, 1
                ),
            )
        )
        z_what_loc = torch.where(what_present_mask, z_what_loc, self.empty_loc)
        z_what_scale = torch.where(what_present_mask, z_what_scale, self.empty_scale)
        z_where = torch.where(present_mask, z_where, self.empty_loc)
        z_depth_loc = torch.where(present_mask, z_depth_loc, self.empty_loc)
        z_depth_scale = torch.where(present_mask, z_depth_scale, self.empty_scale)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Takes images tensors (batch_size x channels x image_size x image_size)
        .. and outputs latent representation tuple
        .. (z_what (loc & scale), z_where, z_present, z_depth (loc & scale))
        """
        where_present_features = self.ssd_backbone(images)
        if self.clone_backbone:
            what_depth_features = self.ssd_backbone_cloned(images)
        else:
            what_depth_features = where_present_features
        z_where = self.where_enc(where_present_features)
        z_present = self.present_enc(where_present_features)
        z_what_loc, z_what_scale = self.what_enc(what_depth_features)
        z_depth_loc, z_depth_scale = self.depth_enc(what_depth_features)
        latents = (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )
        padded_latents = self.pad_latents(latents)
        return self.reset_non_present(padded_latents)


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
        train_what: bool = True,
    ):
        super().__init__()
        self.what_dec = WhatDecoder(z_what_size=z_what_size).requires_grad_(train_what)
        self.where_stn = WhereTransformer(image_size=ssd.image_size[0])
        self.drop = drop_empty
        self.register_buffer("bg_depth", torch.zeros(1))
        self.register_buffer("bg_present", torch.ones(1))
        self.register_buffer("bg_where", torch.tensor([0.5, 0.5, 1.0, 1.0]))
        self.pixel_means = ssd.backbone.PIXEL_MEANS
        self.pixel_stds = ssd.backbone.PIXEL_STDS

    @staticmethod
    def pad_indices(n_present: torch.Tensor) -> torch.Tensor:
        """Using number of objects in chunks create indices
        .. so that every chunk is padded to the same dimension.

        .. Assumes index 0 refers to "starter" (empty) object
        .. Puts background index at the beginning of indices arange

        :param n_present: number of objects in each chunk
        :return: indices for padding tensors
        """
        end_idx = 1
        max_objects = torch.max(n_present)
        indices = []
        for chunk_objects in n_present:
            start_idx = end_idx
            end_idx = end_idx + chunk_objects
            idx_range = torch.cat(
                (
                    torch.tensor(
                        [end_idx - 1], dtype=torch.long, device=n_present.device
                    ),
                    torch.arange(
                        start=start_idx,
                        end=end_idx - 1,
                        dtype=torch.long,
                        device=n_present.device,
                    ),
                ),
                dim=0,
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
    def merge_reconstructions(
        reconstructions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine decoder images into one by weighted sum."""
        weighted_images = reconstructions * functional.softmax(weights, dim=1).view(
            *weights.shape[:2], 1, 1, 1
        )
        return torch.sum(weighted_images, dim=1)

    @staticmethod
    def fill_background(
        merged: torch.Tensor, backgrounds: torch.Tensor
    ) -> torch.Tensor:
        """Fill merged images background with background reconstruction."""
        mask = torch.where(merged < 1e-3, 1.0, 0.0)
        return merged + backgrounds * mask

    def reconstruct_objects(
        self,
        z_what: torch.Tensor,
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render reconstructions and their depths from batch."""
        batch_size = z_what.shape[0]
        z_depth = torch.cat(  # append background depth
            (z_depth, self.bg_depth.expand(batch_size, 1, 1)), dim=1
        )
        z_present = torch.cat(  # append background present
            (z_present, self.bg_present.expand(batch_size, 1, 1)), dim=1
        )
        z_where = torch.cat(  # append background where
            (z_where, self.bg_where.expand(batch_size, 1, 4)), dim=1
        )
        z_what_shape = z_what.shape
        z_where_shape = z_where.shape
        z_depth_shape = z_depth.shape
        if self.drop:
            present_mask = torch.eq(z_present, 1)
            n_present = torch.sum(present_mask, dim=1).squeeze(-1)
            z_what = z_what[present_mask.expand_as(z_what)].view(-1, z_what_shape[-1])
            z_where = z_where[present_mask.expand_as(z_where)].view(
                -1, z_where_shape[-1]
            )
            z_depth = z_depth[present_mask.expand_as(z_depth)].view(
                -1, z_depth_shape[-1]
            )
        z_what_flat = z_what.view(-1, z_what_shape[-1])
        z_where_flat = z_where.view(-1, z_where_shape[-1])
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
            depths = z_depth.where(
                z_present == 1.0, z_depth.new_full((1,), fill_value=-float("inf"))
            )
            reconstructions = torch.cat(
                (reconstructions[:, [-1]], reconstructions[:, :-1]), dim=1
            )
            depths = torch.cat((depths[:, [-1]], depths[:, :-1]), dim=1)
        return reconstructions, depths

    def forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Takes latent variables tensors tuple (z_what, z_where, z_present, z_depth)
        .. and outputs reconstructed images batch
        .. (batch_size x channels x image_size x image_size)
        """
        z_what, z_where, z_present, z_depth = latents
        # render reconstructions
        reconstructions, depths = self.reconstruct_objects(
            z_what, z_where, z_present, z_depth
        )
        # merge reconstructions
        objects, object_weights = reconstructions[:, 1:], depths[:, 1:]
        merged = self.merge_reconstructions(
            reconstructions=objects, weights=object_weights
        )
        output = self.fill_background(merged=merged, backgrounds=reconstructions[:, 0])
        return output


class SSDIR(pl.LightningModule):
    """Single-Shot Detect, Infer, Repeat."""

    def __init__(
        self,
        ssd_model: SSD,
        optimizer: str = "Adam",
        optimizer_kwargs: Optional[List[Tuple[str, Any]]] = None,
        learning_rate: float = 1e-3,
        ssd_lr_multiplier: float = 1.0,
        lr_scheduler: str = "",
        lr_scheduler_kwargs: Optional[List[Tuple[str, Any]]] = None,
        auto_lr_find: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        z_what_size: int = 64,
        z_what_hidden: int = 2,
        z_present_p_prior: float = 0.01,
        z_where_pos_loc_prior: float = 0.5,
        z_where_size_loc_prior: float = 0.2,
        z_where_pos_scale_prior: float = 1.0,
        z_where_size_scale_prior: float = 0.2,
        drop: bool = True,
        z_where_scale_const: float = 0.05,
        z_what_scale_const: Optional[float] = None,
        z_depth_scale_const: Optional[float] = None,
        normalize_elbo: bool = False,
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
        train_backbone_layers: int = -1,
        clone_backbone: bool = False,
        visualize_inference: bool = True,
        visualize_inference_freq: int = 500,
        n_visualize_objects: int = 10,
        visualize_latents: bool = True,
        visualize_latents_freq: int = 5,
        **_kwargs,
    ):
        """
        :param ssd_model: trained SSD to use as backbone
        :param optimizer: optimizer name
        :param learning_rate: learning rate
        :param optimizer_kwargs: optimizer argumnets dictionary
        :param ssd_lr_multiplier: ssd learning rate multiplier (learning rate * mult)
        :param lr_scheduler: LR scheduler name
        :param lr_scheduler_kwargs: LR scheduler arguments dictionary
        :param auto_lr_find: perform auto lr finding
        :param batch_size: mini-batch size for training
        :param num_workers: number of workers for dataloader
        :param pin_memory: pin memory for training
        :param z_what_size: latent what size
        :param z_what_hidden: number of extra hidden layers for what encoder
        :param z_present_p_prior: present prob prior
        :param z_where_pos_loc_prior: prior z_where loc for bbox position
        :param z_where_size_loc_prior: prior z_where loc for bbox size
        :param z_where_pos_scale_prior: prior z_where scale for bbox position
        :param z_where_size_scale_prior: prior z_where scale for bbox size
        :param drop: drop empty objects' latents
        :param z_where_scale_const: z_where scale used in inference
        :param z_what_scale_const: fixed z_what scale (if None - use NN to model)
        :param z_depth_scale_const: fixed z_depth scale (if None - use NN to model)
        :param normalize_elbo: normalize elbo components by tenors' numels
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
        :param train_backbone_layers: n layers to train in the backbone (neg for all)
        :param clone_backbone: clone backbone for depth and what encoders
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
            z_what_hidden=z_what_hidden,
            z_what_scale_const=z_what_scale_const,
            z_depth_scale_const=z_depth_scale_const,
            train_what=train_what,
            train_where=train_where,
            train_present=train_present,
            train_depth=train_depth,
            train_backbone=train_backbone,
            train_backbone_layers=train_backbone_layers,
            clone_backbone=clone_backbone,
        )
        self.decoder = Decoder(
            ssd=ssd_model,
            z_what_size=z_what_size,
            drop_empty=drop,
            train_what=train_what,
        )

        self.optimizer = optimizers[optimizer]
        if optimizer_kwargs is None:
            optimizer_kwargs = []
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.lr = learning_rate
        self.ssd_lr_multiplier = ssd_lr_multiplier
        self.lr_scheduler = lr_schedulers.get(lr_scheduler)
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = []
        self.lr_scheduler_kwargs = dict(lr_scheduler_kwargs)
        self.auto_lr_find = auto_lr_find
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pixel_means = ssd_model.backbone.PIXEL_MEANS
        self.pixel_stds = ssd_model.backbone.PIXEL_STDS
        self._mse_train = pl.metrics.MeanSquaredError()
        self._mse_val = pl.metrics.MeanSquaredError()
        self.mse = {"train": self._mse_train, "val": self._mse_val}

        self.image_size = ssd_model.image_size
        self.flip_train = ssd_model.flip_train
        self.augment_colors_train = ssd_model.augment_colors_train
        self.dataset = ssd_model.dataset
        self.data_dir = ssd_model.data_dir

        self.z_what_size = z_what_size
        self.z_present_p_prior = z_present_p_prior
        self.z_where_loc_prior = [
            z_where_pos_loc_prior,
            z_where_pos_loc_prior,
            z_where_size_loc_prior,
            z_where_size_loc_prior,
        ]
        self.z_where_scale_prior = [
            z_where_pos_scale_prior,
            z_where_pos_scale_prior,
            z_where_size_scale_prior,
            z_where_size_scale_prior,
        ]
        self.z_where_scale_const = z_where_scale_const
        self.drop = drop

        self.normalize_elbo = normalize_elbo
        self._what_coef = what_coef
        self._where_coef = where_coef
        self._present_coef = present_coef
        self._depth_coef = depth_coef
        self._rec_coef = rec_coef

        self.visualize_inference = visualize_inference
        self.visualize_inference_freq = visualize_inference_freq
        self.n_visualize_objects = n_visualize_objects
        self.visualize_latents = visualize_latents
        self.visualize_latents_freq = visualize_latents_freq

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
            "--dataset_name",
            type=str,
            default="MNIST",
            help=f"Used dataset name. Available: {list(datasets.keys())}",
        )
        parser.add_argument(
            "--data_dir", type=str, default="data", help="Dataset files directory"
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help=f"Used optimizer. Available: {list(optimizers.keys())}",
        )
        parser.add_argument(
            "--optimizer_kwargs",
            type=parse_kwargs,
            default=[],
            nargs="*",
            help="Optimizer kwargs in the form of key=value separated by spaces",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="Learning rate used for training the model",
        )
        parser.add_argument(
            "--ssd_lr_multiplier",
            type=float,
            default=1.0,
            help="Learning rate multiplier for training SSD backbone",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="None",
            help=(
                "Used LR scheduler. "
                f"Available: {list(lr_schedulers.keys())}; default: None"
            ),
        )
        parser.add_argument(
            "--lr_scheduler_kwargs",
            type=parse_kwargs,
            default=[],
            nargs="*",
            help="LR scheduler kwargs in the form of key=value separated by spaces",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Mini-batch size used for training the model",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of workers used to load the dataset",
        )
        parser.add_argument(
            "--pin_memory",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Pin data in memory while training",
        )
        parser.add_argument(
            "--z_what_size", type=int, default=64, help="z_what latent size"
        )
        parser.add_argument(
            "--z_what_hidden",
            type=int,
            default=2,
            help="Number of what encoder hidden layers; -1 for backward compatibility",
        )
        parser.add_argument(
            "--z_present_p_prior",
            type=float,
            default=0.01,
            help="z_present probability prior",
        )
        parser.add_argument(
            "--z_where_pos_loc_prior",
            type=float,
            default=0.5,
            help="prior z_where loc for position",
        )
        parser.add_argument(
            "--z_where_size_loc_prior",
            type=float,
            default=0.2,
            help="prior z_where loc for size",
        )
        parser.add_argument(
            "--z_where_pos_scale_prior",
            type=float,
            default=1.0,
            help="prior z_where scale for position",
        )
        parser.add_argument(
            "--z_where_size_scale_prior",
            type=float,
            default=0.2,
            help="prior z_where scale for size",
        )
        parser.add_argument(
            "--drop",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Drop empty objects' latents",
        )
        parser.add_argument(
            "--z_where_scale_const",
            type=float,
            default=0.05,
            help="z_where scale used in inference",
        )
        parser.add_argument(
            "--z_what_scale_const",
            type=float,
            default=None,
            help="constant z_what scale",
        )
        parser.add_argument(
            "--z_depth_scale_const",
            type=float,
            default=None,
            help="constant z_depth scale",
        )
        parser.add_argument(
            "--normalize_elbo",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Normalize elbo components by tenors' numels",
        )
        parser.add_argument(
            "--what_coef",
            type=float,
            default=1.0,
            help="z_what loss component coefficient",
        )
        parser.add_argument(
            "--where_coef",
            type=float,
            default=1.0,
            help="z_where loss component coefficient",
        )
        parser.add_argument(
            "--present_coef",
            type=float,
            default=1.0,
            help="z_present loss component coefficient",
        )
        parser.add_argument(
            "--depth_coef",
            type=float,
            default=1.0,
            help="z_depth loss component coefficient",
        )
        parser.add_argument(
            "--rec_coef",
            type=float,
            default=1.0,
            help="Reconstruction error component coefficient",
        )
        parser.add_argument(
            "--train_what",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train what encoder and decoder",
        )
        parser.add_argument(
            "--train_where",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train where encoder",
        )
        parser.add_argument(
            "--train_present",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train present encoder",
        )
        parser.add_argument(
            "--train_depth",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train depth encoder",
        )
        parser.add_argument(
            "--train_backbone",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train SSD backbone",
        )
        parser.add_argument(
            "--train_backbone_layers",
            type=int,
            default=-1,
            help="Number of final layers to train in the backbone (negative for all)",
        )
        parser.add_argument(
            "--clone_backbone",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Clone SSD backbone for what and depth encoders",
        )
        parser.add_argument(
            "--flip_train",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Flip train images during training",
        )
        parser.add_argument(
            "--augment_colors_train",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Perform random colors augmentation during training",
        )
        parser.add_argument(
            "--visualize_inference",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Log visualizations of model predictions",
        )
        parser.add_argument(
            "--visualize_inference_freq",
            type=int,
            default=500,
            help="How often to perform inference visualization.",
        )
        parser.add_argument(
            "--n_visualize_objects",
            type=int,
            default=5,
            help="Number of objects to visualize",
        )
        parser.add_argument(
            "--visualize_latents",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Log visualizations of model latents",
        )
        parser.add_argument(
            "--visualize_latents_freq",
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

    @property
    def what_coef(self) -> float:
        """Calculate what sampling elbo coefficient."""
        coef = self._what_coef
        if self.normalize_elbo:
            coef /= self.batch_size * (self.n_ssd_features + 1) * self.z_what_size
        return coef

    @property
    def where_coef(self) -> float:
        """Calculate where sampling elbo coefficient."""
        coef = self._where_coef
        if self.normalize_elbo:
            coef /= self.batch_size * (self.n_ssd_features + 1) * 4
        return coef

    @property
    def present_coef(self) -> float:
        """Calculate what sampling elbo coefficient."""
        coef = self._present_coef
        if self.normalize_elbo:
            coef /= self.batch_size * (self.n_ssd_features + 1)
        return coef

    @property
    def depth_coef(self) -> float:
        """Calculate what sampling elbo coefficient."""
        coef = self._depth_coef
        if self.normalize_elbo:
            coef /= self.batch_size * (self.n_ssd_features + 1)
        return coef

    @property
    def rec_coef(self) -> float:
        """Calculate what sampling elbo coefficient."""
        coef = self._rec_coef
        if self.normalize_elbo:
            coef /= self.batch_size * 3 * self.image_size[0] * self.image_size[1]
        return coef

    def model(self, x: torch.Tensor):
        """Pyro model; $$P(x|z)P(z)$$."""
        pyro.module("decoder", self.decoder)
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            z_what_loc = x.new_zeros(  # with background
                batch_size, self.n_ssd_features + 1, self.z_what_size
            )
            z_what_scale = torch.ones_like(z_what_loc)

            z_where_loc = x.new_tensor(self.z_where_loc_prior).expand(
                (batch_size, self.n_ssd_features, 4)
            )
            z_where_scale = x.new_tensor(self.z_where_scale_prior).expand(
                (batch_size, self.n_ssd_features, 4)
            )

            z_present_p = x.new_full(
                (batch_size, self.n_ssd_features, 1),
                fill_value=self.z_present_p_prior,
            )

            z_depth_loc = x.new_zeros((batch_size, self.n_ssd_features, 1))
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
            z_where_scale = torch.full_like(
                z_where_loc, fill_value=self.z_where_scale_const
            )

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

        if (
            self.global_step % self.visualize_inference_freq == 0
            or self.global_step % self.trainer.log_every_n_steps == 0
            or batch_nb == 0
        ):
            with torch.no_grad():
                latents = self.encoder_forward(vis_images)
                reconstructions = self.decoder_forward(latents)

                self.logger.experiment.log(
                    {
                        f"{stage}_mse": self.mse[stage](
                            reconstructions.permute(0, 2, 3, 1),
                            denormalize(
                                vis_images.permute(0, 2, 3, 1),
                                pixel_mean=self.pixel_means,
                                pixel_std=self.pixel_stds,
                            ),
                        )
                    },
                    step=self.global_step,
                )

                if self.visualize_latents:
                    z_what, z_where, z_present, z_depth = latents
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

        optimizer = self.optimizer(
            optimizer_params, lr=self.lr, **self.optimizer_kwargs
        )
        configuration = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(
                optimizer=optimizer, **self.lr_scheduler_kwargs
            )
            configuration["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        return configuration

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
