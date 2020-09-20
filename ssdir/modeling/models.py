"""SSDIR encoder, decoder, model and guide declarations."""
from typing import Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as functional
from pyssd.config import CfgNode, get_config
from pyssd.modeling.checkpoint import CheckPointer
from pyssd.modeling.model import SSD

from ssdir.modeling.depth import DepthEncoder
from ssdir.modeling.present import PresentEncoder
from ssdir.modeling.what import WhatDecoder, WhatEncoder
from ssdir.modeling.where import WhereEncoder, WhereTransformer


class Encoder(nn.Module):
    """ Module encoding input image to latent representation.

    .. latent representation consists of:
       - $$z_{what} ~ N(\\mu^{what}, \\sigma^{what})$$
       - $$z_{where} in R^4$$
       - $$z_{present} ~ Bernoulli(p_{present})$$
       - $$z_{depth} ~ N(\\mu_{depth}, \\sigma_{depth})$$
    """

    def __init__(self, ssd: SSD, z_what_size: int = 64):
        super().__init__()
        self.ssd = ssd
        self.what_enc = WhatEncoder(
            z_what_size=z_what_size, feature_channels=ssd.backbone.out_channels
        )
        self.where_enc = WhereEncoder(ssd_box_predictor=ssd.predictor)
        self.present_enc = PresentEncoder(ssd_box_predictor=ssd.predictor)
        self.depth_enc = DepthEncoder(feature_channels=ssd.backbone.out_channels)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], ...]:
        """ Takes images tensors (batch_size x channels x image_size x image_size)
        .. and outputs latent representation tuple
        .. (z_what (loc & scale), z_where, z_present, z_depth (loc & scale))
        """
        features = self.ssd.backbone(images)
        z_what_loc, z_what_scale = self.what_enc(features)
        z_where = self.where_enc(features)
        z_present = self.present_enc(features)
        z_depth_loc, z_depth_scale = self.depth_enc(features)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )


class Decoder(nn.Module):
    """ Module decoding latent representation.

    .. Pipeline:
       - sort z_depth ascending
       - sort $$z_{what}$$, $$z_{where}$$, $$z_{present}$$ accordingly
       - decode $$z_{what}$$ where $$z_{present} = 1$$
       - transform decoded objects according to $$z_{where}$$
       - merge transformed images based on $$z_{depth}$$
    """

    def __init__(self, ssd: SSD, z_what_size: int = 64, empty_obj_const: float = -1000):
        super().__init__()
        ssd_config = ssd.predictor.config
        self.what_dec = WhatDecoder(z_what_size=z_what_size)
        self.where_stn = WhereTransformer(image_size=ssd_config.DATA.SHAPE[0])
        self.indices = nn.Parameter(self.reconstruction_indices(ssd_config))
        self.empty_obj_const = nn.Parameter(
            torch.tensor(empty_obj_const, dtype=torch.float32)
        )

    @staticmethod
    def reconstruction_indices(ssd_config: CfgNode) -> torch.Tensor:
        """Get indices for reconstructing images.

        .. Caters for the difference between z_what, z_depth and z_where, z_present.
        """
        indices = []
        img_idx = last_img_idx = 0
        for feature_map, boxes_per_loc in zip(
            ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_config.DATA.PRIOR.BOXES_PER_LOC
        ):
            for feature_map_idx in range(feature_map ** 2):
                img_idx = last_img_idx + feature_map_idx
                indices.append(
                    torch.full(
                        size=(boxes_per_loc,), fill_value=img_idx, dtype=torch.float
                    )
                )
            last_img_idx = img_idx + 1
        return torch.cat(indices, dim=0)

    @staticmethod
    def merge_images(images: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Combine decoded images into one by weighted sum."""
        weighted_images = images * weights.view(*weights.shape[:2], 1, 1, 1)
        return torch.sum(weighted_images, dim=1)

    def prepare_merge_weights(self, present: torch.Tensor, depth: torch.Tensor):
        """Merge present tensor with depth tensor and apply softmax."""
        weights = depth.where(present == 1.0, self.empty_obj_const)
        return functional.softmax(weights, dim=1)

    def _render(
        self,
        z_what: torch.Tensor,
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Render single image from batch."""
        z_what_flattened = z_what.view(-1, z_what.shape[-1])
        z_where_flattened = z_where.view(-1, z_where.shape[-1])
        decoded_images = self.what_dec(z_what_flattened)
        transformed_images = self.where_stn(decoded_images, z_where_flattened).view(
            -1, z_what.shape[1], 3, self.where_stn.image_size, self.where_stn.image_size
        )
        merge_weights = self.prepare_merge_weights(present=z_present, depth=z_depth)
        reconstructions = self.merge_images(
            images=transformed_images, weights=merge_weights
        )
        return reconstructions

    def forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """ Takes latent variables tensors tuple (z_what, z_where, z_present, z_depth)
        .. and outputs reconstructed images batch
        .. (batch_size x channels x image_size x image_size)
        """
        z_what, z_where, z_present, z_depth = latents
        # repeat rows to match z_where and z_present
        z_what = z_what.index_select(dim=1, index=self.indices.long())
        z_depth = z_depth.index_select(dim=1, index=self.indices.long())
        # create and apply binary mask for present objects
        return self._render(
            z_what=z_what, z_where=z_where, z_present=z_present, z_depth=z_depth
        )


class SSDIR(nn.Module):
    """Single-Shot Detect, Infer, Repeat."""

    def __init__(
        self,
        z_what_size: int = 64,
        ssd_config: Optional[CfgNode] = None,
        ssd_model_file: str = (
            "vgglite_mnist_sc_SSD-VGGLite_MultiscaleMNIST-0015-09375.pth"
        ),
        z_where_scale_eps: float = 1e-5,
        z_present_p_prior: float = 0.01,
    ):
        super().__init__()
        if ssd_config is None:
            ssd_config = get_config()
        ssd_model = SSD(config=ssd_config)
        ssd_checkpointer = CheckPointer(config=ssd_config, model=ssd_model)
        ssd_checkpointer.load(filename=ssd_model_file)

        self.z_what_size = z_what_size
        self.n_objects = sum(
            features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS
        )
        self.n_ssd_features = sum(
            boxes * features ** 2
            for features, boxes in zip(
                ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_config.DATA.PRIOR.BOXES_PER_LOC
            )
        )
        self.z_where_scale_eps = z_where_scale_eps
        self.z_present_p_prior = z_present_p_prior

        self.encoder = Encoder(ssd=ssd_model, z_what_size=z_what_size)
        self.decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size)

    def encoder_forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Perform forward pass through encoder network."""
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = self.encoder(inputs)
        z_what = dist.Normal(z_what_loc, z_what_scale).sample()
        z_present = dist.Bernoulli(z_present).sample().long()
        z_depth = dist.Normal(z_depth_loc, z_depth_scale).sample()
        return z_what, z_where, z_present, z_depth

    def decoder_forward(self, latents: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        outputs = self.decoder(latents)
        return outputs

    def model(self, x: torch.Tensor):
        """Pyro model; $$P(x|z)P(z)$$."""
        pyro.module("decoder", self.decoder)
        batch_size = x.shape[0]
        with pyro.plate("data"):
            z_what_loc = torch.zeros(
                (batch_size, self.n_objects, self.z_what_size), device=x.device
            )
            z_what_scale = torch.ones_like(z_what_loc)

            z_where_loc = torch.zeros(
                (batch_size, self.n_ssd_features, 4), device=x.device
            )
            z_where_scale = torch.full_like(
                z_where_loc, fill_value=self.z_where_scale_eps
            )

            z_present_p = torch.full(
                (batch_size, self.n_ssd_features, 1),
                fill_value=self.z_present_p_prior,
                dtype=torch.float,
                device=x.device,
            )

            z_depth_loc = torch.zeros((batch_size, self.n_objects, 1), device=x.device)
            z_depth_scale = torch.ones_like(z_depth_loc)

            z_what = pyro.sample(
                "z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2)
            )
            z_where = pyro.sample(
                "z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2)
            )
            z_present = pyro.sample(
                "z_present", dist.Bernoulli(z_present_p).to_event(2)
            )
            z_depth = pyro.sample(
                "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
            )

            output = self.decoder((z_what, z_where, z_present, z_depth))

            pyro.sample(
                "obs", dist.Bernoulli(output).to_event(3), obs=x,
            )

    def guide(self, x: torch.Tensor):
        """Pyro guide; $$q(z|x)$$."""
        pyro.module("encoder", self.encoder)
        with pyro.plate("data"):
            (
                (z_what_loc, z_what_scale),
                z_where_loc,
                z_present_p,
                (z_depth_loc, z_depth_scale),
            ) = self.encoder(x)
            z_where_scale = torch.full_like(
                z_where_loc, fill_value=self.z_where_scale_eps
            )

            pyro.sample("z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2))
            pyro.sample("z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2))
            pyro.sample("z_present", dist.Bernoulli(z_present_p).to_event(2))
            pyro.sample("z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2))
