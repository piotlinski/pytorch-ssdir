"""SSDIR encoder, decoder, model and guide declarations."""
from typing import Tuple, Union

import torch
import torch.nn as nn
from ssd.modeling.model import SSD, CfgNode

from ssdir.modeling import (
    DepthEncoder,
    PresentEncoder,
    WhatDecoder,
    WhatEncoder,
    WhereEncoder,
    WhereTransformer,
)


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
        .. (z_what (mean & std), z_where, z_present, z_depth (mean & std))
        """
        features = self.ssd.backbone(images)
        z_what_mean, z_what_std = self.what_enc(features)
        z_where = self.where_enc(features)
        z_present = self.present_enc(features)
        z_depth_mean, z_depth_std = self.depth_enc(features)
        return (
            (z_what_mean, z_what_std),
            z_where,
            z_present,
            (z_depth_mean, z_depth_std),
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

    def __init__(self, ssd: SSD, z_what_size: int = 64, max_objects: int = 10):
        super().__init__()
        ssd_config = ssd.predictor.config
        self.what_dec = WhatDecoder(z_what_size=z_what_size)
        self.where_stn = WhereTransformer(image_size=ssd_config.DATA.SHAPE[0])
        self.indices = self.reconstruction_indices(ssd_config)
        self.max_objects = max_objects

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
                        size=(boxes_per_loc,), fill_value=img_idx, dtype=torch.long,
                    )
                )
            last_img_idx = img_idx + 1
        return torch.cat(indices, dim=0)

    def _merge_images(self, images: torch.Tensor) -> torch.Tensor:
        """Combine decoded images into one."""
        combined = torch.zeros(3, self.where_stn.image_size, self.where_stn.image_size)
        for image in images:
            combined_zero = combined == 0.0
            image_nonzero = image != 0.0
            mask = combined_zero & image_nonzero
            combined[mask] = image[mask]
        return combined

    def _render(
        self, z_what: torch.Tensor, z_where: torch.Tensor, z_present: torch.Tensor
    ) -> torch.Tensor:
        """Render single image from batch."""
        images = []
        for what, where, present in zip(z_what, z_where, z_present):
            present_mask = present == 1
            what_size = what.shape[-1]
            where_size = where.shape[-1]
            what = what[present_mask.expand_as(what)].view(-1, what_size)
            where = where[present_mask.expand_as(where)].view(-1, where_size)
            decoded_images = self.what_dec(what)
            transformed_images = self.where_stn(decoded_images, where)
            images.append(self._merge_images(transformed_images))
        return torch.stack(images, dim=0)

    def forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """ Takes latent variables tensors tuple (z_what, z_where, z_present, z_depth)
        .. and outputs reconstructed images batch
        .. (batch_size x channels x image_size x image_size)
        """
        z_what, z_where, z_present, z_depth = latents
        # repeat rows to match z_where and z_present
        z_what = torch.index_select(input=z_what, dim=1, index=self.indices)
        z_depth = torch.index_select(input=z_depth, dim=1, index=self.indices)
        _, sort_index = torch.sort(z_depth, dim=1, descending=True)
        sorted_z_what = z_what.gather(dim=1, index=sort_index.expand_as(z_what))[
            :, : self.max_objects, ...
        ]
        sorted_z_where = z_where.gather(dim=1, index=sort_index.expand_as(z_where))[
            :, : self.max_objects, ...
        ]
        sorted_z_present = z_present.gather(
            dim=1, index=sort_index.expand_as(z_present)
        )[:, : self.max_objects, ...]
        return self._render(
            z_what=sorted_z_what, z_where=sorted_z_where, z_present=sorted_z_present
        )
