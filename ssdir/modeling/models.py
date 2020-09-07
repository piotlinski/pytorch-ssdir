"""SSDIR encoder, decoder, model and guide declarations."""
from typing import Tuple, Union

import torch
import torch.nn as nn
from ssd.modeling.model import SSD

from ssdir.modeling import DepthEncoder, PresentEncoder, WhatEncoder, WhereEncoder


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
