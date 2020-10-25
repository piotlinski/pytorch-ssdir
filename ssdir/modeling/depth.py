"""$$z_{depth}$$ encoder"""
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

warnings.filterwarnings(
    "ignore",
    message=(
        "depth_enc.bg_scale was not registered in the param store "
        "because requires_grad=False"
    ),
)


class DepthEncoder(nn.Module):
    """Module encoding input image features to depth latent distribution params."""

    def __init__(self, feature_channels: List[int]):
        super().__init__()
        self.feature_channels = feature_channels
        self.loc_encoders = self._build_depth_encoders()
        self.scale_encoders = self._build_depth_encoders()
        self.bg_scale = nn.Parameter(torch.tensor([0.2]), requires_grad=False)

    def _build_depth_encoders(self) -> nn.ModuleList:
        """Build conv layers list for encoding backbone output."""
        layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for channels in self.feature_channels
        ]
        return nn.ModuleList(layers)

    def forward(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs loc and scale tensors
        .. (batch_size x sum_features(grid*grid) x 1)
        """
        locs = []
        scales = []
        batch_size = features[0].shape[0]
        for feature, loc_encoder, scale_encoder in zip(
            features, self.loc_encoders, self.scale_encoders
        ):
            locs.append(
                loc_encoder(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 1)
            )
            scales.append(
                functional.softplus(scale_encoder(feature))
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 1)
            )

        locs = torch.cat(locs, dim=1)
        scales = torch.cat(scales, dim=1)

        bg_locs, _ = torch.min(locs, dim=1)
        bg_locs = bg_locs.unsqueeze(1) - 1e-3
        bg_scales = self.bg_scale.expand(batch_size, 1, 1)

        return torch.cat((locs, bg_locs), dim=1), torch.cat((scales, bg_scales), dim=1)
