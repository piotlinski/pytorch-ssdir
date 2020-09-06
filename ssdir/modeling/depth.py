"""$$z_{depth}$$ encoder"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional


class DepthEncoder(nn.Module):
    """Module encoding input image features to depth latent distribution params."""

    def __init__(self, feature_channels: List[int]):
        super().__init__()
        self.feature_channels = feature_channels
        self.mean_encoders = self._build_depth_encoders()
        self.std_encoders = self._build_depth_encoders()

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
        """ Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs mean and std tensors
        .. (batch_size x sum_features(grid*grid) x 1)
        """
        means = []
        stds = []
        batch_size = features[0].shape[0]
        for feature, mean_encoder, std_encoder in zip(
            features, self.mean_encoders, self.std_encoders
        ):
            means.append(
                mean_encoder(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 1)
            )
            stds.append(
                functional.softplus(std_encoder(feature))
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 1)
            )

        means = torch.cat(means, dim=1)
        stds = torch.cat(stds, dim=1)

        return means, stds
