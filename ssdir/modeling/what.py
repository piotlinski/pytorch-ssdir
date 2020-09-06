"""$$z_{what}$$ encoder and decoder."""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional


class WhatEncoder(nn.Module):
    """Module encoding input image to what latent distribution params."""

    def __init__(self, z_what_size: int, feature_channels: List[int]):
        self.h_size = z_what_size
        self.feature_channels = feature_channels
        self.mean_encoders = self._build_what_encoders()
        self.std_encoders = self._build_what_encoders()

    def _build_what_encoders(self) -> nn.ModuleList:
        """Build conv layers list for encoding backbone output."""
        layers = [
            nn.Conv2D(
                in_channels=channels,
                out_channels=self.h_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for channels in self.feature_channels
        ]
        return nn.ModuleList(layers)

    def forward(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        means_stds = []
        batch_size = features[0].shape[0]
        for feature, mean_encoder, std_encoder in zip(
            features, self.mean_encoders, self.std_encoders
        ):
            means = (
                mean_encoder(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.h_size)
            )
            stds = (
                functional.softplus(std_encoder(feature))
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.h_size)
            )
            means_stds.append((means, stds))
        return tuple(means_stds)


class WhatDecoder(nn.Module):
    """Module decoding latent what code to individual images."""

    def __init__(self, z_what_size: int, output_sizes: Tuple[int, ...]):
        self.h_size = z_what_size
        self.output_sizes = output_sizes
        self.decoder = self._build_what_decoder()

    def _build_what_decoder(self) -> nn.ModuleList:
        """Build sequential for decoding latent to an image."""
        layers = [
            nn.ConvTranspose2d(self.h_size, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        ]
        return nn.Sequential(layers)

    def forward(self, z_whats: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        decoded = []
        for z_what in z_whats:
            decoded.append(self.decoder(z_what))
        return tuple(decoded)
