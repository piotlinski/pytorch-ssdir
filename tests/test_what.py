"""Test what tools."""
import pytest
import torch

from ssdir.modeling.what import WhatEncoder


@pytest.mark.parametrize("z_what_size", [8, 16, 64])
@pytest.mark.parametrize("feature_channels", [[5], [16, 32], [2, 4, 8]])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_what_encoder_dimensions(z_what_size, feature_channels, batch_size, grid_size):
    """Verify if what encoder z dimensions."""
    inputs = [
        torch.rand(batch_size, feature_channel, grid_size, grid_size)
        for feature_channel in feature_channels
    ]
    encoder = WhatEncoder(z_what_size=z_what_size, feature_channels=feature_channels)
    means, stds = encoder(inputs)
    print(means.shape)
    assert (
        means.shape
        == stds.shape
        == (batch_size, len(feature_channels) * grid_size * grid_size, z_what_size)
    )
