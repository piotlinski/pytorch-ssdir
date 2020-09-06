"""Test what tools."""
import pytest
import torch

from ssdir.modeling.what import WhatDecoder, WhatEncoder


@pytest.mark.parametrize("z_what_size", [8, 16, 64])
@pytest.mark.parametrize("feature_channels", [[5], [16, 32], [2, 4, 8]])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_what_encoder_dimensions(z_what_size, feature_channels, batch_size, grid_size):
    """Verify what encoder z dimensions."""
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


@pytest.mark.parametrize("z_what_size", [8, 16, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("n_objects", [16, 36, 49])
def test_what_decoder_dimensions(z_what_size, batch_size, n_objects):
    """Verify if what decoder output dimensions."""
    z_whats = torch.rand(batch_size, n_objects, z_what_size)
    decoder = WhatDecoder(z_what_size=z_what_size)
    outputs = decoder(z_whats)
    print(outputs.shape)
    assert outputs.shape == (batch_size, n_objects, 3, 64, 64)
