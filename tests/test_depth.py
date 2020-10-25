"""Test depth modules."""
import pytest
import torch

from ssdir.modeling.depth import DepthEncoder


@pytest.mark.parametrize("feature_channels", [[5], [3, 7], [2, 4, 8]])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_depth_encoder_dimensions(feature_channels, batch_size, grid_size):
    """Verify DepthEncoder output dimensions."""
    inputs = [
        torch.rand(batch_size, feature_channel, grid_size, grid_size)
        for feature_channel in feature_channels
    ]
    encoder = DepthEncoder(feature_channels=feature_channels)
    locs, scales = encoder(inputs)
    assert (
        locs.shape
        == scales.shape
        == (batch_size, len(feature_channels) * grid_size ** 2 + 1, 1)
    )


@pytest.mark.parametrize("feature_channels", [[5], [3, 7], [2, 4, 8]])
@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_depth_encoder_background_latent(feature_channels, grid_size):
    """Verify WhereEncoder appending background latent."""
    inputs = [
        torch.rand(1, feature_channel, grid_size, grid_size)
        for feature_channel in feature_channels
    ]
    encoder = DepthEncoder(feature_channels=feature_channels)
    locs, scales = encoder(inputs)
    _, idx = torch.min(locs, dim=1)
    assert idx == locs.shape[1] - 1
    assert scales[:, idx] == encoder.bg_scale


def test_depth_encoder_dtype():
    """Verify DepthEncoder output types."""
    inputs = [torch.rand(3, 4, 5, 5)]
    encoder = DepthEncoder(feature_channels=[4])
    locs, scales = encoder(inputs)
    assert locs.dtype == torch.float
    assert scales.dtype == torch.float
    assert (scales > 0).all()
