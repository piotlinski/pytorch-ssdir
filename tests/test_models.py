"""Test SSDIR models."""
import pytest
import torch

from ssdir.modeling.models import Decoder, Encoder


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_encoder_dimensions(
    z_what_size, batch_size, ssd_model, ssd_config, n_ssd_features
):
    """Verify encoder output dimensions."""
    inputs = torch.rand(batch_size, 3, 300, 300)
    encoder = Encoder(ssd=ssd_model, z_what_size=z_what_size)
    (
        (z_what_mean, z_what_std),
        z_where,
        z_present,
        (z_depth_mean, z_depth_std),
    ) = encoder(inputs)
    n_objects = sum(features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS)
    assert z_what_mean.shape == z_what_std.shape == (batch_size, n_objects, z_what_size)
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_depth_mean.shape == z_depth_std.shape == (batch_size, n_objects, 1)


def test_reconstruction_indices(ssd_config, n_ssd_features):
    """Verify reconstruction indices calculation."""
    indices = Decoder.reconstruction_indices(ssd_config)
    assert indices.shape == (n_ssd_features,)
    assert indices.unique().numel() == sum(
        features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS
    )
    assert (torch.sort(indices)[0] == indices).all()


@pytest.mark.parametrize("max_objects", [5, 10, 15])
def test_decoder_dimensions(max_objects, ssd_model, ssd_config, n_ssd_features):
    """Verify decoder output dimensions."""
    batch_size = 2
    n_objects = sum(features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS)
    z_what_size = 3
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.ones((batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_objects, 1)
    inputs = (z_what, z_where, z_present, z_depth)
    decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size, max_objects=max_objects)
    outputs = decoder(inputs)
    assert outputs.shape == (batch_size, 3, *ssd_config.DATA.SHAPE)
