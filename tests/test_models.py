"""Test SSDIR models."""
from unittest.mock import patch

import pyro
import pytest
import torch

from ssdir.modeling.models import SSDIR, Decoder, Encoder


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_encoder_dimensions(
    z_what_size, batch_size, ssd_model, ssd_config, n_ssd_features
):
    """Verify encoder output dimensions."""
    inputs = torch.rand(batch_size, 3, 300, 300)
    encoder = Encoder(ssd=ssd_model, z_what_size=z_what_size)
    (
        (z_what_loc, z_what_scale),
        z_where,
        z_present,
        (z_depth_loc, z_depth_scale),
    ) = encoder(inputs)
    n_objects = sum(features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS)
    assert (
        z_what_loc.shape == z_what_scale.shape == (batch_size, n_objects, z_what_size)
    )
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_depth_loc.shape == z_depth_scale.shape == (batch_size, n_objects, 1)


def test_reconstruction_indices(ssd_config, n_ssd_features):
    """Verify reconstruction indices calculation."""
    indices = Decoder.reconstruction_indices(ssd_config)
    assert indices.shape == (n_ssd_features,)
    assert indices.unique().numel() == sum(
        features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS
    )
    assert (torch.sort(indices)[0] == indices).all()


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_decoder_dimensions(batch_size, ssd_model, ssd_config, n_ssd_features):
    """Verify decoder output dimensions."""
    n_objects = sum(features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS)
    z_what_size = 3
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_objects, 1)
    inputs = (z_what, z_where, z_present, z_depth)
    decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size)
    outputs = decoder(inputs)
    assert outputs.shape == (batch_size, 3, *ssd_config.DATA.SHAPE)


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@patch("ssdir.modeling.models.CheckPointer")
@patch("ssdir.modeling.models.SSD")
@patch("ssdir.modeling.models.get_config")
def test_ssdir_encoder_forward(
    _get_config_mock,
    ssd_mock,
    _checkpointer_mock,
    z_what_size,
    batch_size,
    ssd_model,
    ssd_config,
    n_ssd_features,
):
    """Verify SSDIR encoder_forward output dimensions and dtypes."""
    ssd_mock.return_value = ssd_model
    model = SSDIR(
        z_what_size=z_what_size, ssd_config_file="test", ssd_model_file="test"
    )

    data_shape = (3, *ssd_config.DATA.SHAPE)
    inputs = torch.rand(batch_size, *data_shape)
    latents = model.encoder_forward(inputs)
    z_what, z_where, z_present, z_depth = latents
    n_objects = sum(features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS)
    assert z_what.shape == (batch_size, n_objects, z_what_size)
    assert z_what.dtype == torch.float
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_where.dtype == torch.float
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_present.dtype == torch.long
    assert z_depth.shape == (batch_size, n_objects, 1)
    assert z_depth.dtype == torch.float


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@patch("ssdir.modeling.models.CheckPointer")
@patch("ssdir.modeling.models.SSD")
@patch("ssdir.modeling.models.get_config")
def test_ssdir_decoder_forward(
    _get_config_mock,
    ssd_mock,
    _checkpointer_mock,
    z_what_size,
    batch_size,
    ssd_model,
    ssd_config,
    n_ssd_features,
):
    """Verify SSDIR encoder_forward output dimensions and dtypes."""
    ssd_mock.return_value = ssd_model
    model = SSDIR(
        z_what_size=z_what_size, ssd_config_file="test", ssd_model_file="test"
    )

    n_objects = sum(features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS)
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_objects, 1)
    latents = (z_what, z_where, z_present, z_depth)
    outputs = model.decoder_forward(latents)
    data_shape = (3, *ssd_config.DATA.SHAPE)
    assert outputs.shape == (batch_size, *data_shape)
    assert outputs.dtype == torch.float


@patch("ssdir.modeling.models.CheckPointer")
@patch("ssdir.modeling.models.SSD")
@patch("ssdir.modeling.models.get_config")
def test_ssdir_model_guide(
    get_config_mock, ssd_mock, _checkpointer_mock, ssd_model, ssd_config
):
    """Validate Pyro setup for SSDIR."""
    pyro.enable_validation()
    pyro.set_rng_seed(0)

    z_what_size = 3
    batch_size = 2
    get_config_mock.return_value = ssd_config
    ssd_mock.return_value = ssd_model

    model = SSDIR(
        z_what_size=z_what_size,
        ssd_config_file="test",
        ssd_model_file="test",
        z_present_p_prior=0.01,
    )

    inputs = torch.rand(batch_size, 3, *ssd_config.DATA.SHAPE)
    model.model(inputs)
    model.guide(inputs)
