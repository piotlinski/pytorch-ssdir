"""Test where modules."""
import pytest
import torch

from ssdir.modeling.where import WhereEncoder, WhereTransformer


def test_where_encoder_dimensions(ssd_config, ssd_model, ssd_features, n_ssd_features):
    """Verify WhereEncoder output dimensions."""
    encoder = WhereEncoder(ssd_box_predictor=ssd_model.predictor, ssd_config=ssd_config)
    outputs = encoder(ssd_features)
    assert outputs.shape == (ssd_features[0].shape[0], n_ssd_features, 4)


@pytest.mark.parametrize("decoded_size", [2, 3])
@pytest.mark.parametrize("image_size", [7, 8])
@pytest.mark.parametrize("batch_size", [3, 4])
@pytest.mark.parametrize("hidden_size", [5, 6])
def test_where_transformer_dimensions(
    decoded_size, image_size, batch_size, hidden_size
):
    """Verify WhereTransformer output dimensions."""
    decoded_images = torch.rand(batch_size, hidden_size, 3, decoded_size, decoded_size)
    z_where = torch.rand(batch_size, hidden_size, 4)
    transformer = WhereTransformer(image_size=image_size)
    outputs = transformer(decoded_images, z_where)
    assert outputs.shape == (batch_size, hidden_size, 3, image_size, image_size)
