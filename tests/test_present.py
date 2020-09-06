"""Test present modules."""
from ssdir.modeling.present import PresentEncoder


def test_present_encoder_dimensions(ssd_model, ssd_features, n_ssd_features):
    """Verify PresentEncoder output dimensions."""
    encoder = PresentEncoder(ssd_box_predictor=ssd_model.predictor)
    outputs = encoder(ssd_features)
    assert outputs.shape == (4, n_ssd_features, 1)
