"""Test where modules."""
from ssdir.modeling.where import WhereEncoder


def test_where_encoder_dimensions(ssd_config, ssd_model, ssd_features):
    """Verify WhereEncoder output dimensions."""
    encoder = WhereEncoder(ssd_box_predictor=ssd_model.predictor, ssd_config=ssd_config)
    outputs = encoder(ssd_features)
    n_boxes = sum(
        boxes * features ** 2
        for features, boxes in zip(
            ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_config.DATA.PRIOR.BOXES_PER_LOC
        )
    )
    assert outputs.shape == (ssd_features[0].shape[0], n_boxes, 4)
