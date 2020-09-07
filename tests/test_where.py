"""Test where modules."""
import pytest
import torch

from ssdir.modeling.where import WhereEncoder, WhereTransformer


def test_where_encoder_dimensions(ssd_model, ssd_features, n_ssd_features):
    """Verify WhereEncoder output dimensions."""
    encoder = WhereEncoder(ssd_box_predictor=ssd_model.predictor)
    outputs = encoder(ssd_features)
    assert outputs.shape == (ssd_features[0].shape[0], n_ssd_features, 4)


def test_where_encoder_dtype(ssd_model, ssd_features):
    encoder = WhereEncoder(ssd_box_predictor=ssd_model.predictor)
    outputs = encoder(ssd_features)
    assert outputs.dtype == torch.float


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


@pytest.mark.parametrize(
    "boxes, image_size, expected",
    [
        (torch.tensor([1.0, 2.0, 3.0, 4.0]), 32, torch.tensor([2.0, 32.0, 64.0])),
        (
            torch.tensor([[10.0, 12.0, 8.0, 6.0], [6.0, 4.0, 2.0, 5.0]]),
            128,
            torch.tensor([[16.0, 1280.0, 1536.0], [10.0, 768.0, 512.0]]),
        ),
        (
            torch.tensor([[[0.5, 0.2, 0.4, 0.3]]]),
            128,
            torch.tensor([[[0.8, 64, 25.6]]]),
        ),
    ],
)
def test_convert_to_sxy(boxes, image_size, expected):
    """Verify converting xywh boxes to sxy."""
    transformer = WhereTransformer(image_size=image_size)
    sxy = transformer.convert_boxes_to_sxy(boxes)
    assert (sxy == expected).all()


def test_expand_where():
    """Verify expanding sxy to transformation matrix."""
    sxy = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[8.0, 7.0, 6.0], [5.0, 4.0, 3.0]]]
    )
    expected = torch.tensor(
        [
            [[[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], [[4.0, 0.0, 5.0], [0.0, 4.0, 6.0]]],
            [[[8.0, 0.0, 7.0], [0.0, 8.0, 6.0]], [[5.0, 0.0, 4.0], [0.0, 5.0, 3.0]]],
        ]
    )
    transformation_mtx = WhereTransformer.expand_where(sxy)
    assert (transformation_mtx == expected).all()
