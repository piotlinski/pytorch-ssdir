"""Test SSDIR models."""
import pyro
import pytest
import torch

from pytorch_ssdir.modeling.models import SSDIR, Decoder, Encoder


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_encoder_dimensions(z_what_size, batch_size, ssd_model, n_ssd_features):
    """Verify encoder output dimensions."""
    inputs = torch.rand(batch_size, 3, 300, 300)
    encoder = Encoder(ssd=ssd_model, z_what_size=z_what_size)
    (
        (z_what_loc, z_what_scale),
        z_where,
        z_present,
        (z_depth_loc, z_depth_scale),
    ) = encoder(inputs)
    n_objects = sum(features ** 2 for features in ssd_model.backbone.feature_maps)
    assert (
        z_what_loc.shape
        == z_what_scale.shape
        == (batch_size, n_objects + 1, z_what_size)
    )
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_depth_loc.shape == z_depth_scale.shape == (batch_size, n_objects, 1)


@pytest.mark.parametrize(
    "modules_enabled",
    [
        [True, True, True, True, True],
        [True, False, False, False, False],
        [True, True, True, True, False],
        [False, False, False, False, False],
    ],
)
def test_disabling_encoder_modules(modules_enabled, ssd_model):
    """Verify if disabling encoder modules influences requires_grad attribute."""
    kwargs_keys = [
        "train_what",
        "train_where",
        "train_present",
        "train_depth",
        "train_backbone",
    ]
    module_names = ["what_enc", "where_enc", "present_enc", "depth_enc", "ssd_backbone"]
    encoder = Encoder(ssd=ssd_model, **dict(zip(kwargs_keys, modules_enabled)))
    for name, requires_grad in zip(module_names, modules_enabled):
        assert all(
            param.requires_grad == requires_grad
            for param in getattr(encoder, name).parameters()
        )


def test_reconstruction_indices(ssd_model, n_ssd_features):
    """Verify reconstruction indices calculation."""
    indices = Decoder.reconstruction_indices(
        feature_maps=ssd_model.backbone.feature_maps,
        boxes_per_loc=ssd_model.backbone.boxes_per_loc,
    )
    assert indices.shape == (n_ssd_features + 1,)
    assert (
        indices.unique().numel()
        == sum(features ** 2 for features in ssd_model.backbone.feature_maps) + 1
    )
    assert (torch.sort(indices)[0] == indices).all()


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([1, 0, 0, 2, 3, 4, 5, 6, 0])),
        (torch.tensor([1, 2, 0]), torch.tensor([1, 0, 2, 3, 0, 0])),
        (torch.tensor([3, 0, 1]), torch.tensor([1, 2, 3, 0, 0, 0, 4, 0, 0])),
    ],
)
def test_pad_indices(n_present, expected):
    """Verify pad indices calculation."""
    indices = Decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present)),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


@pytest.mark.parametrize(
    "merge_function, inputs, weights, expected",
    [
        (
            Decoder.merge_reconstructions_masked,
            torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]], [[[0.5, 0.5], [0.5, 0.5]]]]]),
            torch.tensor([[0.3, 0.2]]),
            torch.tensor([[[[[1.0, 0.5], [0.5, 0.5]]]]]),
        ),
        (
            Decoder.merge_reconstructions_masked,
            torch.tensor(
                [
                    [[[[0.2, 0.0], [0.1, 0.0]]], [[[0.4, 0.3], [0.0, 0.1]]]],
                    [[[[0.4, 0.3], [0.2, 0.1]]], [[[0.5, 0.5], [0.5, 0.5]]]],
                ]
            ),
            torch.tensor([[0.2, 0.5], [0.4, 0.1]]),
            torch.tensor([[[[0.4, 0.3], [0.1, 0.1]]], [[[0.4, 0.3], [0.2, 0.1]]]]),
        ),
        (
            Decoder.merge_reconstructions_weighted,
            torch.ones(1, 2, 3, 5, 5),
            torch.tensor([[0.3, 0.2]]),
            torch.ones(1, 3, 5, 5),
        ),
        (
            Decoder.merge_reconstructions_weighted,
            torch.ones(2, 3, 3, 2, 2),
            torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]),
            torch.ones(2, 3, 2, 2),
        ),
    ],
)
def test_merge_reconstructions(merge_function, inputs, weights, expected):
    """Verify reconstructions merging."""
    merged = merge_function(inputs, weights=weights)
    assert merged.shape == (inputs.shape[0], *inputs.shape[2:])
    assert torch.all(torch.le(torch.abs(merged - expected), 1e-3))


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_decoder_dimensions(batch_size, ssd_model, n_ssd_features):
    """Verify decoder output dimensions."""
    n_objects = sum(features ** 2 for features in ssd_model.backbone.feature_maps)
    z_what_size = 3
    z_what = torch.rand(batch_size, n_objects + 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_objects, 1)
    inputs = (z_what, z_where, z_present, z_depth)
    decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size)
    outputs = decoder(inputs)
    assert outputs.shape == (batch_size, 3, *ssd_model.image_size)


@pytest.mark.parametrize("train_what", [True, False])
def test_disabling_decoder_modules(train_what, ssd_model):
    """Verify if disabling encoder modules influences requires_grad attribute."""
    decoder = Decoder(ssd=ssd_model, train_what=train_what)
    assert all(
        param.requires_grad == train_what for param in decoder.what_dec.parameters()
    )


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_ssdir_encoder_forward(z_what_size, batch_size, ssd_model, n_ssd_features):
    """Verify SSDIR encoder_forward output dimensions and dtypes."""
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
    )

    data_shape = (3, *ssd_model.image_size)
    inputs = torch.rand(batch_size, *data_shape)

    latents = model.encoder_forward(inputs)
    z_what, z_where, z_present, z_depth = latents

    n_objects = sum(features ** 2 for features in ssd_model.backbone.feature_maps)
    assert z_what.shape == (batch_size, n_objects + 1, z_what_size)
    assert z_what.dtype == torch.float
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_where.dtype == torch.float
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_present.dtype == torch.float
    assert z_depth.shape == (batch_size, n_objects, 1)
    assert z_depth.dtype == torch.float


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("drop", [True, False])
def test_ssdir_decoder_forward(
    z_what_size, batch_size, drop, ssd_model, n_ssd_features
):
    """Verify SSDIR encoder_forward output dimensions and dtypes."""
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
    )

    n_objects = sum(features ** 2 for features in ssd_model.backbone.feature_maps)
    z_what = torch.rand(batch_size, n_objects + 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_objects, 1)
    latents = (z_what, z_where, z_present, z_depth)
    outputs = model.decoder_forward(latents)

    data_shape = (3, *ssd_model.image_size)
    assert outputs.shape == (batch_size, *data_shape)
    assert outputs.dtype == torch.float


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_ssdir_forward(z_what_size, batch_size, ssd_model, n_ssd_features):
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
    )

    data_shape = (3, *ssd_model.image_size)
    inputs = torch.rand(batch_size, *data_shape)

    outputs = model(inputs)

    assert outputs.shape == inputs.shape


def test_ssdir_model_guide(ssd_model):
    """Validate Pyro setup for SSDIR."""
    pyro.enable_validation()
    pyro.set_rng_seed(0)

    z_what_size = 3
    batch_size = 2

    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
    )

    inputs = torch.rand(batch_size, 3, *ssd_model.image_size)
    model.model(inputs)
    model.guide(inputs)
