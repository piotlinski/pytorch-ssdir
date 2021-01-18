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
    assert (
        z_what_loc.shape
        == z_what_scale.shape
        == (batch_size, n_ssd_features + 1, z_what_size)
    )
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_depth_loc.shape == z_depth_scale.shape == (batch_size, n_ssd_features, 1)


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


@pytest.mark.parametrize("n_trained", [0, 1, 2, 5])
def test_disabling_backbone_layers(n_trained, ssd_model):
    """Verify if disabling encoder backbone layers disables it effectively."""
    encoder = Encoder(
        ssd=ssd_model, train_backbone=True, train_backbone_layers=n_trained
    )
    for idx, module in enumerate(encoder.ssd_backbone.children()):
        if idx < n_trained:
            assert all(param.requires_grad is True for param in module.parameters())
        else:
            assert all(param.requires_grad is False for param in module.parameters())


def test_cloning_backbone(ssd_model):
    """Verify if disabling encoder backbone layers disables it effectively."""
    encoder = Encoder(ssd=ssd_model, clone_backbone=True)
    assert len(list(encoder.ssd_backbone_cloned.children())) <= len(
        list(encoder.ssd_backbone.children())
    )


@pytest.mark.parametrize("train_backbone", [False, True])
def test_cloning_grads(train_backbone, ssd_model):
    """Verify if train_backbone is used appropriately for backbone and cloned."""
    encoder = Encoder(
        ssd=ssd_model,
        train_backbone=train_backbone,
        clone_backbone=True,
    )
    assert all(
        param.requires_grad is train_backbone
        for param in encoder.ssd_backbone.parameters()
    )
    assert all(
        param.requires_grad is True
        for param in encoder.ssd_backbone_cloned.parameters()
    )
    for backbone_child, cloned_child in zip(
        list(encoder.ssd_backbone.children()),
        list(encoder.ssd_backbone_cloned.children()),
    ):
        assert backbone_child is not cloned_child
        for backbone_param, cloned_param in zip(
            backbone_child.parameters(), cloned_child.parameters()
        ):
            assert backbone_param is not cloned_param


def test_latents_indices(ssd_model, n_ssd_features):
    """Verify latents indices calculation."""
    indices = Encoder.latents_indices(
        feature_maps=ssd_model.backbone.feature_maps,
        boxes_per_loc=ssd_model.backbone.boxes_per_loc,
    )
    assert indices.shape == (n_ssd_features,)
    assert indices.unique().numel() == sum(
        features ** 2 for features in ssd_model.backbone.feature_maps
    )
    assert (torch.sort(indices)[0] == indices).all()


def test_pad_latents(ssd_model, n_ssd_features):
    """Verify if latents are padded appropriately."""
    n_features = sum(features ** 2 for features in ssd_model.backbone.feature_maps)
    encoder = Encoder(ssd=ssd_model)
    z_what_loc = (
        torch.arange(n_features + 1, dtype=torch.float)
        .view(1, -1, 1)
        .expand(1, n_features + 1, 4)
    )
    z_what_scale = (
        torch.arange(n_features + 1, 2 * (n_features + 1), dtype=torch.float)
        .view(1, -1, 1)
        .expand(1, n_features + 1, 4)
    )
    z_where = torch.zeros(1, n_ssd_features, dtype=torch.float)
    z_present = torch.zeros(1, n_ssd_features, dtype=torch.float)
    z_depth_loc = torch.arange(2 * n_features, 3 * n_features, dtype=torch.float).view(
        1, -1, 1
    )
    z_depth_scale = torch.arange(
        3 * n_features, 4 * n_features, dtype=torch.float
    ).view(1, -1, 1)
    (
        (new_z_what_loc, new_z_what_scale),
        _,
        _,
        (new_z_depth_loc, new_z_depth_scale),
    ) = encoder.pad_latents(
        ((z_what_loc, z_what_scale), z_where, z_present, (z_depth_loc, z_depth_scale))
    )
    assert new_z_what_loc.shape == new_z_what_scale.shape == (1, n_ssd_features + 1, 4)
    assert torch.eq(new_z_what_loc[0][0], new_z_what_loc[0][1]).all()
    assert torch.eq(new_z_what_scale[0][2], new_z_what_scale[0][3]).all()
    assert torch.eq(new_z_what_loc[0][8], new_z_what_loc[0][9]).all()
    assert torch.eq(new_z_what_scale[0][16], new_z_what_scale[0][17]).all()
    assert torch.eq(new_z_what_loc[0][400], new_z_what_loc[0][401]).all()
    assert torch.eq(new_z_what_scale[0][562], new_z_what_scale[0][563]).all()
    assert new_z_depth_loc.shape == new_z_depth_scale.shape == (1, n_ssd_features, 1)
    assert torch.eq(new_z_depth_loc[0][204], new_z_depth_loc[0][205]).all()
    assert torch.eq(new_z_depth_scale[0][368], new_z_depth_scale[0][369]).all()
    assert torch.eq(new_z_depth_loc[0][604], new_z_depth_loc[0][605]).all()
    assert torch.eq(new_z_depth_scale[0][628], new_z_depth_scale[0][629]).all()
    assert torch.eq(new_z_depth_loc[0][702], new_z_depth_loc[0][703]).all()
    assert torch.eq(new_z_depth_scale[0][850], new_z_depth_scale[0][851]).all()


def test_reset_non_present(ssd_model):
    """Verify if appropriate latents are reset in encoder."""
    encoder = Encoder(ssd=ssd_model)
    z_what_loc = torch.arange(1, 6, dtype=torch.float).view(1, -1, 1).expand(1, 5, 3)
    z_what_scale = torch.arange(6, 11, dtype=torch.float).view(1, -1, 1).expand(1, 5, 3)
    z_where = torch.arange(1, 5, dtype=torch.float).view(1, -1, 1).expand(1, 4, 4)
    z_present = torch.tensor([1, 0, 0, 1], dtype=torch.float).view(1, -1, 1)
    z_depth_loc = torch.arange(5, 9, dtype=torch.float).view(1, -1, 1)
    z_depth_scale = torch.arange(9, 13, dtype=torch.float).view(1, -1, 1)
    (
        (reset_z_what_loc, reset_z_what_scale),
        reset_z_where,
        reset_z_present,
        (reset_z_depth_loc, reset_z_depth_scale),
    ) = encoder.reset_non_present(
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )
    )
    assert torch.eq(reset_z_what_loc[0][0], z_what_loc[0][0]).all()
    assert torch.eq(reset_z_what_loc[0][3], z_what_loc[0][3]).all()
    assert torch.eq(reset_z_what_scale[0][0], z_what_scale[0][0]).all()
    assert torch.eq(reset_z_what_scale[0][3], z_what_scale[0][3]).all()
    assert torch.eq(reset_z_where[0][0], z_where[0][0]).all()
    assert torch.eq(reset_z_where[0][3], z_where[0][3]).all()
    assert torch.eq(reset_z_depth_loc[0][0], z_depth_loc[0][0]).all()
    assert torch.eq(reset_z_depth_loc[0][3], z_depth_loc[0][3]).all()
    assert torch.eq(reset_z_depth_scale[0][0], z_depth_scale[0][0]).all()
    assert torch.eq(reset_z_depth_scale[0][3], z_depth_scale[0][3]).all()
    assert (reset_z_what_loc[0][1] == reset_z_what_loc[0][2]).all()
    assert (reset_z_what_loc[0][1] == encoder.empty_loc).all()
    assert (reset_z_what_scale[0][1] == reset_z_what_scale[0][2]).all()
    assert (reset_z_what_scale[0][1] == encoder.empty_scale).all()
    assert (reset_z_where[0][1] == reset_z_where[0][2]).all()
    assert (reset_z_where[0][1] == encoder.empty_loc).all()
    assert (reset_z_depth_loc[0][1] == reset_z_depth_loc[0][2]).all()
    assert (reset_z_depth_loc[0][1] == encoder.empty_loc).all()
    assert (reset_z_depth_scale[0][1] == reset_z_depth_scale[0][2]).all()
    assert (reset_z_depth_scale[0][1] == encoder.empty_scale).all()


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([1, 0, 0, 4, 2, 3, 6, 5, 0])),
        (torch.tensor([1, 2, 2]), torch.tensor([1, 0, 3, 2, 5, 4])),
        (torch.tensor([3, 1, 1]), torch.tensor([3, 1, 2, 4, 0, 0, 5, 0, 0])),
    ],
)
def test_pad_indices(n_present, expected):
    """Verify pad indices calculation."""
    indices = Decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present)),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


@pytest.mark.parametrize(
    "inputs, weights, expected",
    [
        (
            torch.ones(1, 2, 3, 5, 5),
            torch.tensor([[0.3, 0.2]]),
            torch.ones(1, 3, 5, 5),
        ),
        (
            torch.ones(2, 3, 3, 2, 2),
            torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]),
            torch.ones(2, 3, 2, 2),
        ),
    ],
)
def test_merge_reconstructions(inputs, weights, expected):
    """Verify reconstructions merging."""
    merged = Decoder.merge_reconstructions(inputs, weights=weights)
    assert merged.shape == (inputs.shape[0], *inputs.shape[2:])
    assert torch.all(torch.le(torch.abs(merged - expected), 1e-3))


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_decoder_dimensions(batch_size, ssd_model, n_ssd_features):
    """Verify decoder output dimensions."""
    z_what_size = 3
    z_what = torch.rand(batch_size, n_ssd_features + 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_ssd_features, 1)
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

    assert z_what.shape == (batch_size, n_ssd_features + 1, z_what_size)
    assert z_what.dtype == torch.float
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_where.dtype == torch.float
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_present.dtype == torch.float
    assert z_depth.shape == (batch_size, n_ssd_features, 1)
    assert z_depth.dtype == torch.float


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("drop", [True, False])
def test_ssdir_decoder_forward(
    z_what_size, batch_size, drop, ssd_model, n_ssd_features
):
    """Verify SSDIR decoder_forward output dimensions and dtypes."""
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
    )

    z_what = torch.rand(batch_size, n_ssd_features + 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_ssd_features, 1)
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
