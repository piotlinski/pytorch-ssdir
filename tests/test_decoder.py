"""Test SSDIR decoder."""
import pytest
import torch

from pytorch_ssdir.modeling.decoder import Decoder


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([1, 0, 0, 4, 2, 3, 6, 5, 0])),
        (torch.tensor([1, 2, 2]), torch.tensor([1, 0, 3, 2, 5, 4])),
        (torch.tensor([3, 1, 1]), torch.tensor([3, 1, 2, 4, 0, 0, 5, 0, 0])),
    ],
)
def test_pad_indices(n_present, expected, ssd_model):
    """Verify pad indices calculation."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4, background=True)
    indices = decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present)),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([0, 1, 0, 0, 0, 2, 3, 4, 0, 5, 6, 0])),
        (torch.tensor([1, 2, 2]), torch.tensor([0, 1, 0, 0, 2, 3, 0, 4, 5])),
        (torch.tensor([3, 1, 1]), torch.tensor([0, 1, 2, 3, 0, 4, 0, 0, 0, 5, 0, 0])),
    ],
)
def test_pad_indices_no_background(n_present, expected, ssd_model):
    """Verify pad indices calculation."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4, background=False)
    indices = decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present) + 1),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


def test_pad_reconstructions(ssd_model):
    """Verify padding reconstructions in Decoder."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4)
    images = (
        torch.arange(1, 5, dtype=torch.float)
        .view(-1, 1, 1, 1)
        .expand(4, 3, decoder.where_stn.image_size, decoder.where_stn.image_size)
    )
    z_depth = torch.arange(5, 9, dtype=torch.float).view(-1, 1)
    n_present = torch.tensor([1, 3])
    padded_images, padded_z_depth = decoder.pad_reconstructions(
        transformed_images=images, z_depth=z_depth, n_present=n_present
    )
    assert padded_images.shape == (2, 3, 3, 300, 300)
    assert padded_z_depth.shape == (2, 3)
    assert torch.equal(padded_images[0][0], images[0])
    assert torch.equal(padded_images[0][1], padded_images[0][2])
    assert torch.equal(padded_images[1][0], images[3])
    assert torch.equal(padded_images[1][1], images[1])
    assert torch.equal(padded_images[1][2], images[2])
    assert padded_z_depth[0][0] == z_depth[0]
    assert padded_z_depth[0][1] == padded_z_depth[0][2] == -float("inf")
    assert padded_z_depth[1][0] == z_depth[3]
    assert padded_z_depth[1][1] == z_depth[1]
    assert padded_z_depth[1][2] == z_depth[2]


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
def test_merge_reconstructions(inputs, weights, expected, ssd_model):
    """Verify reconstructions merging."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4)
    merged = decoder.merge_reconstructions(inputs, weights=weights)
    assert merged.shape == (inputs.shape[0], *inputs.shape[2:])
    assert torch.all(torch.le(torch.abs(merged - expected), 1e-3))


def test_fill_background():
    """Verify adding background to merged reconstructions."""
    background_1 = torch.full((1, 3, 1, 2), fill_value=0.3)
    background_2 = torch.full((1, 3, 1, 2), fill_value=0.7)
    backgrounds = torch.cat((background_1, background_2), dim=0)
    merged = torch.zeros((2, 3, 1, 2))
    merged[0, 1, 0, 1] = 0.4
    merged[1, 2, 0, 1] = 0.5
    merged[0, 2, 0, 0] = 0.9
    merged[1, 0, 0, 1] = 1.0
    filled = Decoder.fill_background(merged, backgrounds)
    assert filled[0, 1, 0, 1] == 0.4
    assert filled[1, 2, 0, 1] == 0.5
    assert filled[0, 2, 0, 0] == 0.9
    assert filled[1, 0, 0, 1] == 1.0
    assert (
        filled[0, 0, 0, 0]
        == filled[0, 0, 0, 1]
        == filled[0, 1, 0, 0]
        == filled[0, 2, 0, 1]
        == 0.3
    )
    assert (
        filled[1, 0, 0, 0]
        == filled[1, 1, 0, 0]
        == filled[1, 1, 0, 1]
        == filled[1, 2, 0, 0]
        == 0.7
    )


class WhatDecoderMock(torch.nn.Module):
    """Module to mock WhatDecoder."""

    def __init__(self, n_objects: int, z_what_size: int):
        self.n_objects = n_objects
        self.z_what_size = z_what_size
        super().__init__()

    def forward(self, z_what: torch.Tensor) -> torch.Tensor:
        return z_what


class WhereTransformerMock(torch.nn.Module):
    """Module to mock WhereTransformer."""

    def __init__(self, n_objects: int, image_size: int):
        self.n_objects = n_objects
        self.image_size = image_size
        super().__init__()

    def forward(
        self, decoded_images: torch.Tensor, where_boxes: torch.Tensor
    ) -> torch.Tensor:
        return (
            (decoded_images * where_boxes)
            .view(self.n_objects, 1, self.image_size, self.image_size)
            .expand(self.n_objects, 3, self.image_size, self.image_size)
        )


@pytest.fixture
def sample_latents():
    """Example latents for testing Decoder."""
    z_what = torch.arange(1, 9, dtype=torch.float).view(2, -1, 1).expand(2, 4, 4)
    z_where = (
        (torch.arange(3, 9, dtype=torch.float) / 10).view(2, -1, 1).expand(2, 3, 4)
    )
    z_present = torch.tensor([[1, 0, 1], [1, 0, 0]], dtype=torch.float).view(2, -1, 1)
    z_depth = torch.arange(7, 13, dtype=torch.float).view(2, -1, 1)
    return z_what, z_where, z_present, z_depth


def test_reconstruct_objects_drop(ssd_model, sample_latents):
    """Verify if reconstructing objects is conducted correctly for drop decoder."""
    decoder = Decoder(ssd=ssd_model, z_what_size=2, drop_empty=True, background=True)
    z_what, z_where, z_present, z_depth = sample_latents

    decoder.what_dec = WhatDecoderMock(n_objects=5, z_what_size=4)
    decoder.where_stn = WhereTransformerMock(n_objects=5, image_size=2)

    def what_decoder_test_hook(module, input, output):
        assert torch.eq(input[0][0], 1.0).all()
        assert torch.eq(input[0][1], 3.0).all()
        assert torch.eq(input[0][2], 4.0).all()
        assert torch.eq(input[0][3], 5.0).all()
        assert torch.eq(input[0][4], 8.0).all()

    decoder.what_dec.register_forward_hook(what_decoder_test_hook)

    def where_transformer_hook(module, input, output):
        images, z_where_flat = input
        assert torch.eq(z_where_flat[0], 0.3).all()
        assert torch.eq(z_where_flat[1], 0.5).all()
        assert torch.equal(z_where_flat[2], decoder.bg_where)
        assert torch.eq(z_where_flat[3], 0.6).all()
        assert torch.equal(z_where_flat[4], decoder.bg_where)

    decoder.where_stn.register_forward_hook(where_transformer_hook)

    reconstructions, depths = decoder.reconstruct_objects(
        z_what, z_where, z_present, z_depth
    )

    assert torch.equal(
        reconstructions[0][0][0].view(4), z_what[0][3] * decoder.bg_where
    )
    assert torch.equal(reconstructions[0][1][0].view(4), z_what[0][0] * z_where[0][0])
    assert depths[0][1] == z_depth[0][0]
    assert torch.equal(reconstructions[0][2][0].view(4), z_what[0][2] * z_where[0][2])
    assert depths[0][2] == z_depth[0][2]
    assert torch.equal(
        reconstructions[1][0][0].view(4), z_what[1][3] * decoder.bg_where
    )
    assert torch.equal(reconstructions[1][1][0].view(4), z_what[1][0] * z_where[1][0])
    assert depths[1][1] == z_depth[1][0]
    assert torch.eq(reconstructions[1][2][0].view(4), 0).all()
    assert depths[1][2] == -float("inf")


def test_reconstruct_objects_no_background(ssd_model, sample_latents):
    """Verify if reconstructing objects is conducted correctly for no-background."""
    decoder = Decoder(ssd=ssd_model, z_what_size=2, drop_empty=True, background=False)
    z_what, z_where, z_present, z_depth = sample_latents
    z_what = z_what[:, :-1]

    decoder.what_dec = WhatDecoderMock(n_objects=3, z_what_size=4)
    decoder.where_stn = WhereTransformerMock(n_objects=3, image_size=2)

    def what_decoder_test_hook(module, input, output):
        assert torch.eq(input[0][0], 1.0).all()
        assert torch.eq(input[0][1], 3.0).all()
        assert torch.eq(input[0][2], 5.0).all()

    decoder.what_dec.register_forward_hook(what_decoder_test_hook)

    def where_transformer_hook(module, input, output):
        images, z_where_flat = input
        assert torch.eq(z_where_flat[0], 0.3).all()
        assert torch.eq(z_where_flat[1], 0.5).all()
        assert torch.eq(z_where_flat[2], 0.6).all()

    decoder.where_stn.register_forward_hook(where_transformer_hook)

    reconstructions, depths = decoder.reconstruct_objects(
        z_what, z_where, z_present, z_depth
    )
    assert torch.eq(reconstructions[0][0][0].view(4), 0).all()
    assert depths[0][0] == -float("inf")
    assert torch.equal(reconstructions[0][1][0].view(4), z_what[0][0] * z_where[0][0])
    assert depths[0][1] == z_depth[0][0]
    assert torch.equal(reconstructions[0][2][0].view(4), z_what[0][2] * z_where[0][2])
    assert depths[0][2] == z_depth[0][2]
    assert torch.eq(reconstructions[1][0][0].view(4), 0).all()
    assert depths[1][0] == -float("inf")
    assert torch.equal(reconstructions[1][1][0].view(4), z_what[1][0] * z_where[1][0])
    assert depths[1][1] == z_depth[1][0]
    assert torch.eq(reconstructions[1][2][0].view(4), 0).all()
    assert depths[1][2] == -float("inf")


def test_reconstruct_objects_no_drop(ssd_model, sample_latents):
    """Verify if reconstructing objects is conducted correctly for no-drop decoder."""
    decoder = Decoder(ssd=ssd_model, z_what_size=2, drop_empty=False, background=True)
    z_what, z_where, z_present, z_depth = sample_latents

    decoder.what_dec = WhatDecoderMock(n_objects=8, z_what_size=4)
    decoder.where_stn = WhereTransformerMock(n_objects=8, image_size=2)

    def what_decoder_test_hook(module, input, output):
        assert torch.eq(input[0][0], 1.0).all()
        assert torch.eq(input[0][1], 2.0).all()
        assert torch.eq(input[0][2], 3.0).all()
        assert torch.eq(input[0][3], 4.0).all()
        assert torch.eq(input[0][4], 5.0).all()
        assert torch.eq(input[0][5], 6.0).all()
        assert torch.eq(input[0][6], 7.0).all()
        assert torch.eq(input[0][7], 8.0).all()

    decoder.what_dec.register_forward_hook(what_decoder_test_hook)

    def where_transformer_hook(module, input, output):
        images, z_where_flat = input
        assert torch.eq(z_where_flat[0], 0.3).all()
        assert torch.eq(z_where_flat[1], 0.4).all()
        assert torch.eq(z_where_flat[2], 0.5).all()
        assert torch.equal(z_where_flat[3], decoder.bg_where)
        assert torch.eq(z_where_flat[4], 0.6).all()
        assert torch.eq(z_where_flat[5], 0.7).all()
        assert torch.eq(z_where_flat[6], 0.8).all()
        assert torch.equal(z_where_flat[7], decoder.bg_where)

    decoder.where_stn.register_forward_hook(where_transformer_hook)

    reconstructions, depths = decoder.reconstruct_objects(
        z_what, z_where, z_present, z_depth
    )

    assert torch.equal(
        reconstructions[0][0][0].view(4), z_what[0][3] * decoder.bg_where
    )
    assert torch.equal(reconstructions[0][1][0].view(4), z_what[0][0] * z_where[0][0])
    assert depths[0][1] == z_depth[0][0]
    assert torch.equal(reconstructions[0][2][0].view(4), z_what[0][1] * z_where[0][1])
    assert depths[0][2] == -float("inf")
    assert torch.equal(reconstructions[0][3][0].view(4), z_what[0][2] * z_where[0][2])
    assert depths[0][3] == z_depth[0][2]
    assert torch.equal(
        reconstructions[1][0][0].view(4), z_what[1][3] * decoder.bg_where
    )
    assert torch.equal(reconstructions[1][1][0].view(4), z_what[1][0] * z_where[1][0])
    assert depths[1][1] == z_depth[1][0]
    assert torch.equal(reconstructions[1][2][0].view(4), z_what[1][1] * z_where[1][1])
    assert depths[1][2] == -float("inf")
    assert torch.equal(reconstructions[1][3][0].view(4), z_what[1][2] * z_where[1][2])
    assert depths[1][3] == -float("inf")


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("background", [True, False])
def test_decoder_dimensions(batch_size, background, ssd_model, n_ssd_features):
    """Verify decoder output dimensions."""
    z_what_size = 3
    z_what = torch.rand(batch_size, n_ssd_features + background * 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 100, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_ssd_features, 1)
    inputs = (z_what, z_where, z_present, z_depth)
    decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size, background=background)
    outputs = decoder(inputs)
    assert outputs.shape == (batch_size, 3, *ssd_model.image_size)


@pytest.mark.parametrize("train_what", [True, False])
def test_disabling_decoder_modules(train_what, ssd_model):
    """Verify if disabling encoder modules influences requires_grad attribute."""
    decoder = Decoder(ssd=ssd_model, train_what=train_what)
    assert all(
        param.requires_grad == train_what for param in decoder.what_dec.parameters()
    )
