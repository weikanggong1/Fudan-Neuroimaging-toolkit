"""Regressions for label ties, whole-sample fill, and mixed-grid pull maps."""
import pytest
import torch

from fnit.synthmorph.spatial import compose, transform


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,point", [((28, 3, 5), (26, 2, 2)), ((32, 3, 8), (30, 2, 5))])
def test_affine_nearest_half_voxel_regression(device, shape, point):
    # Reference affine -> displacement -> coordinate yields j=0.5 here.
    # Direct matrix coordinates can yield 0.5000000596 and select label 1.
    labels = torch.arange(3, device=device, dtype=torch.float32)[None, None, None, :, None]
    labels = labels.expand(1, 1, *shape)
    matrix = torch.tensor([[1.02, .08, 0, .2], [-.03, .95, .04, -.7],
                           [.01, 0, 1.04, .8]], device=device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        actual = transform(labels, matrix, method="nearest", fill_value=-99)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert actual[(0, 0, *point)].item() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("shift,boundary,edge", [(-.25, 0, 10), (.25, 2, 12)])
def test_outside_samples_are_filled_whole_or_clamped(device, method, shift, boundary, edge):
    image = torch.arange(10, 13, device=device, dtype=torch.float32)[None, None, :, None, None]
    image = image.expand(1, 1, 3, 3, 3)
    matrix = torch.eye(4, device=device)
    matrix[0, 3] = shift
    for fill, expected in ((None, edge), (0, 0), (-7, -7)):
        actual = transform(image, matrix, method=method, fill_value=fill)
        assert actual[0, 0, boundary, 1, 1].item() == expected
    if method == "linear":
        assert actual[0, 0, 1, 1, 1].item() == 11 + shift


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_shared_identity_preserves_batched_channels(device, method):
    image = torch.arange(120, device=device, dtype=torch.float32).reshape(2, 2, 2, 3, 5)
    actual = transform(image, torch.eye(4, device=device), method=method)
    assert torch.equal(actual, image)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("right_kind", ["affine", "dense"])
def test_compose_different_resolution_fields(device, right_kind):
    # Left map expands i by 1.5 on a 7x5x3 grid. Right map samples a
    # 3x2x2 output grid at (2*i+1, j+1, k). Combined displacements are
    # therefore (2*i+1.5, 1, 0), independent of the left grid's extent.
    left = torch.zeros(1, 3, 7, 5, 3, device=device)
    left[:, 0] = torch.arange(7, device=device)[:, None, None] / 2
    if right_kind == "affine":
        right = torch.tensor([[2., 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], device=device)
    else:
        right = torch.zeros(1, 3, 3, 2, 2, device=device)
        right[:, 0] = torch.tensor([1., 2., 3.], device=device)[:, None, None]
        right[:, 1] = 1
    actual = compose([left, right], shape=(3, 2, 2))
    expected = torch.zeros(1, 3, 3, 2, 2, device=device)
    expected[:, 0] = torch.tensor([1.5, 3.5, 5.5], device=device)[:, None, None]
    expected[:, 1] = 1
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
