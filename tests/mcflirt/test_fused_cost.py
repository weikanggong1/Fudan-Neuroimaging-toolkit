"""CUDA cost preparation preserves MCFLIRT's float32 row traversal.

Small irregular volumes here check arithmetic contracts only. Real-image
accuracy and elapsed-time comparisons belong to validation, not these tests.
"""

from importlib.util import find_spec

import numpy as np
import pytest
import torch

from fnit.flirt.core import _manual_trilinear
from fnit.mcflirt.core import _normcorr_reduce


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("triton") is None,
    reason="CUDA and Triton are required for fused MCFLIRT preparation",
)


def _tensor_preparation(reference, moving, moving_sizes, coefficients):
    """Literal preparation from the existing scalar motion cost.

    Reference is ZYX; moving is XYZ. In particular, multiplication by an
    absolute x index is not equivalent to the native repeated-addition loop.
    """
    zsize, ysize, xsize = reference.shape
    z, y = torch.meshgrid(
        torch.arange(zsize, dtype=torch.float32, device=moving.device),
        torch.arange(ysize, dtype=torch.float32, device=moving.device),
        indexing="ij",
    )
    coefficients = torch.as_tensor(coefficients, device=moving.device)
    upper = moving.new_tensor([size - 1.0001 for size in moving.shape])
    smooth = moving.new_tensor([1.0 / size for size in moving_sizes])
    origins = torch.stack([
        y * row[1] + z * row[2] + row[3] for row in coefficients
    ])
    directions = coefficients[:, 0]
    xmin = torch.zeros_like(y)
    xmax = torch.full_like(y, float(xsize - 1))
    for axis in range(3):
        direction = directions[axis]
        if abs(float(direction)) < 1e-8:
            outside = (origins[axis] < 0) | (origins[axis] > upper[axis])
            xmin = torch.where(outside, torch.full_like(xmin, xsize), xmin)
        else:
            bound0 = -origins[axis] / direction
            bound1 = (upper[axis] - origins[axis]) / direction
            xmin = torch.maximum(xmin, torch.ceil(torch.minimum(bound0, bound1)))
            xmax = torch.minimum(xmax, torch.floor(torch.maximum(bound0, bound1)))
    xmin = xmin.clamp(0, xsize)
    coordinate = origins + xmin[None] * directions[:, None, None]
    coordinates = []
    for _ in range(xsize):
        coordinates.append(coordinate)
        coordinate = coordinate + directions[:, None, None]
    coordinates = torch.stack(coordinates, -1)
    offsets = torch.arange(xsize, device=moving.device)[None, None, :]
    actual_x = xmin[..., None] + offsets
    valid = actual_x <= xmax[..., None]
    valid &= ((coordinates >= 0) &
              (coordinates <= upper[:, None, None, None])).all(0)
    flat_coordinates = coordinates.reshape(3, -1)
    clamped = torch.minimum(flat_coordinates.clamp_min(0), upper[:, None])
    values = _manual_trilinear(moving, clamped).reshape(zsize, ysize, xsize)
    smooth = smooth[:, None, None, None]
    upper = upper[:, None, None, None]
    weights = torch.where(
        coordinates < smooth, coordinates / smooth,
        torch.where(upper - coordinates < smooth,
                    (upper - coordinates) / smooth, 1.0),
    ).prod(0).clamp_min(0) * valid
    x_index = actual_x.long().clamp_max(xsize - 1)
    reference_values = torch.gather(reference, -1, x_index)
    return reference_values, values, weights


def _assert_bits(actual, expected):
    assert len(actual) == len(expected) == 3
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype == torch.float32
        assert result.device == reference.device
        result = result.detach().cpu().numpy()
        reference = reference.detach().cpu().numpy()
        np.testing.assert_array_equal(result.view(np.uint32), reference.view(np.uint32))


def _images(reference_shape=(5, 7, 17)):
    generator = torch.Generator().manual_seed(41061)
    reference = (torch.rand(reference_shape, generator=generator) * 900 - 80).cuda()
    moving = (torch.rand((17, 19, 23), generator=generator) * 750 - 120).cuda()
    return reference, moving, (1.7, 1.3, 2.2)


def _coefficients():
    identity = np.eye(4, dtype=np.float32)[:3]
    result = [("identity", identity)]
    positive = identity.copy()
    positive[:, 3] = (0.4, 1.2, 0.8)
    result.append(("positive_translation", positive))
    negative = identity.copy()
    negative[:, 3] = (-2.3, -1.7, -0.9)
    result.append(("negative_translation", negative))
    anisotropic = identity.copy()
    anisotropic[:, :3] = np.diag((2.35, 2.9, 1.7))
    anisotropic[:, 3] = (-1.5, 0.3, 0.9)
    result.append(("coarse_reference_grid", anisotropic))
    angle = 0.17
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = identity.copy()
    rotation[:, :3] = ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))
    rotation[:, 3] = (-1.4, 0.7, 0.3)
    result.append(("rotation", rotation))
    reverse = identity.copy()
    reverse[:, 0] = (-0.97, 0.13, -0.03)
    reverse[:, 3] = (16.3, -0.6, 1.1)
    result.append(("negative_x_direction", reverse))
    outside = identity.copy()
    outside[:, 3] = (1000, -1000, 1000)
    result.append(("empty_overlap", outside))
    zero = np.float32(0)
    upper = np.float32(17 - 1.0001)
    boundary_values = (
        ("zero", zero),
        ("below_zero", np.nextafter(zero, np.float32(-np.inf))),
        ("above_zero", np.nextafter(zero, np.float32(np.inf))),
        ("below_upper", np.nextafter(upper, np.float32(-np.inf))),
        ("upper", upper),
        ("above_upper", np.nextafter(upper, np.float32(np.inf))),
    )
    for name, boundary in boundary_values:
        constant = identity.copy()
        constant[:, 0] = 0
        constant[:, 3] = (boundary, 0.75, 0.8)
        result.append(("fov_" + name, constant))
    threshold = np.float32(1e-8)
    for name, direction in (
        ("zero", zero),
        ("tiny_positive", np.float32(0.5e-8)),
        ("tiny_negative", np.float32(-0.5e-8)),
        ("below_threshold", np.nextafter(threshold, zero)),
        ("threshold_float32", threshold),
        ("above_threshold", np.nextafter(threshold, np.float32(np.inf))),
    ):
        near_zero = identity.copy()
        near_zero[:, 0] = (direction, 1, 0)
        near_zero[:, 3] = (-np.float32(1e-8), 0.3, 0.8)
        result.append(("near_zero_" + name, near_zero))
    return result


@pytest.mark.parametrize("case,coefficients", _coefficients(), ids=[
    name for name, _ in _coefficients()
])
def test_fused_preparation_keeps_scalar_sampling_bits(case, coefficients):
    from fnit.mcflirt._cost_cuda import FusedMotionSampler

    reference, moving, moving_sizes = _images()
    sampler = FusedMotionSampler(reference, moving, moving_sizes)
    expected = _tensor_preparation(reference, moving, moving_sizes, coefficients)

    actual = sampler.prepare(coefficients)

    _assert_bits(actual, expected)
    # With identical prepared inputs the unchanged running-count reducer must
    # retain the exact cost, including the empty-overlap sentinel of 1.
    actual_cost = _normcorr_reduce(*actual).detach().cpu().numpy()
    expected_cost = _normcorr_reduce(*expected).detach().cpu().numpy()
    np.testing.assert_array_equal(actual_cost.view(np.uint32), expected_cost.view(np.uint32))
    if case == "empty_overlap":
        assert float(actual_cost) == 1.0


def test_sampler_reuses_buffers_without_stale_values():
    from fnit.mcflirt._cost_cuda import FusedMotionSampler

    reference, moving, moving_sizes = _images((3, 4, 5))
    sampler = FusedMotionSampler(reference, moving, moving_sizes)
    cases = dict(_coefficients())
    for name in ("identity", "negative_translation", "empty_overlap", "identity"):
        coefficients = cases[name]
        actual = sampler.prepare(coefficients)
        expected = _tensor_preparation(reference, moving, moving_sizes, coefficients)
        _assert_bits(actual, expected)
