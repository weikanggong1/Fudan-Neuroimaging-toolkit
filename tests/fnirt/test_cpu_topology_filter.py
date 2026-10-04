"""Filter exactly the original corner scan; fixtures are not benchmarks."""
import numpy as np
import pytest
import torch

from fnit.fnirt._topology_cpu import out_of_range_corners


def original_indices(checks, minimum, maximum):
    nx, ny, nz = checks.shape[1:]
    return np.asarray([(((z * ny + y) * nx + x) * 8) + corner
                       for z in range(nz) for y in range(ny) for x in range(nx) for corner in range(8)
                       if not minimum <= checks[corner, x, y, z] <= maximum], dtype=np.int64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("minimum,maximum", [(0.01, 100.0), (0.2, 5.0), (0.1, 1.3)])
@pytest.mark.parametrize("contiguous", [True, False])
def test_filter_preserves_scalar_thresholds_nan_and_scan_order(dtype, minimum, maximum, contiguous):
    checks = np.random.default_rng(24).normal(1.0, 0.5, (8, 3, 4, 5)).astype(dtype)
    checks.reshape(-1)[:12] = np.asarray([minimum, np.nextafter(dtype(minimum), dtype(-np.inf)), np.nextafter(dtype(minimum), dtype(np.inf)),
                                       maximum, np.nextafter(dtype(maximum), dtype(-np.inf)), np.nextafter(dtype(maximum), dtype(np.inf)),
                                       np.nan, -np.inf, np.inf, 0.0, -0.0, 1.0], dtype=dtype)
    if not contiguous:
        checks = checks.transpose(0, 3, 2, 1)
    expected = original_indices(checks, minimum, maximum)
    actual = out_of_range_corners(checks, float(minimum), float(maximum))
    assert np.array_equal(actual, expected)


def test_entire_valid_volume_has_empty_filter():
    checks = np.ones((8, 3, 4, 5), dtype=np.float32)
    result = out_of_range_corners(checks, 0.2, 5.0)
    assert result.dtype == np.int64
    assert result.shape == (0,)


def original_limiter(gradient, checks, minimum, maximum):
    from fnit.fnirt.topology import _JACOBIAN_OFFSETS
    values = gradient.detach().contiguous().numpy()
    checks = checks.detach().contiguous().numpy()
    nx, ny, nz = values.shape[1:]
    identity = np.eye(3, dtype=np.float64)
    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                for index, offsets in enumerate(_JACOBIAN_OFFSETS):
                    if minimum <= checks[index, x, y, z] <= maximum:
                        continue
                    matrix = np.empty((3, 3), dtype=np.float64)
                    for row in range(3):
                        for column, (dx, dy, dz) in enumerate(offsets):
                            matrix[row, column] = values[3 * row + column, x + dx, y + dy, z + dz]
                    alpha = np.float32(0.0)
                    candidate = matrix
                    determinant = np.linalg.det(candidate)
                    while determinant < minimum or determinant > maximum:
                        alpha = np.float32(alpha + np.float32(0.1))
                        alpha = min(alpha, np.float32(1.0))
                        candidate = (1.0 - float(alpha)) * matrix + float(alpha) * identity
                        determinant = np.linalg.det(candidate)
                    alpha = min(np.float32(alpha + np.float32(0.1)), np.float32(1.0))
                    for row in range(3):
                        for column, (dx, dy, dz) in enumerate(offsets):
                            values[3 * row + column, x + dx, y + dy, z + dz] = np.float32((1.0 - float(alpha)) * matrix[row, column] + float(alpha) * identity[row, column])
    return gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["valid", "folded", "thresholds"])
def test_filtered_limiter_keeps_numpy_det_and_overlapping_updates_bitwise(dtype, mode):
    from fnit.fnirt.topology import gradient_field, jacobian_check, limit_gradient
    axes = torch.stack(torch.meshgrid(torch.arange(4.0), torch.arange(5.0), torch.arange(6.0), indexing="ij")).to(dtype)
    warp = axes.clone()
    if mode != "valid":
        warp = warp + 0.8 * torch.randn(warp.shape, dtype=dtype, generator=torch.Generator().manual_seed(483))
    gradient = gradient_field(warp, (1.0, 1.0, 1.0))
    checks = jacobian_check(warp, (1.0, 1.0, 1.0))
    if mode == "thresholds":
        checks.flatten()[:5] = torch.tensor([0.01, 100.0, float("nan"), float("inf"), -float("inf")], dtype=dtype)
    expected = original_limiter(gradient.clone(), checks, 0.01, 100.0)
    actual, backend = limit_gradient(gradient.clone(), checks, 0.01, 100.0, return_backend=True)
    integer_dtype = torch.int32 if dtype == torch.float32 else torch.int64
    assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))
    assert backend == "numpy-cpu-serial-order-numba-filter"
