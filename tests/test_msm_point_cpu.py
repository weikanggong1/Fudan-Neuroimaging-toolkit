"""CPU geometry must retain Point division at shared triangle boundaries."""
import math

import numpy as np
import pytest
import torch

from fnit.msm import _point_cpu
from fnit.msm._affine import _tangent_basis
from fnit.msm._sphere_map import _area_weights, _normalize


def _scalar_normalize(row):
    length = math.sqrt((row[0]*row[0] + row[1]*row[1]) + row[2]*row[2])
    denominator = length if length > 1e-8 else 1.
    return [value/denominator for value in row]


def test_double_cpu_normalization_matches_ordered_point_arithmetic():
    values = np.random.default_rng(321).normal(size=(71, 3)) * 100
    values[:2] = [[0., 0., 0.], [1e-10, -2e-10, 3e-10]]
    expected = np.asarray([_scalar_normalize(row) for row in values])
    actual = _normalize(torch.from_numpy(values)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_cpu_tangents_follow_major_axis_and_zero_branches():
    normals = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                            [0., 0., 0.], [1., 1., 1.], [-3., 2., 1.]], dtype=torch.float64)
    first, second = _tangent_basis(normals)
    expected_first = []
    for x, y, z in normals.tolist():
        if abs(x) >= abs(y) and abs(x) >= abs(z):
            length = math.sqrt(z*z + y*y)
            tangent = [0., -z/length, y/length] if length else [0., 0., 1.]
        elif abs(y) >= abs(x) and abs(y) >= abs(z):
            length = math.sqrt(z*z + x*x)
            tangent = [-z/length, 0., x/length] if length else [0., 0., 1.]
        else:
            length = math.sqrt(y*y + x*x)
            tangent = [-y/length, x/length, 0.] if length else [1., 0., 0.]
        expected_first.append(tangent)
    expected_first = np.asarray(expected_first)
    expected_second = np.asarray([_scalar_normalize(row)
                                  for row in np.cross(normals.numpy(), expected_first)])
    np.testing.assert_array_equal(first.numpy(), expected_first)
    np.testing.assert_array_equal(second.numpy(), expected_second)


def test_cpu_area_weights_match_scalar_unsigned_areas():
    rng = np.random.default_rng(73)
    triangles, points = rng.normal(size=(19, 3, 3)), rng.normal(size=(19, 3))
    expected = []
    for (a, b, c), point in zip(triangles, points):
        def area(x, y):
            cross = np.cross(x-point, y-point)
            return math.sqrt((cross[0]*cross[0] + cross[1]*cross[1]) + cross[2]*cross[2]) * .5
        first, second, third = area(b, c), area(a, c), area(a, b)
        total = (first + second) + third
        expected.append([first/total, second/total, third/total])
    np.testing.assert_array_equal(_area_weights(torch.from_numpy(triangles),
                                               torch.from_numpy(points)).numpy(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_differentiable_cpu_geometry_keeps_tensor_path(dtype):
    values = torch.tensor([[.7, -.3, .4], [.2, .8, -.1]], dtype=dtype, requires_grad=True)
    assert not _point_cpu.enabled(values)
    first, second = _tangent_basis(_normalize(values))
    loss = (first * torch.tensor([.3, -.7, .2], dtype=dtype)).sum() + second[:, 0].sum()
    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


def test_float32_cpu_geometry_uses_existing_tensor_precision():
    values = torch.tensor([[1.1, -.2, .7]], dtype=torch.float32)
    assert not _point_cpu.enabled(values)
    actual = _normalize(values)
    expected = values / torch.sqrt((values[:, 0]**2 + values[:, 1]**2) + values[:, 2]**2)[:, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
