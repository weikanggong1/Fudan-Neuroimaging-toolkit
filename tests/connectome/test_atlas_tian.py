"""Checks for the discrete FNIRT inverse field operators."""

import torch

from fnit.connectome.atlas_tian import _fill_undefined, _tetrahedral_point


def test_tetrahedral_inverse_recovers_source_position():
    source = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]])
    affine = torch.tensor([[1.2, 0.1, 0.0], [0.0, 0.9, 0.1],
                           [0.0, 0.0, 1.1]], dtype=torch.float64)
    mapped = source.to(torch.float64) @ affine.T + torch.tensor([2., 3., 4.])
    expected = torch.tensor([[0.3, 0.5, 0.8]], dtype=torch.float64)
    point = expected @ affine.T + torch.tensor([2., 3., 4.])
    actual = _tetrahedral_point(mapped, source, point)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_undefined_field_uses_fsl_six_neighbour_boundary_rule():
    values = torch.zeros((5, 3, 3, 3), dtype=torch.float32)
    valid = torch.ones(values.shape[:3], dtype=torch.bool)
    valid[1, 1, 1] = False
    values[0, 1, 1] = 100  # FSL excludes the lower neighbour at index 1.
    values[2, 1, 1] = 2
    values[1, 0, 1] = 3
    values[1, 2, 1] = 4
    values[1, 1, 0] = 5
    values[1, 1, 2] = 6
    actual = _fill_undefined(values, valid)
    torch.testing.assert_close(actual[1, 1, 1], torch.full((3,), 4.0))
