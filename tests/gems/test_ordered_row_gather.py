"""Mathematical checks for repeated-row gradients; not image benchmarks."""

import numpy as np
import pytest
import torch

from fnit.gems.deformation import (ashburner_prior, ordered_row_gather,
                                  prepare_current_geometry,
                                  prepare_deformation_reference,
                                  prepare_vertex_reduction)
from fnit.gems.rasterize import build_block_index, rasterize_priors_compact


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("trailing_shape", ((), (3,), (3, 3)))
def test_ordered_row_gather_preserves_forward_and_repeated_row_derivative(dtype, trailing_shape):
    shape = (5, *trailing_shape)
    values = torch.arange(np.prod(shape), dtype=dtype).reshape(shape).requires_grad_()
    ids = torch.tensor([[2, 0, 2], [1, 2, 0]])
    selected = ordered_row_gather(values, ids)
    assert torch.equal(selected, values[ids])
    gradient = torch.arange(selected.numel(), dtype=dtype).reshape(selected.shape) / 7
    selected.backward(gradient)
    expected = torch.zeros(shape, dtype=torch.float64)
    for row, contribution in zip(ids.reshape(-1), gradient.reshape(ids.numel(), *trailing_shape)):
        expected[row] += contribution.double()
    assert values.grad.dtype == dtype
    torch.testing.assert_close(values.grad, expected.to(dtype), atol=0, rtol=0)


def test_ordered_row_gather_shared_layout_finite_difference_and_empty_ids():
    values = (torch.arange(24, dtype=torch.float64).reshape(4, 3, 2) / 10).requires_grad_()
    ids = torch.tensor([2, 1, 2, 0, 2, 1])[::2]  # Repeated, non-contiguous IDs.
    reduction = prepare_vertex_reduction(ids, len(values))
    assert torch.autograd.gradcheck(lambda data: ordered_row_gather(data, ids, reduction), (values,))
    empty = ordered_row_gather(values, torch.empty((0,), dtype=torch.long))
    gradient, = torch.autograd.grad(empty.sum(), values)
    assert gradient.shape == values.shape and torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_compact_ordered_rows_preserve_priors_objective_and_positive_jacobian(dtype):
    vertices = torch.tensor([[.13, .21, .31], [5.13, .21, .31],
                             [.13, 5.21, .31], [.13, .21, 5.31]], dtype=dtype)
    tetrahedra = torch.tensor([[0, 1, 2, 3]])
    alphas = torch.tensor([[.1, .9], [.3, .7], [.7, .3], [.9, .1]], dtype=dtype)
    shape = (7, 7, 7)
    index = build_block_index(vertices.numpy(), tetrahedra.numpy(), shape, block_size=4)
    xyz = torch.stack(torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij"), -1)
    mask = (xyz > 0).all(-1) & (xyz.sum(-1) <= 5)
    reference = prepare_deformation_reference(vertices, tetrahedra)
    results = []
    for stable in (False, True):
        current = vertices.clone()
        current[1, 0] += .11
        current.requires_grad_()
        geometry = prepare_current_geometry(current, tetrahedra, deterministic_gradient=stable)
        assert geometry.deterministic_gradient is stable
        priors, covered = rasterize_priors_compact(
            current, tetrahedra, alphas, shape, valid_mask=mask, block_index=index,
            background_channel=1, current_geometry=geometry)
        penalty, jacobian = ashburner_prior(
            current, vertices, tetrahedra, .05, reference_geometry=reference,
            current_geometry=geometry, analytic_gradient=True, double_accumulation=True)
        assert torch.all(jacobian > 0)
        weights = torch.arange(priors.numel(), dtype=dtype).reshape(priors.shape) / 11
        objective = (priors * weights).sum(dtype=torch.float64) + penalty
        gradient, = torch.autograd.grad(objective, current)
        assert gradient.dtype == dtype and torch.isfinite(gradient).all()
        results.append((priors.detach(), covered, objective.detach(), gradient))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(results[0][3], results[1][3], atol=tolerance, rtol=tolerance)


def test_default_compact_geometry_does_not_use_ordered_gather(monkeypatch):
    import fnit.gems.rasterize as rasterize
    vertices = torch.tensor([[.1, .1, .1], [4.1, .1, .1],
                             [.1, 4.1, .1], [.1, .1, 4.1]], requires_grad=True)
    tetrahedra = torch.tensor([[0, 1, 2, 3]])
    alphas = torch.tensor([[.1, .9], [.3, .7], [.7, .3], [.9, .1]])
    shape = (6, 6, 6)
    index = build_block_index(vertices.detach().numpy(), tetrahedra.numpy(), shape, block_size=4)
    geometry = prepare_current_geometry(vertices, tetrahedra)
    calls = []
    original = rasterize.ordered_row_gather
    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(rasterize, "ordered_row_gather", observed)
    for stable in (False, True):
        geometry = prepare_current_geometry(vertices, tetrahedra, deterministic_gradient=stable)
        rasterize.rasterize_priors_compact(
            vertices, tetrahedra, alphas, shape, valid_mask=torch.ones(shape, dtype=torch.bool),
            block_index=index, current_geometry=geometry)
        assert len(calls) == (2 if stable else 0)
