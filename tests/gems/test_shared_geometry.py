import numpy as np
import pytest
import torch

from fnit.gems.deformation import (ReferenceGeometry, ashburner_prior,
                                  prepare_current_geometry, prepare_deformation_reference)
from fnit.gems.rasterize import build_block_index, rasterize_priors, rasterize_priors_compact


def _mesh(dtype=torch.float32):
    reference = torch.tensor([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [2, 2, 2]], dtype=dtype)
    transform = reference.new_tensor([[1.2, -.3, .1], [.4, .9, -.1], [.1, .2, 1.1]])
    reference = reference @ transform.T + reference.new_tensor([1.137, 1.211, 1.317])
    # Adjacent cells share vertices and use opposite reference orientations.
    tetra = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 3]])
    vertices = reference + reference.new_tensor([[.02, -.03, .01], [.15, -.04, .06],
                                                [-.03, .05, .04], [.02, .03, -.08], [.03, -.02, .03]])
    alphas = reference.new_tensor([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7], [.3, .5, .2], [.2, .3, .5]])
    return reference, vertices, tetra, alphas


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('analytic', [False, True])
def test_shared_raster_and_prior_match_combined_value_and_vertex_gradient(compact, analytic):
    reference, initial, tetra, alphas = _mesh()
    shape = (7, 7, 7)
    mask = torch.arange(np.prod(shape)).reshape(shape) % 3 != 0
    index = build_block_index(initial.numpy(), tetra.numpy(), shape, block_size=2)
    ref_geometry = prepare_deformation_reference(reference, tetra)

    def evaluate(shared):
        vertices = initial.clone().requires_grad_(True)
        geometry = prepare_current_geometry(vertices, tetra) if shared else None
        if compact:
            priors, covered = rasterize_priors_compact(vertices, tetra, alphas, shape,
                block_index=index, valid_mask=mask, current_geometry=geometry)
        else:
            priors, covered = rasterize_priors(vertices, tetra, alphas, shape,
                block_index=index, current_geometry=geometry)
            priors, covered = priors[:, mask], covered[mask]
        prior, jacobian = ashburner_prior(vertices, reference, tetra, .05,
            reference_geometry=ref_geometry, current_geometry=geometry,
            analytic_gradient=analytic if shared else False)
        coefficients = torch.linspace(.2, 1.3, priors.numel()).reshape_as(priors)
        # Shared vertices receive both raster and prior contributions. Include
        # a Jacobian term to check both outputs of the analytic backward.
        objective = (priors.square() * coefficients).sum() + prior + .13 * jacobian.sum()
        gradient, = torch.autograd.grad(objective, vertices)
        return priors, covered, prior, jacobian, gradient

    ordinary, shared = evaluate(False), evaluate(True)
    assert torch.equal(ordinary[1], shared[1])
    for position in (0, 2, 3, 4):
        torch.testing.assert_close(shared[position], ordinary[position], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('analytic', [False, True])
@pytest.mark.parametrize('kind', ['inverted', 'collapsed', 'tiny_positive'])
def test_shared_prior_preserves_barrier_and_small_positive_determinant(analytic, kind):
    reference = torch.tensor([[0., 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    tetra = torch.tensor([[0, 1, 2, 3]])
    initial = reference.clone()
    initial[3, 2] = {'inverted': -1., 'collapsed': 0., 'tiny_positive': 2e-8}[kind]
    ref_geometry = prepare_deformation_reference(reference, tetra)

    def evaluate(shared):
        vertices = initial.clone().requires_grad_(True)
        current = prepare_current_geometry(vertices, tetra) if shared else None
        cost, jacobian = ashburner_prior(vertices, reference, tetra, .05, invalid_penalty=100.,
            reference_geometry=ref_geometry, current_geometry=current,
            analytic_gradient=analytic if shared else False)
        gradient, = torch.autograd.grad(cost, vertices)
        return cost, jacobian, gradient

    ordinary, shared = evaluate(False), evaluate(True)
    assert torch.isfinite(shared[0]) and torch.isfinite(shared[2]).all()
    pairs = zip(shared[:2], ordinary[:2]) if kind == 'collapsed' else zip(shared, ordinary)
    for actual, expected in pairs:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-5)
    if kind == 'collapsed':
        assert float(shared[0]) == 100.
        assert float(shared[1][0]) == 0.
        torch.testing.assert_close(shared[2][3, 2], torch.tensor(-50.))
        # The barrier derivative is well-defined from its det<=0 branch. The
        # legacy torch determinant backward returns zero at this rank-2 cell.
        moved = initial.clone()
        moved[3, 2] = -1e-3
        moved_cost, _ = ashburner_prior(moved, reference, tetra, .05, invalid_penalty=100.,
            reference_geometry=ref_geometry, current_geometry=prepare_current_geometry(moved, tetra),
            analytic_gradient=analytic)
        torch.testing.assert_close((moved_cost - shared[0]) / -1e-3, shared[2][3, 2], atol=.01, rtol=1e-4)
    elif kind == 'tiny_positive':
        assert 0 < float(shared[1][0]) < torch.finfo(torch.float32).eps


def test_analytic_prior_and_shared_inverse_pass_finite_difference_gradcheck():
    reference, initial, tetra, _ = _mesh(torch.float64)
    ref_geometry = prepare_deformation_reference(reference, tetra)
    vertices = initial.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: prepare_current_geometry(v, tetra).inverse_edges,
                                   (vertices,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradcheck(lambda v: ashburner_prior(v, reference, tetra, .05,
        reference_geometry=ref_geometry, current_geometry=prepare_current_geometry(v, tetra),
        analytic_gradient=True), (vertices,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('analytic', [False, True])
def test_shared_geometry_needs_no_second_inverse_and_accepts_legacy_reference_cache(monkeypatch, analytic):
    reference, initial, tetra, alphas = _mesh()
    vertices = initial.clone().requires_grad_(True)
    current = prepare_current_geometry(vertices, tetra)
    reference_cache = prepare_deformation_reference(reference, tetra)
    legacy_cache = ReferenceGeometry(reference_cache.inverse_edges, reference_cache.volumes)
    index = build_block_index(initial.numpy(), tetra.numpy(), (7, 7, 7), block_size=2)

    def forbidden_inverse(*args, **kwargs):
        raise AssertionError('current geometry was already inverted')

    monkeypatch.setattr(torch.linalg, 'inv', forbidden_inverse)
    monkeypatch.setattr(torch.linalg, 'inv_ex', forbidden_inverse)
    priors, _ = rasterize_priors(vertices, tetra, alphas, (7, 7, 7), block_index=index,
                                 current_geometry=current)
    cost, _ = ashburner_prior(vertices, reference, tetra, .05, reference_geometry=legacy_cache,
                             current_geometry=current, analytic_gradient=analytic)
    gradient, = torch.autograd.grad(priors.square().sum() + cost, vertices)
    assert torch.isfinite(gradient).all()


def test_singular_shared_geometry_has_finite_inverse_and_zero_masked_raster_gradient():
    vertices = torch.tensor([[1., 1, 1], [3, 1, 1], [1, 3, 1], [2, 2, 1]], requires_grad=True)
    tetra = torch.tensor([[0, 1, 2, 3]])
    alphas = torch.tensor([[.8, .2]]).repeat(4, 1)
    current = prepare_current_geometry(vertices, tetra)
    assert current.singular.all() and torch.isfinite(current.inverse_edges).all()
    index = build_block_index(vertices.detach().numpy(), tetra.numpy(), (5, 5, 5))
    priors, covered = rasterize_priors(vertices, tetra, alphas, (5, 5, 5), block_index=index,
                                       current_geometry=current)
    assert not covered.any()
    gradient, = torch.autograd.grad(priors.square().sum(), vertices)
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize('analytic', [False, True])
def test_empty_shared_geometry_prior(analytic):
    reference, initial, _, _ = _mesh()
    tetra = torch.empty((0, 4), dtype=torch.long)
    vertices = initial.requires_grad_(True)
    current = prepare_current_geometry(vertices, tetra)
    cost, jacobian = ashburner_prior(vertices, reference, tetra, .05,
        reference_geometry=prepare_deformation_reference(reference, tetra),
        current_geometry=current, analytic_gradient=analytic)
    gradient, = torch.autograd.grad(cost, vertices)
    assert float(cost) == 0 and jacobian.numel() == 0
    assert torch.count_nonzero(gradient) == 0
