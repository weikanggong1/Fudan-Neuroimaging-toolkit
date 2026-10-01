import numpy as np
import pytest
import torch

from fnit.gems import _raster_triton
from fnit.gems.deformation import prepare_current_geometry
from fnit.gems.rasterize import build_block_index, compact_data_cost


@pytest.fixture
def cuda():
    if not torch.cuda.is_available() or _raster_triton.triton is None:
        pytest.skip("CUDA and Triton required")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield "cuda"
    torch.backends.cuda.matmul.allow_tf32 = previous


def _mesh(device):
    vertices = torch.tensor([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [4, 4, 4]],
                            device=device, dtype=torch.float32)
    tetra = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 3]], device=device)
    return vertices, tetra


def test_hint_reuses_only_strict_interior_and_scans_boundary_and_missing_points(cuda):
    vertices, tetra = _mesh(cuda)
    geometry = prepare_current_geometry(vertices, tetra)
    points = torch.tensor([[[1, 1, 1], [1, 1, 2], [2, 2, 2], [3, 1, 1], [6, 6, 6]]],
                          device=cuda, dtype=torch.float32)
    ids = torch.tensor([[0, 1]], device=cuda)
    mask = torch.ones_like(ids, dtype=torch.bool)
    rows = torch.arange(points.shape[1], device=cuda)
    arguments = (points, ids, mask, geometry.origins, geometry.inverse_edges, geometry.singular, rows)
    exact = _raster_triton.lookup_candidates(*arguments)
    hinted = _raster_triton.lookup_candidates(*arguments, previous_selected=exact[0], return_hint_hits=True)
    assert torch.equal(hinted[0], exact[0]) and torch.equal(hinted[1], exact[1])
    assert torch.equal(hinted[2], torch.tensor([True, False, True, True, False], device=cuda))


def test_hint_falls_back_if_old_cell_became_singular(cuda):
    vertices, tetra = _mesh(cuda)
    geometry = prepare_current_geometry(vertices, tetra)
    points = torch.tensor([[[1., 1, 1]]], device=cuda)
    ids = torch.tensor([[0, 1]], device=cuda)
    mask = torch.ones_like(ids, dtype=torch.bool)
    rows = torch.arange(1, device=cuda)
    geometry.singular[0] = True
    arguments = (points, ids, mask, geometry.origins, geometry.inverse_edges, geometry.singular, rows)
    exact = _raster_triton.lookup_candidates(*arguments)
    hinted = _raster_triton.lookup_candidates(*arguments, previous_selected=torch.zeros(1, device=cuda, dtype=torch.long),
                                             return_hint_hits=True)
    assert torch.equal(hinted[0], exact[0]) and torch.equal(hinted[1], exact[1])
    assert not hinted[2].any()


def _data_inputs(device):
    initial, tetra = _mesh(device)
    initial += initial.new_tensor([1.137, 1.211, 1.317])
    alpha = initial.new_tensor([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7], [.3, .5, .2], [.2, .3, .5]])
    shape = (8, 8, 8)
    valid = torch.arange(np.prod(shape), device=device).reshape(shape) % 3 != 0
    likelihood = -torch.linspace(.1, 9, 3 * int(valid.sum()), device=device).reshape(3, -1)
    index = build_block_index(initial.cpu().numpy(), tetra.cpu().numpy(), shape, block_size=2, margin=3)
    return initial, tetra, alpha, shape, valid, likelihood, index


def _cost_gradient(inputs, displacement, hinted):
    initial, tetra, alpha, shape, valid, likelihood, index = inputs
    vertices = (initial + initial.new_tensor([displacement, -.3 * displacement, .2 * displacement])).requires_grad_(True)
    stats = {}
    cost = compact_data_cost(vertices, tetra, alpha, shape, valid_mask=valid, block_index=index,
        current_geometry=prepare_current_geometry(vertices, tetra), likelihood=likelihood,
        cache_owner_hints=hinted is not None, owner_hints=bool(hinted), hint_stats=stats)
    gradient, = torch.autograd.grad(cost, vertices)
    return cost.detach(), gradient, stats


def test_hint_cache_preserves_cost_and_gradient_through_boundary_crossings(cuda):
    inputs = _data_inputs(cuda)
    _cost_gradient(inputs, 0, False)
    reused = 0
    for displacement in (.01, .12, .4, -.12):
        exact = _cost_gradient(inputs, displacement, None)
        hinted = _cost_gradient(inputs, displacement, True)
        torch.testing.assert_close(hinted[0], exact[0], atol=.001, rtol=2e-6)
        torch.testing.assert_close(hinted[1], exact[1], atol=2e-6, rtol=2e-4)
        reused += int(hinted[2]["reused_points"])
    assert reused > 0


def test_new_index_and_changed_tetrahedron_order_invalidate_hints(cuda):
    inputs = _data_inputs(cuda)
    _cost_gradient(inputs, 0, False)
    assert int(_cost_gradient(inputs, .01, True)[2]["reused_points"]) > 0
    initial, tetra, alpha, shape, valid, likelihood, _ = inputs
    fresh_index = build_block_index(initial.cpu().numpy(), tetra.cpu().numpy(), shape, block_size=2, margin=3)
    fresh = (initial, tetra, alpha, shape, valid, likelihood, fresh_index)
    assert int(_cost_gradient(fresh, .01, True)[2]["reused_points"]) == 0
    tetra[:] = tetra.flip(0)
    assert int(_cost_gradient(inputs, .01, True)[2]["reused_points"]) == 0


def test_double_data_reduction_keeps_point_geometry_and_gradient_fp32(cuda):
    initial, tetra, alpha, shape, valid, likelihood, index = _data_inputs(cuda)
    outcomes = []
    for double in (False, True):
        vertices = initial.clone().requires_grad_(True)
        cost = compact_data_cost(vertices, tetra, alpha, shape, valid_mask=valid, block_index=index,
            current_geometry=prepare_current_geometry(vertices, tetra), likelihood=likelihood,
            double_accumulation=double)
        gradient, = torch.autograd.grad(cost, vertices)
        outcomes.append((cost, gradient))
    assert outcomes[0][0].dtype == torch.float32
    assert outcomes[1][0].dtype == torch.float64
    assert outcomes[1][1].dtype == torch.float32
    torch.testing.assert_close(outcomes[1][0].float(), outcomes[0][0], atol=.001, rtol=1e-6)
    torch.testing.assert_close(outcomes[1][1], outcomes[0][1], atol=2e-6, rtol=1e-5)

