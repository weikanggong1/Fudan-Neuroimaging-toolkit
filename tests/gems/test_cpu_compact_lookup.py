"""Packed CPU lookup preserves ownership, static points, fallbacks and gradients."""

import numba
import numpy as np
import pytest
import torch

from fnit.gems._raster_cpu import lookup_candidates_cpu
from fnit.gems import _raster_cpu_compact as packed
from fnit.gems import rasterize
from fnit.gems.deformation import prepare_current_geometry


def fixture():
    generator = torch.Generator().manual_seed(401)
    origins = torch.rand(256, 3, generator=generator)
    inverses = torch.rand(256, 3, 3, generator=generator) * 2 - 1
    singular = torch.zeros(256, dtype=torch.bool)
    singular[5::19] = True
    batches = []
    for count, width, candidates in ((2, 63, 32), (3, 31, 16), (1, 257, 64)):
        points = torch.rand(count, width, 3, generator=generator) * 3
        ids = torch.randperm(256, generator=generator)[:count * candidates].reshape(count, candidates)
        mask = torch.rand(count, candidates, generator=generator) > .3
        rows = torch.arange(count * width)[::3]
        batches.append((points, ids, mask, torch.arange(count)[:, None], rows))
    return tuple(batches), origins, inverses, singular


def reference(batches, origins, inverses, singular, tolerance=2e-5):
    selected, points, covered = [], [], []
    for point, ids, mask, _, rows in batches:
        chosen, inside = lookup_candidates_cpu(point, ids, mask, origins, inverses, singular,
                                               rows, tolerance=tolerance)
        selected.append(chosen)
        points.append(point.reshape(-1, 3)[rows])
        covered.append(inside)
    return torch.cat(selected), torch.cat(points), torch.cat(covered)


@pytest.mark.parametrize("mode", ["ordinary", "tie", "all_singular", "all_masked", "nan"])
def test_all_batch_results_match_ordered_cpu_reference(mode):
    batches, origins, inverses, singular = fixture()
    if mode == "tie":
        origins[:] = 0
        inverses[:] = torch.eye(3)
    elif mode == "all_singular":
        singular[:] = True
    elif mode == "all_masked":
        for _, _, mask, _, _ in batches:
            mask[:] = False
    elif mode == "nan":
        inverses[::3, 0, 0] = torch.nan
    actual = packed.lookup_compact_cpu(batches, origins, inverses, singular, {})
    for first, second in zip(actual, reference(batches, origins, inverses, singular)):
        assert torch.equal(first, second)


def test_static_point_cache_reuse_and_source_mutation_invalidation():
    batches, origins, inverses, singular = fixture()
    cache = {}
    first = packed.lookup_compact_cpu(batches, origins, inverses, singular, cache)
    origins += .17
    second = packed.lookup_compact_cpu(batches, origins, inverses, singular, cache)
    assert first[1].data_ptr() == second[1].data_ptr()
    for actual, expected in zip(second, reference(batches, origins, inverses, singular)):
        assert torch.equal(actual, expected)
    batches[0][0].add_(1)
    third = packed.lookup_compact_cpu(batches, origins, inverses, singular, cache)
    assert first[1].data_ptr() != third[1].data_ptr()
    for actual, expected in zip(third, reference(batches, origins, inverses, singular)):
        assert torch.equal(actual, expected)


def test_masks_and_rows_are_included_in_cache_version_guard():
    batches, origins, inverses, singular = fixture()
    cache = {}
    packed.lookup_compact_cpu(batches, origins, inverses, singular, cache)
    batches[0][2].logical_not_()
    batches[1][-1].copy_(batches[1][-1].flip(0))
    actual = packed.lookup_compact_cpu(batches, origins, inverses, singular, cache)
    for first, second in zip(actual, reference(batches, origins, inverses, singular)):
        assert torch.equal(first, second)


def test_autocast_and_non_fp32_geometry_request_original_path():
    batches, origins, inverses, singular = fixture()
    assert packed.lookup_compact_cpu(batches, origins.double(), inverses.double(), singular, {}) is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert packed.lookup_compact_cpu(batches, origins, inverses, singular, {}) is None


def test_numba_thread_budget_restores_on_kernel_failure(monkeypatch):
    batches, origins, inverses, singular = fixture()
    before_numba = numba.get_num_threads()
    before_torch = torch.get_num_threads()
    def failure(*args):
        assert numba.get_num_threads() == 1
        raise RuntimeError("private diagnostic failure")
    monkeypatch.setattr(packed, "_lookup_packed", failure)
    try:
        torch.set_num_threads(1)
        with pytest.raises(RuntimeError, match="private diagnostic failure"):
            packed.lookup_compact_cpu(batches, origins, inverses, singular, {})
        assert numba.get_num_threads() == before_numba
    finally:
        torch.set_num_threads(before_torch)


@pytest.mark.parametrize("stable", [False, True])
def test_actual_priors_and_vertex_alpha_gradients_are_exact(monkeypatch, stable):
    shape = (8, 7, 6)
    vertices = torch.tensor([[.13, .21, .17], [5.13, .21, .17],
                             [.73, 5.21, .17], [.13, .61, 5.17]])
    tetrahedra = torch.tensor([[0, 1, 2, 3]])
    alphas = torch.tensor([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7], [.3, .5, .2]])
    mask = torch.arange(np.prod(shape)).reshape(shape) % 3 != 0
    index = rasterize.build_block_index(vertices.numpy(), tetrahedra.numpy(), shape, block_size=4)
    def run():
        current_vertices = vertices.clone().requires_grad_(True)
        current_alphas = alphas.clone().requires_grad_(True)
        geometry = prepare_current_geometry(current_vertices, tetrahedra, deterministic_gradient=stable)
        priors, covered = rasterize.rasterize_priors_compact(
            current_vertices, tetrahedra, current_alphas, shape, valid_mask=mask,
            block_index=index, current_geometry=geometry)
        coefficient = torch.linspace(.1, 1.3, priors.numel()).reshape_as(priors)
        gradient = torch.autograd.grad((priors.square() * coefficient).sum(), (current_vertices, current_alphas))
        return priors, covered, gradient
    actual = run()
    monkeypatch.setattr(packed, "lookup_compact_cpu", lambda *args, **kwargs: None)
    expected = run()
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    for first, second in zip(actual[2], expected[2]):
        assert torch.equal(first, second)
