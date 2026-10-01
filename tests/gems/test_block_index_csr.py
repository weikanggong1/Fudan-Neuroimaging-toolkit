"""CPU candidate-index regression against the original tetrahedron traversal."""

import math

import numpy as np
import pytest

from fnit.gems.rasterize import build_block_index


def _original_candidates(vertices, tetrahedra, shape, block_size, margin):
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    nblocks = tuple(int(math.ceil(s / block_size)) for s in shape)
    candidates = [[] for _ in range(np.prod(nblocks))]
    if len(tetrahedra):
        xyz = vertices[tetrahedra]
        lo = np.maximum(np.floor(xyz.min(1) - margin).astype(int), 0)
        hi = np.minimum(np.ceil(xyz.max(1) + margin).astype(int), np.asarray(shape) - 1)
        for tid in range(len(tetrahedra)):
            if np.any(hi[tid] < lo[tid]):
                continue
            blo, bhi = lo[tid] // block_size, hi[tid] // block_size
            for bx in range(blo[0], bhi[0] + 1):
                for by in range(blo[1], bhi[1] + 1):
                    base = (bx * nblocks[1] + by) * nblocks[2]
                    for bz in range(blo[2], bhi[2] + 1):
                        candidates[base + bz].append(tid)
    return tuple(np.asarray(ids, dtype=np.int64) for ids in candidates)


def _assert_candidates_equal(actual, expected):
    assert len(actual) == len(expected)
    assert all(ids.dtype == np.int64 and ids.ndim == 1 for ids in actual)
    np.testing.assert_array_equal([len(ids) for ids in actual], [len(ids) for ids in expected])
    if actual:
        np.testing.assert_array_equal(np.concatenate(actual), np.concatenate(expected))


@pytest.mark.parametrize("block_size", [1, 4, 8, 13])
@pytest.mark.parametrize("margin", [-0.25, 0., .2, 3.])
def test_negative_boundary_sliver_and_nonintersecting_boxes_keep_exact_order(block_size, margin):
    vertices = np.asarray([
        [1, 1, 1], [7, 1, 1], [1, 6, 1], [1, 1, 8],
        [-4, -2, -1], [2, -2, -1], [-4, 3, -1], [-4, -2, 4],
        [21, 15, 17], [27, 15, 17], [21, 21, 17], [21, 15, 23],
        [4, 4, 4], [11, 4, 4], [4, 11, 4], [4, 4, 4.0001],
        [-10, -10, -10], [-9, -10, -10], [-10, -9, -10], [-10, -10, -9],
        [40, 40, 40], [42, 40, 40], [40, 42, 40], [40, 40, 42],
    ], dtype=np.float64)
    tetrahedra = np.arange(len(vertices)).reshape(-1, 4)[[3, 1, 5, 0, 2, 4, 0]]
    shape = (23, 17, 19)
    result = build_block_index(vertices, tetrahedra, shape, block_size, margin)
    expected = _original_candidates(vertices, tetrahedra, shape, block_size, margin)
    assert result.shape == shape and result.block_size == block_size
    _assert_candidates_equal(result.candidates, expected)


@pytest.mark.parametrize("shape", [(17, 11, 9), (0, 11, 9)])
def test_empty_tetrahedra(shape):
    vertices, tetrahedra = np.empty((0, 3)), np.empty((0, 4), dtype=np.int64)
    result = build_block_index(vertices, tetrahedra, shape, block_size=4)
    _assert_candidates_equal(result.candidates, _original_candidates(vertices, tetrahedra, shape, 4, 1.))


def test_all_tetrahedra_outside_the_grid():
    vertices = np.asarray([[-8, -8, -8], [-6, -8, -8], [-8, -6, -8], [-8, -8, -6]], float)
    tetrahedra = np.asarray([[0, 1, 2, 3]])
    result = build_block_index(vertices, tetrahedra, (17, 11, 9), block_size=4, margin=0)
    assert all(len(ids) == 0 for ids in result.candidates)
    _assert_candidates_equal(result.candidates, _original_candidates(vertices, tetrahedra, (17, 11, 9), 4, 0))


def test_expansion_crosses_chunk_boundary_without_reordering():
    vertices = np.asarray([[0, 0, 0], [32, 0, 0], [0, 32, 0], [0, 0, 32]], float)
    tetrahedra = np.tile([[0, 1, 2, 3]], (40, 1))
    shape = (32, 32, 32)
    result = build_block_index(vertices, tetrahedra, shape, block_size=1, margin=0)
    assert sum(len(ids) for ids in result.candidates) == 1_310_720
    _assert_candidates_equal(result.candidates, _original_candidates(vertices, tetrahedra, shape, 1, 0))


def test_high_pair_fallback_does_not_allocate_global_sort_buffers(monkeypatch):
    import fnit.gems.rasterize as rasterize

    def sorting_would_exceed_budget(*args, **kwargs):
        raise AssertionError("large-pair fallback must avoid global argsort")

    monkeypatch.setattr(rasterize.np, "argsort", sorting_would_exceed_budget)
    vertices = np.asarray([[0, 0, 0], [25, 0, 0], [0, 25, 0], [0, 0, 25]], float)
    tetrahedra = np.tile([[0, 1, 2, 3]], (540, 1))
    result = build_block_index(vertices, tetrahedra, (25, 25, 25), block_size=1, margin=0)
    assert sum(len(ids) for ids in result.candidates) == 8_437_500
    expected = np.arange(540, dtype=np.int64)
    for ids in result.candidates:
        np.testing.assert_array_equal(ids, expected)
