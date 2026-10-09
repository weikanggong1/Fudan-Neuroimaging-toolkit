"""较小网格仍须保留所有源球/bbox候选；模拟测试不替代真实放置。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.place_surface_candidates_torch import conservative_face_candidates_torch


def row_sets(offsets, ids):
    return [set(ids[offsets[i]:offsets[i+1]].tolist()) for i in range(len(offsets)-1)]


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("cells", [2, 3])
def test_complete_spheres_and_boundary(cells):
    rng = np.random.default_rng(4816)
    sources = rng.normal(size=(130, 3)).astype(np.float64)
    sources[:3] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]
    queries = np.concatenate((np.zeros((1, 3)), sources[::7]))
    radii = np.linspace(1, 2, len(queries), dtype=np.float64)
    offsets, ids, info = conservative_face_candidates_torch(
        source_centers=sources, query_centers=queries, radii=radii, device="cpu",
        query_chunk_size=8, maximum_chunk_candidates=1000, grid_cells_per_axis=cells,
    )
    actual = row_sets(offsets, ids)
    expected = [set(np.flatnonzero(np.linalg.norm(sources-q, axis=1) <= r).tolist())
                for q, r in zip(queries, radii)]
    assert actual == expected
    assert info["candidate_truncation"] is False
    assert info["grid_cells_per_axis"] == cells


def test_grids_keep_same_bbox_and_shared_vertex_candidates():
    rng = np.random.default_rng(530)
    sources = rng.normal(size=(80, 3)).astype(np.float64)
    queries = sources[::3].copy()
    radii = np.full(len(queries), 2.0, np.float64)
    source_faces = np.arange(len(sources)*3, dtype=np.int32).reshape(-1, 3)
    query_faces = source_faces[::3].copy()
    arguments = dict(source_centers=sources, query_centers=queries, radii=radii,
        source_low=sources-.4, source_high=sources+.4,
        query_low=queries-.4, query_high=queries+.4, motion_bound=.2,
        source_faces=source_faces, query_faces=query_faces,
        device="cpu", query_chunk_size=16, maximum_chunk_candidates=400)
    baseline = conservative_face_candidates_torch(**arguments, grid_cells_per_axis=2)
    candidate = conservative_face_candidates_torch(**arguments, grid_cells_per_axis=3)
    assert row_sets(*baseline[:2]) == row_sets(*candidate[:2])
    assert candidate[2]["adaptive_chunk_reductions"] > 0


@pytest.mark.parametrize("invalid", [0, 1, 4, 2.5, None])
def test_invalid_grid_count(invalid):
    with pytest.raises(ValueError, match="grid_cells_per_axis"):
        conservative_face_candidates_torch(
            source_centers=np.zeros((1, 3), np.float64),
            query_centers=np.zeros((1, 3), np.float64), radii=np.ones(1, np.float64),
            grid_cells_per_axis=invalid, device="cpu",
        )
