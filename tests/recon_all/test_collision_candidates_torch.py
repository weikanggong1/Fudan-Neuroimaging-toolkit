"""球/有界AABB候选完整性；人工几何仅算子回归，真实接受参考另测。"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from fnit.recon_all.place_surface_candidates_torch import conservative_face_candidates_torch


def test_full_sphere_candidates_and_adaptive_chunks():
    generator = np.random.default_rng(833)
    sources = generator.normal(size=(400, 3))
    queries = generator.normal(size=(60, 3))
    radii = generator.uniform(.1, 2.4, size=60)
    offsets, ids, info = conservative_face_candidates_torch(sources, queries, radii,
        device="cpu", query_chunk_size=30, maximum_chunk_candidates=1200)
    expected = cKDTree(sources).query_ball_point(queries, radii)
    assert offsets.shape == (61,) and offsets.dtype == np.int64 and ids.dtype == np.int32
    for query, reference in enumerate(expected):
        assert set(reference) == set(ids[offsets[query]:offsets[query+1]])
    assert info["adaptive_chunk_reductions"] > 0 and info["candidate_truncation"] is False


def test_checked_motion_boxes_and_shared_vertices():
    random = np.random.default_rng(684)
    triangles = random.normal(size=(120, 3, 3)).astype(np.float32)
    centers = triangles.mean(1, dtype=np.float64)
    low, high = triangles.min(1), triangles.max(1)
    faces = np.arange(360, dtype=np.int32).reshape(-1, 3)
    faces[1, 0] = faces[0, 0]
    radii = np.full(12, 2.5, np.float64)
    bound = .25
    offsets, ids, info = conservative_face_candidates_torch(centers, centers[:12], radii,
        source_low=low, source_high=high, query_low=low[:12], query_high=high[:12],
        motion_bound=bound, source_faces=faces, query_faces=faces[:12], device="cpu")
    original = cKDTree(centers).query_ball_point(centers[:12], radii)
    for query, reference in enumerate(original):
        expected = {face for face in reference if
            np.all(high[face].astype(np.float64)+2*bound >= low[query]) and
            np.all(low[face].astype(np.float64)-2*bound <= high[query]) and
            not np.any(faces[face, :, None] == faces[query, None, :])}
        assert expected == set(ids[offsets[query]:offsets[query+1]])
    assert info["retained_candidates"] < info["sphere_candidates"]


def test_zero_radius_empty_and_failure_are_explicit():
    points = np.array([[0.,0,0], [1.,0,0], [0.,2,0]])
    offsets, ids, _ = conservative_face_candidates_torch(points, points, np.zeros(3), device="cpu")
    np.testing.assert_array_equal(offsets, [0,1,2,3])
    np.testing.assert_array_equal(ids, [0,1,2])
    offsets, ids, _ = conservative_face_candidates_torch(np.empty((0,3)), points, np.zeros(3), device="cpu")
    np.testing.assert_array_equal(offsets, [0,0,0,0])
    assert len(ids) == 0
    with pytest.raises(ValueError, match="indexed"):
        conservative_face_candidates_torch(points, points, np.zeros(3), device="cuda")
    with pytest.raises(MemoryError, match="single complete query"):
        conservative_face_candidates_torch(points, points, np.ones(3)*10,
                                           maximum_chunk_candidates=1, device="cpu")
    with pytest.raises(ValueError, match="required together"):
        conservative_face_candidates_torch(points, points, np.zeros(3), source_low=points, device="cpu")
