"""源逐方向标记、共享顶点/面rip排除与显式Torch契约；真实回归单独报告。"""

import numpy as np
import pytest

from fnit.recon_all import place_surface_intersection_marking as marking


@pytest.fixture
def crossing():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [.25, .25, -1], [.25, .25, 1], [.75, .25, 0]], np.float32)
    return vertices, np.array([[0, 1, 2], [3, 4, 5]], np.int32)


@pytest.mark.parametrize("backend", ["numba", "torch"])
def test_crossing_and_face_rip_exclusion(crossing, backend):
    xyz, faces = crossing
    marked, count = marking.mark_source_intersections(xyz, faces, predicate_backend=backend, device="cpu")
    assert count == 2 and marked.all()
    marked, count = marking.mark_source_intersections(xyz, faces, face_ripped=np.array([True, False]),
                                                    predicate_backend=backend, device="cpu")
    assert count == 0 and not marked.any()


def test_shared_vertex_is_excluded_before_predicate(crossing):
    xyz, faces = crossing
    faces[1, 0] = 0
    marked, count = marking.mark_source_intersections(xyz, faces)
    assert count == 0 and not marked.any()


def test_compiled_candidate_filter_preserves_order_and_exact_boundaries():
    # fa=0；4mm边界保留，下一double排除；面冻结/共享顶点/桶不交各自排除。
    centers = np.zeros((7, 3), np.float64)
    centers[1, 0] = 4.
    centers[2, 0] = np.nextafter(4., np.inf)
    radii = np.zeros(7, np.float64)
    faces = np.arange(21, dtype=np.int32).reshape(7, 3)
    faces[4, 2] = faces[0, 0]
    flags = np.zeros(7, bool)
    flags[3] = True
    lows = np.zeros((7, 3), np.int64)
    highs = np.zeros((7, 3), np.int64)
    lows[5, 0] = highs[5, 0] = 1
    candidates = np.array([6, 1, 0, 2, 3, 4, 5], np.int64)
    first, second = marking._candidate_pairs_from_csr(
        centers, radii, lows, highs, faces, flags, 0,
        np.array([0, 7], np.int64), candidates)
    np.testing.assert_array_equal(first, [0, 0])
    np.testing.assert_array_equal(second, [6, 1])


def test_compiled_candidate_filter_respects_chunk_origin_and_empty_rows():
    # CSR首行为空；实际查询面从3起，不能误按局部行号匹配共享顶点。
    centers = np.zeros((6, 3), np.float64)
    faces = np.arange(18, dtype=np.int32).reshape(6, 3)
    radii = np.zeros(6, np.float64)
    cells = np.zeros((6, 3), np.int64)
    first, second = marking._candidate_pairs_from_csr(
        centers, radii, cells, cells, faces, np.zeros(6, bool), 3,
        np.array([0, 0, 2], np.int64), np.array([5, 2], np.int64))
    np.testing.assert_array_equal(first, [4])
    np.testing.assert_array_equal(second, [5])


def test_compiled_radius_and_bucket_filter_matches_numpy_at_boundary():
    # 同原NumPy的sum/<=判定比较：半径相等与±1ULP、整数桶接触和不交。
    radii = np.array([.25, .125, .375, .5, .125, .375, .5], np.float64)
    limit = radii[1:] + radii[0] + 4.
    centers = np.zeros((7, 3), np.float64)
    centers[1:, 0] = limit
    centers[2, 0] = np.nextafter(centers[2, 0], np.inf)
    centers[3, 0] = np.nextafter(centers[3, 0], 0.)
    low = np.zeros((7, 3), np.int64)
    high = np.ones((7, 3), np.int64)
    low[4, 1] = high[4, 1] = 1  # exactly touching integer boundary
    low[5, 1] = high[5, 1] = 2  # one whole bucket apart
    order = np.array([6, 5, 4, 3, 2, 1], np.int64)
    near = np.sum((centers[order]-centers[0])**2, axis=1) <= (radii[order]+radii[0]+4.)**2
    overlap = np.all(low[0] <= high[order], axis=1) & np.all(low[order] <= high[0], axis=1)
    first, second = marking._candidate_pairs_from_csr(
        centers, radii, low, high, np.arange(21, dtype=np.int32).reshape(7, 3),
        np.zeros(7, bool), 0, np.array([0, 6], np.int64), order)
    np.testing.assert_array_equal(first, np.zeros(np.count_nonzero(near & overlap), np.int64))
    np.testing.assert_array_equal(second, order[near & overlap])


def test_explicit_smaller_complete_grid_preserves_source_rules(crossing):
    xyz,faces=crossing
    for flags in (None,np.array([True,False])):
        control=marking.mark_source_intersections(xyz,faces,face_ripped=flags,
            predicate_backend="torch",device="cpu",candidate_grid_cells_per_axis=2)
        candidate=marking.mark_source_intersections(xyz,faces,face_ripped=flags,
            predicate_backend="torch",device="cpu",candidate_grid_cells_per_axis=3)
        np.testing.assert_array_equal(candidate[0],control[0]);assert candidate[1]==control[1]


def test_nondefault_grid_is_not_ignored(crossing):
    with pytest.raises(ValueError,match="requires torch predicate"):
        marking.mark_source_intersections(*crossing,candidate_grid_cells_per_axis=3)
    from fnit.recon_all.place_surface_final_cleanup import repair_intersections
    with pytest.raises(ValueError,match="requires source_torch"):
        repair_intersections(*crossing,np.zeros(len(crossing[0]),bool),candidate_grid_cells_per_axis=3)


def test_reverse_positive_does_not_mark_forward_face(crossing, monkeypatch):
    from fnit.recon_all import place_surface_collision_torch as source
    calls = []

    def ordered(first, second):
        calls.append(1)
        return np.full(len(first), len(calls) == 2, bool)

    monkeypatch.setattr(source, "_source_pairs", ordered)
    xyz, faces = crossing
    marked, count = marking.mark_source_intersections(xyz, faces)
    assert calls == [1, 1] and count == 1
    assert np.array_equal(marked, np.array([False, False, False, True, True, True]))


def test_predicate_positive_without_shared_hash_bucket_is_not_marked(crossing, monkeypatch):
    from fnit.recon_all import place_surface_collision_torch as source
    monkeypatch.setattr(source, "_source_pairs", lambda first, second: np.ones(len(first), bool))
    monkeypatch.setattr(marking, "_sample_mht_voxels", lambda triangle: {tuple(triangle[0])})
    marked, count = marking.mark_source_intersections(*crossing)
    assert count == 0 and not marked.any()


@pytest.mark.parametrize("backend", ["numba", "torch"])
def test_small_area_source_plane_tolerance_keeps_shared_bucket_pair(backend):
    # 源平面距离未归一化：几何AABB不重叠仍可能通过固定源谓词。
    first = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0]], np.float32)
    vertices = np.concatenate((first, first + np.float32(.0001)))
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    marked, count = marking.mark_source_intersections(vertices, faces,
        predicate_backend=backend, device="cpu")
    assert count == 2 and marked.all()


@pytest.mark.parametrize("kwargs", [{"predicate_backend": "bad"}, {"predicate_backend": "torch"},
                                  {"block_faces": 0}, {"block_pairs": .5}, {"face_ripped": [True]}])
def test_invalid_configuration_fails(crossing, kwargs):
    with pytest.raises(ValueError):
        marking.mark_source_intersections(*crossing, **kwargs)


def test_cleanup_reuses_only_identical_geometry_marks(monkeypatch):
    from fnit.recon_all.place_surface_final_cleanup import repair_intersections
    calls = []

    def marker(vertices, faces, **kwargs):
        calls.append(vertices.copy())
        return np.ones(len(vertices), bool), 1

    monkeypatch.setattr(marking, "mark_source_intersections", marker)
    vertices = np.zeros((3, 3), np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    _, report = repair_intersections(vertices, faces, np.zeros(3, bool), marking_backend="source_numba")
    assert len(calls) == report["marker_evaluations"] == 1
    assert report["marker_identical_geometry_cache_hits"] == 16
    assert report["smoothing_iterations"] == 1600


def test_cleanup_invalidates_marks_after_coordinate_change(monkeypatch):
    from fnit.recon_all.place_surface_final_cleanup import repair_intersections
    calls = []

    def marker(vertices, faces, **kwargs):
        calls.append(vertices.copy())
        return np.ones(len(vertices), bool), 1 if len(calls) == 1 else 0

    monkeypatch.setattr(marking, "mark_source_intersections", marker)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    _, report = repair_intersections(vertices, faces, np.zeros(3, bool), marking_backend="source_numba")
    assert len(calls) == report["marker_evaluations"] == 2
    assert report["marker_identical_geometry_cache_hits"] == 0
    assert not np.array_equal(calls[0], calls[1])
