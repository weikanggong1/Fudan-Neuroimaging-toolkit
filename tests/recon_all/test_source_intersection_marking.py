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
