"""缓存的动态依赖、源顺序、占用更新和不可变基底合同；非MRI benchmark。"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fnit.recon_all.topology_face_detection_cache import (
    detect_patch_faces_cached, prepare_face_detection_context,
)
from fnit.recon_all.topology_retessellation_python import detect_patch_faces
from fnit.recon_all.topology_retessellation_dynamic import retessellate_and_detect_candidate


def compare(xyz, vertices, neighbors, faces, context=None):
    args = dict(canonical=xyz, vertices=vertices, neighbors=neighbors, external_faces=faces)
    if context is None:
        context = prepare_face_detection_context(base_faces=faces, vertex_count=len(xyz))
    old, old_report = detect_patch_faces(**args)
    new, new_report = detect_patch_faces_cached(**args, context=context)
    assert old.dtype == new.dtype and old.shape == new.shape and old.tobytes() == new.tobytes()
    assert old_report == new_report
    return new_report


@pytest.fixture
def fixture():
    xyz = np.asarray([[1, 0, 1], [0, 1, 1], [-1, -1, 1], [0, 0, 1]], np.float32)
    neighbors = [[1, 2], [0, 2], [1, 0], []]
    return xyz, np.asarray([0, 1, 2], np.int32), neighbors, np.empty((0, 3), np.int32)


def test_complete_dynamic_some_and_all_use_explicit_cache(fixture):
    xyz, _, _, faces = fixture
    context = prepare_face_detection_context(base_faces=faces, vertex_count=4)
    for mode in ("some", "all"):
        args = dict(canonical=xyz, base_faces=faces, neighbors=[[], [], [], []],
            candidate_edges=np.asarray([[0, 3], [0, 1], [1, 2], [2, 0]], np.int32),
            edge_used=np.zeros(4, np.int32), ordering=np.arange(4), inside_vertices=np.arange(4),
            inside_discarded=np.zeros(4, np.bool_), border_vertices=np.empty(0, np.int32),
            hull_vertices=np.empty(0, np.int32), mode=mode)
        old = retessellate_and_detect_candidate(**args)
        new = retessellate_and_detect_candidate(**args, face_detection_context=context)
        assert old.keys() == new.keys()
        for name in old:
            if isinstance(old[name], np.ndarray):
                assert old[name].shape == new[name].shape and old[name].dtype == new[name].dtype
                assert old[name].tobytes() == new[name].tobytes()
            else:
                assert old[name] == new[name]


def test_face_order_dynamic_neighbor_order_and_repeated_base_faces(fixture):
    xyz, vertices, neighbors, faces = fixture
    for order in (vertices, vertices[::-1]):
        for rows in (neighbors, [row[::-1] for row in neighbors]):
            compare(xyz, order, rows, faces)
            compare(xyz, order, rows, np.asarray([[0, 1, 2], [0, 1, 2]], np.int32))


def test_current_neighbors_are_never_cached(fixture):
    xyz, vertices, neighbors, faces = fixture
    context = prepare_face_detection_context(base_faces=faces, vertex_count=4)
    accepted = compare(xyz, vertices, neighbors, faces, context)
    # 相同基底，但新增一个真实球面内点；必须重新消费本次邻接。
    contained_rows = [[1, 2, 3], [0, 2, 3], [1, 0, 3], [0, 1, 2]]
    rejected = compare(xyz, vertices, contained_rows, faces, context)
    assert accepted["accepted_faces"] > 0
    assert rejected["rejected_contains_neighbor"] > 0
    assert not context.face_identities and not context.edge_counts


def test_edge_overlay_preserves_two_face_capacity():
    xyz = np.asarray([[1, 0, 1], [0, 1, 1], [-1, -1, 1], [0, 0, -1],
                      [1, 0, -1], [0, 1, -1]], np.float32)
    faces = np.asarray([[0, 1, 4], [1, 0, 5]], np.int32)
    rows = [[1, 2, 4, 5], [0, 2, 4, 5], [0, 1], [], [0, 1], [0, 1]]
    report = compare(xyz, np.asarray([0, 1, 2], np.int32), rows, faces)
    assert report["rejected_fully_occupied_edge"] > 0


def test_changed_base_and_mutable_metadata_rejected(fixture):
    xyz, vertices, neighbors, _ = fixture
    faces = np.asarray([[0, 1, 2]], np.int32)
    context = prepare_face_detection_context(base_faces=faces, vertex_count=4)
    with pytest.raises(ValueError):
        context.faces.flags.writeable = True
    with pytest.raises(TypeError):
        context.edge_counts[(0, 1)] = 10
    faces[0] = [0, 2, 1]
    with pytest.raises(ValueError, match="complete external_faces"):
        detect_patch_faces_cached(canonical=xyz, vertices=vertices, neighbors=neighbors,
                                  external_faces=faces, context=context)
    context.faces.shape = (1, 1, 3)
    with pytest.raises(ValueError, match="immutable"):
        detect_patch_faces_cached(canonical=xyz, vertices=vertices, neighbors=neighbors,
                                  external_faces=faces, context=context)


def test_two_candidates_sharing_context_do_not_share_dynamic_overlay(fixture):
    xyz, vertices, neighbors, faces = fixture
    context = prepare_face_detection_context(base_faces=faces, vertex_count=4)
    with ThreadPoolExecutor(max_workers=2) as workers:
        calls = [workers.submit(compare, xyz, order, neighbors, faces, context)
                 for order in (vertices, vertices[::-1])]
        assert all(call.result()["accepted_faces"] for call in calls)
    assert context.faces.shape == (0, 3) and not context.face_identities and not context.edge_counts
