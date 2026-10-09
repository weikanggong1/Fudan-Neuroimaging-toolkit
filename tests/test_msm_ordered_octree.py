"""Generated geometry and literal arithmetic gates; these are not MRI benchmarks."""
import numpy as np
import pytest
import torch

from fnit.msm import _fastpd_native
from fnit.msm._execution import current_statistics, execution_budget
from fnit.msm._sphere_map import RadialSphereMap
from fnit.msm.msmsulc import _ico, _sphere_warp


def deformed_shared_edges():
    vertices, faces = _ico(2)
    queries, _ = _ico(4)
    moved = vertices.copy()
    alpha = .9
    moved[0] = (1-alpha)*vertices[0] + alpha*vertices[82]
    moved *= 100/np.linalg.norm(moved, axis=1, keepdims=True)
    triangles = moved[faces]
    assert np.all(np.sum(np.cross(triangles[:, 1]-triangles[:, 0],
                                 triangles[:, 2]-triangles[:, 0])*triangles.mean(1), axis=1) > 0)
    warped = _sphere_warp(torch.from_numpy(queries), vertices, faces, torch.from_numpy(moved),
                          'cpu', 'optimized', source_precision=True).numpy()
    return moved, faces, warped


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('source_precision', [False, True])
def test_deformed_shared_edges_use_fixed_original_octree_faces(device, source_precision):
    vertices, faces, query = deformed_shared_edges()
    # Independent pinned af5c... newMSM Octree API measured these generated
    # queries. No real subject coordinates or registered sphere is stored.
    rows = [299, 327, 407, 1141, 1143, 1246, 1252]
    expected_faces = [76, 92, 139, 76, 76, 92, 92]
    for execution in ('reference', 'optimized'):
        mapper = RadialSphereMap(vertices, faces, device, execution=execution,
                                 source_precision=source_precision)
        ids, weights, patches = mapper.weights(torch.as_tensor(query[rows], device=device), batch_size=3)
        assert patches.cpu().tolist() == expected_faces
        assert torch.equal(ids.cpu(), torch.as_tensor(faces[expected_faces]))
        assert torch.isfinite(weights).all()
        torch.testing.assert_close(weights.sum(-1), torch.ones(len(rows), dtype=torch.float64, device=device),
                                   rtol=0, atol=3e-16)


def test_native_cpu_workers_keep_ordered_projection_byte_exact():
    vertices, faces, query = deformed_shared_edges()
    values = []
    for threads in (1, 4):
        with execution_budget(threads):
            mapper = RadialSphereMap(vertices, faces, 'cpu')
            nodes = mapper._leaf_nodes(torch.from_numpy(query))
            values.append(mapper._native_selection(torch.from_numpy(query), nodes))
            mapper.weights(torch.from_numpy(query))
            assert current_statistics()['octree_native_queries'] == len(query)
    for first, second in zip(*values):
        assert torch.equal(first, second)


def ordered_manual(vertices, faces, query, leaf, siblings, threads=1):
    v = np.asarray(vertices, np.float64); f = np.asarray(faces, np.int64)
    q = np.asarray(query, np.float64)
    offset = np.array([0, len(leaf)], np.int64)
    fallback = np.array([0, len(siblings)], np.int64)
    raw = _fastpd_native.source_ordered_selection(
        v.tobytes(), f.tobytes(), q.tobytes(), np.zeros(len(q), np.int32).tobytes(),
        offset.tobytes(), np.asarray(leaf, np.int32).tobytes(),
        fallback.tobytes(), np.asarray(siblings, np.int32).tobytes(), len(v), len(f), len(q), threads)
    return np.frombuffer(raw, np.float64).reshape(-1, 4)


def test_native_leaf_pool_precedes_better_global_candidate_and_keeps_strict_ties():
    vertices = [[-20, -20, 100], [20, -20, 100], [0, 20, 100],
                [-2, -2, 100], [2, -2, 100], [0, 2, 100]]
    faces = [[0, 1, 2], [3, 4, 5]]
    assert ordered_manual(vertices, faces, [[0, 0, 90]], [0], [1])[0, 0] == 0
    assert ordered_manual(vertices, faces, [[0, 0, 90]], [0, 1], [])[0, 0] == 1
    # Exact distance ties preserve candidate order, including sibling order.
    vertices = [[0, 0, 100], [10, 0, 100], [0, 10, 100], [10, 10, 100]]
    faces = [[0, 1, 2], [1, 3, 2]]
    assert ordered_manual(vertices, faces, [[5, 5, 100]], [], [1, 0])[0, 0] == 1


def test_native_empty_leaf_uses_direct_siblings_then_nearest_corner():
    vertices = [[0, 0, 100], [10, 0, 100], [0, 10, 100], [10, 10, 100]]
    faces = [[0, 1, 2], [1, 3, 2]]
    found = ordered_manual(vertices, faces, [[2, 3, 90]], [], [0, 1])
    np.testing.assert_allclose(found[0], [0, 2*(100/90), 3*(100/90), 100], rtol=0, atol=1e-14)
    # Both triangles reject this projected point; the closest sibling corner
    # selects face 1, while the final projection remains outside its triangle.
    last = ordered_manual(vertices, faces, [[20, 20, 100]], [], [0, 1])
    np.testing.assert_array_equal(last[0], [1, 20, 20, 100])


def test_native_selection_rejects_invalid_query_and_candidate_buffers():
    vertices = [[0, 0, 100], [10, 0, 100], [0, 10, 100]]
    faces = [[0, 1, 2]]
    with pytest.raises(ValueError, match='within'):
        ordered_manual(vertices, faces, [[102, 0, 100]], [0], [])
    with pytest.raises(ValueError, match='outside the mesh'):
        ordered_manual(vertices, faces, [[0, 0, 100]], [2], [])
    with pytest.raises(ValueError, match='float64 vertices'):
        _fastpd_native.build_ordered_face_octree(b'', b'', 3, 1)
