"""检查有序关联面的整数语义和成熟法向内核的逐位输出。"""

import numpy as np
import pytest
import torch

from fnit.recon_all.mris_register_nonlinear import ordered_face_incidence
from fnit.recon_all.place_surface_normals import (
    _normals, initial_vertex_normals, ordered_face_csr)


def _old_rows(faces, nvertices):
    # 冻结旧 Python 定义：按面及角点顺序追加，保留重复项。
    rows = [[] for _ in range(nvertices)]
    for face_no, face in enumerate(faces):
        for corner, vertex in enumerate(face):
            rows[vertex].append((face_no, corner))
    return rows


def _old_padded(faces, nvertices):
    rows = _old_rows(faces, nvertices)
    degrees = np.asarray([len(row) for row in rows], dtype=np.int64)
    width = int(degrees.max(initial=0))
    face_indices = np.zeros((nvertices, width), dtype=np.int64)
    corners = np.zeros_like(face_indices)
    for vertex, row in enumerate(rows):
        for index, (face_no, corner) in enumerate(row):
            face_indices[vertex, index] = face_no
            corners[vertex, index] = corner
    return face_indices, corners, degrees


@pytest.mark.parametrize("faces,nvertices", [
    (np.asarray([[0, 1, 2], [2, 0, 3]], dtype=np.int64), 5),
    (np.asarray([[0, 0, 1], [0, 0, 1], [2, 2, 2]], dtype=np.int32), 4),
    (np.empty((0, 3), dtype=np.int64), 3),
])
def test_csr_and_padded_match_old_face_corner_order(faces, nvertices):
    rows = _old_rows(faces, nvertices)
    offsets, face_ids, corners = ordered_face_csr(faces=faces, nvertices=nvertices)
    assert offsets.dtype == face_ids.dtype == corners.dtype == np.int64
    assert len(face_ids) == len(corners) == faces.size
    for vertex, expected in enumerate(rows):
        assert list(zip(face_ids[offsets[vertex]:offsets[vertex + 1]],
                        corners[offsets[vertex]:offsets[vertex + 1]])) == expected
    observed = ordered_face_incidence(faces=torch.from_numpy(faces), nvertices=nvertices)
    for expected, value in zip(_old_padded(faces, nvertices), observed):
        np.testing.assert_array_equal(value.numpy(), expected)
        assert value.dtype == torch.int64 and value.device.type == "cpu"


def test_normals_keep_old_float32_kernel_bit_patterns_and_isolated_zero():
    vertices = np.asarray([[1.5, -0.2, 0.3], [-0.4, 1.1, 0.7],
                           [0.2, 0.3, 1.6], [-0.8, -0.6, -0.5],
                           [2.1, 0.7, -0.2]], dtype=np.float32)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
                        [0, 0, 1], [0, 2, 1]], dtype=np.int32)
    rows = _old_rows(faces, len(vertices))
    offsets = np.zeros(len(vertices) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(row) for row in rows])
    face_ids = np.asarray([face for row in rows for face, _ in row], dtype=np.int32)
    corners = np.asarray([corner for row in rows for _, corner in row], dtype=np.int32)
    expected = _normals(vertices, faces, face_ids, corners, offsets)
    observed = initial_vertex_normals(vertices=vertices, triangles=faces)
    np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))
    assert np.count_nonzero(observed[4]) == 0


@pytest.mark.parametrize("faces", [
    np.asarray([[-1, 0, 1]], dtype=np.int64),
    np.asarray([[0, 1, 3]], dtype=np.int64),
    np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
    np.asarray([[0, 1]], dtype=np.int64),
])
def test_invalid_indices_or_structure_raise_before_kernel(faces):
    with pytest.raises(ValueError):
        ordered_face_csr(faces=faces, nvertices=3)
    with pytest.raises(ValueError):
        ordered_face_incidence(faces=torch.from_numpy(faces), nvertices=3)
    with pytest.raises(ValueError):
        initial_vertex_normals(vertices=np.zeros((3, 3), dtype=np.float32), triangles=faces)


@pytest.mark.parametrize("nvertices", [-1, 3.5, True])
def test_invalid_vertex_count_raises(nvertices):
    with pytest.raises(ValueError):
        ordered_face_csr(faces=np.empty((0, 3), dtype=np.int64), nvertices=nvertices)
