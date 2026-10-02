"""顺序几何内核的小网格回归；不能代替真实 T1 benchmark。

双精度平滑/完整 remesh 指纹来自冻结 FNIT 764607c 原路径；验证面顺序、
原位更新及累加顺序。quick scatter 用原 np.add.at 作为相同 float32 规则参考。
"""

import hashlib
import heapq

import numpy as np

from fnit.recon_all.mris_remesh_python import Mesh, remesh_geometry, smooth
from fnit.recon_all.sphere_quick_python import _accumulate_corner_normals


def _mesh():
    t = (1 + np.sqrt(5)) / 2
    vertices = np.asarray([
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ], np.float64)
    faces = np.asarray([
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ], np.int32)
    vertices = (vertices * np.asarray([37.3, 40.13, 33.9])
                + np.asarray([.9123456, -4.31415, 7.0123]))
    return vertices, faces


def test_smooth_retains_frozen_ordered_float64_geometry():
    vertices, faces = _mesh()
    mesh = Mesh([tuple(row) for row in vertices], faces.tolist())
    smooth(mesh=mesh, repeats=2)
    result = np.asarray(mesh.points, dtype="<f8")
    assert hashlib.sha256(result.tobytes()).hexdigest() == (
        "f2cb82c66d99796c772945a6090a47f503f67fe36eeaa2a5a2ec88af0f77fea6")
    np.testing.assert_array_equal(mesh.faces, faces)


def test_remesh_keeps_frozen_split_collapse_and_smooth_result():
    vertices, faces = _mesh()
    origin = np.asarray([.9123456, -4.31415, 7.0123])
    vertices = (vertices - origin) * np.asarray([1., 1.7, .42]) + origin
    result, result_faces = remesh_geometry(vertices=vertices, faces=faces, iterations=3)
    assert result.shape == (8, 3) and result_faces.shape == (12, 3)
    data = result.astype("<f4").tobytes() + result_faces.astype("<i4").tobytes()
    assert hashlib.sha256(data).hexdigest() == (
        "5657a57d7a43d9fe35a42b251723a54856945acfaa23daf5ccaab53d95882b40")


def test_heapify_keeps_length_ties_in_source_edge_order():
    for descending in (False, True):
        old = []
        values = [((-(i % 7), -i) if descending else (i % 7, i)) for i in range(53)]
        for value in values:
            heapq.heappush(old, value)
        new = values.copy()
        heapq.heapify(new)
        assert [heapq.heappop(old) for _ in values] == [heapq.heappop(new) for _ in values]


def test_reused_corner_kernel_keeps_float32_face_corner_accumulation():
    # 大小混合数值验证不能改成改变加法顺序的批量/并行归约。
    ids = np.asarray([2, 0, 1, 2, 1, 0, 2, 0, 1], np.int32)
    terms = np.asarray([
        [2**25, .1, -2**25], [.03, 2**25, 4], [1, -2**25, .01],
        [1, .2, -1], [2**25, .5, .02], [9, 1, -2**25],
        [-2**25, .4, 2**25], [.07, -2**25, 2**25], [-2**25, .9, .04],
    ], np.float32)
    reference = np.zeros((3, 3), np.float32)
    np.add.at(reference, ids, terms)
    result = _accumulate_corner_normals(ids=ids, corners=terms, nvertices=3)
    np.testing.assert_array_equal(result, reference)
