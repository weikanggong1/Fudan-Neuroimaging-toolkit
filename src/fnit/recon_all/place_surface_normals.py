"""Source-order vertex normals for the initial placement surface."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _unit(vector: np.ndarray) -> None:
    length = np.float32(np.sqrt(np.float32(
        np.float32(vector[0] * vector[0] + vector[1] * vector[1]) + vector[2] * vector[2]
    )))
    if length > 0:
        vector[0] = np.float32(vector[0] / length)
        vector[1] = np.float32(vector[1] / length)
        vector[2] = np.float32(vector[2] / length)


@njit(cache=True)
def _normals(xyz: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, corners: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        normal = np.zeros(3, dtype=np.float32)
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            face = faces[face_ids[entry]]
            corner = corners[entry]
            previous = face[(corner + 2) % 3]
            following = face[(corner + 1) % 3]
            v0 = np.empty(3, dtype=np.float32)
            v1 = np.empty(3, dtype=np.float32)
            for axis in range(3):
                v0[axis] = np.float32(xyz[vertex, axis] - xyz[previous, axis])
                v1[axis] = np.float32(xyz[following, axis] - xyz[vertex, axis])
            _unit(v0)
            _unit(v1)
            face_normal = np.array([
                np.float32(-v1[1] * v0[2] + v0[1] * v1[2]),
                np.float32(v1[0] * v0[2] - v0[0] * v1[2]),
                np.float32(-v1[0] * v0[1] + v0[0] * v1[1]),
            ], dtype=np.float32)
            _unit(face_normal)
            normal[0] = np.float32(normal[0] + face_normal[0])
            normal[1] = np.float32(normal[1] + face_normal[1])
            normal[2] = np.float32(normal[2] + face_normal[2])
        _unit(normal)
        result[vertex] = normal
    return result


def initial_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """按原有面顺序计算每顶点单位法向，复用已有单精度数值内核。

    vertices 为 (N, 3) 坐标，triangles 为 (F, 3) 有序顶点索引；输入转换为
    float32/int32。坐标单位不影响单位法向；返回 (N, 3) float32，无关联面
    的顶点返回零向量。稳定排序只构造关联索引，不重排面或更改累计顺序。
    属于 mris_sphere / mris_place_surface 的内部几何步骤，无独立 CLI。
    非法索引沿用 NumPy/Numba 的错误行为；真实回归与计时见性能记录。
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    counts = np.bincount(faces.ravel(), minlength=len(xyz))
    offsets = np.empty(len(xyz) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(faces.ravel(), kind="stable")
    face_ids = (order // 3).astype(np.int32)
    corners = (order % 3).astype(np.int32)
    return _normals(xyz, faces, face_ids, corners, offsets)
