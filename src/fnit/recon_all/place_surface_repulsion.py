"""First placement surface repulsion on the original vertex hash grid."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_normals import _unit, FaceNormalTopology, ordered_face_csr


@njit(cache=True)
def _original_normals(xyz: np.ndarray, faces: np.ndarray, ids: np.ndarray, corners: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        normal = np.zeros(3, dtype=np.float32)
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            face = faces[ids[entry]]
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
            normal[0] = np.float32(normal[0] + face_normal[0])
            normal[1] = np.float32(normal[1] + face_normal[1])
            normal[2] = np.float32(normal[2] + face_normal[2])
        _unit(normal)
        result[vertex] = normal
    return result


def original_vertex_normals(vertices: np.ndarray, triangles: np.ndarray, *,
                            topology: FaceNormalTopology | None = None) -> np.ndarray:
    """原surface法向：同序(N,3)mm坐标/(F,3)面→float32单位法向。

    topology=None；可传同网格整数CSR，仍重算坐标及原定义法向。
    与current法向不同：累加前不归一化每个面法向，不能互换。
    非法拓扑抛ValueError，属于mris_place_surface内部步骤，没有独立CLI。
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    faces = np.asarray(triangles, dtype=np.int32)
    if topology is None:
        offsets, ids, corners = ordered_face_csr(triangles, nvertices=len(xyz))
    else:
        topology.validate(triangles, len(xyz))
        offsets, ids, corners = topology.offsets, topology.face_ids, topology.corners
    return _original_normals(xyz, faces, ids, corners, offsets)


def vertex_buckets(current: np.ndarray, original: np.ndarray, ripped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """原候选桶完整接口；三组同N坐标/固定mask→int32 CSR，不截断候选。

    坐标surface RAS/mm；先float32加1000再向零取整，与原规则一致。
    一次调用建立索引；连续迭代应显式复用OriginalVertexBuckets。
    当前坐标每次重算键，桶内保持原顶点升序；非法数组抛ValueError。
    """
    return OriginalVertexBuckets(original, ripped).query(current)


@njit(cache=True)
def _query_vertex_buckets(current_keys, sorted_keys, vertex_ids, ripped):
    offsets = np.zeros(len(current_keys) + 1, dtype=np.int32)
    starts = np.zeros(len(current_keys), dtype=np.int64)
    for vertex in range(len(current_keys)):
        offsets[vertex + 1] = offsets[vertex]
        if ripped[vertex]:
            continue
        x, y, z = current_keys[vertex]
        low, high = 0, len(sorted_keys)
        while low < high:
            mid = (low + high) // 2
            sx, sy, sz = sorted_keys[mid]
            if sx < x or (sx == x and (sy < y or (sy == y and sz < z))):
                low = mid + 1
            else:
                high = mid
        starts[vertex] = low
        stop = low
        while stop < len(sorted_keys):
            sx, sy, sz = sorted_keys[stop]
            if sx != x or sy != y or sz != z:
                break
            stop += 1
        offsets[vertex + 1] += stop - low
    flat = np.empty(offsets[-1], dtype=np.int32)
    for vertex in range(len(current_keys)):
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            flat[entry] = vertex_ids[starts[vertex] + entry - offsets[vertex]]
    return offsets, flat


class OriginalVertexBuckets:
    """固定原表面/rip的整数索引；当前几何每次查询，不缓存变化坐标。

    original为float32(N,3)surface RAS/mm，ripped为(N,)布尔mask；
    两者的索引信息构造时复制并冻结。query(current)同N有限坐标返回
    offsets(N+1)/candidate_ids(M) int32，桶内原顶点顺序，完整M无上限。
    原表面或rip改变须新建实例；非法shape/非有限值抛ValueError。
    属于mris_place_surface内部步骤，无独立官方CLI、设备或精度开关。
    """
    def __init__(self, original: np.ndarray, ripped: np.ndarray):
        xyz = np.asarray(original, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("original must contain finite (N, 3) coordinates")
        self.ripped = np.array(ripped, dtype=np.bool_, copy=True)
        if self.ripped.shape != (len(xyz),):
            raise ValueError("ripped must have shape (N,)")
        self.nvertices = len(xyz)
        keys = (xyz + np.float32(1000)).astype(np.int32)
        ids = np.flatnonzero(~self.ripped).astype(np.int32)
        selected = keys[ids]
        order = np.lexsort((ids, selected[:, 2], selected[:, 1], selected[:, 0]))
        self.keys, self.ids = selected[order], ids[order]
        for array in (self.keys, self.ids, self.ripped):
            array.flags.writeable = False

    def query(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """当前同N坐标→完整桶候选CSR；保留向零取整与顶点插入次序。"""
        xyz = np.asarray(current, dtype=np.float32)
        if xyz.shape != (self.nvertices, 3) or not np.isfinite(xyz).all():
            raise ValueError("current must contain finite matching (N, 3) coordinates")
        keys = (xyz + np.float32(1000)).astype(np.int32)
        return _query_vertex_buckets(keys, self.keys, self.ids, self.ripped)


@njit(cache=True)
def _gradient(
    xyz: np.ndarray, normals: np.ndarray, original: np.ndarray,
    original_normals: np.ndarray, ripped: np.ndarray, cropped: np.ndarray,
    offsets: np.ndarray, candidates: np.ndarray, weight: float,
) -> np.ndarray:
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex] or cropped[vertex]:
            continue
        sx = sy = sz = np.float32(0.0)
        x, y, z = xyz[vertex]
        nx, ny, nz = normals[vertex]
        for index in range(offsets[vertex], offsets[vertex + 1]):
            other = candidates[index]
            dx = np.float32(x - original[other, 0])
            dy = np.float32(y - original[other, 1])
            dz = np.float32(z - original[other, 2])
            dot = np.float32(np.float32(dx * original_normals[other, 0] + dy * original_normals[other, 1]) + dz * original_normals[other, 2])
            if dot > 1:
                continue
            dot = min(max(float(dot), -40.0), 40.0)
            scale = weight * (1.0 - dot) ** 4.0
            sx = np.float32(sx + scale * float(nx))
            sy = np.float32(sy + scale * float(ny))
            sz = np.float32(sz + scale * float(nz))
        result[vertex, 0] = sx
        result[vertex, 1] = sy
        result[vertex, 2] = sz
    return result


def surface_repulsion_gradient(
    vertices: np.ndarray, normals: np.ndarray, original_vertices: np.ndarray,
    original_normals: np.ndarray, ripped: np.ndarray,
    offsets: np.ndarray, candidates: np.ndarray, *, weight: float = 5.0,
    cropped: np.ndarray | None = None,
) -> np.ndarray:
    """Return original-surface repulsion, excluding collision-cropped vertices."""
    return _gradient(
        np.asarray(vertices, dtype=np.float32), np.asarray(normals, dtype=np.float32),
        np.asarray(original_vertices, dtype=np.float32), np.asarray(original_normals, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_),
        np.zeros(len(vertices), dtype=np.bool_) if cropped is None else np.asarray(cropped, dtype=np.bool_),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(candidates, dtype=np.int32), float(np.float32(weight)),
    )
