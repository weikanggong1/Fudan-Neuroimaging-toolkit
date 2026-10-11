"""Source-order curvature-adaptive smoothing for an already retessellated patch.

The first-candidate validator derives histogram means from native diagnostic
plots and MGH metadata. Native-free histogram construction and subsequent MRI
matching remain separate topology steps.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from .place_surface_normals import FaceNormalTopology


def _incident_arrays(*, faces, vertex_count, topology=None):
    """复用当前面版本的显式CSR；None保留原完整有序构造。

    topology必须是FaceNormalTopology，绑定同N及完全相同的有序整数面。
    返回原Numba内核使用的int32 offsets/face_ids；不缓存坐标或法向。
    不兼容上下文或int32容量溢出抛ValueError，不截断索引。
    """
    if topology is not None:
        if not isinstance(topology, FaceNormalTopology):
            raise ValueError('topology must be FaceNormalTopology or None')
        topology.validate(faces, vertex_count)
        if len(topology.face_ids) > np.iinfo(np.int32).max:
            raise ValueError('incident entries exceed int32 capacity')
        return topology.offsets.astype(np.int32), topology.face_ids.astype(np.int32)
    slots = [[] for _ in range(vertex_count)]
    for fi, face in enumerate(faces):
        for vertex in face:
            slots[int(vertex)].append(fi)
    offsets = np.zeros(vertex_count + 1, np.int32)
    for vertex, row in enumerate(slots):
        offsets[vertex + 1] = offsets[vertex] + len(row)
    return offsets, np.asarray([fi for row in slots for fi in row], np.int32)

@njit
def _smooth_core(xyz, faces, inside, neighbors, degree, offsets, face_ids,
                 k1_mean, k2_mean):
    current = xyz.copy()
    face_normal = np.zeros((len(faces), 3), np.float32)
    rmin = np.float32(-1.0 / k1_mean)
    rmax = np.float32(-1.0 / k2_mean)
    e = np.float32((np.float32(1.0 / rmin) + np.float32(1.0 / rmax)) / np.float32(2))
    fscale = np.float32(6.0 / np.float32(np.float32(1.0 / rmin) - np.float32(1.0 / rmax)))
    for _ in range(25):
        for fi in range(len(faces)):
            a, b, c = faces[fi]
            ax = np.float32(current[b, 0] - current[a, 0])
            ay = np.float32(current[b, 1] - current[a, 1])
            az = np.float32(current[b, 2] - current[a, 2])
            bx = np.float32(current[c, 0] - current[a, 0])
            by = np.float32(current[c, 1] - current[a, 1])
            bz = np.float32(current[c, 2] - current[a, 2])
            nx = np.float32(ay * bz - az * by)
            ny = np.float32(az * bx - ax * bz)
            nz = np.float32(ax * by - ay * bx)
            length = np.sqrt(np.float32(nx * nx + ny * ny + nz * nz))
            if length < 1e-5:
                length = np.float32(1.0)
            face_normal[fi, 0] = np.float32(nx / length)
            face_normal[fi, 1] = np.float32(ny / length)
            face_normal[fi, 2] = np.float32(nz / length)
        next_xyz = current.copy()
        for v in inside:
            nx = ny = nz = np.float32(0)
            for slot in range(offsets[v], offsets[v + 1]):
                fi = face_ids[slot]
                nx = np.float32(nx + face_normal[fi, 0])
                ny = np.float32(ny + face_normal[fi, 1])
                nz = np.float32(nz + face_normal[fi, 2])
            length = np.sqrt(np.float32(nx * nx + ny * ny + nz * nz))
            if length < 1e-5:
                length = np.float32(1.0)
            nx = np.float32(nx / length)
            ny = np.float32(ny / length)
            nz = np.float32(nz / length)
            x, y, z = current[v]
            sx = sy = sz = sd = np.float32(0)
            n = 0
            while n < degree[v]:
                other = neighbors[v, n]
                dx = np.float32(current[other, 0] - x)
                dy = np.float32(current[other, 1] - y)
                dz = np.float32(current[other, 2] - z)
                sx = np.float32(sx + dx)
                sy = np.float32(sy + dy)
                sz = np.float32(sz + dz)
                sd = np.float32(sd + np.sqrt(np.float32(dx * dx + dy * dy + dz * dz)))
                n += 2  # source increments n in the loop body and in the for header
            denom = np.float32(n)
            sx = np.float32(sx / denom)
            sy = np.float32(sy / denom)
            sz = np.float32(sz / denom)
            sd = np.float32(sd / denom)
            nc = np.float32(sx * nx + sy * ny + sz * nz)
            sxn = np.float32(nc * nx)
            syn = np.float32(nc * ny)
            szn = np.float32(nc * nz)
            sxt = np.float32(sx - sxn)
            syt = np.float32(sy - syn)
            szt = np.float32(sz - szn)
            radius = np.float32(sd * sd / np.float32(2 * abs(nc))) if nc != 0 else np.float32(np.inf)
            gain = np.float32((1.0 + np.tanh(float(np.float32(fscale * np.float32(np.float32(1.0 / radius) - e))))) / 2.0)
            next_xyz[v, 0] = np.float32(float(x) + 0.1 * float(np.float32(sxt + gain * sxn)))
            next_xyz[v, 1] = np.float32(float(y) + 0.1 * float(np.float32(syt + gain * syn)))
            next_xyz[v, 2] = np.float32(float(z) + 0.1 * float(np.float32(szt + gain * szn)))
        current = next_xyz
    return current


def defect_smooth_type2(xyz: np.ndarray, faces: np.ndarray, inside_vertices: np.ndarray,
                        runtime_neighbors: list[list[int]],
                        k1_mean: float, k2_mean: float, *, topology=None) -> np.ndarray:
    """保留内点25轮源有序平滑，输出独立float32(N,3)坐标。

    xyz为surface RAS/mm坐标，faces为同顶点顺序整数(F,3)，inside_vertices
    为保留内点(K,)，runtime_neighbors为各内点原顺序邻接；曲率均值
    k1_mean/k2_mean单位mm⁻¹，来自全局SFF1直方图。topology默认None，
    沿用完整有序入射构造；可显式传同一面版本FaceNormalTopology复用CSR，
    换面/顶点数必须重建，坐标/法向仍逐轮重算。上下文不匹配抛ValueError。
    空内点返回独立坐标副本；原25轮、双自增邻域规则及求和顺序不变。
    这是mris_fix_topology内部defectSmooth步骤，没有独立CLI或文件读写。
    """
    points = np.ascontiguousarray(xyz, np.float32)
    triangles = np.ascontiguousarray(faces, np.int32)
    inside = np.ascontiguousarray(inside_vertices, np.int32)
    if topology is not None:
        # 验证原faces，不接受先截断浮点索引再匹配上下文。
        offsets, face_ids = _incident_arrays(faces=faces, vertex_count=len(points), topology=topology)
    if len(inside) == 0:
        return points.copy()
    degree = np.zeros(len(points), np.int32)
    maximum = max(map(len, runtime_neighbors))
    neighbors = np.zeros((len(points), maximum), np.int32)
    for vertex, row in zip(inside, runtime_neighbors):
        degree[vertex] = len(row)
        neighbors[vertex, :len(row)] = row
    if topology is None:
        offsets, face_ids = _incident_arrays(faces=triangles, vertex_count=len(points))
    return _smooth_core(points, triangles, inside, neighbors, degree,
                        offsets, face_ids, np.float32(k1_mean), np.float32(k2_mean))
