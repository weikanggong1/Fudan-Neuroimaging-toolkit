"""候选拓扑的独立面法向与显式顶点法向；不运行原版程序。

面缓存 original_area 与 FACE.area、VERTEX.area 是不同状态，不能互换。
当前 fitness 调用仅更新面法向；顶点法向函数供实际调用它的匹配阶段使用。
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

from .topology_angle_pseudonormals import _indices, _topology


@njit(cache=True, fastmath=False)
def _face_geometry(xyz, faces, selected):
    normals = np.empty((len(selected), 3), np.float32)
    original_area = np.empty(len(selected), np.float32)
    fallback = np.zeros(len(selected), np.uint8)
    epsilon = np.float32(np.finfo(np.float32).eps)
    for row in range(len(selected)):
        a, b = np.empty(3, np.float32), np.empty(3, np.float32)
        triangle = faces[selected[row]]
        for axis in range(3):
            a[axis] = np.float32(xyz[triangle[1], axis] - xyz[triangle[0], axis])
            b[axis] = np.float32(xyz[triangle[2], axis] - xyz[triangle[0], axis])
        for attempt in range(2):
            nx = np.float32(np.float32(a[1] * b[2]) - np.float32(b[1] * a[2]))
            ny = np.float32(np.float32(a[2] * b[0]) - np.float32(b[2] * a[0]))
            nz = np.float32(np.float32(a[0] * b[1]) - np.float32(b[0] * a[1]))
            squared = np.float32(np.float32(nx * nx + ny * ny) + nz * nz)
            length = np.float32(math.sqrt(squared))
            if abs(length) >= epsilon:
                break
            fallback[row] = attempt + 1
            if attempt == 0:
                for axis in range(3):
                    # Source subtracts float coordinates before double *100.
                    a[axis] = np.float32(100.0 * np.float64(np.float32(xyz[triangle[2], axis] - xyz[triangle[1], axis])))
                    b[axis] = np.float32(100.0 * np.float64(np.float32(xyz[triangle[0], axis] - xyz[triangle[1], axis])))
            else:
                length = np.float32(1)
        normals[row, 0] = np.float32(nx / length)
        normals[row, 1] = np.float32(ny / length)
        normals[row, 2] = np.float32(nz / length)
        original_area[row] = np.float32(length / np.float32(2))
    return normals, original_area, fallback


def candidate_face_normals(*, original_vertices, faces, selected_faces) -> dict:
    """原候选几何→指定面序的源单位法向与缓存 original_area。

    original_vertices 为 float32(N,3) orig surface RAS/mm；faces 为同版
    整数(F,3)，selected_faces 为原顺序整数(K,)面号，可重复。全部参数
    必填，返回 face_normals float32(K,3)、cached_original_area_mm2
    float32(K,)及fallback uint8(K,)（0普通、1放大重试、2两次近零）。
    保留源 F32 cross/sqrt 与 FZERO=FLT_EPSILON；近零时换角点并将边
    放大100，仍近零用length=1，缓存面积因此可能为0.5，不是物理面积。
    不修改 FACE.area/VERTEX.area、顶点法向、ripflag 或任何输入，不写
    文件、不调用原程序；属于 mris_fix_topology 内部步骤，无独立CLI。
    非法类型/形状/索引/非有限输入抛ValueError，非有限输出抛
    FloatingPointError。尚不包含完整候选生成、fitness 或 GA。
    """
    xyz, triangles = np.asarray(original_vertices), np.asarray(faces)
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('finite float32 original_vertices (N,3) required')
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz))):
        raise ValueError('integer same-version faces (F,3) within original vertices required')
    selected = _indices(selected_faces, len(triangles), 'selected_faces')
    normal, area, fallback = _face_geometry(np.ascontiguousarray(xyz), np.ascontiguousarray(triangles, np.int64), selected)
    if not np.isfinite(normal).all() or not np.isfinite(area).all():
        raise FloatingPointError('nonfinite candidate face cache')
    return dict(face_normals=normal, cached_original_area_mm2=area, fallback=fallback)


@njit(cache=True, fastmath=False)
def _selected_vertex_normals(normals, offsets, incident, selected):
    output = np.zeros((len(selected), 3), np.float32)
    epsilon = np.float32(np.finfo(np.float32).eps)
    for row in range(len(selected)):
        vertex = selected[row]
        nx = ny = nz = np.float32(0)
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            face = incident[entry]
            nx = np.float32(nx + normals[face, 0])
            ny = np.float32(ny + normals[face, 1])
            nz = np.float32(nz + normals[face, 2])
        length = np.float32(math.sqrt(np.float32(np.float32(nx * nx + ny * ny) + nz * nz)))
        if abs(length) < epsilon:
            length = np.float32(1)
        output[row, 0], output[row, 1], output[row, 2] = np.float32(nx / length), np.float32(ny / length), np.float32(nz / length)
    return output


def candidate_vertex_normals(*, faces, face_normals, incident_offsets,
        incident_faces, selected_vertices) -> np.ndarray:
    """按源 incident 面序累计并单位化指定顶点法向；调用者显式选择执行。

    faces 是整数(F,3)，face_normals float32(F,3)无量纲；incident_offsets
    整数(N+1,)、incident_faces整数(E,)为当前原顺序CSR，保留重复面。
    selected_vertices 为原顺序整数(K,)顶点，可重复；全部参数必填。
    返回独立float32(K,3)，孤立/正反法向抵消时为零；近零长度用1，
    不跳过ripped面，不重算面积或angle伪法向。当前fitness源码未调用
    该步骤，不能为方便而自动更新fitness输入的旧顶点法向。
    类型/CSR/面归属非法抛ValueError，非有限输出抛FloatingPointError。
    无文件/设备/精度副作用，属于原拓扑匹配内部步骤，无独立官方CLI。
    """
    rows, normals = np.asarray(incident_offsets), np.asarray(face_normals)
    if rows.ndim != 1 or len(rows) < 1:
        raise ValueError('complete incident_offsets required')
    triangles, offsets, incident = _topology(faces, rows, incident_faces, len(rows) - 1)
    if normals.dtype != np.float32 or normals.shape != (len(triangles), 3) or not np.isfinite(normals).all():
        raise ValueError('finite float32 face_normals (F,3) required')
    selected = _indices(selected_vertices, len(rows) - 1, 'selected_vertices')
    for vertex in selected:
        if np.any(np.all(triangles[incident[offsets[vertex]:offsets[vertex + 1]]] != vertex, axis=1)):
            raise ValueError('incident face does not contain selected vertex')
    result = _selected_vertex_normals(np.ascontiguousarray(normals), offsets, incident, selected)
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite candidate vertex normals')
    return result
