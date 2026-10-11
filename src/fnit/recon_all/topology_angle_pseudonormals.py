"""拓扑距离场的角度加权伪法向和源有序相邻面查找。

独立 Python/Numba 实现固定 d932 的 computeVertexPseudoNormal 与
findOtherEdgeFace。复用调用者当前几何、面法向和有序 CSR，不重算
面法向、不运行原版程序；尚不包含完整有符号距离或 GA。
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


def _topology(faces, offsets, face_ids, nvertices):
    triangles, rows, entries = map(np.asarray, (faces, offsets, face_ids))
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or rows.shape != (nvertices + 1,) or entries.ndim != 1
            or any(not np.issubdtype(value.dtype, np.integer) for value in (triangles, rows, entries))):
        raise ValueError("integer faces (F,3), incident offsets (N+1,) and face IDs (E,) required")
    if (rows[0] != 0 or rows[-1] != len(entries) or np.any(rows[1:] < rows[:-1])
            or np.any(rows < 0) or np.any(rows > len(entries))):
        raise ValueError("complete monotonic incident-face CSR required")
    if (triangles.size and (triangles.min() < 0 or triangles.max() >= nvertices)
            or entries.size and (entries.min() < 0 or entries.max() >= len(triangles))):
        raise ValueError("face or incident-face index outside current topology")
    return (np.ascontiguousarray(triangles, np.int64),
            np.ascontiguousarray(rows, np.int64), np.ascontiguousarray(entries, np.int64))


def _indices(indices, limit, name):
    values = np.asarray(indices)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or values.size and (values.min() < 0 or values.max() >= limit)):
        raise ValueError(name + " requires ordered integer indices within the current array")
    return np.ascontiguousarray(values, np.int64)


@njit(cache=True, fastmath=False, error_model="numpy")
def _vertex_pseudonormals(xyz, faces, normals, offsets, face_ids, corners, selected):
    output = np.zeros((len(selected), 3), np.float32)
    for row in range(len(selected)):
        vertex = selected[row]
        for slot in range(offsets[vertex], offsets[vertex + 1]):
            face_id, corner = face_ids[slot], corners[slot]
            following = (corner + 1) % 3
            previous = (corner + 2) % 3
            face = faces[face_id]
            if (face[corner] != vertex or face[following] == vertex
                    or face[previous] == vertex or face[following] == face[previous]):
                raise ValueError("incident face/corner does not describe a nondegenerate triangle")
            ax = np.float32(xyz[face[following], 0] - xyz[vertex, 0])
            ay = np.float32(xyz[face[following], 1] - xyz[vertex, 1])
            az = np.float32(xyz[face[following], 2] - xyz[vertex, 2])
            bx = np.float32(xyz[face[previous], 0] - xyz[vertex, 0])
            by = np.float32(xyz[face[previous], 1] - xyz[vertex, 1])
            bz = np.float32(xyz[face[previous], 2] - xyz[vertex, 2])
            dot = np.float32(np.float32(ax * bx + ay * by) + az * bz)
            aa = np.float32(np.float32(ax * ax + ay * ay) + az * az)
            bb = np.float32(np.float32(bx * bx + by * by) + bz * bz)
            # The pinned C++17 translation unit resolves sqrt(float) and
            # both divisions to float, not double. MIN/MAX then promote
            # their comparison to double, before alpha's float writeback.
            cosine = np.float32(np.float32(dot / math.sqrt(aa)) / math.sqrt(bb))
            # C MIN/MAX's argument order also determines NaN propagation.
            clipped = 1.0 if 1.0 < cosine else cosine
            clipped = 0.0 if 0.0 >= clipped else clipped
            alpha = np.float32(clipped)
            # acos(float) also returns float in the original translation unit.
            alpha = np.float32(math.acos(alpha))
            for axis in range(3):
                output[row, axis] = np.float32(output[row, axis] + np.float32(alpha * normals[face_id, axis]))
    return output


def angle_weighted_vertex_pseudonormals(*, original_vertices, faces, face_normals,
        incident_offsets, incident_faces, incident_corners, selected_vertices) -> np.ndarray:
    """当前 orig 几何及源 incident CSR→选中顶点的未归一化伪法向。

    original_vertices 是 (N,3) surface RAS/mm，转换 float32；faces 是
    同版本 (F,3) 整数面，face_normals 是该版本 (F,3) 无量纲源面法向。
    incident_offsets(N+1,)/incident_faces(E,)/incident_corners(E,) 是
    原 vt.f/vt.n 的整数 CSR；corners 为0/1/2。不排序、不删除重复项，
    不因面 ripflag 跳过原函数仍消费的关联面。selected_vertices(K,)
    是原顺序整数顶点索引，可重复；孤立顶点返回零向量。
    返回独立 (K,3) float32，按每个顶点原面顺序累计 angle*face_normal，
    cosine 按源夹到[0,1]，两次 alpha float32 写回，最终不单位化。
    所有参数必填；不改输入、设备精度或文件。类型、形状、CSR和
    角点关系非法抛ValueError；非有限输出抛FloatingPointError。
    这是 mris_fix_topology 的内部几何步骤，没有独立原软件CLI；
    不是通用单位顶点法向，不能替换 white/pial 成熟法向函数。
    """
    xyz, normals = np.asarray(original_vertices, np.float32), np.asarray(face_normals, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("original_vertices requires finite (N,3) surface RAS coordinates")
    triangles, offsets, face_ids = _topology(faces, incident_offsets, incident_faces, len(xyz))
    corners = np.asarray(incident_corners)
    if (normals.shape != (len(triangles), 3) or not np.isfinite(xyz).all()
            or not np.isfinite(normals).all() or corners.shape != face_ids.shape
            or not np.issubdtype(corners.dtype, np.integer) or np.any(corners < 0) or np.any(corners > 2)):
        raise ValueError("finite matching face normals and incident corners 0/1/2 required")
    selected = _indices(selected_vertices, len(xyz), "selected_vertices")
    result = _vertex_pseudonormals(np.ascontiguousarray(xyz), triangles, np.ascontiguousarray(normals),
        offsets, face_ids, np.ascontiguousarray(corners, np.int64), selected)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite source angle-weighted pseudonormal")
    return result


@njit(cache=True, fastmath=False)
def _other_edge_faces(faces, offsets, face_ids, selected):
    output = np.empty((len(selected), 3), np.int64)
    for row in range(len(selected)):
        current = selected[row]
        for edge in range(3):
            a, b = faces[current, edge], faces[current, (edge + 1) % 3]
            found = current
            for first in range(offsets[a], offsets[a + 1]):
                peer = face_ids[first]
                if peer == current:
                    continue
                for second in range(offsets[b], offsets[b + 1]):
                    if peer == face_ids[second]:
                        found = peer
                        break
                if found != current:
                    break
            output[row, edge] = found
    return output


def other_edge_faces(*, faces, incident_offsets, incident_faces, selected_faces) -> np.ndarray:
    """源 face/incident CSR→每面01/12/20三条边的原有序otherFace索引。

    faces(F,3)、incident_offsets(N+1,)、incident_faces(E,) 为同一版本
    整数数组；selected_faces(Q,) 为原顺序面号，可重复。返回(Q,3)
    int64。两端关联面按各自原顺序查找，第一个非自身公共面胜出；
    找不到时按源码返回自身，不要求输入中间候选已是二流形。
    不排序、去重、跳过ripped面或做最终网格验收。所有参数必填，
    无坐标、单位、文件或设备参数；非法形状/CSR/索引抛ValueError。
    这是 mris_fix_topology 内部步骤，没有独立原软件CLI。
    """
    rows = np.asarray(incident_offsets)
    if rows.ndim != 1 or len(rows) < 1:
        raise ValueError("incident_offsets requires a complete vertex CSR")
    triangles, offsets, face_ids = _topology(faces, rows, incident_faces, len(rows) - 1)
    selected = _indices(selected_faces, len(triangles), "selected_faces")
    return _other_edge_faces(triangles, offsets, face_ids, selected)
