"""有序顶点法向的五轮退化修复候选；返回共享几何的完整变更状态。"""
from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np
from numba import njit

from .inflate_topology import _one_ring
from .place_surface_normals import FaceNormalTopology, _unit


@njit(cache=True, fastmath=False)
def _fnv_delta(trial: int, vertex: int, counter: int) -> np.float32:
    # 固定平台的三个32位word按小端字节处理；低32位与原unsigned long完全相同。
    value = np.uint64(0x811C9DC5)
    for word in (trial, vertex, counter):
        for shift in (0, 8, 16, 24):
            byte = np.uint64((word >> shift) & 255)
            value = ((value ^ byte) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
    fraction = float(value) / 4294967295.0
    return np.float32((0.001 - (-0.001)) * fraction + (-0.001))


@njit(cache=True, fastmath=False)
def _cross(first, second):
    return np.array((np.float32(-second[1]*first[2]+first[1]*second[2]),
                     np.float32(second[0]*first[2]-first[0]*second[2]),
                     np.float32(-second[0]*first[1]+first[0]*second[1])), np.float32)


@njit(cache=True, fastmath=False)
def _area_normal(xyz, face):
    # 该缓存保存原始叉积，不能用单位面法向替代。
    return _cross(xyz[face[2]]-xyz[face[0]], xyz[face[1]]-xyz[face[2]])


@njit(cache=True, fastmath=False)
def _evaluate_vertex(xyz, faces, offsets, face_ids, corners, face_ripped,
                     vertex, unitize):
    normal = np.zeros(3, np.float32)
    area = np.float32(0)
    count = 0
    for entry in range(offsets[vertex], offsets[vertex+1]):
        fno = face_ids[entry]
        if face_ripped[fno]:
            continue
        face = faces[fno]
        corner = corners[entry]
        first = xyz[vertex]-xyz[face[(corner+2) % 3]]
        second = xyz[face[(corner+1) % 3]]-xyz[vertex]
        raw = _cross(first, second)
        squared = np.float32(np.float32(raw[0]*raw[0]+raw[1]*raw[1])+raw[2]*raw[2])
        area = np.float32(area+np.float32(np.sqrt(float(squared))/2.0))
        _unit(first); _unit(second)
        contribution = _cross(first, second)
        if unitize:
            _unit(contribution)
        for axis in range(3):
            normal[axis] = np.float32(normal[axis]+contribution[axis])
        count += 1
    squared = np.float32(np.float32(normal[0]*normal[0]+normal[1]*normal[1])+normal[2]*normal[2])
    length = np.float32(np.sqrt(float(squared)))
    _unit(normal)
    return normal, area, count, length


@njit(cache=True, fastmath=False)
def _repair(xyz, faces, offsets, face_ids, corners, neighbors, degrees,
            ripped, face_ripped, normals, area, origarea, face_normals, border,
            planar, unitize, fix_vertex_area):
    for fno in range(len(faces)):
        face = faces[fno]
        if face_ripped[fno]:
            for vertex in face:
                border[vertex] = True
        elif not (ripped[face[0]] and ripped[face[1]] and ripped[face[2]]):
            face_normals[fno] = _area_normal(xyz, face)
    pending = np.flatnonzero(~ripped)
    counts = np.zeros(5, np.int64)
    changed = np.zeros(len(xyz), np.bool_)
    completed_trials = 0
    moved_vertices = 0
    history = np.zeros((5, len(xyz)), np.bool_)
    for trial in range(5):
        if len(pending) == 0:
            break
        failed = np.empty(len(pending), np.int64)
        failed_size = 0
        for vertex in pending:
            normal, value, count, length = _evaluate_vertex(
                xyz, faces, offsets, face_ids, corners, face_ripped, vertex, unitize)
            if count == 0 or length > 0:
                value = np.float32(float(value)/(3.0 if fix_vertex_area else 2.0))
                area[vertex] = value
                if origarea[vertex] < 0:
                    origarea[vertex] = value
                normals[vertex] = normal
            else:
                failed[failed_size] = vertex
                failed_size += 1
        pending = np.sort(failed[:failed_size])
        counts[trial] = failed_size
        history[trial, pending] = True
        completed_trials += 1
        # 源循环在trial=4仍扰动；不会进行第六次法向计算。
        for vertex in pending:
            counter = 0
            for axis in range(2 if planar else 3):
                xyz[vertex, axis] = np.float32(xyz[vertex, axis]+_fnv_delta(trial, vertex, counter))
                counter += 1
            changed[vertex] = True
            moved_vertices += 1
            if not planar:
                for column in range(degrees[vertex]):
                    adjacent = neighbors[vertex, column]
                    for axis in range(3):
                        xyz[adjacent, axis] = np.float32(xyz[adjacent, axis]+_fnv_delta(trial, vertex, counter))
                        counter += 1
                    changed[adjacent] = True
                    moved_vertices += 1
                for entry in range(offsets[vertex], offsets[vertex+1]):
                    fno = face_ids[entry]
                    if not face_ripped[fno]:
                        face_normals[fno] = _area_normal(xyz, faces[fno])
    return pending, counts[:completed_trials], history[:completed_trials], changed, moved_vertices


@dataclass(frozen=True)
class NormalRepairResult:
    """完整返回状态；已成功邻居的法向可能按源顺序保留扰动前值。"""
    positions: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    original_areas: np.ndarray
    face_normals: np.ndarray
    border: np.ndarray
    unresolved_vertices: np.ndarray
    pending_counts: np.ndarray
    pending_history: np.ndarray
    changed_vertices: np.ndarray
    coordinate_update_count: int


def repair_vertex_normals(*, positions: np.ndarray, faces: np.ndarray,
                          topology: FaceNormalTopology | None = None,
                          status: str = "sphere", unitize: bool = True,
                          fix_vertex_area: bool = True,
                          ripped: np.ndarray | None = None,
                          face_ripped: np.ndarray | None = None,
                          initial_normals: np.ndarray | None = None,
                          initial_areas: np.ndarray | None = None,
                          original_areas: np.ndarray | None = None,
                          initial_face_normals: np.ndarray | None = None,
                          initial_border: np.ndarray | None = None) -> NormalRepairResult:
    """返回原算法五轮有序修复的独立CPU候选，输入不原地修改。

    positions为float32(N,3)surface RAS/mm，faces为整数(F,3)有序面。
    topology=None时复用FaceNormalTopology/Numba一环构造；缓存须同网格。
    status默认sphere；plane/cut仅移动失败点x/y，其余状态移动xyz及有序
    一环邻居。unitize=True对应角点单位化；fix_vertex_area=True面积除3，
    False除2。ripped/face_ripped默认全False；初始normal/area/face_normal
    默认零，original_areas默认-1，border默认False，均先复制。
    返回坐标、法向、面积(mm²)、origarea、原始面叉积(mm²)、border、每轮
    pending及未解决顶点；真正未解决分支不添加第六轮、epsilon或随机库。
    无关联活跃面的顶点按源算法成功且法向为零。源重试只重算失败顶点，
    不重算已成功但被移动的邻居；调用方必须显式接纳几何并失效动态缓存。
    非有限/非float32坐标、非法索引、形状/状态/缓存不符或非小端平台
    抛ValueError。未解决顶点通过返回值报告，不假装完整metric properties。
    属于MRIScomputeNormals内部步骤，无独立官方CLI，不调用原版程序。
    """
    xyz = np.asarray(positions)
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("positions must be finite float32 (N,3)")
    if sys.byteorder != "little":
        raise ValueError("fixed fnv stream requires little-endian source platform")
    if status not in {"sphere", "spherical_patch", "parameterized_sphere", "surface", "plane", "cut"}:
        raise ValueError("unsupported source surface status")
    if not isinstance(unitize, bool) or not isinstance(fix_vertex_area, bool):
        raise ValueError("unitize and fix_vertex_area must be bool")
    if topology is None:
        topology = FaceNormalTopology(triangles=faces, nvertices=len(xyz))
    else:
        topology.validate(faces, len(xyz))
    neighbors, degrees = _one_ring(topology.faces, topology.face_ids, topology.corners, topology.offsets)
    nvertices, nfaces = len(xyz), len(topology.faces)

    def array(value, shape, dtype, fill=0):
        if value is None:
            return np.full(shape, fill, dtype=dtype)
        source = np.asarray(value)
        if source.shape != shape or (np.issubdtype(dtype, np.floating) and not np.isfinite(source).all()):
            raise ValueError("initial state differs from mesh shape or is nonfinite")
        if dtype == np.bool_ and source.dtype != np.bool_:
            raise ValueError("rip and border flags must be boolean arrays")
        return np.array(source, dtype=dtype, copy=True)

    normals = array(initial_normals, (nvertices,3), np.float32)
    areas = array(initial_areas, (nvertices,), np.float32)
    original = array(original_areas, (nvertices,), np.float32, -1)
    raw_face = array(initial_face_normals, (nfaces,3), np.float32)
    border = array(initial_border, (nvertices,), np.bool_)
    vertex_rips = array(ripped, (nvertices,), np.bool_)
    face_rips = array(face_ripped, (nfaces,), np.bool_)
    coordinates = np.array(xyz, copy=True)
    pending, counts, history, changed, updates = _repair(
        coordinates, topology.faces, topology.offsets, topology.face_ids, topology.corners,
        neighbors, degrees, vertex_rips, face_rips, normals, areas, original, raw_face, border,
        status in {"plane", "cut"}, unitize, fix_vertex_area)
    return NormalRepairResult(coordinates, normals, areas, original, raw_face, border,
                              pending, counts, history, np.flatnonzero(changed), int(updates))
