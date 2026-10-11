"""拓扑候选的部分状态保存/恢复；不恢复评分统计，不调用原版程序。

使用独立 NumPy 缓冲保存源 DVS 字段。恢复以新字典发布，输入不修改；
与源一样仅缩回有效面数，不复制保存时的全网格面/曲率/RP 状态。
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import time
import numpy as np

TOPOLOGY_COLUMNS = ("nsize_cur", "nsize_max", "vtotal", "vnum", "v2num", "v3num", "num")


def _integers(value, name, *, ndim=1):
    array = np.asarray(value)
    if array.size == 0 and isinstance(value, (list, tuple)):
        array = array.astype(np.int32)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(name + " must contain ordered integers")
    if array.size and (array.min() < np.iinfo(np.int32).min
                       or array.max() > np.iinfo(np.int32).max):
        raise ValueError(name + " exceeds int32 storage")
    return np.asarray(array, dtype=np.int32)


def _readonly(value, dtype=None):
    value = np.array(value, dtype=dtype, copy=True, order="C")
    value.setflags(write=False)
    return value


def _surface_arrays(surface_state):
    if not isinstance(surface_state, Mapping):
        raise ValueError("surface_state must be an explicit field mapping")
    required = ("faces", "nfaces", "original_vertices", "vertex_normals", "vertex_rip",
                "vertex_topology", "neighbor_storage", "incident_faces", "incident_corners")
    if any(key not in surface_state for key in required):
        raise ValueError("complete DVS geometry/topology fields required")
    xyz = np.asarray(surface_state["original_vertices"], dtype=np.float32)
    normals = np.asarray(surface_state["vertex_normals"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape:
        raise ValueError("same-order original_vertices/vertex_normals (N,3) required")
    count = len(xyz)
    topology = _integers(surface_state["vertex_topology"], "vertex_topology", ndim=2)
    faces = _integers(surface_state["faces"], "faces", ndim=2)
    rip = np.asarray(surface_state["vertex_rip"])
    if topology.shape != (count, 7) or faces.shape[1:] != (3,):
        raise ValueError("vertex_topology (N,7) and face storage (capacity,3) required")
    if rip.shape != (count,) or rip.dtype.kind not in "biu":
        raise ValueError("vertex_rip must contain N original rip flags")
    nfaces = surface_state["nfaces"]
    if isinstance(nfaces, (bool, np.bool_)) or not isinstance(nfaces, (int, np.integer)):
        raise ValueError("logical nfaces must be an integer")
    if nfaces < 0 or nfaces > len(faces):
        raise ValueError("logical nfaces exceeds face storage")
    rows = []
    for key in ("neighbor_storage", "incident_faces", "incident_corners"):
        container = surface_state[key]
        if not isinstance(container, Sequence) or len(container) != count:
            raise ValueError(key + " must contain one ordered storage row per vertex")
        rows.append(container)
    return xyz, normals, rip, topology, rows, int(nfaces)


def _sizes(topology):
    cur, maximum, total, first, second, third, num = map(int, topology)
    if cur < 0 or cur > 3 or maximum < cur or maximum > 3 or min(total, first, num) < 0:
        raise ValueError("valid source nsize/neighbor/incident counts required")
    expected = (0, first, second, third)[cur]
    # Restore invokes MRIS_setNsizeCur then asserts vtotal matches its snapshot.
    if total != expected:
        raise ValueError("vtotal must equal the original nsizeCur count for restorable DVS")
    stored = (0, first, second, third)[maximum]
    if stored < 0:
        raise ValueError("nsizeMax storage count cannot be negative")
    return max(stored, total), num


def _record_rows(topology, neighbors, face_ids, corners, vertex_count, nfaces):
    size, num = _sizes(topology)
    n = _integers(neighbors, "neighbor_storage row")
    f = _integers(face_ids, "incident_faces row")
    c = _integers(corners, "incident_corners row")
    if len(n) < size or len(f) < num or len(c) < num:
        raise ValueError("source row shorter than vsize or num; no undefined memory read")
    if n[:size].size and (n[:size].min() < 0 or n[:size].max() >= vertex_count):
        raise ValueError("saved neighbor index outside vertex storage")
    if f[:num].size and (f[:num].min() < 0 or f[:num].max() >= nfaces):
        raise ValueError("saved incident face outside logical nfaces")
    if c[:num].size and (c[:num].min() < 0 or c[:num].max() > 2):
        raise ValueError("source triangle corner must be 0, 1, or 2")
    return _readonly(n[:size]), _readonly(f[:num]), _readonly(c[:num], np.uint8)


@dataclass(frozen=True)
class DefectVertexSnapshot:
    """独立只读 DVS 记录；字段只涵盖源实际恢复的数据。"""
    vertex_count: int
    nfaces: int
    profile: str
    vertex_numbers: np.ndarray
    restore_vertex: np.ndarray
    original_vertices: np.ndarray
    vertex_normals: np.ndarray
    vertex_topology: np.ndarray
    neighbor_storage: tuple[np.ndarray, ...]
    incident_faces: tuple[np.ndarray, ...]
    incident_corners: tuple[np.ndarray, ...]
    vsize_exceeds_vtotal: np.ndarray
    seconds: float


def record_candidate_state(*, surface_state, defect_vertices, convex_hull_vertices,
                           border_vertices, vertex_translation, profile="chull"):
    """保存原 DVS 字段，返回不与候选缓冲别名的 DefectVertexSnapshot。

    surface_state 为第2节列出的完整字典，orig/normal float32(N,3)，
    topology int32(N,7) 列序见 TOPOLOGY_COLUMNS；存储行须包含到nsizeMax。
    defect/chull/border 为源网格有序索引，translation 是源→corrected
    int32映射；任何负映射跳过。profile='chull' 默认对应固定源error-mode=1，
    'border' 对应另一个编译分支。原rip顶点不保存可恢复字段，rip本身不保存。
    保存逻辑nfaces、orig(mm surface RAS)、normals(无量纲)、7计数和有序行。
    源vsize>vtotal分支安全分配vsize并记录风险，不复制原malloc越界。
    输入不修改、不读写文件；非法shape/索引/容量或不可恢复计数报ValueError。
    """
    started = time.perf_counter()
    xyz, normals, rip, topology, rows, nfaces = _surface_arrays(surface_state)
    if profile not in ("chull", "border"):
        raise ValueError("profile must be chull or border")
    translation = _integers(vertex_translation, "vertex_translation")
    defect = _integers(defect_vertices, "defect_vertices")
    boundary = _integers(convex_hull_vertices if profile == "chull" else border_vertices,
                         "selected hull/border")
    source_vertices = np.concatenate((defect, boundary))
    if source_vertices.size and (source_vertices.min() < 0 or source_vertices.max() >= len(translation)):
        raise ValueError("defect/hull/border index outside translation")
    mapped = translation[source_vertices]
    if mapped.size and np.any(mapped >= len(xyz)):
        raise ValueError("positive vertex translation outside corrected mesh")
    restore = mapped >= 0
    restore[restore] &= rip[mapped[restore]] == 0
    count = len(mapped)
    saved_xyz = np.zeros((count, 3), np.float32)
    saved_normals = np.zeros((count, 3), np.float32)
    saved_topology = np.zeros((count, 7), np.int32)
    saved_rows, saved_faces, saved_corners, risks = [], [], [], []
    for index, vno in enumerate(mapped):
        if not restore[index]:
            empty = _readonly([], np.int32)
            saved_rows.append(empty); saved_faces.append(empty)
            saved_corners.append(_readonly([], np.uint8)); risks.append(False)
            continue
        vno = int(vno)
        n, f, c = _record_rows(topology[vno], rows[0][vno], rows[1][vno], rows[2][vno], len(xyz), nfaces)
        saved_xyz[index] = xyz[vno]; saved_normals[index] = normals[vno]
        saved_topology[index] = topology[vno]
        saved_rows.append(n); saved_faces.append(f); saved_corners.append(c)
        risks.append(len(n) > int(topology[vno, 2]))
    return DefectVertexSnapshot(len(xyz), nfaces, profile, _readonly(mapped), _readonly(restore),
        _readonly(saved_xyz), _readonly(saved_normals), _readonly(saved_topology),
        tuple(saved_rows), tuple(saved_faces), tuple(saved_corners), _readonly(risks, np.bool_),
        time.perf_counter() - started)


def restore_candidate_state(*, surface_state, snapshot, mode="full"):
    """原DVS部分恢复，返回 (new_surface_state, report)，原输入不修改。

    snapshot 必须是 record_candidate_state 的原同网格快照。mode='full'
    默认恢复orig/normal、nsize计数、原max邻域存储和incident face/corner；
    'faces' 只恢复num/incident。两者均缩回原logical nfaces，不恢复面内容、
    rip、currentXYZ、curvbak、marks、曲率、缓存或RP累积统计。额外字典键
    保留当前值；未变缓冲可共享，改动字段独立发布，异常前不写任何输入。
    原snapshot ripped/负映射始终跳过，不依据当前rip重新选择。
    num/vsize为0时保留当前零有效长度的存储，符合原realloc(0)避免策略。
    形状/容量/快照损坏或当前nfaces少于原记录报ValueError；本函数只检查
    恢复数据的内存安全，不替代调用方完整邻接/面/最终网格QC。
    """
    started = time.perf_counter()
    xyz, normals, rip, topology, rows, nfaces = _surface_arrays(surface_state)
    if not isinstance(snapshot, DefectVertexSnapshot) or mode not in ("full", "faces"):
        raise ValueError("recorded DefectVertexSnapshot and full/faces restore required")
    if snapshot.vertex_count != len(xyz) or snapshot.nfaces < 0 or nfaces < snapshot.nfaces:
        raise ValueError("same corrected vertex storage and at least recorded nfaces required")
    count = len(snapshot.vertex_numbers)
    if (snapshot.restore_vertex.shape != (count,) or snapshot.original_vertices.shape != (count, 3)
            or snapshot.vertex_normals.shape != (count, 3) or snapshot.vertex_topology.shape != (count, 7)
            or any(len(container) != count for container in
                   (snapshot.neighbor_storage, snapshot.incident_faces, snapshot.incident_corners))):
        raise ValueError("damaged DVS snapshot dimensions")
    # Validate every saved payload before allocating/publishing any changed field.
    prepared = []
    for index, vno in enumerate(snapshot.vertex_numbers):
        if not snapshot.restore_vertex[index]:
            continue
        if vno < 0 or vno >= len(xyz):
            raise ValueError("damaged DVS mapped vertex")
        n, f, c = _record_rows(snapshot.vertex_topology[index], snapshot.neighbor_storage[index],
            snapshot.incident_faces[index], snapshot.incident_corners[index], len(xyz), snapshot.nfaces)
        prepared.append((index, int(vno), n, f, c))
    result = dict(surface_state)
    output_topology = topology.copy()
    output_neighbors = list(rows[0])
    output_faces, output_corners = list(rows[1]), list(rows[2])
    if mode == "full":
        output_xyz, output_normals = xyz.copy(), normals.copy()
    for index, vno, n, f, c in prepared:
        num = int(snapshot.vertex_topology[index, 6])
        output_topology[vno, 6] = num
        if num:
            output_faces[vno] = f.copy(); output_corners[vno] = c.copy()
        if mode == "full":
            output_xyz[vno] = snapshot.original_vertices[index]
            output_normals[vno] = snapshot.vertex_normals[index]
            output_topology[vno, :6] = snapshot.vertex_topology[index, :6]
            if len(n):
                output_neighbors[vno] = n.copy()
    result.update(nfaces=snapshot.nfaces, vertex_topology=output_topology,
                  incident_faces=output_faces, incident_corners=output_corners)
    if mode == "full":
        result.update(original_vertices=output_xyz, vertex_normals=output_normals,
                      neighbor_storage=output_neighbors)
    return result, dict(mode=mode, restored_records=len(prepared),
        skipped_records=count-len(prepared), logical_faces_before=nfaces,
        logical_faces_after=snapshot.nfaces,
        vsize_exceeds_vtotal=int(np.count_nonzero(snapshot.vsize_exceeds_vtotal)),
        original_coordinate_restored_vertices=[vno for _, vno, _, _, _ in prepared] if mode == "full" else [],
        topology_cache_invalidation_required=True,
        original_input_unchanged=True, whole_ga=False,
        seconds=time.perf_counter()-started)
