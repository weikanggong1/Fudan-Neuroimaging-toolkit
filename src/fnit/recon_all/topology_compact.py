"""源序移除 rip 顶点/面；不修补网格，不执行原版程序。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import time

import numpy as np

from .topology_genetic_base import COPIED_VECTOR_FIELDS, COPIED_SCALAR_FIELDS


def _integers(value, shape, name, *, maximum=None):
    """转换前检查类型和 int32 范围，不截断浮点或溢出编号。"""
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iu':
        raise ValueError(name+' requires an integer array with shape '+str(shape))
    if a.size and (a.min() < np.iinfo(np.int32).min
                  or a.max() > np.iinfo(np.int32).max):
        raise ValueError(name+' exceeds source int32 storage')
    if maximum is not None and a.size and (a.min() < 0 or a.max() >= maximum):
        raise ValueError(name+' outside logical source storage')
    return np.array(a, np.int32, copy=True)


def _flags(value, count, name):
    a = np.asarray(value)
    if (a.shape != (count,) or a.dtype.kind not in 'biu'
            or a.size and (a.min() < -128 or a.max() > 127)):
        raise ValueError(name+' requires explicit source signed-char/bool flags')
    return a != 0


def compact_ripped_surface_elements(*, surface_state, metadata) -> dict:
    """完整旧网格状态→源序压缩的新网格、两张映射和耗时。

    surface_state 为同一版本的显式字典：五个 COPIED_VECTOR_FIELDS
    float32(N,3)，15个 COPIED_SCALAR_FIELDS float32(N,)，faces 整数
    (capacity,3)、nfaces 为实际 F、vertex_rip(N)、face_rip(capacity)、
    vertex_topology int32(N,7)、完整有序 neighbor_storage/incident_faces/
    incident_corners。坐标 surface RAS/mm，法向无量纲，H/k1/k2 为
    mm⁻¹，K 为 mm⁻²。metadata 必含 use_real_ras/status/origxyz_status/
    volume_geometry/hemisphere；状态不从参考或缺省值补齐。

    仅保留 rip==0 顶点及面，按旧编号升序建立 vertex_translation(N,)
    和 face_translation(F,)（删除项 −1）。保留邻接和 incident 原行序，
    过滤被删除编号后映射；悬空但未 rip 的边仍保留，不能从面重建一环。
    新 topology nsizeCur/Max=1、vtotal=vnum、v2num/v3num=0。复制上述
    源白名单的全部位，包括标量 NaN；E/fx/area/marks/rip/面法向等原
    新分配未复制字段明确为零，不把旧缓存当压缩后有效状态。元信息深
    复制，volume_geometry.register_mat 为新对象原 null，外层原调用
    复制 hemisphere/fname 的语义在本接口保留。

    返回 surface_state、metadata、两张映射、kept_source_vertices/
    faces、removed_*、seconds。所有参数必填，无设备/线程/精度或文件
    副作用。非法类型/索引/容量/不完整 incidence，或未 rip 面引用 rip
    顶点报 ValueError，输入不修改；不自动删这样的面以掩盖 GA 错误。
    不做 SOAP、metric、Euler/流形/自相交验收或发布。对应固定
    MRISremoveRippedSurfaceElements 内部步骤，没有独立原软件 CLI。
    """
    started = time.perf_counter()
    if not isinstance(surface_state, Mapping):
        raise ValueError('complete explicit surface_state mapping required')
    if any(k not in surface_state for k in (*COPIED_VECTOR_FIELDS,
           *COPIED_SCALAR_FIELDS, 'faces', 'nfaces', 'face_rip', 'vertex_rip',
           'vertex_topology', 'neighbor_storage', 'incident_faces', 'incident_corners')):
        raise ValueError('complete copied attributes and source topology required')
    original = np.asarray(surface_state['original_vertices'])
    if original.ndim != 2 or original.shape[1:] != (3,):
        raise ValueError('original_vertices float32(N,3) required')
    n = len(original)
    arrays = {}
    for name in COPIED_VECTOR_FIELDS:
        a = np.asarray(surface_state[name])
        if a.shape != (n,3) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('finite copied float32(N,3) required: '+name)
        arrays[name] = a
    for name in COPIED_SCALAR_FIELDS:
        a = np.asarray(surface_state[name])
        if a.shape != (n,) or a.dtype != np.float32:
            raise ValueError('copied float32(N,) required: '+name)
        arrays[name] = a
    faces = np.asarray(surface_state['faces'])
    nf = surface_state['nfaces']
    if (isinstance(nf, (bool,np.bool_)) or not isinstance(nf, (int,np.integer))
            or faces.ndim != 2 or faces.shape[1:] != (3,)
            or not 0 <= nf <= len(faces)):
        raise ValueError('explicit logical nfaces within ordered face capacity required')
    triangles = _integers(faces[:nf], (int(nf),3), 'faces', maximum=n)
    vertex_rip = _flags(surface_state['vertex_rip'], n, 'vertex_rip')
    face_flags = np.asarray(surface_state['face_rip'])
    if face_flags.ndim != 1 or len(face_flags) != len(faces):
        raise ValueError('full face capacity rip flags required')
    face_rip = _flags(face_flags, len(faces), 'face_rip')[:nf]
    top = _integers(surface_state['vertex_topology'], (n,7), 'vertex_topology')
    if np.any(top < 0):
        raise ValueError('source topology counts must be nonnegative')
    for name in ('neighbor_storage','incident_faces','incident_corners'):
        if not isinstance(surface_state[name], Sequence) or len(surface_state[name]) != n:
            raise ValueError(name+' requires a complete ordered row per source vertex')
    meta_keys = ('use_real_ras','status','origxyz_status','volume_geometry','hemisphere')
    if not isinstance(metadata, Mapping) or any(k not in metadata for k in meta_keys):
        raise ValueError('complete explicit source geometry/status metadata required')

    kept_v = np.flatnonzero(~vertex_rip).astype(np.int32)
    kept_f = np.flatnonzero(~face_rip).astype(np.int32)
    vtrans = np.full(n, -1, np.int32);vtrans[kept_v] = np.arange(len(kept_v), dtype=np.int32)
    ftrans = np.full(nf, -1, np.int32);ftrans[kept_f] = np.arange(len(kept_f), dtype=np.int32)
    if np.any(vertex_rip[triangles[kept_f]]):
        raise ValueError('non-ripped face references a ripped vertex; upstream topology is invalid')
    output_faces = vtrans[triangles[kept_f]].astype(np.int32)
    row_neighbors, row_faces, row_corners = [], [], []
    out_top = np.zeros((len(kept_v),7), np.int32)
    seen = np.zeros((nf,3), np.bool_)
    # Validate complete logical incidence even for removed rows; never use
    # uninitialized entries after vt.num or vt.vnum.
    validated = []
    for old in range(n):
        degree, count = map(int, top[old,[3,6]])
        values = []
        for name, size, limit in (('neighbor_storage',degree,n),
                                 ('incident_faces',count,nf), ('incident_corners',count,3)):
            raw = np.asarray(surface_state[name][old])
            if raw.ndim != 1 or len(raw) < size:
                raise ValueError('logical row exceeds ordered storage: '+name)
            if size == 0 and isinstance(surface_state[name][old], (list,tuple)):
                raw = np.empty(0, np.int32)
            values.append(_integers(raw[:size], (size,), name, maximum=limit))
        adjacent, incident, corner = values
        if count and not np.all(triangles[incident,corner] == old):
            raise ValueError('incident face/corner does not belong to its source vertex')
        packed = incident.astype(np.int64)*3+corner
        if len(np.unique(packed)) != len(packed) or np.any(seen[incident,corner]):
            raise ValueError('duplicated source face/corner incidence')
        seen[incident,corner] = True
        validated.append(values)
    if not seen.all():
        raise ValueError('source incidence must cover every logical face corner')
    for new, old in enumerate(kept_v):
        adjacent, incident, corner = validated[old]
        neighbors = vtrans[adjacent[~vertex_rip[adjacent]]].copy()
        retained = ~face_rip[incident]
        incident = ftrans[incident[retained]].copy()
        corner = corner[retained].astype(np.uint8)
        row_neighbors.append(neighbors);row_faces.append(incident);row_corners.append(corner)
        out_top[new,0:2] = 1;out_top[new,2:4] = len(neighbors);out_top[new,6] = len(incident)
    state = {key: np.array(a[kept_v], copy=True) for key,a in arrays.items()}
    nv, ff = len(kept_v), len(kept_f)
    state.update(faces=output_faces, nfaces=ff, vertex_topology=out_top,
        neighbor_storage=row_neighbors, incident_faces=row_faces, incident_corners=row_corners,
        vertex_rip=np.zeros(nv,np.int32),vertex_marks=np.zeros(nv,np.int32),
        vertex_border=np.zeros(nv,np.int32),vertex_area=np.zeros(nv,np.float32),
        tangent_e1=np.zeros((nv,3),np.float32),tangent_e2=np.zeros((nv,3),np.float32),
        flattened_vertices=np.zeros((nv,3),np.float32),undefval=np.zeros(nv,np.float32),
        old_undefval=np.zeros(nv,np.float32),face_rip=np.zeros(ff,np.int32),
        face_area=np.zeros(ff,np.float32),face_normals=np.zeros((ff,3),np.float32))
    meta = copy.deepcopy(dict(metadata))
    if isinstance(meta['volume_geometry'], Mapping) and 'register_mat' in meta['volume_geometry']:
        meta['volume_geometry']['register_mat'] = None
    meta.update(surface_type='MRIS_TRIANGULAR_SURFACE',surface_type_code=3,
        nsize=1,max_nsize=1,vertex_capacity=int(1.5*nv+0.5),face_capacity=int(1.5*ff+0.5))
    return dict(surface_state=state,metadata=meta,vertex_translation=vtrans,face_translation=ftrans,
        kept_source_vertices=kept_v,kept_source_faces=kept_f,
        removed_vertices=n-nv,removed_faces=int(nf)-ff,seconds=time.perf_counter()-started,
        source_final_compaction=True,complete_ga=False,final_mesh_quality='not_assessed')
