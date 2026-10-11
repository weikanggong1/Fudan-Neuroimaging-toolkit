"""固定面局部patch的静态CSR缓存与完整目标空间查询，不改动态窄相。

仅省略不会被顺序更新访问的查询行；全体目标三角面仍在空间树中。
复用已有子体积顺序和编译接受核，不执行原版阶段命令。
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
from scipy.spatial import cKDTree

from .place_surface_collision import subvolume_assignment
from .place_surface_snapshot import _ordered_snapshot_step


class FixedPatchCollisionCache:
    """同一固定拓扑的缓存；坐标、包围盒、空间候选每次重新计算。

    faces为int32(F,3)，selected为bool(N,)。复制两者并设为只读，
    缓存固定有序incident CSR与实际查询面；N/F/索引非法立即报错。
    不跨网格重用；step必须提供同shape float32坐标/提案。返回同序
    float32坐标、原完整更新顺序及候选/计时诊断，无文件I/O。
    """

    def __init__(self, *, faces: np.ndarray, selected: np.ndarray):
        raw_faces, raw_selected = np.asarray(faces), np.asarray(selected)
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in 'iu':
            raise ValueError('faces must be integer (F,3)')
        if raw_selected.ndim != 1 or raw_selected.dtype != np.bool_:
            raise ValueError('selected must be bool (N,)')
        if np.any(raw_faces < 0) or np.any(raw_faces >= len(raw_selected)):
            raise ValueError('face vertex outside selected mesh')
        self.faces = np.array(raw_faces, dtype=np.int32, copy=True, order='C')
        self.selected = np.array(raw_selected, dtype=bool, copy=True)
        self.faces.flags.writeable = self.selected.flags.writeable = False
        flat = self.faces.ravel()
        self.incident = np.argsort(flat, kind='stable') // 3
        self.incident_offsets = np.zeros(len(self.selected)+1, np.int64)
        self.incident_offsets[1:] = np.cumsum(np.bincount(flat, minlength=len(self.selected)))
        self.query_faces = np.flatnonzero(np.any(self.selected[self.faces], axis=1))
        self.faces_sha256 = hashlib.sha256(self.faces.astype('<i4').tobytes()).hexdigest()
        self.selected_sha256 = hashlib.sha256(self.selected.tobytes()).hexdigest()

    def step(self, *, current: np.ndarray, proposal: np.ndarray,
             faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """完整即时接受；cache只复用拓扑，查询范围随当前坐标变化。"""
        started = time.perf_counter()
        if not np.array_equal(faces,self.faces):
            raise ValueError('cache belongs to a different ordered topology')
        for values in (current, proposal):
            if (values.shape != (len(self.selected), 3) or values.dtype != np.float32
                    or not np.isfinite(values).all()):
                raise ValueError('finite float32 (N,3) current/proposal required')
        if not np.array_equal(proposal[~self.selected], current[~self.selected]):
            raise ValueError('unselected vertices must remain fixed')
        xyz, next_xyz = np.ascontiguousarray(current), np.ascontiguousarray(proposal)
        triangles = self.faces
        geometry, _, vertex_svi = subvolume_assignment(xyz, triangles, next_xyz, ~self.selected)
        order = np.concatenate([np.flatnonzero(vertex_svi == region) for region in range(65)])
        points = xyz[triangles]
        centers = points.mean(axis=1, dtype=np.float64)
        radii = np.linalg.norm(points.astype(np.float64)-centers[:,None,:], axis=2).max(axis=1)
        maximum_radius = float(radii.max(initial=0))
        tree = cKDTree(centers)
        maximum_motion = float(np.linalg.norm(next_xyz.astype(np.float64)-xyz,axis=1).max(initial=0))
        bound = 2.0*maximum_motion + 0.01
        counts = np.zeros(len(triangles), np.int64)
        chunks = []
        build_started = time.perf_counter()
        # Query rows only; all target centers/faces remain in the original tree.
        for start in range(0, len(self.query_faces), 4096):
            query = self.query_faces[start:start+4096]
            rows = tree.query_ball_point(centers[query], ((radii[query]+maximum_radius)+1.0)+3.0*bound,
                                         return_sorted=False,workers=1)
            sizes = np.fromiter((len(row) for row in rows), np.int64, count=len(rows))
            counts[query] = sizes
            chunks.append(np.fromiter((item for row in rows for item in row),np.int32,count=int(sizes.sum())))
        candidate_offsets = np.empty(len(triangles)+1,np.int64)
        candidate_offsets[0] = 0
        np.cumsum(counts,out=candidate_offsets[1:])
        candidates = np.concatenate(chunks) if chunks else np.empty(0,np.int32)
        build_done = time.perf_counter()
        # Keep the exact existing ordered update, shared-vertex exclusions,
        # live candidate coordinates, bounds guard and triangle predicate.
        result = _ordered_snapshot_step(
            xyz,triangles,next_xyz,order,self.incident,self.incident_offsets,
            np.empty((len(xyz),0),np.int32),np.empty((len(xyz),0),np.bool_),
            np.zeros_like(xyz),False,np.zeros_like(xyz),False,
            geometry,vertex_svi,np.float32(.01),centers,maximum_radius,
            candidate_offsets,candidates,bound)
        diagnostics = {'effective_candidate_backend':'snapshot_active_query_rows',
            'source_faces':len(triangles),'query_faces':len(self.query_faces),
            'candidate_pairs':len(candidates),'motion_bound_mm':bound,
            'fixed_faces_sha256':self.faces_sha256,'selected_vertices_sha256':self.selected_sha256,
            'candidate_build_seconds':build_done-build_started,
            'ordered_acceptance_seconds':time.perf_counter()-build_done,
            'complete_step_seconds':time.perf_counter()-started,
            'static_incident_CSR_reused':True,'all_target_faces_retained':True}
        return result,order,diagnostics
