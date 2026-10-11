"""独立的局部残余相交求解；不修改原 SOAP 复现路径或固定顶点。

这是显式实验接口，不等价于 FreeSurfer 的 MRISremoveIntersections。
全部候选使用原源有向谓词/MHT 与独立双精度谓词，完整覆盖改变的面。
"""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial import cKDTree

from .place_surface_collision import _sample_mht_voxels
from .place_surface_collision_torch import _source_pairs
from .mris_remove_intersection_python import _triangles_intersect
from .place_surface_intersection_marking import _candidate_pairs_from_csr


def _pair_sets(vertices, faces, query_faces=None):
    """返回源有向面对和独立 F64 无向面对；完整查询，不截断候选。"""
    if len(faces) < 2:
        return set(), set()
    points = np.asarray(vertices, np.float32)[faces]
    p64 = points.astype(np.float64)
    centers = p64.mean(axis=1)
    radii = np.linalg.norm(p64-centers[:, None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    all_faces = query_faces is None
    query = np.arange(len(faces)) if all_faces else np.unique(query_faces)
    selected = set(map(int, query))
    face_flags = np.zeros(len(faces), bool)
    source, independent = set(), set()
    buckets = {}
    low = np.trunc(points.min(axis=1).astype(np.float64)+1000.-1e-5).astype(np.int64)
    high = np.trunc(points.max(axis=1).astype(np.float64)+1000.+1e-5).astype(np.int64)
    for start in range(0, len(query), 256):
        ids = query[start:start+256]
        rows = tree.query_ball_point(centers[ids], radii[ids]+radii.max()+4.)
        if all_faces:
            offsets = np.empty(len(rows)+1, np.int64); offsets[0] = 0
            offsets[1:] = np.cumsum([len(row) for row in rows])
            neighbors = np.concatenate(rows, dtype=np.int64)
            a, b = _candidate_pairs_from_csr(centers, radii, low, high, faces,
                                            face_flags, int(ids[0]), offsets, neighbors)
        else:
            first, second = [], []
            for fa, nearby in zip(ids, rows):
                for fb in nearby:
                    if fb == fa or (fb in selected and fb < fa):
                        continue
                    first.append(min(fa, fb)); second.append(max(fa, fb))
            a, b = np.asarray(first, np.int64), np.asarray(second, np.int64)
        if not len(a): continue
        keep = np.all(faces[a, :, None] != faces[b, None, :], axis=(1, 2))
        distance = np.sum((centers[a]-centers[b])**2, axis=1)
        keep &= distance <= (radii[a]+radii[b]+4.)**2
        a, b, distance = a[keep], b[keep], distance[keep]
        source_keep = np.all(low[a] <= high[b], axis=1) & np.all(low[b] <= high[a], axis=1)
        sa, sb = a[source_keep], b[source_keep]
        forward, reverse = _source_pairs(points[sa], points[sb]), _source_pairs(points[sb], points[sa])
        for slot in np.flatnonzero(forward | reverse):
            fa, fb = int(sa[slot]), int(sb[slot])
            if fa not in buckets:
                buckets[fa] = _sample_mht_voxels(points[fa])
            if fb not in buckets:
                buckets[fb] = _sample_mht_voxels(points[fb])
            if buckets[fa].isdisjoint(buckets[fb]):
                continue
            if forward[slot]: source.add((fa, fb))
            if reverse[slot]: source.add((fb, fa))
        independent_keep = distance <= (radii[a]+radii[b]+1e-5)**2
        independent_keep &= np.all(p64[a].min(axis=1) <= p64[b].max(axis=1)+1e-5, axis=1)
        independent_keep &= np.all(p64[b].min(axis=1) <= p64[a].max(axis=1)+1e-5, axis=1)
        for fa, fb in zip(a[independent_keep], b[independent_keep]):
            if _triangles_intersect(p64[fa], p64[fb]):
                independent.add((int(fa), int(fb)))
    return source, independent


def _face_normals(vertices, faces):
    tri = vertices.astype(np.float64)[faces]
    return np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])


def _checked(vertices, faces, fixed_vertices, white_vertices):
    xyz = np.asarray(vertices)
    white = np.asarray(white_vertices)
    triangles = np.asarray(faces)
    fixed = np.asarray(fixed_vertices)
    if xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("finite float32 vertices (N,3) in surface RAS/mm required")
    if white.shape != xyz.shape or not np.isfinite(white).all() or not np.issubdtype(white.dtype, np.floating):
        raise ValueError("finite ordered own-white coordinates (N,3) required")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("integer ordered faces (F,3) required")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError("face vertex outside ordered domain")
    if fixed.dtype != np.bool_ or fixed.shape != (len(xyz),):
        raise ValueError("explicit boolean fixed_vertices (N,) required")
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("face repeats a vertex")
    return xyz.copy(), triangles.astype(np.int32), fixed.copy(), white.astype(np.float64)


def resolve_residual_intersections(*, vertices: np.ndarray, faces: np.ndarray,
                                   fixed_vertices: np.ndarray, white_vertices: np.ndarray,
                                   max_displacement_mm: float = 1.0, minimum_area_ratio: float = .2,
                                   max_passes: int = 4) -> tuple[np.ndarray, dict]:
    """显式局部候选求解，返回新 float32 坐标及完整接受/失败诊断。

    vertices/white_vertices 为同网格 (N,3) surface RAS/mm；white必须为
    自产同索引 white，只提供局部候选方向，不复制整个参考表面。
    faces 为 (F,3) 有序整数，始终不改。fixed_vertices 为 (N,) bool，
    调用者必须合并 rip 与全部 cortex 外点，固定在进入函数的原坐标。
    max_displacement_mm 默认1mm是候选算法上限；minimum_area_ratio
    默认0.2限制每个受影响面的面积与初始值之比；max_passes默认4。
    这些不是整例指标等效阈值。无半精度、不初始化GPU/更改TF32。

    固定候选表为0/1/2邻接环、自产white方向及含固定邻居的局部均值
    方向，步长为上限乘0.05/0.1/0.2/0.35/0.5/0.75/1；浮点写入前
    乘0.9999避免量化越过上限。源有向及F64两组阳性面对都必须严格
    减少，且全局不能新增阳性面对；所有受影响面需正面积、相对初始
    法向点积为正。局部检查完整覆盖所有坐标改变的面，最终再全量
    双谓词复检。不能消除时返回status=failed_residual，无占位输出。
    非法dtype/shape/参数、退化输入面及验证器不同意候选会明确失败；
    不修改输入，原repair_intersections行为不变。
    """
    wall = time.perf_counter()
    original, triangles, fixed, white = _checked(vertices, faces, fixed_vertices, white_vertices)
    if not np.isfinite(max_displacement_mm) or max_displacement_mm <= 0:
        raise ValueError("positive finite max_displacement_mm required")
    if not np.isfinite(minimum_area_ratio) or not 0 < minimum_area_ratio <= 1:
        raise ValueError("minimum_area_ratio must be in (0,1]")
    if isinstance(max_passes, (bool, np.bool_)) or not isinstance(max_passes, (int, np.integer)) or max_passes < 1:
        raise ValueError("positive integer max_passes required")
    initial_normals = _face_normals(original, triangles)
    initial_area2 = np.linalg.norm(initial_normals, axis=1)
    if np.any(initial_area2 <= 0):
        raise ValueError("positive initial triangle areas required")
    result = original.copy()
    source, independent = _pair_sets(result, triangles)
    initial_source, initial_independent = source.copy(), independent.copy()
    diagnostics = {"schema_version": 1, "algorithm": "independent_local_residual_solver_v1",
                   "source_directed_pairs_before": len(source), "float64_pairs_before": len(independent),
                   "fixed_vertices": int(fixed.sum()), "parameters": {
                       "max_displacement_mm": float(max_displacement_mm), "minimum_area_ratio": float(minimum_area_ratio),
                       "max_passes": int(max_passes), "rings": [0, 1, 2],
                       "step_fractions": [.05, .1, .2, .35, .5, .75, 1.], "quantization_safety_factor": .9999},
                   "candidate_trace": [], "accepted": [], "global_checks": [],
                   "strict_reproduction": "independent_algorithm_not_source_reproduction",
                   "overall_metric_equivalence": "not_assessed"}
    adjacency = [set() for _ in range(len(result))]
    for a, b, c in triangles:
        adjacency[a].update((int(b), int(c))); adjacency[b].update((int(a), int(c))); adjacency[c].update((int(a), int(b)))
    for pass_number in range(max_passes):
        bad_faces = {a for a, b in source} | {x for pair in independent for x in pair}
        if not bad_faces:
            break
        active = set(map(int, triangles[list(bad_faces)].ravel()))
        active.difference_update(map(int, np.flatnonzero(fixed)))
        best = None
        for ring in (0, 1, 2):
            if ring:
                active.update({w for v in tuple(active) for w in adjacency[v] if not fixed[w]})
            indices = np.asarray(sorted(active), np.int64)
            if not len(indices): continue
            affected = np.flatnonzero(np.isin(triangles, indices).any(axis=1))
            current_source, current_independent = _pair_sets(result, triangles, affected)
            smoothing = np.array([result[sorted(adjacency[v])].astype(np.float64).mean(axis=0) for v in indices])
            for direction_name, target in (("own_white_direction", white[indices]), ("fixed_boundary_mean", smoothing)):
                direction = target-result[indices].astype(np.float64)
                length = np.linalg.norm(direction, axis=1)
                direction /= np.maximum(length[:, None], np.finfo(np.float64).tiny)
                for fraction in (.05, .1, .2, .35, .5, .75, 1.):
                    candidate = result.copy()
                    next_xyz = result[indices].astype(np.float64) + direction*(max_displacement_mm*fraction*.9999)
                    delta = next_xyz-original[indices]
                    scale = np.minimum(1., max_displacement_mm*.9999/np.maximum(np.linalg.norm(delta, axis=1), np.finfo(np.float64).tiny))
                    candidate[indices] = (original[indices]+delta*scale[:, None]).astype(np.float32)
                    normals = _face_normals(candidate, triangles[affected])
                    displacement = np.linalg.norm(candidate.astype(np.float64)-original, axis=1)
                    reason = None
                    if displacement.max(initial=0.) > max_displacement_mm: reason = "total_displacement_limit"
                    elif np.any(np.sum(normals*initial_normals[affected], axis=1) <= 0): reason = "face_orientation"
                    elif np.any(np.linalg.norm(normals, axis=1) < minimum_area_ratio*initial_area2[affected]): reason = "face_area"
                    row = {"pass": pass_number, "ring": ring, "direction": direction_name,
                           "step_fraction": fraction, "moving_vertices": len(indices), "affected_faces": len(affected)}
                    if reason is None:
                        candidate_source, candidate_independent = _pair_sets(candidate, triangles, affected)
                        row.update(source_directed_pairs=len(candidate_source), float64_pairs=len(candidate_independent))
                        if not candidate_source <= current_source or not candidate_independent <= current_independent:
                            reason = "new_positive_pair"
                        elif not len(candidate_source) < len(current_source) or not len(candidate_independent) < len(current_independent):
                            reason = "not_strictly_reduced_in_both_detectors"
                        else:
                            final_source = (source-current_source) | candidate_source
                            final_independent = (independent-current_independent) | candidate_independent
                            score = (len(final_source)+len(final_independent), float(displacement.max(initial=0.)), float(displacement.sum()))
                            if best is None or score < best[0]:
                                row["decision"] = "eligible"
                                best = (score, candidate, final_source, final_independent, row.copy())
                    row["decision"] = reason or "eligible"
                    diagnostics["candidate_trace"].append(row)
        if best is None: break
        _, result, source, independent, selected = best
        diagnostics["accepted"].append(selected)
    # 最终完整面对复检不依赖局部表或总计数，可捕获任何新增阳性面对。
    final_source, final_independent = _pair_sets(result, triangles)
    diagnostics["global_checks"].append({"full_source_pair_set_matches_local_update": final_source == source,
                                           "full_float64_pair_set_matches_local_update": final_independent == independent,
                                           "no_new_source_directed_pairs": final_source <= initial_source,
                                           "no_new_float64_pairs": final_independent <= initial_independent})
    if not all(diagnostics["global_checks"][0].values()):
        raise RuntimeError("complete global pair validation disagrees with accepted candidates")
    if not np.array_equal(result[fixed], original[fixed]):
        raise RuntimeError("fixed vertex changed")
    displacement = np.linalg.norm(result.astype(np.float64)-original, axis=1)
    diagnostics.update(status="passed_zero_intersections" if not final_source and not final_independent else "failed_residual",
                       source_directed_pairs_after=len(final_source), float64_pairs_after=len(final_independent),
                       source_intersecting_faces_after=len({a for a, b in final_source}),
                       float64_intersecting_faces_after=len({x for pair in final_independent for x in pair}),
                       changed_vertices=int(np.count_nonzero(displacement)), fixed_different_elements=0,
                       mean_displacement_mm=float(displacement.mean()), p99_displacement_mm=float(np.quantile(displacement, .99)),
                       max_displacement_mm=float(displacement.max(initial=0.)),
                       full_resolver_seconds=time.perf_counter()-wall)
    return result, diagnostics
