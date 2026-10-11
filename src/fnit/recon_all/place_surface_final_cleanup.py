"""Pial medial-wall pinning and native-order intersection repair."""

from __future__ import annotations

import numpy as np

from .mris_remove_intersection_python import mark_intersections
from .place_surface_smoothing import _ordered_neighbors


def pial_cleanup_fixed_vertices(*, ripped: np.ndarray, cortex_vertices: np.ndarray) -> np.ndarray:
    """pial最终清理的冻结集合：原rip与cortex外已pin顶点的并集。

    ripped为(N,)bool，cortex_vertices为一维整数顶点序号（范围0..N-1）。
    返回独立(N,)bool，不修改优化器rip或label；空cortex表示全部pin。
    hippocampus/amygdala在优化中可动，但pin到white后不能被SOAP再次移动。
    不解除rip、不改目标/候选/接受、不把残余相交清零。非法shape/类型/越界
    抛ValueError。示例和真实冻结输入修复前后证据见task2链审计专页。
    """
    flags=np.asarray(ripped)
    cortex=np.asarray(cortex_vertices)
    if flags.ndim!=1 or flags.dtype!=np.bool_:
        raise ValueError("one-dimensional boolean ripped required")
    if cortex.ndim!=1 or not np.issubdtype(cortex.dtype,np.integer):
        raise ValueError("one-dimensional integer cortex vertices required")
    if cortex.size and (cortex.min()<0 or cortex.max()>=len(flags)):
        raise ValueError("cortex vertex outside ordered pial mesh")
    fixed=np.ones(len(flags),np.bool_)
    fixed[cortex]=False
    fixed|=flags
    return fixed


def pial_cleanup_mask(*, ripped: np.ndarray, cortex_vertices: np.ndarray,
                      constraint_mode: str = "source_rip") -> np.ndarray:
    """选取清理rip mask；它同时限制移动顶点和SOAP邻居贡献。

    source_rip默认返回原优化rip的独立副本，保留非rip、cortex外邻居。
    cortex_pinned_union显式返回原rip与cortex外顶点并集；这些新增固定
    顶点既不移动，也不参与邻居平均。这是强化诊断策略，不是源算法
    等价修复或仅冻结移动域的算法。输入格式与上方并集函数一致。
    两种模式均不改变原rip/标签，非法模式在分配或计算前拒绝。
    """
    if constraint_mode not in ("source_rip", "cortex_pinned_union"):
        raise ValueError("cleanup_constraint_mode must be source_rip or cortex_pinned_union")
    union = pial_cleanup_fixed_vertices(ripped=ripped, cortex_vertices=cortex_vertices)
    return np.asarray(ripped).copy() if constraint_mode == "source_rip" else union


def pin_medial_wall(pial: np.ndarray, white: np.ndarray, cortex_vertices: np.ndarray) -> np.ndarray:
    """Match MRISpinMedialWallToWhite on the ordered vertex arrays."""
    result = np.asarray(pial, dtype=np.float32).copy()
    outside = np.ones(len(result), dtype=np.bool_)
    outside[np.asarray(cortex_vertices, dtype=np.int64)] = False
    result[outside] = np.asarray(white, dtype=np.float32)[outside]
    return result


def repair_intersections(
    vertices: np.ndarray, faces: np.ndarray, ripped: np.ndarray,
    *, marking_backend: str = "legacy", device: str | None = None,
    candidate_grid_cells_per_axis: int = 2,
    ordered_neighbor_offsets: np.ndarray | None = None,
    ordered_neighbors: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """源顺序100次soap-bubble/轮；新有向marker仍为显式实验选项。

    vertices为(N,3)surface RAS/mm，faces为(M,3)有序整数，ripped为(N,)
    bool顶点冻结标记。返回float32坐标(N,3)和相交数/迭代诊断dict。
    marking_backend默认legacy，保留已验证pial的旧默认；source_numba/
    source_torch使用源逐面方向和MHT桶规则，后者需要显式device。不放宽
    非零残余，修复停滞时按源规则返回最佳状态，由调用阶段检查质量。
    有向后端在同一次清理中仅复用逐元素完全相同的坐标标记，停滞时不
    重建空间索引；任一坐标变化即失效，不跨阶段缓存。错误后端或缺少
    Torch设备抛ValueError。candidate_grid_cells_per_axis默认2；显式3
    仅用于source_torch完整候选索引，不裁剪面对或改变soap/停止规则，
    其他后端非默认值报错。ordered_neighbor_offsets/ordered_neighbors
    默认均None，完整保持pial从面构建邻接的已验证路径；显式时两者
    必须同时提供，分别为整数(N+1,)/(E,)一环CSR，按原顺序保留未被
    面支撑的邻接。转换前检查类型/范围/CSR，不能用faces重建替换。
    显式CSR复用源F32双pass SOAP，其100轮、marker/停止/最佳恢复不变；
    不修改输入。零相交也先校验CSR，非法数据ValueError。
    对应mris_remove_intersection内部MRISremoveIntersections。
    """
    if marking_backend not in ("legacy", "source_numba", "source_torch"):
        raise ValueError("invalid intersection marking_backend")
    if marking_backend == "source_torch" and device is None:
        raise ValueError("source_torch marking requires explicit device")
    if candidate_grid_cells_per_axis not in (2,3):
        raise ValueError("candidate_grid_cells_per_axis must be 2 or 3")
    if candidate_grid_cells_per_axis!=2 and marking_backend!="source_torch":
        raise ValueError("nondefault cleanup candidate grid requires source_torch")
    csr = None
    if (ordered_neighbor_offsets is None) != (ordered_neighbors is None):
        raise ValueError("both ordered neighbor CSR arrays must be provided")
    if ordered_neighbor_offsets is not None:
        from .topology_compact import _integers
        raw = np.asarray(ordered_neighbors)
        if raw.ndim != 1:
            raise ValueError("ordered_neighbors must be a one-dimensional integer array")
        adjacent = _integers(raw, raw.shape, "ordered_neighbors", maximum=len(vertices))
        offsets = _integers(ordered_neighbor_offsets, (len(vertices)+1,), "ordered_neighbor_offsets")
        if offsets[0] != 0 or offsets[-1] != len(adjacent) or np.any(np.diff(offsets.astype(np.int64)) < 0):
            raise ValueError("complete ordered neighbor CSR required")
        csr = (offsets, adjacent)
    marker = mark_intersections
    marker_calls = marker_evaluations = marker_cache_hits = 0
    if marking_backend != "legacy":
        from .place_surface_intersection_marking import mark_source_intersections
        previous_coordinates = previous_marks = previous_count = None

        def marker(xyz, tris):
            nonlocal previous_coordinates, previous_marks, previous_count
            nonlocal marker_calls, marker_evaluations, marker_cache_hits
            marker_calls += 1
            if previous_coordinates is not None and np.array_equal(previous_coordinates, xyz):
                marker_cache_hits += 1
                return previous_marks.copy(), previous_count
            marked, count = mark_source_intersections(
                xyz, tris, predicate_backend="torch" if marking_backend == "source_torch" else "numba",
                device=device,candidate_grid_cells_per_axis=candidate_grid_cells_per_axis)
            marker_evaluations += 1
            previous_coordinates, previous_marks, previous_count = xyz.copy(), marked.copy(), count
            return marked, count
    result = np.asarray(vertices, dtype=np.float32).copy()
    faces = np.asarray(faces, dtype=np.int32)
    ripped = np.asarray(ripped, dtype=np.bool_)
    marked, count = marker(result, faces)
    if count == 0:
        diagnostics = {"intersecting_faces_before": 0, "intersecting_faces_after": 0,
                       "marked_vertices": 0, "smoothing_cycles": 0}
        if csr is not None:
            diagnostics.update(smoothing_neighbor_domain="explicit_ordered_CSR",
                               ordered_neighbor_entries=len(csr[1]))
        return result, diagnostics
    if csr is None:
        neighbors, valid, _ = _ordered_neighbors(faces, len(result))
    first_count, first_marked = count, int(marked.sum())
    best, minimum = result.copy(), len(result)
    old_count, no_progress, cycles = len(result), 0, 0
    smoothed = 0
    trace = [count]
    while count:
        if count > old_count or count == old_count and no_progress >= 0:
            no_progress += 1
            if no_progress > 15:
                break
        else:
            no_progress = 0 if count < old_count else no_progress + 1
            if count < minimum:
                minimum, best = count, result.copy()
        old_count = count
        moving = np.flatnonzero(marked & ~ripped)
        smoothed += len(moving)
        if csr is not None:
            from .topology_finish_geometry import soap_bubble_vertex_positions
            result = soap_bubble_vertex_positions(
                vertices=result, neighbor_offsets=csr[0], neighbors=csr[1],
                vertex_rip=ripped, vertex_marks=np.where(marked, 0, 1).astype(np.int32),
                iterations=100)["vertices"]
        else:
            for _ in range(100):
                next_xyz = result.copy()
                for vertex in moving:
                    x, y, z = (np.float32(value) for value in result[vertex])
                    n = np.float32(1)
                    for slot in range(neighbors.shape[1]):
                        if not valid[vertex, slot]:
                            continue
                        other = neighbors[vertex, slot]
                        if ripped[other]:
                            continue
                        x = np.float32(x + result[other, 0])
                        y = np.float32(y + result[other, 1])
                        z = np.float32(z + result[other, 2])
                        n = np.float32(n + np.float32(1))
                    next_xyz[vertex, 0] = np.float32(x / n)
                    next_xyz[vertex, 1] = np.float32(y / n)
                    next_xyz[vertex, 2] = np.float32(z / n)
                result = next_xyz
        cycles += 1
        if cycles > 101:
            break
        marked, count = marker(result, faces)
        trace.append(count)
    if count > minimum:
        result = best
        _, count = marker(result, faces)
    diagnostics = {"intersecting_faces_before": first_count,
                    "intersecting_faces_after": count,
                    "intersecting_faces_trace": trace,
                    "marked_vertices": first_marked,
                    "smoothed_vertices": smoothed,
                    "smoothing_cycles": cycles,
                    "smoothing_iterations": 100 * cycles}
    if marking_backend != "legacy":
        diagnostics.update(marker_calls=marker_calls, marker_evaluations=marker_evaluations,
                           marker_identical_geometry_cache_hits=marker_cache_hits)
    if csr is not None:
        diagnostics.update(smoothing_neighbor_domain="explicit_ordered_CSR",
                           ordered_neighbor_entries=len(csr[1]))
    return result, diagnostics
