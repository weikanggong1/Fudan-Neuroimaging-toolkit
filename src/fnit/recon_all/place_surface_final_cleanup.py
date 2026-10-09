"""Pial medial-wall pinning and native-order intersection repair."""

from __future__ import annotations

import numpy as np

from .mris_remove_intersection_python import mark_intersections
from .place_surface_smoothing import _ordered_neighbors


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
) -> tuple[np.ndarray, dict]:
    """源顺序100次soap-bubble/轮；新有向marker仍为显式实验选项。

    vertices为(N,3)surface RAS/mm，faces为(M,3)有序整数，ripped为(N,)
    bool顶点冻结标记。返回float32坐标(N,3)和相交数/迭代诊断dict。
    marking_backend默认legacy，保留已验证pial的旧默认；source_numba/
    source_torch使用源逐面方向和MHT桶规则，后者需要显式device。不放宽
    非零残余，修复停滞时按源规则返回最佳状态，由调用阶段检查质量。
    有向后端在同一次清理中仅复用逐元素完全相同的坐标标记，停滞时不
    重建空间索引；任一坐标变化即失效，不跨阶段缓存。错误后端或缺少
    Torch设备抛ValueError。对应mris_remove_intersection。
    """
    if marking_backend not in ("legacy", "source_numba", "source_torch"):
        raise ValueError("invalid intersection marking_backend")
    if marking_backend == "source_torch" and device is None:
        raise ValueError("source_torch marking requires explicit device")
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
                device=device)
            marker_evaluations += 1
            previous_coordinates, previous_marks, previous_count = xyz.copy(), marked.copy(), count
            return marked, count
    result = np.asarray(vertices, dtype=np.float32).copy()
    faces = np.asarray(faces, dtype=np.int32)
    ripped = np.asarray(ripped, dtype=np.bool_)
    marked, count = marker(result, faces)
    if count == 0:
        return result, {"intersecting_faces_before": 0, "intersecting_faces_after": 0,
                        "marked_vertices": 0, "smoothing_cycles": 0}
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
    return result, diagnostics
