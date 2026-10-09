"""固定源码逐面相交标记；复用现有源谓词与1mm MHT采样。

修复旧无序pair将一次方向的结果同时赋给两面的假设。属于
mrisMarkIntersections(FillHoles=0)内部步骤，没有独立原软件CLI。
本模块为显式实验接口，未替换生产清理默认。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .place_surface_collision import _sample_mht_voxels


def mark_source_intersections(
    vertices: np.ndarray, faces: np.ndarray, *, face_ripped: np.ndarray | None = None,
    predicate_backend: str = "numba", device: str | None = None,
    block_faces: int = 256, block_pairs: int = 65536,
) -> tuple[np.ndarray, int]:
    """按有向源谓词与共享MHT桶标记每个面，再汇总受影响顶点。

    vertices为(N,3)surface RAS/mm，按原软件顶点存储转float32；faces
    为(M,3)有序整数索引。face_ripped为可选(M,)bool面冻结标记，默认
    全False；它不是顶点冻结标记。返回(N,)bool顶点标记和相交面数。
    predicate_backend默认numba；torch需显式device，支持cpu/CUDA，不
    改精度设置、不使用半精度、不静默回退。源谓词按FP64/固定阈值。
    block_faces默认256、block_pairs默认65536，只限制批次，完整候选
    不截断。非有限坐标、非法shape/索引/后端或非正批次会抛ValueError。

    球半径额外4mm仅用于产生1mm源hash桶可能共有的完整候选；不把
    几何AABB当作带容差源谓词的排除条件。最终阳性必须通过已有源
    MHT采样桶共享验证，未共享的pair不标记。
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("expected vertices (N,3), faces (M,3)")
    if not np.isfinite(xyz).all() or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("coordinates must be finite and faces must be integer indices")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError("face index is outside vertices")
    if predicate_backend not in ("numba", "torch") or any(
        not isinstance(size, (int, np.integer)) or size < 1 for size in (block_faces, block_pairs)
    ):
        raise ValueError("invalid predicate backend or block size")
    if predicate_backend == "torch" and device is None:
        raise ValueError("torch predicate requires explicit device")
    triangles = triangles.astype(np.int32, copy=False)
    face_flags = np.zeros(len(triangles), dtype=bool) if face_ripped is None else np.asarray(face_ripped, dtype=bool)
    if face_flags.shape != (len(triangles),):
        raise ValueError("face_ripped must have shape (M,)")
    face_marks = np.zeros(len(triangles), dtype=bool)
    marks = np.zeros(len(xyz), dtype=bool)
    if len(triangles) < 2:
        return marks, 0
    from .place_surface_collision_torch import _source_pairs
    if predicate_backend == "torch":
        from .place_surface_collision_torch import triangle_pairs_intersect_torch

    points = xyz[triangles]
    centers = points.mean(axis=1, dtype=np.float64)
    radii = np.linalg.norm(points.astype(np.float64) - centers[:, None], axis=2).max(axis=1)
    # 源1mm桶采样是三角形内的凸组合与两个样点桶之间的整数路径。
    # 所有路径坐标都位于整数化corner AABB内；边界pad仅保护FP64累加。
    # 不使用几何AABB，以免删掉源平面容差判为相交的近共面pair。
    cell_low = np.trunc(points.min(axis=1).astype(np.float64) + 1000. - 1e-5).astype(np.int64)
    cell_high = np.trunc(points.max(axis=1).astype(np.float64) + 1000. + 1e-5).astype(np.int64)
    maximum = radii.max()
    bucket_cache: dict[int, set] = {}

    def evaluate(first, second):
        a, b = points[first], points[second]
        if predicate_backend == "numba":
            forward, reverse = _source_pairs(a, b), _source_pairs(b, a)
        else:
            forward, _ = triangle_pairs_intersect_torch(a, b, device=device, chunk_size=block_pairs)
            reverse, _ = triangle_pairs_intersect_torch(b, a, device=device, chunk_size=block_pairs)
            forward, reverse = forward.cpu().numpy(), reverse.cpu().numpy()
        for slot in np.flatnonzero(forward | reverse):
            fa, fb = int(first[slot]), int(second[slot])
            if fa not in bucket_cache:
                bucket_cache[fa] = _sample_mht_voxels(points[fa])
            if fb not in bucket_cache:
                bucket_cache[fb] = _sample_mht_voxels(points[fb])
            if bucket_cache[fa].isdisjoint(bucket_cache[fb]):
                continue
            # 与源逐fno查询一样，每一方向只标查询面，不能OR后同时标双面。
            face_marks[fa] |= forward[slot]
            face_marks[fb] |= reverse[slot]

    if predicate_backend == "torch":
        # 复用已完成的GPU完整候选空间索引；整数桶锚点区间相交是
        # 源采样桶共有的必要条件，避免逐面cKDTree/Python列表。
        from .place_surface_candidates_torch import conservative_face_candidates_torch
        boxes_low = cell_low.astype(np.float64) - 1000.
        boxes_high = cell_high.astype(np.float64) - 1000.
        offsets, ids, _ = conservative_face_candidates_torch(
            centers, centers, radii + maximum + 4., device=device,
            source_low=boxes_low, source_high=boxes_high,
            query_low=boxes_low, query_high=boxes_high, motion_bound=0.,
            source_faces=triangles, query_faces=triangles)
        first = np.repeat(np.arange(len(triangles), dtype=np.int64), np.diff(offsets))
        keep = (first < ids) & ~face_flags[first] & ~face_flags[ids]
        first, second = first[keep], ids[keep]
        for start in range(0, len(first), block_pairs):
            evaluate(first[start:start + block_pairs], second[start:start + block_pairs])
        marks[triangles[face_marks].ravel()] = True
        return marks, int(face_marks.sum())

    tree = cKDTree(centers)
    for offset in range(0, len(triangles), block_faces):
        end = min(offset + block_faces, len(triangles))
        nearby = tree.query_ball_point(centers[offset:end], radii[offset:end] + maximum + 4.)
        first_parts, second_parts = [], []
        for fa, other in enumerate(nearby, offset):
            if face_flags[fa]:
                continue
            other = np.asarray(other, dtype=np.int64)
            other = other[(other > fa) & ~face_flags[other]]
            if not len(other):
                continue
            near = np.sum((centers[other] - centers[fa]) ** 2, axis=1) <= (radii[other] + radii[fa] + 4.) ** 2
            other = other[near]
            cells_overlap = np.all(cell_low[fa] <= cell_high[other], axis=1) & np.all(
                cell_low[other] <= cell_high[fa], axis=1)
            other = other[cells_overlap]
            other = other[np.all(triangles[fa, :, None] != triangles[other, None, :], axis=(1, 2))]
            first_parts.append(np.full(len(other), fa, dtype=np.int64))
            second_parts.append(other)
        if not first_parts:
            continue
        first, second = np.concatenate(first_parts), np.concatenate(second_parts)
        for start in range(0, len(first), block_pairs):
            evaluate(first[start:start + block_pairs], second[start:start + block_pairs])
    marks[triangles[face_marks].ravel()] = True
    return marks, int(face_marks.sum())
