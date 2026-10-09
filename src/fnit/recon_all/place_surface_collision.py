"""Source-order surface placement triangle intersection probe.

The exact Möller triangle test follows FreeSurfer's `tritri.cpp`. The
static-mesh broadphase is diagnostic; ordered asynchronous updates remain open.
"""

from __future__ import annotations

import numpy as np
from itertools import chain
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=True)
def _interval(v: np.ndarray, d: np.ndarray) -> tuple[float, float, bool]:
    d01 = d[0] * d[1]
    d02 = d[0] * d[2]
    if d01 > 0:
        first, second, third = 2, 0, 1
    elif d02 > 0:
        first, second, third = 1, 0, 2
    elif d[1] * d[2] > 0 or d[0] != 0:
        first, second, third = 0, 1, 2
    elif d[1] != 0:
        first, second, third = 1, 0, 2
    elif d[2] != 0:
        first, second, third = 2, 0, 1
    else:
        return 0.0, 0.0, True
    a = v[first] + (v[second] - v[first]) * d[first] / (d[first] - d[second])
    b = v[first] + (v[third] - v[first]) * d[first] / (d[first] - d[third])
    return min(a, b), max(a, b), False


@njit(cache=True)
def _edge_hit(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, x: int, y: int) -> bool:
    ax, ay = b[x] - a[x], b[y] - a[y]
    bx, by = c[x] - d[x], c[y] - d[y]
    cx, cy = a[x] - c[x], a[y] - c[y]
    f = ay * bx - ax * by
    h = by * cx - bx * cy
    if (f > 0 and 0 <= h <= f) or (f < 0 and f <= h <= 0):
        e = ax * cy - ay * cx
        return (0 <= e <= f) if f > 0 else (f <= e <= 0)
    return False


@njit(cache=True)
def _point_in_tri(a: np.ndarray, triangle: np.ndarray, x: int, y: int) -> bool:
    distances = np.zeros(3, dtype=np.float64)
    for edge in range(3):
        p, q = triangle[edge], triangle[(edge + 1) % 3]
        aa, bb = q[y] - p[y], -(q[x] - p[x])
        cc = -aa * p[x] - bb * p[y]
        distances[edge] = aa * a[x] + bb * a[y] + cc
    return distances[0] * distances[1] > 0 and distances[0] * distances[2] > 0


@njit(cache=True)
def _coplanar(a: np.ndarray, b: np.ndarray, normal: np.ndarray) -> bool:
    absolute = np.abs(normal)
    if absolute[0] > absolute[1]:
        x, y = (1, 2) if absolute[0] > absolute[2] else (0, 1)
    else:
        x, y = (0, 1) if absolute[2] > absolute[1] else (0, 2)
    for edge in range(3):
        p, q = a[edge], a[(edge + 1) % 3]
        for other in range(3):
            if _edge_hit(p, q, b[other], b[(other + 1) % 3], x, y):
                return True
    return _point_in_tri(a[0], b, x, y) or _point_in_tri(b[0], a, x, y)


@njit(cache=True)
def triangles_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    """FreeSurfer's double-precision Möller test on two float32 triangles."""
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    n1 = np.cross(aa[1] - aa[0], aa[2] - aa[0])
    d1 = -np.dot(n1, aa[0])
    du = np.array([np.dot(n1, bb[i]) + d1 for i in range(3)])
    absolute = np.abs(du)
    if np.any(absolute > 1e-6) and du[0] * du[1] > 0 and du[0] * du[2] > 0:
        return False
    for i in range(3):
        if absolute[i] < 1e-6:
            du[i] = 0.0
    if absolute[0] < 1e-5 and absolute[1] < 1e-5 and absolute[2] < 1e-6:
        du[:] = 0.0

    n2 = np.cross(bb[1] - bb[0], bb[2] - bb[0])
    d2 = -np.dot(n2, bb[0])
    dv = np.array([np.dot(n2, aa[i]) + d2 for i in range(3)])
    absolute = np.abs(dv)
    if np.any(absolute > 1e-6) and dv[0] * dv[1] > 0 and dv[0] * dv[2] > 0:
        return False
    for i in range(3):
        if absolute[i] < 1e-6:
            dv[i] = 0.0
    if absolute[0] < 1e-5 and absolute[1] < 1e-5 and absolute[2] < 1e-6:
        dv[:] = 0.0

    direction = np.cross(n1, n2)
    axis = 0
    if abs(direction[1]) > abs(direction[axis]):
        axis = 1
    if abs(direction[2]) > abs(direction[axis]):
        axis = 2
    a0, a1, coplanar = _interval(aa[:, axis], dv)
    if coplanar:
        return _coplanar(aa, bb, n1)
    b0, b1, coplanar = _interval(bb[:, axis], du)
    if coplanar:
        return _coplanar(aa, bb, n1)
    return not (a1 < b0 or b1 < a0)


def static_collision_mask(
    vertices: np.ndarray, faces: np.ndarray, proposed: np.ndarray,
    ripped: np.ndarray, selected: np.ndarray,
) -> np.ndarray:
    """Test selected proposed moves against the original mesh only."""
    xyz = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    next_xyz = np.asarray(proposed, dtype=np.float32)
    face_ripped = np.zeros(len(triangles), dtype=np.bool_)
    face_points = xyz[triangles]
    centers = face_points.mean(axis=1, dtype=np.float64)
    radii = np.linalg.norm(face_points.astype(np.float64) - centers[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centers)
    flat_vertices = triangles.ravel()
    order = np.argsort(flat_vertices, kind="stable")
    offsets = np.zeros(len(xyz) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(flat_vertices, minlength=len(xyz)))
    incident = order // 3
    result = np.zeros(len(selected), dtype=np.bool_)
    for position, vertex in enumerate(selected):
        vertex = int(vertex)
        if ripped[vertex] or np.array_equal(xyz[vertex], next_xyz[vertex]):
            continue
        for face_id in incident[offsets[vertex]:offsets[vertex + 1]]:
            if face_ripped[face_id]:
                continue
            corners = triangles[face_id]
            moved = xyz[corners].copy()
            moved[np.flatnonzero(corners == vertex)[0]] = next_xyz[vertex]
            center = moved.mean(axis=0, dtype=np.float64)
            radius = np.linalg.norm(moved.astype(np.float64) - center, axis=1).max()
            for other in tree.query_ball_point(center, radius + radii.max()):
                if face_ripped[other]:
                    continue
                if np.linalg.norm(center - centers[other]) > radius + radii[other]:
                    continue
                if np.any(np.isin(triangles[other], corners)):
                    continue
                if triangles_intersect(moved, face_points[other]):
                    result[position] = True
                    break
            if result[position]:
                break
    return result


@njit(cache=True)
def _subvolume_index(point: np.ndarray, geometry: np.ndarray) -> int:
    index = np.zeros((2, 3), dtype=np.int32)
    for axis in range(3):
        low = geometry[2 * axis]
        length = geometry[6 + axis]
        verge = geometry[9 + axis]
        value = point[axis]
        index[0, axis] = min(3, int(np.float32(np.float32(np.float32(value - verge) - low) / length)))
        index[1, axis] = min(3, int(np.float32(np.float32(np.float32(value + verge) - low) / length)))
        if index[0, axis] != index[1, axis]:
            return 64
    return index[0, 0] * 16 + index[0, 1] * 4 + index[0, 2]


@njit(cache=True)
def _assign_faces(bounds: np.ndarray, faces: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    result = np.zeros(len(faces), dtype=np.int32)
    for face_id in range(len(faces)):
        low = np.empty(3, dtype=np.float32)
        high = np.empty(3, dtype=np.float32)
        for axis in range(3):
            low[axis] = bounds[faces[face_id, 0], axis, 0]
            high[axis] = bounds[faces[face_id, 0], axis, 1]
            for corner in range(1, 3):
                vertex = faces[face_id, corner]
                low[axis] = min(low[axis], bounds[vertex, axis, 0])
                high[axis] = max(high[axis], bounds[vertex, axis, 1])
        first = _subvolume_index(low, geometry)
        second = _subvolume_index(high, geometry)
        result[face_id] = first if first == second else 64
    return result


@njit(cache=True)
def _assign_vertices(face_svi, incident, offsets, ripped):
    """Assign independent vertices using the same complete incident-face order."""
    result = np.full(len(ripped), -1, dtype=np.int32)
    for vertex in range(len(ripped)):
        start, end = offsets[vertex], offsets[vertex + 1]
        if ripped[vertex] or start == end:
            continue
        region = face_svi[incident[start]]
        for slot in range(start + 1, end):
            if face_svi[incident[slot]] != region:
                region = 64
                break
        result[vertex] = region
    return result


def subvolume_assignment(
    vertices: np.ndarray, faces: np.ndarray, proposed: np.ndarray, ripped: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed first-step geometry plus ordered face/vertex subvolumes."""
    xyz = np.asarray(vertices, dtype=np.float32)
    next_xyz = np.asarray(proposed, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    bounds = np.stack((np.minimum(xyz, next_xyz), np.maximum(xyz, next_xyz)), axis=-1)
    low = bounds[:, :, 0].min(axis=0)
    high = bounds[:, :, 1].max(axis=0)
    lengths = np.float32(np.float32(high - low) / np.float32(4))
    verge = np.float32(np.float32(lengths * np.float32(0.02)) + np.float32(0.01))
    geometry = np.concatenate((np.stack((low, high), axis=1).ravel(), lengths, verge)).astype(np.float32)
    face_svi = _assign_faces(bounds, triangles, geometry)
    flat = triangles.ravel()
    order = np.argsort(flat, kind="stable") // 3
    offsets = np.zeros(len(xyz) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(flat, minlength=len(xyz)))
    vertex_svi = _assign_vertices(face_svi, order, offsets, ripped)
    return geometry, face_svi, vertex_svi



@njit(cache=True)
def _moved_face_geometry(
    current: np.ndarray, triangles: np.ndarray, face_id: int,
    vertex: int, endpoint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    corners = triangles[face_id]
    moved = np.empty((3, 3), dtype=np.float32)
    center = np.empty(3, dtype=np.float64)
    low = np.empty(3, dtype=np.float32)
    high = np.empty(3, dtype=np.float32)
    for corner in range(3):
        for axis in range(3):
            moved[corner, axis] = endpoint[axis] if corners[corner] == vertex else current[corners[corner], axis]
    for axis in range(3):
        center[axis] = (float(moved[0, axis]) + float(moved[1, axis]) + float(moved[2, axis])) / 3.0
        low[axis] = min(moved[0, axis], moved[1, axis], moved[2, axis])
        high[axis] = max(moved[0, axis], moved[1, axis], moved[2, axis])
    radius = 0.0
    for corner in range(3):
        dx = float(moved[corner, 0]) - center[0]
        dy = float(moved[corner, 1]) - center[1]
        dz = float(moved[corner, 2]) - center[2]
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        radius = max(radius, distance)
    return moved, center, radius, low, high


@njit(cache=True)
def _candidate_collision(
    current: np.ndarray, triangles: np.ndarray, moved: np.ndarray,
    corners: np.ndarray, low: np.ndarray, high: np.ndarray,
    nearby: np.ndarray,
) -> int:
    for face_id in nearby:
        face = triangles[face_id]
        touching = False
        for first in range(3):
            for second in range(3):
                if face[first] == corners[second]:
                    touching = True
                    break
            if touching:
                break
        if touching:
            continue
        overlap = True
        for axis in range(3):
            c0 = current[face[0], axis]
            c1 = current[face[1], axis]
            c2 = current[face[2], axis]
            if max(c0, c1, c2) < low[axis] or min(c0, c1, c2) > high[axis]:
                overlap = False
                break
        if not overlap:
            continue
        candidate = np.empty((3, 3), dtype=np.float32)
        for corner in range(3):
            for axis in range(3):
                candidate[corner, axis] = current[face[corner], axis]
        if triangles_intersect(moved, candidate):
            return int(face_id) + 1
    return 0


@njit(cache=True)
def _incident_face_geometry(current, triangles, face_ids, vertex, endpoint):
    """同一顶点试步：逐面复用原几何内核；返回Fv个完整几何，没有并行更新。"""
    count = len(face_ids)
    moved = np.empty((count, 3, 3), dtype=np.float32)
    centers = np.empty((count, 3), dtype=np.float64)
    radii = np.empty(count, dtype=np.float64)
    lows = np.empty((count, 3), dtype=np.float32)
    highs = np.empty((count, 3), dtype=np.float32)
    for row in range(count):
        m, center, radius, low, high = _moved_face_geometry(
            current, triangles, face_ids[row], vertex, endpoint)
        moved[row], centers[row], radii[row], lows[row], highs[row] = m, center, radius, low, high
    return moved, centers, radii, lows, highs


@njit(cache=True)
def _incident_faces_collide(current, triangles, face_ids, moved, low, high, offsets, nearby):
    """同一顶点按原关联面/候选次序碰撞检查，首次命中后停止。"""
    for row in range(len(face_ids)):
        if _candidate_collision(current, triangles, moved[row], triangles[face_ids[row]],
                                low[row], high[row], nearby[offsets[row]:offsets[row+1]]):
            return True
    return False


def _vertex_collision_batch(current, triangles, face_ids, vertex, endpoint, tree, maximum_radius):
    """同顶点批量KD查询；current不修改，完整候选无上限，返回是否碰撞。

    current(N,3)float32及(F,3)int32面为surface RAS/mm；face_ids为原序关联面，
    vertex为顶点编号、endpoint为float32(3,)试步终点；tree和最大半径来自
    本次固定输入面。radius仍依次加maximum_radius与1mm，未缩小搜索范围。
    明确return_sorted=False保持单点查询次序；只适用于无拒绝试步MHT状态。
    不写坐标，无TF32/低精度开关；属于mris_place_surface内部步骤，无独立CLI。
    """
    if len(face_ids) == 0:
        return False
    moved, centers, radii, lows, highs = _incident_face_geometry(
        current, triangles, face_ids, vertex, endpoint)
    lists = tree.query_ball_point(centers, (radii + maximum_radius) + 1.0, return_sorted=False)
    counts = np.fromiter((len(row) for row in lists), dtype=np.int64, count=len(lists))
    offsets = np.empty(len(lists) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    nearby = np.fromiter(chain.from_iterable(lists), dtype=np.int32, count=int(offsets[-1]))
    return _incident_faces_collide(current, triangles, face_ids, moved, lows, highs, offsets, nearby)



@njit(cache=True)
def _project_close_neighbors(
    current: np.ndarray, vertex: int, neighbors: np.ndarray, neighbor_valid: np.ndarray,
    initial_offset: np.ndarray, geometry: np.ndarray, subvolume: int, min_neighbor_mm: np.float32,
) -> tuple[np.ndarray, bool]:
    """Apply mrisRemoveNeighborGradientComponent before the triangle collision test."""
    x, y, z = current[vertex, 0], current[vertex, 1], current[vertex, 2]
    odx, ody, odz = initial_offset[0], initial_offset[1], initial_offset[2]
    for slot in range(neighbors.shape[1]):
        if not neighbor_valid[vertex, slot]:
            continue
        other = neighbors[vertex, slot]
        dx = np.float32(current[other, 0] - x)
        dy = np.float32(current[other, 1] - y)
        dz = np.float32(current[other, 2] - z)
        squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
        dist = np.float32(np.sqrt(float(squared)))
        if dist == 0.0 or dist > min_neighbor_mm:
            continue
        dx, dy, dz = np.float32(dx / dist), np.float32(dy / dist), np.float32(dz / dist)
        dot = np.float32(np.float32(dx * odx + dy * ody) + dz * odz)
        if dot <= 0.0:
            continue
        next_dx = np.float32(odx - np.float32(dot * dx))
        next_dy = np.float32(ody - np.float32(dot * dy))
        next_dz = np.float32(odz - np.float32(dot * dz))
        if subvolume != 64:
            endpoint = np.empty(3, dtype=np.float32)
            endpoint[0] = np.float32(x + next_dx)
            endpoint[1] = np.float32(y + next_dy)
            endpoint[2] = np.float32(z + next_dz)
            if _subvolume_index(endpoint, geometry) != subvolume:
                return initial_offset.copy(), False
        odx, ody, odz = next_dx, next_dy, next_dz
    output = np.empty(3, dtype=np.float32)
    output[0], output[1], output[2] = odx, ody, odz
    return output, True

def _sample_mht_voxels(triangle: np.ndarray) -> set[tuple[int, int, int]]:
    """Sample a triangle into the pinned FreeSurfer 1 mm face-hash buckets."""
    a, b, c = np.asarray(triangle, dtype=np.float64)
    steps = max(1, int(np.ceil(2.0 * max(np.linalg.norm(a - b), np.linalg.norm(a - c)))))
    db, dc = (a - b) / steps, (a - c) / steps
    pb, pc = b.copy(), c.copy()
    result: set[tuple[int, int, int]] = set()

    def voxel(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(int(float(value) + 1000.0) for value in point)

    def path(old: tuple[int, int, int], new: tuple[int, int, int]) -> None:
        axes = [range(x, y + (1 if y >= x else -1), 1 if y >= x else -1)
                for x, y in zip(old, new)]
        for index, (x, y, z) in enumerate(
            (x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]
        ):
            if index:
                result.add((x, y, z))

    old_b = old_c = (0, 0, 0)
    for main in range(steps + 1):
        if main:
            pb += db
            pc += dc
        vb, vc = voxel(pb), voxel(pc)
        if main == 0:
            result.update((vb, vc))
        else:
            if old_b != vb:
                path(old_b, vb)
            if old_c != vc:
                path(old_c, vc)
        old_b, old_c = vb, vc
        if vb == vc:
            continue
        rung_steps = max(1, int(np.ceil(2.0 * np.linalg.norm(pc - pb))))
        rung_delta = (pc - pb) * (0.9995 / rung_steps)
        rung = pb.copy()
        old = vb
        for _ in range(rung_steps):
            rung += rung_delta
            new = voxel(rung)
            if old != new:
                path(old, new)
                old = new
    return result


def _retry_face_in_mht(
    moved: np.ndarray, other: int, faces: np.ndarray,
    original: np.ndarray, trial: np.ndarray, current: np.ndarray,
    order_rank: np.ndarray, current_rank: int,
) -> bool:
    """Replay the retained face-hash voxels for one candidate after a rejected trial."""
    corners = faces[other]
    stored = _sample_mht_voxels(trial[corners])
    face_xyz = original[corners].copy()
    prior = sorted((int(order_rank[vertex]), slot) for slot, vertex in enumerate(corners)
                   if order_rank[vertex] < current_rank)
    for _, slot in prior:
        stored.difference_update(_sample_mht_voxels(face_xyz))
        face_xyz[slot] = current[corners[slot]]
        stored.update(_sample_mht_voxels(face_xyz))
    return bool(stored.intersection(_sample_mht_voxels(moved)))


def asynchronous_first_step(
    vertices: np.ndarray, faces: np.ndarray, proposed: np.ndarray,
    ripped: np.ndarray, *, limit: int | None = None,
    regions: tuple[int, ...] | None = None,
    fast: bool = True,
    offsets: np.ndarray | None = None,
    accepted_offsets: np.ndarray | None = None,
    stale_mht_trial: np.ndarray | None = None,
    ordered_neighbors: tuple[np.ndarray, np.ndarray] | None = None,
    min_neighbor_mm: float = 0.01,
    candidate_backend: str = "tree",
    candidate_device: str | None = None,
    candidate_diagnostics: dict | None = None,
    candidate_grid_cells_per_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay sorted subvolumes with dynamic triangle collision tests.

    `proposed` contains the clipped pre-collision endpoint. Optional `offsets`
    retain its unrounded float32 displacement for close-neighbor projection.
    The initial cKDTree is widened by 1 mm to cover all 0.3 mm moves.
    `stale_mht_trial` reproduces face-bucket retention after a rejected trial.
    `candidate_backend="snapshot"` uses a trial-local conservative CSR and
    compiled ordered updates for the first fast trial; retained-MHT retries and
    slow diagnostics use the original tree path. Default remains tree.
    `candidate_backend="torch_snapshot"` uses the same compiled ordered updates
    with PyTorch conservative spatial candidates, guarded initial AABB motion
    bounds and static shared-vertex exclusions. `candidate_device` must name an
    indexed CUDA target (cpu only for regression); `candidate_diagnostics` can
    receive candidate counts, build/acceptance wall times and the effective
    backend. Coordinates and offsets are float32 surface RAS/mm; face indices
    and returned order address the unchanged mesh. Invalid input, exceeded
    motion bounds and CUDA/OOM errors propagate. No candidate limit truncation.
    """
    if candidate_backend not in ("tree", "snapshot", "torch_snapshot"):
        raise ValueError("candidate_backend must be tree, snapshot or torch_snapshot")
    if candidate_backend == "torch_snapshot" and candidate_device is None:
        raise ValueError("torch_snapshot requires an explicit candidate_device")
    if candidate_grid_cells_per_axis not in (2, 3):
        raise ValueError("candidate_grid_cells_per_axis must be 2 or 3")
    xyz = np.asarray(vertices, dtype=np.float32)
    if stale_mht_trial is not None and not fast:
        raise ValueError("retained MHT replay requires fast collision mode")
    if accepted_offsets is not None and offsets is None:
        raise ValueError("accepted_offsets requires unrounded offsets")
    triangles = np.asarray(faces, dtype=np.int32)
    next_xyz = np.asarray(proposed, dtype=np.float32)
    geometry, _, vertex_svi = subvolume_assignment(xyz, triangles, next_xyz, ripped)
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float32)
        if ordered_neighbors is None:
            from .place_surface_smoothing import _ordered_neighbors
            neighbor_indices, neighbor_valid, _ = _ordered_neighbors(triangles, len(xyz))
        else:
            neighbor_indices, neighbor_valid = ordered_neighbors
    order = np.concatenate([
        np.flatnonzero(vertex_svi == region)
        for region in (range(65) if regions is None else regions)
    ])
    if limit is not None:
        order = order[:limit]
    order_rank = np.full(len(xyz), len(order), dtype=np.int32)
    order_rank[order] = np.arange(len(order), dtype=np.int32)
    trial = np.asarray(stale_mht_trial, dtype=np.float32) if stale_mht_trial is not None else None
    current = xyz.copy()
    original_triangles = xyz[triangles]
    initial_centers = original_triangles.mean(axis=1, dtype=np.float64)
    initial_radii = np.linalg.norm(
        original_triangles.astype(np.float64) - initial_centers[:, None, :], axis=2,
    ).max(axis=1)
    tree = cKDTree(initial_centers)
    flat = triangles.ravel()
    incident = np.argsort(flat, kind="stable") // 3
    incident_offsets = np.zeros(len(xyz) + 1, dtype=np.int64)
    incident_offsets[1:] = np.cumsum(np.bincount(flat, minlength=len(xyz)))
    maximum_radius = float(initial_radii.max())
    if candidate_backend in ("snapshot", "torch_snapshot") and fast and trial is None:
        from .place_surface_snapshot import snapshot_ordered_step
        result = snapshot_ordered_step(xyz, triangles, next_xyz, order, incident, incident_offsets,
            neighbor_indices if offsets is not None else None,
            neighbor_valid if offsets is not None else None, offsets, accepted_offsets,
            geometry, vertex_svi, min_neighbor_mm, tree, initial_centers, initial_radii, maximum_radius,
            candidate_device=candidate_device if candidate_backend == "torch_snapshot" else None,
            candidate_diagnostics=candidate_diagnostics,
            candidate_grid_cells_per_axis=candidate_grid_cells_per_axis)
        return result, order
    if candidate_diagnostics is not None:
        candidate_diagnostics["effective_candidate_backend"]="tree_retained_mht" if trial is not None else "tree"
    for vertex in order:
        if np.array_equal(next_xyz[vertex], xyz[vertex]):
            continue
        endpoint = next_xyz[vertex]
        final_offset = offsets[vertex] if offsets is not None else None
        if offsets is not None:
            projected, valid = _project_close_neighbors(
                current, int(vertex), neighbor_indices, neighbor_valid,
                offsets[vertex], geometry, int(vertex_svi[vertex]),
                np.float32(min_neighbor_mm),
            )
            if not valid:
                continue
            endpoint = np.float32(xyz[vertex] + projected)
            final_offset = projected
        collision = False
        incident_faces = incident[incident_offsets[vertex]:incident_offsets[vertex + 1]]
        if fast and trial is None:
            collision = _vertex_collision_batch(
                current, triangles, incident_faces, int(vertex), endpoint, tree, maximum_radius)
        for face_id in (() if fast and trial is None else incident_faces):
            corners = triangles[face_id]
            if fast:
                moved, center, radius, low, high = _moved_face_geometry(
                    current, triangles, face_id, int(vertex), endpoint,
                )
            else:
                moved = current[corners].copy()
                moved[np.flatnonzero(corners == vertex)[0]] = endpoint
                center = moved.mean(axis=0, dtype=np.float64)
                radius = np.linalg.norm(moved.astype(np.float64) - center, axis=1).max()
                low, high = moved.min(axis=0), moved.max(axis=0)
            nearby = np.asarray(tree.query_ball_point(center, radius + maximum_radius + 1.0), dtype=np.int32)
            if fast:
                hit = _candidate_collision(current, triangles, moved, corners, low, high, nearby)
                if trial is None:
                    collision = hit > 0
                else:
                    while hit:
                        other = hit - 1
                        if _retry_face_in_mht(
                            moved, other, triangles, xyz, trial, current,
                            order_rank, int(order_rank[vertex]),
                        ):
                            collision = True
                            break
                        nearby = nearby[np.flatnonzero(nearby == other)[0] + 1:]
                        hit = _candidate_collision(current, triangles, moved, corners, low, high, nearby)
            else:
                if len(nearby):
                    neighbors = triangles[nearby]
                    touching = np.any(neighbors[:, :, None] == corners[None, None, :], axis=(1, 2))
                    nearby = nearby[~touching]
                if len(nearby):
                    points = current[triangles[nearby]]
                    overlap = np.all(points.max(axis=1) >= low, axis=1) & np.all(points.min(axis=1) <= high, axis=1)
                    for candidate in points[overlap]:
                        if triangles_intersect(moved, candidate):
                            collision = True
                            break
            if collision:
                break
        if collision:
            if accepted_offsets is not None:
                accepted_offsets[vertex] = 0.0
        else:
            current[vertex] = endpoint
            if accepted_offsets is not None:
                accepted_offsets[vertex] = final_offset
    return current, order
