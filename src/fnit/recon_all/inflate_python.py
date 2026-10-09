"""Python translation of the standard recon-all ``mris_inflate`` surface call.

This follows the FreeSurfer 8.2.0 default spring, distance, momentum, and
stopping equations. The implementation currently runs on CPU with NumPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numba import njit

from .inflate_center_python import center_vertices
from .smooth_surface_python import ordered_neighbors


def normalize(vectors: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.sum(vectors * vectors, axis=-1, dtype=np.float32))
    return np.divide(vectors, length[..., None], out=np.zeros_like(vectors),
                     where=length[..., None] > 0)


@njit(cache=True)
def _accumulate_corner_normals(ids: np.ndarray, corners: np.ndarray,
                               nvertices: int) -> np.ndarray:
    result = np.zeros((nvertices, 3), np.float32)
    for index in range(len(ids)):
        vertex = ids[index]
        for axis in range(3):
            result[vertex, axis] = np.float32(
                result[vertex, axis] + corners[index, axis])
    return result


def vertex_normals(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = xyz[faces]
    v0 = normalize(vertices - np.roll(vertices, 1, axis=1))
    v1 = normalize(np.roll(vertices, -1, axis=1) - vertices)
    corner = np.empty_like(v0)
    corner[..., 0] = -v1[..., 1] * v0[..., 2] + v0[..., 1] * v1[..., 2]
    corner[..., 1] = v1[..., 0] * v0[..., 2] - v0[..., 0] * v1[..., 2]
    corner[..., 2] = -v1[..., 0] * v0[..., 1] + v0[..., 0] * v1[..., 1]
    accumulated = _accumulate_corner_normals(
        faces.reshape(-1).astype(np.int32, copy=False),
        normalize(corner).reshape(-1, 3), len(xyz))
    return normalize(accumulated)


def matrix(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    degree = np.fromiter((len(row) for row in rows), np.int32, count=len(rows))
    indices = np.zeros((len(rows), int(degree.max())), np.int32)
    for vertex, row in enumerate(rows):
        indices[vertex, :len(row)] = row
    return indices, degree


def two_ring_neighbors(one_ring: list[list[int]]) -> list[list[int]]:
    rows = []
    for center, neighbors in enumerate(one_ring):
        row = list(neighbors)
        seen = set(row)
        seen.add(center)
        for neighbor in neighbors:
            for candidate in one_ring[neighbor]:
                if candidate not in seen:
                    seen.add(candidate)
                    row.append(candidate)
        rows.append(row)
    return rows


def face_area_total(xyz: np.ndarray, faces: np.ndarray) -> np.float32:
    edge_a = xyz[faces[:, 1]] - xyz[faces[:, 0]]
    edge_b = xyz[faces[:, 2]] - xyz[faces[:, 0]]
    cross = np.cross(edge_a, edge_b)
    squared = cross[:, 0] * cross[:, 0]
    squared += cross[:, 1] * cross[:, 1]
    squared += cross[:, 2] * cross[:, 2]
    area = np.float32(0.5) * np.sqrt(squared)
    return np.float32(np.sum(area.astype(np.float64)))


def bounding_center(xyz: np.ndarray) -> np.ndarray:
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    return np.float32(0.5) * (low.astype(np.float64) + high.astype(np.float64)).astype(np.float32)


def postprocess(xyz: np.ndarray, faces: np.ndarray, original_area: np.float32) -> np.ndarray:
    """Match main's ``MRIScenter`` then ``MRISscaleBrainArea`` coordinate math."""
    current_area = face_area_total(xyz, faces)
    centered = center_vertices(xyz)
    scale = np.sqrt(np.float32(original_area / current_area))
    if scale == 1:
        return centered
    center = bounding_center(centered)
    return (centered - center) * scale + center


def distances(xyz: np.ndarray, indices: np.ndarray, degree: np.ndarray) -> np.ndarray:
    result = np.zeros(indices.shape, np.float32)
    for slot in range(indices.shape[1]):
        valid = degree > slot
        delta = xyz[valid] - xyz[indices[valid, slot]]
        squared = delta[:, 0] * delta[:, 0]
        squared += delta[:, 1] * delta[:, 1]
        squared += delta[:, 2] * delta[:, 2]
        result[valid, slot] = np.sqrt(squared)
    return result


def distance_gradient(xyz: np.ndarray, normals: np.ndarray, indices: np.ndarray,
                      degree: np.ndarray, original_dist: np.ndarray,
                      current_dist: np.ndarray, original_area: np.float32,
                      current_area: np.float32, weight: np.float32,
                      average_neighbors: np.float32) -> np.ndarray:
    gradient = np.zeros_like(xyz)
    if weight == 0:
        return gradient
    scale = np.sqrt(np.float32(original_area / current_area))
    for slot in range(indices.shape[1]):
        valid = degree > slot
        delta = current_dist[valid, slot] - original_dist[valid, slot] / scale
        direction = xyz[indices[valid, slot]] - xyz[valid]
        squared = direction[:, 0] * direction[:, 0]
        squared += direction[:, 1] * direction[:, 1]
        squared += direction[:, 2] * direction[:, 2]
        length = np.sqrt(squared)
        reciprocal = np.divide(np.float32(1), length, out=np.ones_like(length),
                               where=length > 0)
        gradient[valid] += direction * reciprocal[:, None] * delta[:, None]
    gradient *= np.float32(1 / average_neighbors)
    normal_component = gradient[:, 0] * normals[:, 0]
    normal_component += gradient[:, 1] * normals[:, 1]
    normal_component += gradient[:, 2] * normals[:, 2]
    gradient -= normal_component[:, None] * normals
    return np.float32(weight * gradient)


@njit(cache=True)
def _average_gradient_numba(gradient: np.ndarray, indices: np.ndarray,
                            degree: np.ndarray, reciprocal: np.ndarray,
                            count: int) -> np.ndarray:
    current = gradient.copy()
    for _ in range(count):
        updated = current.copy()
        for vertex in range(len(current)):
            for slot in range(degree[vertex]):
                neighbor = indices[vertex, slot]
                for axis in range(3):
                    updated[vertex, axis] = np.float32(
                        updated[vertex, axis] + current[neighbor, axis])
        for vertex in range(len(current)):
            for axis in range(3):
                updated[vertex, axis] = np.float32(
                    updated[vertex, axis] * reciprocal[vertex])
        current = updated
    return current


def average_gradient(gradient: np.ndarray, one_indices: np.ndarray,
                     one_degree: np.ndarray, count: int) -> np.ndarray:
    reciprocal = np.float32(1) / (one_degree + 1).astype(np.float32)
    return _average_gradient_numba(
        np.asarray(gradient, np.float32), one_indices, one_degree,
        reciprocal, count)


@njit(cache=True)
def neighbor_spring(xyz: np.ndarray, indices: np.ndarray,
                    degree: np.ndarray) -> np.ndarray:
    spring = np.zeros_like(xyz)
    for slot in range(indices.shape[1]):
        for vertex in range(len(xyz)):
            if degree[vertex] <= slot:
                continue
            neighbor = indices[vertex, slot]
            for axis in range(3):
                spring[vertex, axis] = np.float32(
                    spring[vertex, axis] +
                    np.float32(xyz[neighbor, axis] - xyz[vertex, axis]))
    return spring


def add_spring(gradient: np.ndarray, xyz: np.ndarray, normals: np.ndarray,
               one_indices: np.ndarray, one_degree: np.ndarray,
               original_area: np.float32, current_area: np.float32) -> np.ndarray:
    spring = neighbor_spring(xyz, one_indices, one_degree)
    dist_scale = np.sqrt(np.float32(original_area / current_area))
    spring *= (dist_scale / one_degree.astype(np.float32))[:, None]
    dot = spring[:, 0] * normals[:, 0]
    dot += spring[:, 1] * normals[:, 1]
    dot += spring[:, 2] * normals[:, 2]
    dot_average = np.float32(np.sum(dot.astype(np.float64)) / len(xyz))
    gradient += spring
    gradient -= dot_average * normals
    return gradient


@njit(cache=True)
def rms_tangent_height(xyz: np.ndarray, normals: np.ndarray,
                       indices: np.ndarray, degree: np.ndarray) -> float:
    """FreeSurfer's ring-two tangent-plane RMS used for early stopping."""
    squared_sum = 0.0
    count = 0
    for slot in range(indices.shape[1]):
        for vertex in range(len(xyz)):
            if degree[vertex] <= slot:
                continue
            neighbor = indices[vertex, slot]
            dx = float(xyz[neighbor, 0] - xyz[vertex, 0])
            dy = float(xyz[neighbor, 1] - xyz[vertex, 1])
            dz = float(xyz[neighbor, 2] - xyz[vertex, 2])
            length_squared = dx * dx + dy * dy + dz * dz
            if length_squared <= 1e-12:
                continue
            dot = (dx * float(normals[vertex, 0]) +
                   dy * float(normals[vertex, 1]) +
                   dz * float(normals[vertex, 2]))
            squared_sum += dot * dot / length_squared
            count += 1
    return np.sqrt(squared_sum / count)


def inflate_updates(xyz: np.ndarray, faces: np.ndarray, niterations: int = 10,
                    first_averages: int = 16, snapshot=None,
                    rms_target: float | None = 0.015, *,
                    sulc: np.ndarray | None = None, diagnostics: dict | None = None) -> np.ndarray:
    """执行原有完整默认积分，可显式累积 sulc 和记录子段墙钟。

    xyz 为 FP32(N,3) surface RAS/mm，faces 为有序整数(F,3)。niterations
    默认每档10步，first_averages默认16，依次16/8/4/2/1/0；rms_target
    默认0.015。snapshot可接收(step,xyz)。返回新FP32坐标，不改xyz。
    sulc 若给定须为可写FP32(N,)零数组，逐步原地累积更新前法向投影；
    此处不去均值，文件完整接口最后处理。diagnostics若给定接收各子段
    秒数/步数/RMS；计时不改算法。对应固定 mris_inflate 的内部积分。
    """
    import time
    xyz = np.asarray(xyz, np.float32).copy()
    if sulc is not None and (sulc.shape != (len(xyz),) or sulc.dtype != np.float32
                             or not sulc.flags.writeable or np.any(sulc != 0)):
        raise ValueError("sulc must be a writable zero float32 vector (N,)")
    timings = {}
    def timed(name, function, *args):
        if diagnostics is None:
            return function(*args)
        begin = time.perf_counter()
        result = function(*args)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - begin
        return result
    setup_begin = time.perf_counter()
    one = ordered_neighbors(faces, len(xyz))
    one_indices, one_degree = matrix(one)
    two_indices, two_degree = matrix(two_ring_neighbors(one))
    average_neighbors = np.float32(np.sum(two_degree, dtype=np.int64) / len(xyz))
    original_area = face_area_total(xyz, faces)
    original_dist = distances(xyz, two_indices, two_degree)
    previous = np.zeros_like(xyz)
    step = 0
    normals = vertex_normals(xyz, faces)
    area = original_area
    current_dist = original_dist.copy()
    reached = False
    rms = None
    if diagnostics is not None:
        timings["topology_initial_metric_normals_including_jit"] = time.perf_counter() - setup_begin
    for averages in (16, 8, 4, 2, 1, 0):
        if averages > first_averages:
            continue
        weight = np.float32(np.float32(0.1) * np.sqrt(averages))
        for _ in range(niterations):
            gradient = timed("distance_gradient", distance_gradient, xyz, normals, two_indices, two_degree,
                                         original_dist, current_dist, original_area,
                                         area, weight, average_neighbors)
            gradient = timed("ordered_averaging", average_gradient, gradient, one_indices, one_degree, averages)
            gradient = timed("normalized_spring", add_spring, gradient, xyz, normals, one_indices, one_degree,
                                  original_area, area)
            update_begin = time.perf_counter()
            previous = (gradient.astype(np.float64) * float(np.float32(0.9)) +
                        (np.float32(0.9) * previous).astype(np.float64)).astype(np.float32)
            xyz += previous
            if sulc is not None:
                projection = previous[:, 0] * normals[:, 0]
                projection += previous[:, 1] * normals[:, 1]
                projection += previous[:, 2] * normals[:, 2]
                sulc += projection
            if diagnostics is not None:
                timings["momentum_and_sulc"] = timings.get("momentum_and_sulc", 0.0) + time.perf_counter() - update_begin
            normals = timed("normals", vertex_normals, xyz, faces)
            area = timed("face_area", face_area_total, xyz, faces)
            current_dist = timed("distances", distances, xyz, two_indices, two_degree)
            step += 1
            if snapshot is not None:
                snapshot(step, xyz)
            rms = (timed("rms_tangent_height", rms_tangent_height, xyz, normals, two_indices, two_degree)
                   if rms_target is not None else None)
            if rms_target is not None and rms < rms_target:
                reached = True
                break
        if reached:
            break
    if diagnostics is not None:
        diagnostics.update({"steps": step, "stopped_by_rms": reached,
                            "final_rms": rms, "timings_seconds": timings})
    return xyz


def inflate_surface(input_path: str | Path, output_path: str | Path) -> None:
    """Write the inflated triangle surface for the standard recon-all call."""
    raw = Path(input_path).read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError("expected FreeSurfer triangle surface")
    start = raw.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(raw[start:start + 4], "big")
    nfaces = int.from_bytes(raw[start + 4:start + 8], "big")
    xyz_start = start + 8
    xyz_end = xyz_start + 12 * nvertices
    faces_end = xyz_end + 12 * nfaces
    if len(raw) < faces_end:
        raise ValueError("truncated triangle surface")
    xyz = np.frombuffer(raw[xyz_start:xyz_end], dtype=">f4").astype(np.float32).reshape(-1, 3)
    faces = np.frombuffer(raw[xyz_end:faces_end], dtype=">i4").astype(np.int32).reshape(-1, 3)
    updated = inflate_updates(xyz, faces)
    final = postprocess(updated, faces, face_area_total(xyz, faces))
    with Path(output_path).open("wb") as stream:
        stream.write(b"\xff\xff\xfecreated by fnit\n\n")
        stream.write(raw[start:xyz_start])
        stream.write(np.asarray(final, dtype=">f4").tobytes())
        stream.write(raw[xyz_end:])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    inflate_surface(args.input, args.output)


if __name__ == "__main__":
    main()
