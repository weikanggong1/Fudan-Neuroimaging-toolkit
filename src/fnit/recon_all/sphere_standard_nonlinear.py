"""First nonlinear-area epoch of FreeSurfer's conventional sphere unfold."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .place_surface_normals import initial_vertex_normals
from .smooth_surface_python import ordered_neighbors
from .sphere_standard_line_search import _distance_sse, first_epoch_line_search
from .sphere_standard_unfold import _distance_force, _face_geometry


def one_ring_metric(faces: np.ndarray, nvertices: int,
                    full_offsets: np.ndarray, full_neighbors: np.ndarray,
                    full_distances: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    """Select native fold-repair rows while retaining the old avg_nbrs."""
    adjacent = ordered_neighbors(np.asarray(faces, np.int32), nvertices)
    counts = np.asarray([len(row) for row in adjacent], np.int64)
    offsets = np.empty(nvertices + 1, np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    neighbors = np.empty(offsets[-1], np.int32)
    distances = np.empty(offsets[-1], np.float32)
    for vertex, row in enumerate(adjacent):
        length = len(row)
        begin = full_offsets[vertex]
        if not np.array_equal(full_neighbors[begin:begin + length], row):
            raise ValueError(f"full metric row {vertex} does not begin with one ring")
        target = offsets[vertex]
        neighbors[target:target + length] = row
        distances[target:target + length] = full_distances[begin:begin + length]
    old_average_neighbors = np.float32(len(full_neighbors) / nvertices)
    return offsets, neighbors, distances, old_average_neighbors


@njit
def _nonlinear_area_force(xyz: np.ndarray, faces: np.ndarray,
                          normals: np.ndarray, area: np.ndarray,
                          original_area: np.ndarray, area_scale: float,
                          distance_gradient: np.ndarray,
                          weight: float, k: float) -> np.ndarray:
    gradient = distance_gradient.copy()
    max_ratio = 400.0 / k
    for fno in range(len(faces)):
        a, b, c = faces[fno]
        ratio = min(max(area_scale * np.float64(area[fno]), -max_ratio), max_ratio)
        scale = weight / (1.0 + math.exp(k * ratio))
        delta = scale * (area_scale * np.float64(area[fno])
                         - np.float64(original_area[fno]))
        nx, ny, nz = normals[fno]
        ux = np.float32(xyz[b, 0] - xyz[a, 0])
        uy = np.float32(xyz[b, 1] - xyz[a, 1])
        uz = np.float32(xyz[b, 2] - xyz[a, 2])
        vx = np.float32(xyz[c, 0] - xyz[a, 0])
        vy = np.float32(xyz[c, 1] - xyz[a, 1])
        vz = np.float32(xyz[c, 2] - xyz[a, 2])
        anx = np.float32(uy * nz - uz * ny)
        any_ = np.float32(uz * nx - ux * nz)
        anz = np.float32(ux * ny - uy * nx)
        bnx = np.float32(vy * nz - vz * ny)
        bny = np.float32(vz * nx - vx * nz)
        bnz = np.float32(vx * ny - vy * nx)
        gradient[a, 0] = np.float32(gradient[a, 0]
                                    + np.float32(np.float32(-anx + bnx) * delta))
        gradient[a, 1] = np.float32(gradient[a, 1]
                                    + np.float32(np.float32(-any_ + bny) * delta))
        gradient[a, 2] = np.float32(gradient[a, 2]
                                    + np.float32(np.float32(-anz + bnz) * delta))
        gradient[b, 0] = np.float32(gradient[b, 0] + np.float32(bnx * -delta))
        gradient[b, 1] = np.float32(gradient[b, 1] + np.float32(bny * -delta))
        gradient[b, 2] = np.float32(gradient[b, 2] + np.float32(bnz * -delta))
        gradient[c, 0] = np.float32(gradient[c, 0] + np.float32(anx * delta))
        gradient[c, 1] = np.float32(gradient[c, 1] + np.float32(any_ * delta))
        gradient[c, 2] = np.float32(gradient[c, 2] + np.float32(anz * delta))
    return gradient


@njit
def _nonlinear_area_sse(area: np.ndarray, area_scale: float, k: float) -> float:
    total = 0.0
    max_ratio = 400.0 / k
    for fno in range(len(area)):
        ratio = min(max(-max_ratio, area_scale * np.float64(area[fno])), max_ratio)
        total += math.log(1.0 + math.exp(k * ratio)) / k - ratio
    return total


def nonlinear_epoch_gradient(vertices: np.ndarray, faces: np.ndarray,
                             smoothwm_vertices: np.ndarray,
                             offsets: np.ndarray, neighbors: np.ndarray,
                             original_distances: np.ndarray,
                             old_average_neighbors: np.float32,
                             distance_weight: float = 1e-6,
                             nonlinear_weight: float = 1.0,
                             k: float = 10.0, *,
                             normal_topology=None, original_metric=None) -> tuple[np.ndarray, dict]:
    """按原顺序计算球面梯度，坐标为surface RAS/mm。

    normal_topology=None可复用本次同面的FaceNormalTopology；original_metric=None
    可传同一smoothwm的(float32(F,)绝对面积,float32总面积mm²)，两者只读。
    默认None沿原即时计算；其余输入/权重和返回梯度/报告保持原定义。
    缓存不能跨拓扑或metric版本使用，法向和现版面积每次重算。
    """
    xyz = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32)
    area, face_normals = _face_geometry(xyz, faces)
    if original_metric is None:
        original_face_area, _ = _face_geometry(
            np.asarray(smoothwm_vertices, np.float32), faces)
        original_face_area = np.abs(original_face_area)
        original_total = np.float32(np.sum(original_face_area, dtype=np.float64))
    else:
        original_face_area, original_total = original_metric
    current_total = np.float32(4 * math.pi * 100 * 100)
    area_scale = np.float32(original_total / current_total)
    distance_scale = np.float32(np.sqrt(np.float32(original_total / current_total)))
    normals = initial_vertex_normals(xyz, faces, topology=normal_topology)
    distance = _distance_force(xyz, normals, offsets, neighbors,
                               original_distances, distance_scale,
                               np.float32(distance_weight),
                               old_average_neighbors)
    total = _nonlinear_area_force(xyz, faces, face_normals, area,
                                  original_face_area, float(area_scale), distance,
                                  nonlinear_weight, k)
    return total, {"negative_faces": int(np.count_nonzero(area < 0)),
                   "original_area_mm2": float(original_total),
                   "area_scale": float(area_scale),
                   "distance_scale": float(distance_scale),
                   "old_average_neighbors": float(old_average_neighbors),
                   "distance_weight": distance_weight,
                   "nonlinear_weight": nonlinear_weight, "k": k}


def nonlinear_epoch_sse(vertices: np.ndarray, faces: np.ndarray,
                        offsets: np.ndarray, neighbors: np.ndarray,
                        original_distances: np.ndarray,
                        original_total_area: np.float32,
                        distance_weight: float = 1e-6,
                        nonlinear_weight: float = 1.0,
                        k: float = 10.0) -> dict:
    """Evaluate native logistic-area plus distance objective at one trial."""
    xyz = np.asarray(vertices, np.float32)
    area, _ = _face_geometry(xyz, np.asarray(faces, np.int32))
    negative_area = np.float32(-np.sum(area[area < 0], dtype=np.float64))
    total_area = np.float32(4 * math.pi * 100 * 100)
    denominator = (np.float32(total_area - negative_area)
                   if negative_area < total_area else total_area)
    distance_scale = math.sqrt(float(np.float32(original_total_area / denominator)))
    area_scale = float(np.float32(original_total_area / total_area))
    distance = _distance_sse(xyz, offsets, neighbors, original_distances,
                             distance_scale)
    nonlinear = _nonlinear_area_sse(area, area_scale, k)
    # Native INTEGRATION_PARMS stores both weights as float32.
    distance_weight = float(np.float32(distance_weight))
    nonlinear_weight = float(np.float32(nonlinear_weight))
    return {"total": distance_weight * distance + nonlinear_weight * nonlinear,
            "weighted_distance": distance_weight * distance,
            "weighted_nonlinear_area": nonlinear_weight * nonlinear,
            "negative_faces": int(np.count_nonzero(area < 0)),
            "distance_scale": distance_scale, "area_scale": area_scale}


def nonlinear_epoch_line_search(vertices: np.ndarray, gradient: np.ndarray,
                                faces: np.ndarray, offsets: np.ndarray,
                                neighbors: np.ndarray,
                                original_distances: np.ndarray,
                                original_face_area: np.ndarray,
                                original_total_area: np.float32,
                                distance_weight: float = 1e-6,
                                nonlinear_weight: float = 1.0,
                                k: float = 10.0) -> dict:
    def objective(xyz: np.ndarray) -> dict:
        return nonlinear_epoch_sse(xyz, faces, offsets, neighbors,
                                   original_distances, original_total_area,
                                   distance_weight, nonlinear_weight, k)

    return first_epoch_line_search(vertices, gradient, faces, offsets, neighbors,
                                   original_distances, original_face_area,
                                   original_total_area, distance_weight,
                                   objective=objective)
