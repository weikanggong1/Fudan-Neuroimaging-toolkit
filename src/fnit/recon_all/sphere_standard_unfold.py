"""First conventional ``MRISunfold`` epoch force on a fixed sphere checkpoint.

This is a checkpoint-level translation of the distance and negative-face-area
terms for the default nonquick sphere call. It does not implement fold repair,
line minimization, later epochs, or a complete ``mris_sphere`` stage.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .place_surface_normals import initial_vertex_normals


@njit(cache=True, fastmath=False)
def _sphere_radius_units(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.empty(len(xyz), np.float32)
    unit = np.empty_like(xyz)
    for vertex in range(len(xyz)):
        x, y, z = xyz[vertex]
        length = np.float32(math.sqrt(np.float64(x) * np.float64(x)
                                      + np.float64(y) * np.float64(y)
                                      + np.float64(z) * np.float64(z)))
        radius[vertex] = length
        inverse = np.float32(np.float32(1) / length) if length > 0 else np.float32(0)
        unit[vertex, 0] = np.float32(x * inverse)
        unit[vertex, 1] = np.float32(y * inverse)
        unit[vertex, 2] = np.float32(z * inverse)
    return radius, unit


@njit(cache=True, inline="always", fastmath=False)
def _spherical_distance(xyz: np.ndarray, radius: np.ndarray,
                        unit: np.ndarray, first: int, second: int) -> np.float32:
    normalizer = radius[second]
    dot = np.float32(np.float64(unit[first, 0]) * np.float64(xyz[second, 0])
                     + np.float64(unit[first, 1]) * np.float64(xyz[second, 1])
                     + np.float64(unit[first, 2]) * np.float64(xyz[second, 2]))
    if abs(float(dot)) > float(normalizer):
        normalizer = np.float32(abs(float(dot)))
    # Native dot / norm has float operands; assignment to double occurs later.
    cosine = np.float64(np.float32(dot / normalizer))
    if cosine < float(np.float32(0.99)):
        angle = np.float32(math.acos(cosine))
    else:
        angle = np.float32(math.sqrt(2.0 * (1.0 - cosine)))
    return np.float32(angle * radius[first])


@njit(parallel=True, cache=True, fastmath=False)
def _distance_force(xyz: np.ndarray, normals: np.ndarray,
                    offsets: np.ndarray, neighbors: np.ndarray,
                    original_distances: np.ndarray, scale: np.float32,
                    weight: np.float32,
                    average_neighbors: np.float32 = np.float32(0)) -> np.ndarray:
    gradient = np.zeros_like(xyz)
    radius, unit = _sphere_radius_units(xyz)
    if average_neighbors <= 0:
        average_neighbors = np.float32(len(neighbors) / len(xyz))
    norm = np.float32(np.float32(1) / average_neighbors)
    for vertex in prange(len(xyz)):
        dx = np.float32(0)
        dy = np.float32(0)
        dz = np.float32(0)
        for p in range(offsets[vertex], offsets[vertex + 1]):
            other = neighbors[p]
            x = np.float32(xyz[other, 0] - xyz[vertex, 0])
            y = np.float32(xyz[other, 1] - xyz[vertex, 1])
            z = np.float32(xyz[other, 2] - xyz[vertex, 2])
            squared = np.float32(np.float32(x * x + y * y) + z * z)
            length = np.float32(np.sqrt(squared))
            if length < 1e-6:
                continue
            current_distance = _spherical_distance(xyz, radius, unit, vertex, other)
            delta = np.float32(current_distance
                               - np.float32(original_distances[p] / scale))
            inverse = np.float32(np.float32(1) / length)
            dx = np.float32(dx + np.float32(np.float32(x * inverse) * delta))
            dy = np.float32(dy + np.float32(np.float32(y * inverse) * delta))
            dz = np.float32(dz + np.float32(np.float32(z * inverse) * delta))
        dx = np.float32(dx * norm)
        dy = np.float32(dy * norm)
        dz = np.float32(dz * norm)
        nx, ny, nz = normals[vertex]
        component = np.float32(np.float32(dx * nx + dy * ny) + dz * nz)
        dx = np.float32(dx - np.float32(component * nx))
        dy = np.float32(dy - np.float32(component * ny))
        dz = np.float32(dz - np.float32(component * nz))
        gradient[vertex, 0] = np.float32(dx * weight)
        gradient[vertex, 1] = np.float32(dy * weight)
        gradient[vertex, 2] = np.float32(dz * weight)
    return gradient


@njit(parallel=True, cache=True, fastmath=False)
def _face_geometry(xyz: np.ndarray, faces: np.ndarray,
                   ) -> tuple[np.ndarray, np.ndarray]:
    area = np.empty(len(faces), np.float32)
    normals = np.empty((len(faces), 3), np.float32)
    for fno in prange(len(faces)):
        a, b, c = faces[fno]
        u0 = np.float32(xyz[b, 0] - xyz[a, 0])
        u1 = np.float32(xyz[b, 1] - xyz[a, 1])
        u2 = np.float32(xyz[b, 2] - xyz[a, 2])
        v0 = np.float32(xyz[c, 0] - xyz[a, 0])
        v1 = np.float32(xyz[c, 1] - xyz[a, 1])
        v2 = np.float32(xyz[c, 2] - xyz[a, 2])
        nx = np.float32(u1 * v2 - u2 * v1)
        ny = np.float32(u2 * v0 - u0 * v2)
        nz = np.float32(u0 * v1 - u1 * v0)
        length = np.float32(np.sqrt(np.float32(
            np.float32(nx * nx + ny * ny) + nz * nz)))
        area[fno] = np.float32(length * np.float32(0.5))
        # V3_NORMALIZE keeps a sub-FLT_EPSILON cross product unscaled.
        if length >= np.float32(1.1920928955078125e-7):
            inverse = np.float32(np.float32(1) / length)
            nx = np.float32(nx * inverse)
            ny = np.float32(ny * inverse)
            nz = np.float32(nz * inverse)
        if np.float32(
            np.float32(np.float32(xyz[a, 0] + xyz[b, 0] + xyz[c, 0]) * nx
                       + np.float32(xyz[a, 1] + xyz[b, 1] + xyz[c, 1]) * ny)
            + np.float32(xyz[a, 2] + xyz[b, 2] + xyz[c, 2]) * nz
        ) < 0:
            area[fno] = -area[fno]
            nx, ny, nz = -nx, -ny, -nz
        normals[fno, 0] = nx
        normals[fno, 1] = ny
        normals[fno, 2] = nz
    return area, normals


@njit(cache=True, fastmath=False)
def _area_force(xyz: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                area: np.ndarray, original_area: np.ndarray,
                scale: np.float32, initial_gradient: np.ndarray) -> np.ndarray:
    gradient = initial_gradient.copy()
    for fno in range(len(faces)):
        if area[fno] > 0:
            continue
        a, b, c = faces[fno]
        delta = np.float32(np.float32(scale * area[fno]) - original_area[fno])
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
        gradient[b, 0] = np.float32(gradient[b, 0] - np.float32(bnx * delta))
        gradient[b, 1] = np.float32(gradient[b, 1] - np.float32(bny * delta))
        gradient[b, 2] = np.float32(gradient[b, 2] - np.float32(bnz * delta))
        gradient[c, 0] = np.float32(gradient[c, 0] + np.float32(anx * delta))
        gradient[c, 1] = np.float32(gradient[c, 1] + np.float32(any_ * delta))
        gradient[c, 2] = np.float32(gradient[c, 2] + np.float32(anz * delta))
    return gradient


def first_epoch_gradient(vertices: np.ndarray, faces: np.ndarray,
                         smoothwm_vertices: np.ndarray,
                         offsets: np.ndarray, neighbors: np.ndarray,
                         original_distances: np.ndarray,
                         distance_weight: float = 0.1,
                         *, normal_topology=None, original_metric=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
    # MRIScomputeMetricProperties overrides total_area for MRIS_SPHERE.
    positive_face_area = np.float32(np.sum(area[area >= 0], dtype=np.float64))
    current_total = np.float32(math.pi * 100.0 * 100.0 * 4.0)
    distance_scale = np.float32(np.sqrt(np.float32(original_total / current_total)))
    area_scale = np.float32(original_total / current_total)
    vertex_normals = initial_vertex_normals(xyz, faces, topology=normal_topology)
    distance = _distance_force(xyz, vertex_normals, offsets, neighbors,
                               original_distances, distance_scale, np.float32(distance_weight))
    negative_area = _area_force(xyz, faces, face_normals, area,
                                original_face_area, area_scale,
                                np.zeros_like(distance))
    # Native adds face forces onto the existing distance gradient in face order.
    total = _area_force(xyz, faces, face_normals, area,
                        original_face_area, area_scale, distance)
    report = {"negative_faces": int(np.count_nonzero(area < 0)),
              "original_area_mm2": float(original_total),
              "current_positive_face_area_mm2": float(positive_face_area),
              "sphere_metric_total_area_mm2": float(current_total),
              "distance_scale": float(distance_scale),
              "area_scale": float(area_scale),
              "distance_weight": float(distance_weight)}
    return distance, negative_area, total, report
