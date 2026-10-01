"""First conventional-sphere line-search SSE on a fixed MRISunfold checkpoint.

This only handles the first epoch's distance and negative-face-area weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numba import njit, prange

from .sphere_standard_unfold import _face_geometry, _sphere_radius_units, _spherical_distance


@njit(parallel=True, cache=True)
def _distance_sse_rows(xyz, radius, unit, offsets, neighbors, original_distances, scale):
    values = np.empty(len(xyz), np.float64)
    for vertex in prange(len(xyz)):
        vertex_sse = 0.0
        for p in range(offsets[vertex], offsets[vertex + 1]):
            current = _spherical_distance(xyz, radius, unit, vertex, neighbors[p])
            delta = scale * np.float64(current) - np.float64(original_distances[p])
            vertex_sse += delta * delta
        values[vertex] = vertex_sse
    return values


@njit(cache=True)
def _distance_sse(xyz: np.ndarray, offsets: np.ndarray, neighbors: np.ndarray,
                  original_distances: np.ndarray, scale: float) -> float:
    """按原邻居序计算每顶点float64 SSE，再按顶点序串行合计。

    xyz是float32(N,3)surface RAS/mm，CSR偏移/邻居和原距离同索引，
    scale为原目标函数比例；返回原定义float64总量。独立行并行，不改变
    求和顺序、float32球面距离或线搜索规则，无fastmath。
    属于mris_sphere内部步骤，没有独立CLI。
    """
    radius, unit = _sphere_radius_units(xyz)
    values = _distance_sse_rows(xyz, radius, unit, offsets, neighbors,
                                original_distances, scale)
    total = 0.0
    for vertex in range(len(xyz)):
        total += values[vertex]
    return total


@njit
def _negative_area_sse(area: np.ndarray, original_area: np.ndarray,
                       scale: float) -> float:
    total = 0.0
    for face in range(len(area)):
        if area[face] < 0:
            delta = scale * np.float64(area[face]) - np.float64(original_area[face])
            total += delta * delta
    return total


def first_epoch_sse(vertices: np.ndarray, faces: np.ndarray,
                    offsets: np.ndarray, neighbors: np.ndarray,
                    original_distances: np.ndarray,
                    original_face_area: np.ndarray,
                    original_total_area: np.float32,
                    distance_weight: float = 0.1) -> dict:
    """Evaluate one fixed sphere epoch distance and negative-area SSE."""
    xyz = np.asarray(vertices, np.float32)
    area, _ = _face_geometry(xyz, np.asarray(faces, np.int32))
    negative_area = np.float32(-np.sum(area[area < 0], dtype=np.float64))
    total_area = np.float32(4 * math.pi * 100 * 100)
    denominator = (np.float32(total_area - negative_area)
                   if negative_area < total_area else total_area)
    distance_scale = math.sqrt(float(np.float32(original_total_area / denominator)))
    area_scale = float(np.float32(original_total_area / total_area))
    distance = _distance_sse(xyz, offsets, neighbors, original_distances, distance_scale)
    negative = _negative_area_sse(area, original_face_area, area_scale)
    # INTEGRATION_PARMS stores l_dist as float before the double SSE product.
    weight = float(np.float32(distance_weight))
    return {"total": weight * distance + negative,
            "weighted_distance": weight * distance,
            "negative_area": negative,
            "negative_faces": int(np.count_nonzero(area < 0)),
            "metric_negative_area_mm2": float(negative_area),
            "distance_scale": distance_scale,
            "area_scale": area_scale}


def _matrix_multiply_float32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.zeros((left.shape[0], right.shape[1]), np.float32)
    for row in range(left.shape[0]):
        for col in range(right.shape[1]):
            value = np.float32(0)
            for k in range(left.shape[1]):
                value = np.float32(value + np.float32(left[row, k] * right[k, col]))
            output[row, col] = value
    return output


def _quadratic_fit_float32(times: list[float], scores: list[float]) -> tuple[float, float, float]:
    x = np.asarray([[dt * dt, 2 * dt, 1] for dt in times], np.float32)
    y = np.asarray(scores, np.float32).reshape(3, 1)
    normal = _matrix_multiply_float32(x.T, x)
    right = _matrix_multiply_float32(x.T, y)
    # VXL vnl_inverse<float,3,3> uses a float32 determinant and adjugate.
    a, b, c = normal[0]
    d, e, f = normal[1]
    g, h, i = normal[2]
    determinant = a*e*i - a*h*f - d*b*i + d*h*c + g*b*f - g*e*c
    inverse_det = np.float32(1) / determinant
    inverse = np.asarray([
        [e*i - f*h, h*c - i*b, b*f - c*e],
        [f*g - d*i, a*i - c*g, d*c - f*a],
        [d*h - e*g, b*g - a*h, a*e - b*d]], np.float32)
    coeff = _matrix_multiply_float32(np.float32(inverse * inverse_det), right).ravel()
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def _quadratic_candidate_allowed(a: float, predicted_dt: float, best_dt: float) -> bool:
    return (abs(a) >= float(np.finfo(np.float32).eps)
            and np.isfinite(predicted_dt)
            and best_dt / 10 < predicted_dt < 10 * best_dt)


def first_epoch_line_search(vertices: np.ndarray, gradient: np.ndarray,
                            faces: np.ndarray, offsets: np.ndarray,
                            neighbors: np.ndarray, original_distances: np.ndarray,
                            original_face_area: np.ndarray,
                            original_total_area: np.float32,
                            distance_weight: float = 0.1,
                            objective: Callable[[np.ndarray], dict] | None = None) -> dict:
    """Select the first-epoch step from trial SSE, without reading a native dt."""
    from .sphere_python import project_radially

    xyz = np.asarray(vertices, np.float32)
    grad = np.asarray(gradient, np.float32)
    sq = np.float32(np.float32(grad[:, 0] * grad[:, 0]
                               + grad[:, 1] * grad[:, 1])
                    + grad[:, 2] * grad[:, 2])
    lengths = np.sqrt(sq.astype(np.float64))
    sum_delta = 0.0
    for length in lengths:
        sum_delta += float(length)  # Native mrisLineMinimize accumulates by vertex order.
    mean_delta = sum_delta / len(lengths)
    max_delta = float(np.max(lengths))
    min_dt = 0.001 / mean_delta
    max_dt = 12.2 / mean_delta

    def trial(dt: float) -> dict:
        if dt:
            shifted = (xyz.astype(np.float64)
                       + dt * grad.astype(np.float64)).astype(np.float32)
            positions = project_radially(shifted, already_sphere=True)
        else:
            positions = xyz
        if objective is not None:
            return objective(positions)
        return first_epoch_sse(positions, faces, offsets, neighbors,
                               original_distances, original_face_area,
                               original_total_area, distance_weight)

    starting = trial(0)
    best_sse = starting["total"]
    best_dt = 0.0
    initial_candidates = []
    dt = min_dt
    while dt < max_dt:
        score = trial(dt)["total"]
        initial_candidates.append({"dt": dt, "sse": score})
        if score <= best_sse:
            best_sse, best_dt = score, dt
        dt *= 10
    if best_dt == 0:
        best_dt = min_dt / 10
        best_sse = trial(best_dt)["total"]
    bracket_dt = [best_dt / 2, best_dt, best_dt * 1.5]
    bracket_sse = [trial(bracket_dt[0])["total"], best_sse,
                   trial(bracket_dt[2])["total"]]
    a, b, c = _quadratic_fit_float32(bracket_dt, bracket_sse)
    candidates = [{"dt": t, "sse": s} for t, s in zip(bracket_dt, bracket_sse)]
    candidates.append({"dt": 0.0, "sse": starting["total"]})
    predicted_dt = float(np.float32(-b / a)) if a else float("nan")
    if _quadratic_candidate_allowed(a, predicted_dt, best_dt):
        candidates.append({"dt": predicted_dt, "sse": trial(predicted_dt)["total"]})
    selected_index = int(np.argmin([entry["sse"] for entry in candidates]))
    return {"mean_delta": mean_delta, "max_delta": max_delta,
            "min_dt": min_dt, "max_dt": max_dt,
            "starting_sse": starting,
            "initial_candidates": initial_candidates,
            "quadratic": {"a": a, "b": b, "c": c,
                          "predicted_dt": predicted_dt},
            "candidates": candidates,
            "selected_index": selected_index,
            "selected_dt": candidates[selected_index]["dt"]}
