"""Fused ordered CPU containment for the cached radial sphere mapper.

The cached normals come from the existing PyTorch geometry calculation.
Each candidate retains float64 operation order, exact distance ties and
the smallest mesh face ID. Only independent query rows run in parallel;
CUDA and the reference execution do not import this module.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False, inline="always")
def _dot(a, b):
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]


@njit(cache=True, fastmath=False, inline="always")
def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


@njit(cache=True, fastmath=False, inline="always")
def _subtract(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


@njit(cache=True, fastmath=False, inline="always")
def _norm(a):
    return np.sqrt(_dot(a, a))


@njit(cache=True, fastmath=False, inline="always", error_model="numpy")
def _distance(point, a, b, c):
    # Triangle::dist_to_point: finite edges in (a,b), (a,c), (b,c)
    # order, followed by the three vertices. Comparisons are strict.
    best = np.inf
    for first, second in ((a, b), (a, c), (b, c)):
        edge = _subtract(second, first)
        first_delta = _subtract(point, first)
        second_delta = _subtract(point, second)
        valid = _dot(first_delta, edge) > 0 and _dot(second_delta, edge) < 0
        if valid:
            value = _norm(_cross(first_delta, second_delta)) / _norm(edge)
            if value < best:
                best = value
    for corner in (a, b, c):
        value = _norm(_subtract(point, corner))
        if value < best:
            best = value
    return best


def _select_rows(points, nearest, incident, triangles, normals, tops, ab, bc, ca):
    rows = len(points)
    faces = np.empty(rows, dtype=np.int64)
    projections = np.empty((rows, 3), dtype=np.float64)
    exists = np.empty(rows, dtype=np.bool_)
    ambiguous = np.empty(rows, dtype=np.bool_)
    for row in prange(rows):
        point = (points[row, 0], points[row, 1], points[row, 2])
        # The tensor implementation returns candidate zero if every distance
        # is infinite. Keep its projected point for the common fallback.
        selected_face = incident[nearest[row, 0], 0]
        selected_projection = (np.nan, np.nan, np.nan)
        minimum = np.inf
        containing = 0
        candidate_number = 0
        for column in range(nearest.shape[1]):
            for index in range(incident.shape[1]):
                face = incident[nearest[row, column], index]
                scale = tops[face] / _dot(normals[face], point)
                projected = (point[0] * scale, point[1] * scale, point[2] * scale)
                if candidate_number == 0:
                    selected_projection = projected
                candidate_number += 1
                a, b, c = triangles[face, 0], triangles[face, 1], triangles[face, 2]
                inside = (np.isfinite(projected[0]) and np.isfinite(projected[1])
                          and np.isfinite(projected[2])
                          and _dot(_cross(_subtract(b, a), _subtract(projected, a)), ab[face]) > -1e-8
                          and _dot(_cross(_subtract(c, b), _subtract(projected, b)), bc[face]) > -1e-8
                          and _dot(_cross(_subtract(a, c), _subtract(projected, c)), ca[face]) > -1e-8)
                if inside:
                    containing += 1
                    distance = _distance(projected, a, b, c)
                    if distance < minimum or (distance == minimum and face < selected_face):
                        minimum = distance
                        selected_face = face
                        selected_projection = projected
        faces[row] = selected_face
        projections[row, 0] = selected_projection[0]
        projections[row, 1] = selected_projection[1]
        projections[row, 2] = selected_projection[2]
        exists[row] = np.isfinite(minimum)
        ambiguous[row] = containing > 1
    return faces, projections, exists, ambiguous


# Release the GIL so independently budgeted L/R workers can overlap.
_serial = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_select_rows)
_parallel = njit(cache=True, fastmath=False, parallel=True, nogil=True, error_model="numpy")(_select_rows)


def select_faces(*arguments, cpu_threads):
    """Bound row workers by the caller and hemisphere budgets; restore mask."""
    import torch
    from numba import get_num_threads, set_num_threads
    previous = get_num_threads()
    budget = min(previous, torch.get_num_threads(), cpu_threads)
    if budget < 2:
        return _serial(*arguments)
    changed = budget != previous
    if changed:
        set_num_threads(budget)
    try:
        return _parallel(*arguments)
    finally:
        if changed:
            set_num_threads(previous)
