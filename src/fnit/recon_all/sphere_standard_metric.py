"""Source-order sparse original metric for conventional FreeSurfer ``mris_sphere``.

This reproduces the fixed 8.2.0 ``MRISsampleDistances`` neighborhood policy:
three complete topology rings plus eight seeded samples from each ring 3–7.
The returned CSR rows include the intentional third-ring duplicates and the
serial reciprocal-distance averaging. It does not perform ``MRISunfold``.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from numba import njit

from .smooth_surface_python import ordered_neighbors
from .sphere_standard_python import FreeSurferSphereRandom


@njit

def _distance(xyz: np.ndarray, a: int, b: int) -> np.float32:
    dx = np.float32(xyz[a, 0] - xyz[b, 0])
    dy = np.float32(xyz[a, 1] - xyz[b, 1])
    dz = np.float32(xyz[a, 2] - xyz[b, 2])
    squared = np.float32(dx * dx + dy * dy)
    squared = np.float32(squared + dz * dz)
    return np.float32(np.sqrt(squared))


@njit

def _random_u32(state: np.ndarray) -> np.uint64:
    position = int(state[37])
    first = state[(37 + position - 24) % 37]
    result = (first - state[position] - state[38]) & np.uint64(0xffffffff)
    if result < first:
        state[38] = 0
    if result > first:
        state[38] = 1
    state[position] = result
    state[37] = (position + 1) % 37
    return result


@njit

def _random_index(state: np.ndarray, count: int) -> int:
    first = _random_u32(state)
    second = _random_u32(state)
    value = float(first) / 4294967295.0 + float(second) / 18446744065119617025.0
    return int(float(np.float32(value)) * (count - 1) + 0.5)


@njit

def _angle(xyz: np.ndarray, vertex: int, a: int, b: int) -> np.float32:
    ax = np.float32(xyz[a, 0] - xyz[vertex, 0])
    ay = np.float32(xyz[a, 1] - xyz[vertex, 1])
    az = np.float32(xyz[a, 2] - xyz[vertex, 2])
    bx = np.float32(xyz[b, 0] - xyz[vertex, 0])
    by = np.float32(xyz[b, 1] - xyz[vertex, 1])
    bz = np.float32(xyz[b, 2] - xyz[vertex, 2])
    la = math.sqrt(np.float64(ax) * np.float64(ax)
                   + np.float64(ay) * np.float64(ay)
                   + np.float64(az) * np.float64(az))
    lb = math.sqrt(np.float64(bx) * np.float64(bx)
                   + np.float64(by) * np.float64(by)
                   + np.float64(bz) * np.float64(bz))
    normalizer = la * lb
    if normalizer < 1e-6:
        return np.float32(0)
    dot = np.float32(np.float32(ax * bx + ay * by) + az * bz)
    normalizer = max(normalizer, abs(float(dot)))
    return np.float32(math.acos(float(dot) / normalizer))


@njit

def _sample_rows(xyz: np.ndarray, adjacent_offsets: np.ndarray,
                 adjacent: np.ndarray, state: np.ndarray, limit: int,
                 row_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    nvertices = len(xyz)
    marks = np.zeros(nvertices, dtype=np.int8)
    metric = np.zeros(nvertices, dtype=np.float32)
    candidates = np.empty(20000, dtype=np.int32)  # FreeSurfer MAX_NBHD_VERTICES.
    ring_starts = np.empty(9, dtype=np.int32)
    selected = np.empty((limit, row_width), dtype=np.int32)
    distance = np.empty((limit, row_width), dtype=np.float32)
    counts = np.zeros(limit, dtype=np.int32)
    max_candidate_count = 0
    for vertex in range(limit):
        candidates[0] = vertex
        marks[vertex] = 1
        ring_starts[0] = 0
        ring_starts[1] = 1
        candidate_count = 1
        for depth in range(1, 8):
            old_start = ring_starts[depth - 1]
            old_end = ring_starts[depth]
            ring_start = candidate_count
            for p in range(old_start, old_end):
                prior = candidates[p]
                for j in range(adjacent_offsets[prior], adjacent_offsets[prior + 1]):
                    other = adjacent[j]
                    if marks[other] != 0:
                        continue
                    if candidate_count == len(candidates):
                        raise ValueError("seven-ring neighborhood exceeds source bound")
                    marks[other] = depth
                    candidates[candidate_count] = other
                    candidate_count += 1
            ring_starts[depth + 1] = candidate_count
            for p in range(ring_start, candidate_count):
                current = candidates[p]
                best = np.float32(1e10)
                for j in range(adjacent_offsets[current], adjacent_offsets[current + 1]):
                    other = adjacent[j]
                    if marks[other] == 0 or marks[other] == depth:
                        continue
                    step = _distance(xyz, current, other)
                    if depth > 1:
                        step = np.float32(step / np.float32(1.09))
                    trial = np.float32(step + metric[other])
                    if trial < best:
                        best = trial
                metric[current] = (_distance(xyz, current, vertex)
                                   if depth <= 2 else best)
            for p in range(ring_start, candidate_count):
                current = candidates[p]
                best = metric[current]
                for j in range(adjacent_offsets[current], adjacent_offsets[current + 1]):
                    other = adjacent[j]
                    if marks[other] != depth:
                        continue
                    step = _distance(xyz, current, other)
                    if depth > 1:
                        step = np.float32(step / np.float32(1.09))
                    trial = np.float32(metric[other] + step)
                    if trial < best:
                        best = trial
                metric[current] = best
        if candidate_count > max_candidate_count:
            max_candidate_count = candidate_count
        count = 0
        for p in range(1, ring_starts[4]):
            if count == row_width:
                raise ValueError("sample row exceeds allocated width")
            other = candidates[p]
            selected[vertex, count] = other
            distance[vertex, count] = _distance(xyz, vertex, other)
            count += 1
        for depth in range(3, 8):
            begin = ring_starts[depth]
            found = ring_starts[depth + 1] - begin
            if found <= 8:
                for p in range(begin, begin + found):
                    if count == row_width:
                        raise ValueError("sample row exceeds allocated width")
                    other = candidates[p]
                    selected[vertex, count] = other
                    distance[vertex, count] = metric[other]
                    count += 1
                continue
            available = np.ones(found, dtype=np.uint8)
            sampled_start = count
            min_angle = np.float32(0.9 * 2.0 * math.pi / 8.0)
            for _ in range(8):
                tries = 0
                while True:
                    index = _random_index(state, found)
                    if available[index] == 0:
                        continue
                    other = candidates[begin + index]
                    accepted = True
                    for prior in range(sampled_start, count):
                        if _angle(xyz, vertex, other,
                                  selected[vertex, prior]) < min_angle:
                            accepted = False
                            break
                    tries += 1
                    if tries > found:
                        min_angle = np.float32(min_angle * np.float32(0.75))
                        tries = 0
                    if accepted or abs(float(min_angle)) < 1e-6:
                        if count == row_width:
                            raise ValueError("sample row exceeds allocated width")
                        selected[vertex, count] = other
                        distance[vertex, count] = metric[other]
                        count += 1
                        available[index] = 0
                        break
        counts[vertex] = count
        for p in range(candidate_count):
            marks[candidates[p]] = 0
    return counts, selected, distance, max_candidate_count


@njit

def _reciprocal_average(offsets: np.ndarray, selected: np.ndarray,
                        distance: np.ndarray) -> int:
    matched = 0
    for vertex in range(len(offsets) - 1):
        for p in range(offsets[vertex], offsets[vertex + 1]):
            other = selected[p]
            for q in range(offsets[other], offsets[other + 1]):
                if selected[q] == vertex:
                    mean = np.float32(np.float32(distance[p] + distance[q])
                                      / np.float32(2))
                    distance[p] = mean
                    distance[q] = mean
                    matched += 1
                    break
    return matched


def sample_standard_metric_matrix(vertices: np.ndarray, faces: np.ndarray,
                                  seed: int = 1234,
                                  limit: int | None = None,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return sampled CSR rows, their pre-symmetry distances, and timings.

    A limited prefix is useful for inspecting the VNL stream; full reciprocal
    averaging requires all vertices and is performed by ``average_standard_metric``.
    """
    xyz = np.asarray(vertices, dtype=np.float32)
    start = perf_counter()
    neighbor_rows = ordered_neighbors(np.asarray(faces, np.int32), len(xyz))
    adjacent_offsets = np.zeros(len(xyz) + 1, np.int64)
    adjacent_offsets[1:] = np.cumsum([len(row) for row in neighbor_rows])
    adjacent = np.fromiter((v for row in neighbor_rows for v in row),
                           dtype=np.int32, count=int(adjacent_offsets[-1]))
    topology_seconds = perf_counter() - start
    rng = FreeSurferSphereRandom(seed)
    state = np.asarray(rng.values + [rng.position, rng.borrow], np.uint64)
    limit = len(xyz) if limit is None else min(int(limit), len(xyz))
    start = perf_counter()
    counts, table, values, max_candidates = _sample_rows(
        xyz, adjacent_offsets, adjacent, state, limit, 256)
    sampling_seconds = perf_counter() - start
    offsets = np.zeros(limit + 1, np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    indices = np.empty(offsets[-1], np.int32)
    distances = np.empty(offsets[-1], np.float32)
    for row in range(limit):
        begin, end = offsets[row:row + 2]
        indices[begin:end] = table[row, :counts[row]]
        distances[begin:end] = values[row, :counts[row]]
    report = {"vertices": limit, "samples": int(offsets[-1]),
              "max_candidates": int(max_candidates),
              "topology_seconds": topology_seconds,
              "sampling_seconds_including_jit": sampling_seconds}
    return offsets, indices, distances, report


def average_standard_metric(offsets: np.ndarray, indices: np.ndarray,
                            distances: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Average reciprocal entries in native vertex and neighbor order."""
    if len(offsets) < int(indices.max(initial=-1)) + 2:
        raise ValueError("reciprocal averaging needs a complete surface")
    result = np.asarray(distances, np.float32).copy()
    started = perf_counter()
    matched = _reciprocal_average(np.asarray(offsets, np.int64),
                                  np.asarray(indices, np.int32), result)
    return result, matched, perf_counter() - started
