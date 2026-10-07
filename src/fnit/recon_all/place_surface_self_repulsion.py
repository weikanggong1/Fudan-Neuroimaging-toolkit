"""FreeSurfer 8.2 white-surface self-repulsion on the current vertex grid."""

from __future__ import annotations

import numpy as np
from numba import njit

from .place_surface_repulsion import vertex_buckets


def vertex_buckets_current(
    vertices: np.ndarray, ripped: np.ndarray, *, resolution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current-position MHT bucket members in insertion order.

    Placement forces use the default 1 mm grid. ``MRIScomputeSSE`` instead
    creates a grid at the current mean one-/two-ring vertex spacing.
    """
    if resolution == 1.0:
        return vertex_buckets(vertices, vertices, ripped)
    xyz = np.asarray(vertices, dtype=np.float32)
    spacing = np.float32(resolution)
    key = np.float32(np.float32(xyz / spacing) + np.float32(1000)).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for vertex in range(len(key)):
        if not ripped[vertex]:
            buckets.setdefault(tuple(key[vertex]), []).append(vertex)
    offsets = np.zeros(len(key) + 1, dtype=np.int32)
    flat: list[int] = []
    for vertex in range(len(key)):
        if not ripped[vertex]:
            flat.extend(buckets.get(tuple(key[vertex]), ()))
        offsets[vertex + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int32)


@njit(cache=True)
def _mean_vertex_spacing(
    xyz: np.ndarray, ripped: np.ndarray,
    neighbor_offsets: np.ndarray, neighbors: np.ndarray,
) -> float:
    total = 0.0
    count = 0
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        x, y, z = xyz[vertex]
        for index in range(neighbor_offsets[vertex], neighbor_offsets[vertex + 1]):
            other = neighbors[index]
            dx = np.float32(xyz[other, 0] - x)
            dy = np.float32(xyz[other, 1] - y)
            dz = np.float32(xyz[other, 2] - z)
            squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
            total += np.sqrt(np.float64(squared))
            count += 1
    return total / count


def mean_vertex_spacing(
    vertices: np.ndarray, ripped: np.ndarray,
    neighbor_offsets: np.ndarray, neighbors: np.ndarray,
) -> float:
    """Return ``MRIScomputeTotalVertexSpacingStats`` mean for current mesh."""
    return _mean_vertex_spacing(
        np.asarray(vertices, dtype=np.float32), np.asarray(ripped, dtype=np.bool_),
        np.asarray(neighbor_offsets, dtype=np.int32), np.asarray(neighbors, dtype=np.int32),
    )


@njit(cache=True)
def _force(
    xyz: np.ndarray, ripped: np.ndarray,
    bucket_offsets: np.ndarray, bucket_members: np.ndarray,
    neighbor_offsets: np.ndarray, neighbors: np.ndarray,
    weight: float,
) -> np.ndarray:
    result = np.zeros_like(xyz)
    # Mark each vertex's one/two-ring neighbors once per source vertex.
    # This preserves source-order accumulation while avoiding an O(degree)
    # scan for every bucket candidate.
    neighbor_marks = np.full(len(xyz), -1, dtype=np.int32)
    for vertex in range(len(xyz)):
        for neighbor in range(neighbor_offsets[vertex], neighbor_offsets[vertex + 1]):
            neighbor_marks[neighbors[neighbor]] = vertex
        if ripped[vertex]:
            continue
        x, y, z = xyz[vertex]
        sx = sy = sz = np.float32(0)
        count = 0
        for slot in range(bucket_offsets[vertex], bucket_offsets[vertex + 1]):
            other = bucket_members[slot]
            if other == vertex or ripped[other]:
                continue
            if neighbor_marks[other] == vertex:
                continue
            dx = np.float32(x - xyz[other, 0])
            dy = np.float32(y - xyz[other, 1])
            dz = np.float32(z - xyz[other, 2])
            squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
            norm = np.float32(np.sqrt(np.float64(squared)))
            distance = np.float32(norm + np.float32(0.25))
            power = distance
            for _ in range(6):
                power = np.float32(power * distance)
            scale = 4.0 / float(power)
            if norm == 0:
                norm = np.float32(1)
            dx = np.float32(dx / norm)
            dy = np.float32(dy / norm)
            dz = np.float32(dz / norm)
            sx = np.float32(float(sx) + scale * float(dx))
            sy = np.float32(float(sy) + scale * float(dy))
            sz = np.float32(float(sz) + scale * float(dz))
            count += 1
        if count:
            scale = weight / count
            result[vertex, 0] = np.float32(sx * scale)
            result[vertex, 1] = np.float32(sy * scale)
            result[vertex, 2] = np.float32(sz * scale)
    return result


def self_repulsion_gradient(
    vertices: np.ndarray, ripped: np.ndarray,
    bucket_offsets: np.ndarray, bucket_members: np.ndarray,
    neighbor_offsets: np.ndarray, neighbors: np.ndarray,
    *, weight: float = 5.0,
) -> np.ndarray:
    """Return ``mrisComputeRepulsiveTerm`` force for one white placement step.

    ``neighbors`` is the source-order one- plus two-ring vertex list. The
    current-position bucket and all topology arrays may be reused to compute
    other terms for the same mesh, but buckets must be rebuilt after a step.
    """
    return _force(
        np.asarray(vertices, dtype=np.float32), np.asarray(ripped, dtype=np.bool_),
        np.asarray(bucket_offsets, dtype=np.int32), np.asarray(bucket_members, dtype=np.int32),
        np.asarray(neighbor_offsets, dtype=np.int32), np.asarray(neighbors, dtype=np.int32),
        float(weight),
    )


@njit(cache=True)
def _energy(
    xyz: np.ndarray, ripped: np.ndarray,
    bucket_offsets: np.ndarray, bucket_members: np.ndarray,
    neighbor_offsets: np.ndarray, neighbors: np.ndarray,
    weight: float,
) -> float:
    total = 0.0
    neighbor_marks = np.full(len(xyz), -1, dtype=np.int32)
    for vertex in range(len(xyz)):
        for neighbor in range(neighbor_offsets[vertex], neighbor_offsets[vertex + 1]):
            neighbor_marks[neighbors[neighbor]] = vertex
        if ripped[vertex]:
            continue
        x, y, z = xyz[vertex]
        vertex_energy = 0.0
        for slot in range(bucket_offsets[vertex], bucket_offsets[vertex + 1]):
            other = bucket_members[slot]
            if other == vertex or ripped[other]:
                continue
            if neighbor_marks[other] == vertex:
                continue
            dx = np.float32(xyz[other, 0] - x)
            dy = np.float32(xyz[other, 1] - y)
            dz = np.float32(xyz[other, 2] - z)
            squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
            distance = np.float32(np.sqrt(np.float64(squared)) + 0.25)
            cube = np.float32(np.float32(distance * distance) * distance)
            power = np.float32(cube * cube)
            vertex_energy += 1.0 / float(power)
        total += vertex_energy
    return weight * total


def self_repulsion_energy(
    vertices: np.ndarray, ripped: np.ndarray,
    bucket_offsets: np.ndarray, bucket_members: np.ndarray,
    neighbor_offsets: np.ndarray, neighbors: np.ndarray,
    *, weight: float = 5.0,
) -> float:
    """Return current-vertex ``mrisComputeRepulsiveEnergy`` SSE term."""
    return _energy(
        np.asarray(vertices, dtype=np.float32), np.asarray(ripped, dtype=np.bool_),
        np.asarray(bucket_offsets, dtype=np.int32), np.asarray(bucket_members, dtype=np.int32),
        np.asarray(neighbor_offsets, dtype=np.int32), np.asarray(neighbors, dtype=np.int32),
        float(weight),
    )
