"""Integer-key connected components for six-dimensional BWAS edges."""

import numpy as np
from numba import njit


@njit(cache=True)
def _find(parent, index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


@njit(cache=True)
def _edge_components(left, right, neighbors, nvox):
    count = len(left)
    capacity = 1
    while capacity < 2*count:
        capacity *= 2
    keys = np.full(capacity, -1, dtype=np.int64)
    values = np.full(capacity, -1, dtype=np.int32)
    mask = capacity-1
    for edge in range(count):
        key = np.int64(left[edge])*nvox + right[edge]
        slot = key & mask
        while keys[slot] != -1 and keys[slot] != key:
            slot = (slot+1) & mask
        keys[slot] = key
        values[slot] = edge

    parent = np.arange(count, dtype=np.int32)
    for edge in range(count):
        for ni in neighbors[left[edge]]:
            if ni < 0:
                continue
            for nj in neighbors[right[edge]]:
                if nj < 0:
                    continue
                key = np.int64(ni)*nvox + nj
                slot = key & mask
                while keys[slot] != -1:
                    if keys[slot] == key:
                        other = values[slot]
                        if other < edge:
                            first = _find(parent, edge)
                            second = _find(parent, other)
                            if first != second:
                                parent[second] = first
                        break
                    slot = (slot+1) & mask
    roots = np.empty(count, dtype=np.int32)
    for edge in range(count):
        roots[edge] = _find(parent, edge)
    return roots


def edge_components(edges: list[tuple[int, int, float]], coords: np.ndarray) -> np.ndarray:
    """Return one component root per edge under the original 18-neighbor rule."""
    nvox = len(coords)
    volume = np.full(tuple(coords.max(axis=0)+1), -1, dtype=np.int32)
    volume[tuple(coords.T)] = np.arange(nvox, dtype=np.int32)
    offsets = [(x, y, z) for x in (-1, 0, 1)
               for y in (-1, 0, 1) for z in (-1, 0, 1)
               if x*x+y*y+z*z < 1.5**2]
    neighbors = np.full((nvox, len(offsets)), -1, dtype=np.int32)
    shape = np.asarray(volume.shape)
    for column, offset in enumerate(offsets):
        shifted = coords + offset
        valid = np.all((shifted >= 0) & (shifted < shape), axis=1)
        neighbors[valid, column] = volume[tuple(shifted[valid].T)]
    left = np.fromiter((edge[0] for edge in edges), dtype=np.int32, count=len(edges))
    right = np.fromiter((edge[1] for edge in edges), dtype=np.int32, count=len(edges))
    return _edge_components(left, right, neighbors, nvox)
