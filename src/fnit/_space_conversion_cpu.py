"""CPU RF-ANTs nearest-vertex lookup with grid_sample rounding semantics.

The caller constructs the same normalized float32 coordinates as the CUDA
path.  Only the CPU gather is compiled; no CUDA tensor or resampling kernel
uses this module.  Disabling fastmath preserves the unnormalization and
nearest-even rounding used by PyTorch with ``align_corners=True``.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False, inline="always")
def _nearest_index(coordinate, size):
    voxel = np.float32(np.float32(np.float32(coordinate + np.float32(1))
                                 / np.float32(2)) * np.float32(size - 1))
    if not np.isfinite(voxel):
        return -1
    return int(np.rint(voxel))


@njit(cache=True, fastmath=False, parallel=True)
def project_surface_layer(grid, mask, left_vertices, right_vertices,
                          left_values, right_values):
    """Project one output plane; vertex maps retain MATLAB row-column order."""
    result = np.zeros((grid.shape[0], left_values.shape[1]), np.float32)
    for row in prange(grid.shape[0]):
        x = _nearest_index(grid[row, 0], mask.shape[0])
        y = _nearest_index(grid[row, 1], mask.shape[1])
        z = _nearest_index(grid[row, 2], mask.shape[2])
        if (x < 0 or y < 0 or z < 0 or x >= mask.shape[0]
                or y >= mask.shape[1] or z >= mask.shape[2]):
            continue
        if not (mask[x, y, z] > np.float32(0.5)):
            continue
        # MRIread/MATLAB uses (row, column, depth); NIfTI uses (x, y, z).
        left = int(left_vertices[y, x, z])
        right = int(right_vertices[y, x, z])
        for frame in range(left_values.shape[1]):
            value = np.float32(0)
            if left > 0:
                value = np.float32(value + left_values[left - 1, frame])
            if right > 0:
                value = np.float32(value + right_values[right - 1, frame])
            result[row, frame] = value
    return result
