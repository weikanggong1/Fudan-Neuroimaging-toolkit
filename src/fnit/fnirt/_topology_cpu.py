"""Select CPU limiter corners without changing the original update order."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def out_of_range_corners(checks, minimum, maximum):
    """Flat indices in z/y/x/corner order; comparisons promote to double."""
    nx, ny, nz = checks.shape[1:]
    count = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for corner in range(8):
                    value = np.float64(checks[corner, x, y, z])
                    if not minimum <= value <= maximum:
                        count += 1
    result = np.empty(count, dtype=np.int64)
    position = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for corner in range(8):
                    value = np.float64(checks[corner, x, y, z])
                    if not minimum <= value <= maximum:
                        result[position] = (((z * ny + y) * nx + x) * 8) + corner
                        position += 1
    return result
