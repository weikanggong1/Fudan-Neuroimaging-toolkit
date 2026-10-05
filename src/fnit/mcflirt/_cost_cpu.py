"""Ordered CPU MCFLIRT sampling; the CUDA implementation is independent.

Reuse FLIRT's scalar interpolation and CPU thread budget. Every coordinate,
weight, intensity product and row/plane sum retains float32 operation order.
The two cumulative count sums stay in PyTorch, whose CPU accumulator differs
from an ordinary float32 prefix sum. Rows alone may run in parallel.
"""

import numpy as np
from numba import njit, prange

from ..flirt._cpu import _dispatch_with_cpu_budget, _trilinear


def _sample_rows_impl(reference, moving, coefficients, upper, smooth):
    nz, ny, nx = reference.shape
    rows = np.zeros((6, nz, ny), dtype=np.float32)
    for row in prange(nz * ny):
        z, y = row // ny, row % ny
        origin = np.empty(3, dtype=np.float32)
        xmin, xmax = np.float32(0), np.float32(nx - 1)
        for axis in range(3):
            origin[axis] = np.float32(np.float32(np.float32(y) * coefficients[axis, 1])
                                    + np.float32(np.float32(z) * coefficients[axis, 2]))
            origin[axis] = np.float32(origin[axis] + coefficients[axis, 3])
            direction = coefficients[axis, 0]
            if abs(direction) < 1e-8:
                if origin[axis] < 0 or origin[axis] > upper[axis]:
                    xmin = np.float32(nx)
            else:
                bound0 = np.float32(-origin[axis] / direction)
                bound1 = np.float32(np.float32(upper[axis] - origin[axis]) / direction)
                xmin = max(xmin, np.float32(np.ceil(min(bound0, bound1))))
                xmax = min(xmax, np.float32(np.floor(max(bound0, bound1))))
        xmin = min(max(xmin, np.float32(0)), np.float32(nx))
        coordinate = np.empty(3, dtype=np.float32)
        for axis in range(3):
            coordinate[axis] = np.float32(origin[axis] + np.float32(xmin * coefficients[axis, 0]))
        total = np.zeros(6, dtype=np.float32)
        for offset in range(nx):
            actual_x = np.float32(xmin + np.float32(offset))
            valid = actual_x <= xmax
            weight = np.float32(1)
            px, py, pz = coordinate[0], coordinate[1], coordinate[2]
            sx = min(max(px, np.float32(0)), upper[0])
            sy = min(max(py, np.float32(0)), upper[1])
            sz = min(max(pz, np.float32(0)), upper[2])
            for axis in range(3):
                position = coordinate[axis]
                valid = valid and position >= 0 and position <= upper[axis]
                distance = np.float32(upper[axis] - position)
                taper = (np.float32(position / smooth[axis]) if position < smooth[axis]
                         else np.float32(distance / smooth[axis]) if distance < smooth[axis]
                         else np.float32(1))
                weight = np.float32(weight * taper)
            weight = np.float32(max(weight, np.float32(0)) * np.float32(valid))
            value = _trilinear(moving, sx, sy, sz)
            ref = reference[z, y, min(int(actual_x), nx - 1)]
            wr, wv = np.float32(weight * ref), np.float32(weight * value)
            terms = (weight, wr, np.float32(wr * ref), wv,
                     np.float32(wv * value), np.float32(wr * value))
            for term in range(6):
                total[term] = np.float32(total[term] + terms[term])
            for axis in range(3):
                coordinate[axis] = np.float32(coordinate[axis] + coefficients[axis, 0])
        for term in range(6):
            rows[term, z, y] = total[term]
    sums = np.zeros(6, dtype=np.float32)
    for term in range(6):
        for z in range(nz):
            plane = np.float32(0)
            for y in range(ny):
                plane = np.float32(plane + rows[term, z, y])
            sums[term] = np.float32(sums[term] + plane)
    return rows[0].copy(), sums


_serial = njit(cache=True, fastmath=False, error_model="numpy")(_sample_rows_impl)
_parallel = njit(cache=True, fastmath=False, parallel=True, error_model="numpy")(_sample_rows_impl)


def motion_cost_rows(reference, moving, coefficients, upper, smooth):
    arguments = (reference, moving, coefficients, upper, smooth)
    # The motion pyramid makes thousands of small calls interleaved with Torch
    # reductions. Avoid waking a second OpenMP pool for these calls. Larger
    # volumes still use the caller's bounded Numba pool; neither path changes
    # process-wide Torch/Numba settings or the ordered per-row arithmetic.
    if reference.size < 262144:
        return _serial(*arguments)
    return _dispatch_with_cpu_budget(_parallel, _serial, arguments)
