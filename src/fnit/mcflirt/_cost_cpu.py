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
        # Every intermediate keeps the original float32 operation order.
        o0 = np.float32(np.float32(np.float32(y) * coefficients[0, 1])
                        + np.float32(np.float32(z) * coefficients[0, 2]))
        o1 = np.float32(np.float32(np.float32(y) * coefficients[1, 1])
                        + np.float32(np.float32(z) * coefficients[1, 2]))
        o2 = np.float32(np.float32(np.float32(y) * coefficients[2, 1])
                        + np.float32(np.float32(z) * coefficients[2, 2]))
        o0 = np.float32(o0 + coefficients[0, 3])
        o1 = np.float32(o1 + coefficients[1, 3])
        o2 = np.float32(o2 + coefficients[2, 3])
        origin = (o0, o1, o2)
        xmin, xmax = np.float32(0), np.float32(nx - 1)
        for axis in range(3):
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
        c0 = np.float32(o0 + np.float32(xmin * coefficients[0, 0]))
        c1 = np.float32(o1 + np.float32(xmin * coefficients[1, 0]))
        c2 = np.float32(o2 + np.float32(xmin * coefficients[2, 0]))
        t0, t1, t2 = np.float32(0), np.float32(0), np.float32(0)
        t3, t4, t5 = np.float32(0), np.float32(0), np.float32(0)
        for offset in range(nx):
            actual_x = np.float32(xmin + np.float32(offset))
            valid = actual_x <= xmax
            weight = np.float32(1)
            sx = min(max(c0, np.float32(0)), upper[0])
            sy = min(max(c1, np.float32(0)), upper[1])
            sz = min(max(c2, np.float32(0)), upper[2])
            coordinate = (c0, c1, c2)
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
            t0 = np.float32(t0 + weight)
            t1 = np.float32(t1 + wr)
            t2 = np.float32(t2 + np.float32(wr * ref))
            t3 = np.float32(t3 + wv)
            t4 = np.float32(t4 + np.float32(wv * value))
            t5 = np.float32(t5 + np.float32(wr * value))
            c0 = np.float32(c0 + coefficients[0, 0])
            c1 = np.float32(c1 + coefficients[1, 0])
            c2 = np.float32(c2 + coefficients[2, 0])
        rows[0, z, y] = t0
        rows[1, z, y] = t1
        rows[2, z, y] = t2
        rows[3, z, y] = t3
        rows[4, z, y] = t4
        rows[5, z, y] = t5
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
