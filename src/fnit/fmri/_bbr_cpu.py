"""Fused CPU BBR operators with separately rounded interpolation operations.

CUDA retains its existing kernels. CPU reuses FLIRT's Numba thread-budget
dispatcher, including restoration of the caller's previous Numba setting.
"""
import math
import numpy as np
from numba import njit, prange

from ..flirt._cpu import _dispatch_with_cpu_budget


def _smooth_axis_impl(data, coefficients, axis):
    output = np.empty_like(data)
    nx, ny, nz = data.shape
    radius = len(coefficients) // 2
    for index in prange(data.size):
        x, remainder = index // (ny * nz), index % (ny * nz)
        y, z = remainder // nz, remainder % nz
        value = np.float32(0)
        for tap in range(len(coefficients)):
            offset = tap - radius
            ix, iy, iz = x, y, z
            if axis == 0: ix += offset
            elif axis == 1: iy += offset
            else: iz += offset
            sample = np.float32(0)
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                sample = data[ix, iy, iz]
            value = np.float32(np.float64(value) + np.float64(sample) * coefficients[tap])
        output[x, y, z] = value
    return output


@njit(cache=True, fastmath=False, inline="always")
def _corner(image, x, y, z):
    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and 0 <= z < image.shape[2]:
        return image[x, y, z]
    return np.float32(0)


@njit(cache=True, fastmath=False, inline="always")
def _sample_zero(image, x, y, z):
    ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    dx, dy, dz = np.float32(x - np.float32(ix)), np.float32(y - np.float32(iy)), np.float32(z - np.float32(iz))
    v000, v001 = _corner(image, ix, iy, iz), _corner(image, ix, iy, iz+1)
    v010, v011 = _corner(image, ix, iy+1, iz), _corner(image, ix, iy+1, iz+1)
    v100, v101 = _corner(image, ix+1, iy, iz), _corner(image, ix+1, iy, iz+1)
    v110, v111 = _corner(image, ix+1, iy+1, iz), _corner(image, ix+1, iy+1, iz+1)
    a = np.float32(np.float32(np.float32(v100-v000)*dx)+v000)
    b = np.float32(np.float32(np.float32(v101-v001)*dx)+v001)
    c = np.float32(np.float32(np.float32(v110-v010)*dx)+v010)
    d = np.float32(np.float32(np.float32(v111-v011)*dx)+v011)
    e, f = np.float32(np.float32(np.float32(c-a)*dy)+a), np.float32(np.float32(np.float32(d-b)*dy)+b)
    return np.float32(np.float32(np.float32(f-e)*dz)+e)


@njit(cache=True, fastmath=False, inline="never")
def _side_value(image, points, transforms, candidate, point, side):
    x = points[side, point, 0] * transforms[candidate, 0, 0] + points[side, point, 1] * transforms[candidate, 0, 1]
    x = x + points[side, point, 2] * transforms[candidate, 0, 2]
    x = x + transforms[candidate, 0, 3]
    y = points[side, point, 0] * transforms[candidate, 1, 0] + points[side, point, 1] * transforms[candidate, 1, 1]
    y = y + points[side, point, 2] * transforms[candidate, 1, 2]
    y = y + transforms[candidate, 1, 3]
    z = points[side, point, 0] * transforms[candidate, 2, 0] + points[side, point, 1] * transforms[candidate, 2, 1]
    z = z + points[side, point, 2] * transforms[candidate, 2, 2]
    z = z + transforms[candidate, 2, 3]
    return _sample_zero(image, np.float32(x), np.float32(y), np.float32(z))


def _cost_updates_impl(image, points, transforms):
    count = points.shape[1]
    updates = np.empty((len(transforms), count), dtype=np.float64)
    for index in prange(updates.size):
        candidate, point = index // count, index % count
        grey = np.float64(_side_value(image, points, transforms, candidate, point, 0))
        white = np.float64(_side_value(image, points, transforms, candidate, point, 1))
        total = grey + white
        difference = 200.0 * (grey-white) / total if abs(total) > 1e-6 else 0.0
        updates[candidate, point] = 1.0 + math.tanh(-0.5 * difference)
    return updates

_smooth_serial = njit(cache=True, fastmath=False)(_smooth_axis_impl)
_smooth_parallel = njit(cache=True, fastmath=False, parallel=True)(_smooth_axis_impl)
_cost_serial = njit(cache=True, fastmath=False, error_model="numpy")(_cost_updates_impl)
_cost_parallel = njit(cache=True, fastmath=False, parallel=True, error_model="numpy")(_cost_updates_impl)


def smooth_axis(data, coefficients, axis):
    return _dispatch_with_cpu_budget(_smooth_parallel, _smooth_serial, (data, coefficients, axis))


def cost_updates(image, points, transforms):
    arguments = (image, points, transforms)
    # Local searches evaluate one matrix at a time and alternate with Torch
    # means. Small batches stay serial to avoid competing OpenMP pools.
    if len(transforms) * points.shape[1] < 262144:
        return _cost_serial(*arguments)
    return _dispatch_with_cpu_budget(_cost_parallel, _cost_serial, arguments)
