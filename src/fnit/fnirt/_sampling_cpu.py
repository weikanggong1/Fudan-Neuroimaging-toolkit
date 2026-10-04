"""CPU FP32 FNIRT sampling with the original scalar rounding sequence.

The optional fast path returns values, validity and explicit voxel gradients.
Unsupported tensors retain registration's tensor sampler. CUDA callers do
not import this module. No image or coordinate values are cached between
calls; finite checks, allocation and the original FP32 products remain local.
"""
import numpy as np
import torch
from numba import njit, prange, get_num_threads, set_num_threads

F0 = np.float32(0)
F1 = np.float32(1)
TOL = np.float32(1e-8)


@njit(cache=True, fastmath=False, error_model="numpy", inline="never")
def _point(data, coordinates, index, sx, sy, sz, bounds, derivatives):
    x, y, z = coordinates[0, index], coordinates[1, index], coordinates[2, index]
    valid = (np.float32(x + TOL) >= F0 and x <= bounds[0]
             and np.float32(y + TOL) >= F0 and y <= bounds[1]
             and np.float32(z + TOL) >= F0 and z <= bounds[2])
    x0, y0, z0 = np.int64(np.floor(x)), np.int64(np.floor(y)), np.int64(np.floor(z))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    wx, wy, wz = (np.float32(x - np.float32(x0)),
                  np.float32(y - np.float32(y0)),
                  np.float32(z - np.float32(z0)))
    ix0, ix1 = min(max(x0, 0), sx - 1), min(max(x1, 0), sx - 1)
    iy0, iy1 = min(max(y0, 0), sy - 1), min(max(y1, 0), sy - 1)
    iz0, iz1 = min(max(z0, 0), sz - 1), min(max(z1, 0), sz - 1)
    vx0, vx1 = 0 <= x0 < sx, 0 <= x1 < sx
    vy0, vy1 = 0 <= y0 < sy, 0 <= y1 < sy
    vz0, vz1 = 0 <= z0 < sz, 0 <= z1 < sz
    b00, b01 = ix0 * (sy * sz) + iy0 * sz, ix0 * (sy * sz) + iy1 * sz
    b10, b11 = ix1 * (sy * sz) + iy0 * sz, ix1 * (sy * sz) + iy1 * sz
    # Retain value * inside even outside, preserving negative-zero products.
    v000 = np.float32(data[b00 + iz0] * np.float32(vx0 and vy0 and vz0))
    v001 = np.float32(data[b00 + iz1] * np.float32(vx0 and vy0 and vz1))
    v010 = np.float32(data[b01 + iz0] * np.float32(vx0 and vy1 and vz0))
    v011 = np.float32(data[b01 + iz1] * np.float32(vx0 and vy1 and vz1))
    v100 = np.float32(data[b10 + iz0] * np.float32(vx1 and vy0 and vz0))
    v101 = np.float32(data[b10 + iz1] * np.float32(vx1 and vy0 and vz1))
    v110 = np.float32(data[b11 + iz0] * np.float32(vx1 and vy1 and vz0))
    v111 = np.float32(data[b11 + iz1] * np.float32(vx1 and vy1 and vz1))
    omz, omy = np.float32(F1 - wz), np.float32(F1 - wy)
    gx, gy, gz = F0, F0, F0
    if derivatives:
        t11 = np.float32(np.float32(omz * v000) + np.float32(wz * v001))
        t12 = np.float32(np.float32(omz * v010) + np.float32(wz * v011))
        t13 = np.float32(np.float32(omz * v100) + np.float32(wz * v101))
        t14 = np.float32(np.float32(omz * v110) + np.float32(wz * v111))
        gx = np.float32(np.float32(omy * np.float32(t13 - t11))
                        + np.float32(wy * np.float32(t14 - t12)))
        gy = np.float32(np.float32(np.float32(F1 - wx) * np.float32(t12 - t11))
                        + np.float32(wx * np.float32(t14 - t13)))
    t11 = np.float32(np.float32(omy * v000) + np.float32(wy * v010))
    t12 = np.float32(np.float32(omy * v001) + np.float32(wy * v011))
    t13 = np.float32(np.float32(omy * v100) + np.float32(wy * v110))
    t14 = np.float32(np.float32(omy * v101) + np.float32(wy * v111))
    t21 = np.float32(np.float32(np.float32(F1 - wx) * t11) + np.float32(wx * t13))
    t22 = np.float32(np.float32(np.float32(F1 - wx) * t12) + np.float32(wx * t14))
    if derivatives:
        gz = np.float32(t22 - t21)
    value = np.float32(np.float32(omz * t21) + np.float32(wz * t22))
    valid_float = np.float32(valid)
    value = np.float32(value * valid_float)
    if derivatives:
        gx, gy, gz = (np.float32(gx * valid_float),
                      np.float32(gy * valid_float),
                      np.float32(gz * valid_float))
    return value, valid, gx, gy, gz


def _loop(data, coordinates, shape, bounds, derivatives, values, valid, gradient):
    sx, sy, sz = shape
    for index in prange(values.size):
        value, is_valid, gx, gy, gz = _point(
            data, coordinates, index, sx, sy, sz, bounds, derivatives)
        values[index], valid[index] = value, is_valid
        if derivatives:
            gradient[0, index], gradient[1, index], gradient[2, index] = gx, gy, gz


_serial = njit(cache=True, fastmath=False, error_model="numpy")(_loop)
_parallel = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_loop)


def try_sample_cpu(volume, coordinates, *, derivatives=True):
    """Return exact fused results or None for the caller's original fallback.

    This sampler has no image cache: every call observes all image
    and coordinate bytes, including in-place mutations and exceptional values.
    Finite scans and output allocation are included in diagnostic API timings.
    """
    if (not isinstance(volume, torch.Tensor) or not isinstance(coordinates, torch.Tensor)
            or volume.device.type != "cpu" or coordinates.device.type != "cpu"
            or volume.dtype != torch.float32 or coordinates.dtype != torch.float32
            or volume.requires_grad or coordinates.requires_grad
            or volume.layout != torch.strided or coordinates.layout != torch.strided
            or volume.is_neg() or coordinates.is_neg()
            or volume.is_conj() or coordinates.is_conj()
            or not volume.is_contiguous() or not coordinates.is_contiguous()
            or volume.ndim != 3 or coordinates.ndim < 2 or coordinates.shape[0] != 3
            or not volume.numel() or not coordinates.numel()):
        return None
    # Forward-mode dual tensors can carry a tangent with requires_grad=False;
    # NumPy would silently discard it, so retain the tensor implementation.
    try:
        if (torch.autograd.forward_ad.unpack_dual(volume).tangent is not None
                or torch.autograd.forward_ad.unpack_dual(coordinates).tangent is not None):
            return None
        image, query = volume.numpy(), coordinates.numpy().reshape(3, -1)
    except (RuntimeError, TypeError):
        # Batched/functional tensor transforms can also prevent a NumPy view;
        # their original tensor path owns those semantics.
        return None
    if not np.isfinite(image).all() or not np.isfinite(query).all():
        return None
    # The tensor fallback owns extreme floor-to-long/lo+1 semantics.  Staying
    # well inside int64 avoids undefined conversions or neighbour overflow.
    if np.min(query) <= -(2**30) or np.max(query) >= 2**30:
        return None
    count = query.shape[1]
    values = np.empty(count, np.float32)
    valid = np.empty(count, np.bool_)
    gradient = np.empty((3, count) if derivatives else (3, 0), np.float32)
    bounds = np.array([float(size - 1) + 1e-8 for size in image.shape], np.float32)
    args = (image.reshape(-1), query, image.shape, bounds, derivatives, values, valid, gradient)
    previous = get_num_threads()
    budget = min(previous, torch.get_num_threads())
    if budget < 2:
        _serial(*args)
    else:
        changed = budget != previous
        if changed:
            set_num_threads(budget)
        try:
            _parallel(*args)
        finally:
            if changed:
                set_num_threads(previous)
    # Finite image/coordinates may still overflow a derivative subtraction.
    # Keep original NaN payload and exceptional arithmetic in that case.
    if not np.isfinite(values).all() or (derivatives and not np.isfinite(gradient).all()):
        return None
    spatial_shape = tuple(coordinates.shape[1:])
    return (torch.from_numpy(values.reshape(spatial_shape)),
            torch.from_numpy(valid.reshape(spatial_shape)),
            torch.from_numpy(gradient.reshape((3, *spatial_shape))) if derivatives else None)
