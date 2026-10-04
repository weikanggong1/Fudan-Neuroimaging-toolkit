"""CPU Gaussian fusion preserving the float store after every offset."""
from __future__ import annotations

import numpy as np
from numba import config, get_num_threads, njit, prange
import torch


@njit(inline="always", fastmath=False, error_model="numpy")
def _smooth_voxel(volume, weights, axis, single, output, batch, channel, x, y, z):
    radius = weights.shape[0] // 2
    position = x if axis == 0 else y if axis == 1 else z
    length = volume.shape[axis + 2]
    start = max(0, radius - position)
    stop = min(weights.shape[0], radius + length - position)
    value = np.float64(0.0)
    for index in range(start, stop):
        offset = index - radius
        sx = x + offset if axis == 0 else x
        sy = y + offset if axis == 1 else y
        sz = z + offset if axis == 2 else z
        product = np.float64(volume[batch, channel, sx, sy, sz]) * weights[index]
        value = value + product
        if single:
            value = np.float64(np.float32(value))
    output[batch, channel, x, y, z] = value


def _smooth_axis(volume, weights, axis, single, output, fastest):
    if fastest == 0:
        for outer in prange(volume.shape[0] * volume.shape[1] * volume.shape[4]):
            batch = outer // (volume.shape[1] * volume.shape[4])
            channel = (outer // volume.shape[4]) % volume.shape[1]
            z = outer % volume.shape[4]
            for y in range(volume.shape[3]):
                for x in range(volume.shape[2]):
                    _smooth_voxel(volume, weights, axis, single, output, batch, channel, x, y, z)
    elif fastest == 1:
        for outer in prange(volume.shape[0] * volume.shape[1] * volume.shape[4]):
            batch = outer // (volume.shape[1] * volume.shape[4])
            channel = (outer // volume.shape[4]) % volume.shape[1]
            z = outer % volume.shape[4]
            for x in range(volume.shape[2]):
                for y in range(volume.shape[3]):
                    _smooth_voxel(volume, weights, axis, single, output, batch, channel, x, y, z)
    else:
        for outer in prange(volume.shape[0] * volume.shape[1] * volume.shape[2]):
            batch = outer // (volume.shape[1] * volume.shape[2])
            channel = (outer // volume.shape[2]) % volume.shape[1]
            x = outer % volume.shape[2]
            for y in range(volume.shape[3]):
                for z in range(volume.shape[4]):
                    _smooth_voxel(volume, weights, axis, single, output, batch, channel, x, y, z)


_serial = njit(cache=True, fastmath=False, error_model="numpy")(_smooth_axis)
_parallel = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_smooth_axis)


def gaussian_blur_cpu(volume, kernels):
    """Use two image buffers; independent voxels retain serial offset sums."""
    if volume.device.type != "cpu" or volume.requires_grad or volume.ndim != 5 or volume.dtype not in (torch.float32, torch.float64):
        raise ValueError("CPU fused Gaussian requires a non-differentiable FP32/FP64 5D CPU image")
    budget = torch.get_num_threads()
    threads = get_num_threads() if config.NUMBA_NUM_THREADS <= budget else 1
    kernel = _parallel if 1 < threads <= budget else _serial
    buffers = [torch.empty_like(volume), torch.empty_like(volume)]
    result = volume
    single = volume.dtype == torch.float32
    for axis, weights in enumerate(kernels):
        output = buffers[axis % 2]
        fastest = min(range(3), key=lambda dimension: result.stride(dimension + 2))
        kernel(result.numpy(), np.asarray(weights, dtype=np.float64), axis, single, output.numpy(), fastest)
        result = output
    return result
