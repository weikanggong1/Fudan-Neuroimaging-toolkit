"""One CPU pass for the original multiply-then-square elementwise steps."""
from __future__ import annotations

import numpy as np
from numba import config, get_num_threads, njit, prange
import torch


@njit(inline="always", fastmath=False, error_model="numpy")
def _square_voxel(field, multiplier, channel, x, y, z):
    # Both operands already have the field dtype. The first product is rounded
    # to that same dtype before the second product; no FMA or reduction here.
    value = field[channel, x, y, z] * multiplier
    field[channel, x, y, z] = value * value


def _multiply_square(field, multiplier, fastest):
    if fastest == 1:
        for z in prange(field.shape[3]):
            for channel in range(field.shape[0]):
                for x in range(field.shape[1]):
                    for y in range(field.shape[2]):
                        _square_voxel(field, multiplier, channel, x, y, z)
    elif fastest == 0:
        for z in prange(field.shape[3]):
            for channel in range(field.shape[0]):
                for y in range(field.shape[2]):
                    for x in range(field.shape[1]):
                        _square_voxel(field, multiplier, channel, x, y, z)
    else:
        for x in prange(field.shape[1]):
            for channel in range(field.shape[0]):
                for y in range(field.shape[2]):
                    for z in range(field.shape[3]):
                        _square_voxel(field, multiplier, channel, x, y, z)


_serial = njit(cache=True, fastmath=False, error_model="numpy")(_multiply_square)
_parallel = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_multiply_square)


def _multiply_square_flat(field, multiplier):
    for index in prange(field.size):
        value = field[index] * multiplier
        field[index] = value * value


_flat_serial = njit(cache=True, fastmath=False, error_model="numpy")(_multiply_square_flat)
_flat_parallel = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_multiply_square_flat)


def multiply_square_(field, multiplier):
    """Preserve field layout and both scalar products; leave sum to PyTorch."""
    if (field.device.type != "cpu" or field.dtype not in (torch.float32, torch.float64)
            or field.ndim != 4 or field.requires_grad):
        raise ValueError("CPU bending fusion requires a non-differentiable FP32/FP64 4D field")
    budget = torch.get_num_threads()
    threads = get_num_threads() if config.NUMBA_NUM_THREADS <= budget else 1
    parallel = 1 < threads <= budget
    factor = np.float32(multiplier) if field.dtype == torch.float32 else np.float64(multiplier)
    # Einsum's physical order also interleaves the channel dimension. A sorted
    # stride view is contiguous without a copy and permits ordinary SIMD for
    # two independent products. Keep the original Tensor for its later sum.
    physical = field.permute(tuple(sorted(range(4), key=lambda axis: field.stride(axis), reverse=True)))
    if physical.is_contiguous():
        kernel = _flat_parallel if parallel else _flat_serial
        kernel(physical.view(-1).numpy(), factor)
        return field
    kernel = _parallel if parallel else _serial
    fastest = min(range(3), key=lambda axis: field.stride(axis + 1))
    kernel(field.numpy(), factor, fastest)
    return field
