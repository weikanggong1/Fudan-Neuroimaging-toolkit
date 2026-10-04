"""Strict FP64 CPU normalization for repeated coefficient-field queries.

Only contiguous, non-differentiable CPU FP64 coordinates use this helper.
The original FP32 grid path, CUDA, autograd and other layouts retain their
existing tensor expressions. No dtype is narrowed and fastmath is disabled.
"""
from __future__ import annotations

import numpy as np
from numba import config, get_num_threads, njit, prange
import torch


def _normalize_float64_grid64(coordinates, shape, output):
    for index in prange(coordinates.shape[1]):
        for axis in range(3):
            if shape[axis] == 1:
                output[index, 2 - axis] = np.float64(0)
            else:
                value = np.float64(np.float64(2) * coordinates[axis, index])
                value = np.float64(value / np.float64(shape[axis] - 1))
                output[index, 2 - axis] = np.float64(value - np.float64(1))



_serial_float64_grid64 = njit(cache=True, fastmath=False, error_model="numpy")(_normalize_float64_grid64)
_parallel_float64_grid64 = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_normalize_float64_grid64)


def normalize_into(coordinates, shape, output):
    """Write FP64 from contiguous CPU views within the existing thread budget."""
    if coordinates.dtype != np.float64 or output.dtype != np.float64:
        raise ValueError("Fused float64 grid requires float64 coordinates and output")
    budget = torch.get_num_threads()
    threads = get_num_threads() if config.NUMBA_NUM_THREADS <= budget else 1
    kernel = _parallel_float64_grid64 if 1 < threads <= budget else _serial_float64_grid64
    kernel(coordinates, tuple(int(size) for size in shape), output)
