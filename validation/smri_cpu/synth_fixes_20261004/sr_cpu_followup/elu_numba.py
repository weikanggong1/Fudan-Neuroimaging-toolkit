"""Diagnostic independent FP32 Eigen ELU; no fastmath, no production import."""
import platform
from pathlib import Path

if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
    raise RuntimeError('this diagnostic packet implementation requires Linux x86-64; production never imports it')
flags = Path('/proc/cpuinfo').read_text().split('flags', 1)[1].split('\n', 1)[0].split()
if 'sse2' not in flags:
    raise RuntimeError('SSE2 is required for this diagnostic packet implementation')

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False, cache=True)
def _elu_flat(source, result):
    for i in prange(source.size):
        value = source[i]
        if not value < np.float32(0):
            result[i] = value
        elif value < np.float32(-104):
            result[i] = np.float32(-1)
        else:
            power = np.floor(value * np.float32(1.44269504088896341) + np.float32(.5))
            remainder = power * np.float32(-.693359375) + value
            remainder = power * np.float32(2.12194440e-4) + remainder
            square = remainder * remainder
            even = square * np.float32(1.37449637986719608306884765625e-3) + np.float32(4.166965186595916748046875e-2)
            odd = square * np.float32(8.36894474923610687255859375e-3) + np.float32(.16666518151760101318359375)
            even = square * even + np.float32(.49999988079071044921875)
            exponential = square * (remainder * odd + even) + (remainder + np.float32(1))
            exponential = np.ldexp(exponential, np.int32(power))
            result[i] = exponential - np.float32(1)


def elu_numpy(source):
    """Return a new contiguous FP32 array using strict scalar operation order."""
    source = np.ascontiguousarray(source, dtype=np.float32)
    result = np.empty_like(source)
    _elu_flat(source.reshape(-1), result.reshape(-1))
    return result


def elu_torch(value):
    """Diagnostic only: preserve a CPU FP32 tensor's contiguous/CL3D format."""
    import torch
    if value.device.type != 'cpu' or value.dtype != torch.float32 or value.requires_grad:
        raise ValueError('diagnostic ELU requires CPU FP32 inference without autograd')
    result = torch.empty_like(value)
    if value.is_contiguous(memory_format=torch.channels_last_3d):
        source = value.permute(0, 2, 3, 4, 1).numpy().reshape(-1)
        output = result.permute(0, 2, 3, 4, 1).numpy().reshape(-1)
    elif value.is_contiguous():
        source = value.numpy().reshape(-1)
        output = result.numpy().reshape(-1)
    else:
        raise ValueError('diagnostic ELU needs a contiguous or channels-last 3D tensor')
    _elu_flat(source, output)
    return result
