"""Scalar C math for the ordered FAST CPU path.

Torch's vector exp/log can differ by one float32 ULP from expf/logf.
Those differences can change FAST's discrete partial-volume grid choice.
Only independent elements are parallelized; CUDA uses the existing Torch
operations. No FSL executable or FSL shared library is loaded.
"""

import ctypes
from ctypes.util import find_library
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=1)
def _kernels():
    # CUDA never needs the CPU compiler or its import/startup cost.
    from numba import njit, prange

    # libm is supplied by the platform C runtime, not by an imaging package.
    library_name = find_library("m")
    library = ctypes.CDLL(library_name) if library_name else ctypes.CDLL(None)
    try:
        exp32, exp64 = library.expf, library.exp
        log32, log64 = library.logf, library.log
    except AttributeError as error:
        raise RuntimeError("FAST ordered CPU math requires C expf/logf/exp/log symbols") from error
    for function in (exp32, log32):
        function.argtypes, function.restype = [ctypes.c_float], ctypes.c_float
    for function in (exp64, log64):
        function.argtypes, function.restype = [ctypes.c_double], ctypes.c_double

    # ctypes function pointers depend on this process, so no disk JIT cache.
    @njit(parallel=True, fastmath=False)
    def evaluate32(values, logarithm):
        output = np.empty_like(values)
        for index in prange(values.size):
            output[index] = log32(values[index]) if logarithm else exp32(values[index])
        return output

    @njit(parallel=True, fastmath=False)
    def evaluate64(values, logarithm):
        output = np.empty_like(values)
        for index in prange(values.size):
            output[index] = log64(values[index]) if logarithm else exp64(values[index])
        return output

    return evaluate32, evaluate64


def _evaluate_cpu(tensor, logarithm):
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError("FAST ordered CPU math requires float32 or float64")
    if tensor.requires_grad:
        raise ValueError("FAST ordered CPU math is an inference-only operation")
    values = tensor.contiguous().numpy().reshape(-1)
    evaluate32, evaluate64 = _kernels()
    function = evaluate32 if tensor.dtype == torch.float32 else evaluate64
    output = function(values, logarithm)
    return torch.from_numpy(output.reshape(tensor.shape))


def exp(tensor):
    if tensor.is_cuda:
        return torch.exp(tensor)
    return _evaluate_cpu(tensor, False)


def log(tensor):
    if tensor.is_cuda:
        return torch.log(tensor)
    return _evaluate_cpu(tensor, True)
