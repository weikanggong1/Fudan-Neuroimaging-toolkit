"""Isolated full-stage adapter for the real-mesh persistent OpenMP pilot.

Set FNIT_AVERAGE_BASE_SOURCE and FNIT_AVERAGE_PILOT_LIBRARY explicitly.
The accepted production module is loaded unchanged under a private name. Only
its CPU NumPy averaging kernel is replaced; this is not a product backend.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
from numba import get_num_threads

source = Path(os.environ["FNIT_AVERAGE_BASE_SOURCE"])
name = "fnit.recon_all._persistent_average_pilot_base"
spec = importlib.util.spec_from_file_location(name, source)
base = importlib.util.module_from_spec(spec)
sys.modules[name] = base
spec.loader.exec_module(base)
library = ctypes.CDLL(str(Path(os.environ["FNIT_AVERAGE_PILOT_LIBRARY"]).resolve()))
function = library.fnit_average_persistent
function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3 + [ctypes.c_int, ctypes.c_void_p]
function.restype = ctypes.c_int


def average_numpy(gradient, neighbors, degrees, reciprocals, iterations):
    arrays = [np.ascontiguousarray(value) for value in (gradient, neighbors, degrees, reciprocals)]
    output = np.empty_like(arrays[0])
    error = function(*(value.ctypes.data for value in arrays), len(degrees),
                     neighbors.shape[1], iterations, get_num_threads(), output.ctypes.data)
    if error:
        raise RuntimeError("diagnostic C++ averaging status " + str(error))
    return output


base._average_numpy = average_numpy
three_hop_neighbor_total = base.three_hop_neighbor_total
average_gradients_exact_cpu = base.average_gradients_exact_cpu
RegistrationGradientAverager = base.RegistrationGradientAverager
