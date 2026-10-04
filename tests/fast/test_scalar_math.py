"""C scalar precision matters at discrete FAST PVE decision boundaries."""

import ctypes
from ctypes.util import find_library

import numpy as np
import pytest
import torch

from fnit.fast import _fsl_math
from fnit.fast._fsl_cpu import thread_budget


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["exp", "log"])
def test_scalar_math_preserves_c_precision_strides_and_input(dtype, name):
    values = torch.tensor([[0.03125, 0.10000000149011612, 0.5, 1.0],
                           [1.125, 1.9999998807907104, 4.25, 8.125]], dtype=dtype).T
    original = values.clone()
    library = ctypes.CDLL(find_library("m"))
    scalar = getattr(library, name + ("f" if dtype == torch.float32 else ""))
    ctype = ctypes.c_float if dtype == torch.float32 else ctypes.c_double
    scalar.argtypes, scalar.restype = [ctype], ctype
    expected = np.asarray([scalar(float(v)) for v in values.flatten()],
                          dtype=values.numpy().dtype).reshape(values.shape)
    with thread_budget(1):
        first = getattr(_fsl_math, name)(values)
    with thread_budget(4):
        second = getattr(_fsl_math, name)(values)
    np.testing.assert_array_equal(first.numpy(), expected)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert first.dtype == dtype and first.shape == values.shape


def test_cpu_math_does_not_replace_global_torch_operations():
    original_exp, original_log = torch.exp, torch.log
    _fsl_math.exp(torch.ones(2))
    _fsl_math.log(torch.ones(2))
    assert torch.exp is original_exp and torch.log is original_log
