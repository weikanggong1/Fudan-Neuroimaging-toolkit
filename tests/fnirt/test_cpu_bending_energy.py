"""Preserve dense energy reductions; numerical fixtures are not benchmarks."""
import pytest
import torch

from fnit.fnirt.spline import BendingOperator, fsl_control_shape
from fnit.fnirt._bending_cpu import multiply_square_


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("multiplier", [1.0, 2.0 ** 0.5])
@pytest.mark.parametrize("fastest", [0, 1, 2])
def test_fused_elementwise_keeps_every_product_bit(threads, dtype, multiplier, fastest):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        values = torch.randn((3, 5, 7, 9), dtype=dtype, generator=torch.Generator().manual_seed(29))
        tiny = torch.finfo(dtype).tiny
        values.reshape(-1)[:8] = torch.tensor([0.0, -0.0, tiny, -tiny, tiny / 2, -tiny / 2, 1e-10, -1e-10], dtype=dtype)
        order = (0,) + tuple(axis + 1 for axis in range(3) if axis != fastest) + (fastest + 1,)
        inverse_order = tuple(order.index(axis) for axis in range(4))
        actual = values.permute(order).contiguous().permute(inverse_order)
        expected = actual.clone(memory_format=torch.preserve_format)
        expected.mul_(multiplier).square_()
        stride = actual.stride()
        result = multiply_square_(actual, multiplier)
        integer = torch.int32 if dtype == torch.float32 else torch.int64
        assert result is actual
        assert torch.equal(actual.view(integer), expected.view(integer))
        assert actual.stride() == stride
        assert torch.get_num_threads() == threads
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_elementwise_supports_non_dense_view_without_copy(dtype):
    backing = torch.randn((3, 5, 7, 18), dtype=dtype, generator=torch.Generator().manual_seed(31))
    field = backing[..., ::2]
    untouched = backing[..., 1::2].clone()
    expected = field.clone().mul_(2.0 ** 0.5).square_()
    multiply_square_(field, 2.0 ** 0.5)
    integer = torch.int32 if dtype == torch.float32 else torch.int64
    assert torch.equal(field.view(integer), expected.view(integer))
    assert torch.equal(backing[..., 1::2], untouched)


def test_parallel_fusion_matches_torch_in_isolated_budget():
    import os
    from pathlib import Path
    import subprocess
    import sys
    environment = dict(os.environ)
    environment.update(NUMBA_NUM_THREADS="8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    source = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = str(source)
    code = """
import torch
from numba import get_num_threads
from fnit.fnirt._bending_cpu import multiply_square_
torch.set_num_threads(8)
assert get_num_threads() == 8
for dtype in (torch.float32, torch.float64):
    for layout in (False, True):
        field = torch.randn((3, 19, 23, 29), dtype=dtype, generator=torch.Generator().manual_seed(61))
        if layout:
            field = field.permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)
        expected = field.clone(memory_format=torch.preserve_format).mul_(2.0 ** 0.5).square_()
        multiply_square_(field, 2.0 ** 0.5)
        integer = torch.int32 if dtype == torch.float32 else torch.int64
        assert torch.equal(field.view(integer), expected.view(integer))
assert torch.get_num_threads() == get_num_threads() == 8
"""
    subprocess.run([sys.executable, "-c", code], env=environment, check=True, timeout=90)


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shape,spacing", [((7, 9, 11), (1, 2, 3)), ((9, 13, 11), (8, 8, 8))])
def test_streamed_energy_keeps_dense_bits_and_coefficients(threads, dtype, channels, shape, spacing):
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        operator = BendingOperator(shape, spacing, (1.0, 1.5, 2.0), device="cpu", dtype=dtype, execution="optimized")
        coefficients = torch.randn((channels, *fsl_control_shape(shape, spacing)), dtype=dtype, generator=torch.Generator().manual_seed(57))
        before = coefficients.clone()
        expected = sum((field.square().sum() for field in operator.forward(coefficients)), coefficients.new_zeros(()))
        actual = operator.energy(coefficients)
        integer_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))
        assert torch.equal(coefficients, before)
        assert torch.get_num_threads() == threads
    finally:
        torch.set_num_threads(prior_threads)


def test_differentiable_energy_retains_reference_graph():
    shape, spacing = (7, 9, 11), (2, 3, 2)
    operator = BendingOperator(shape, spacing, (1.0, 1.5, 2.0), device="cpu", dtype=torch.float64, execution="optimized")
    coefficients = torch.randn((1, *fsl_control_shape(shape, spacing)), dtype=torch.float64, generator=torch.Generator().manual_seed(93), requires_grad=True)
    expected = sum((field.square().sum() for field in operator.forward(coefficients)), coefficients.new_zeros(()))
    expected_gradient = torch.autograd.grad(expected, coefficients)[0]
    actual = operator.energy(coefficients)
    actual_gradient = torch.autograd.grad(actual, coefficients)[0]
    assert torch.equal(actual, expected)
    assert torch.equal(actual_gradient, expected_gradient)


def test_reference_execution_keeps_forward_contract(monkeypatch):
    operator = BendingOperator((3, 4, 5), (1, 1, 1), (1, 1, 1), device="cpu", dtype=torch.float64, execution="reference")
    sentinel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    monkeypatch.setattr(operator, "forward", lambda coefficients: (sentinel,))
    assert operator.energy(torch.zeros((1, 3, 4, 5), dtype=torch.float64)) == 14.0
