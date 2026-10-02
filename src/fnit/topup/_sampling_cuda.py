"""Fused float32 periodic cubic sampling for TOPUP's fixed image coefficients.

``sample_cubic_with_derivatives_cuda`` returns the sampled intensity, its three
voxel-coordinate derivatives, and the non-PE field-of-view validity mask. The
mask is supplied separately: TOPUP's caller owns its Jacobian and frame masks.
``sample_cubic_cuda`` exposes the same values with first-order autograd for the
coordinates only. Image spline coefficients are fixed inputs. The analytic
interface is also suitable for matrix-free solvers without second-order
PyTorch autograd. Inputs/outputs are float32; spline weights and accumulation
use double precision, following ``miscmaths-2203.2/splinterpolator.h`` order-3
sampling. No TF32 matrix product is used.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _periodic_index(lower, tap: tl.constexpr, size: tl.constexpr):
    # Triton/CUDA remainder truncates toward zero; torch.remainder uses the
    # positive divisor's sign. Correct a negative remainder explicitly.
    index = (lower - 1 + tap) % size
    return tl.where(index < 0, index + size, index)


@triton.jit
def _official_start(coordinate):
    # splinterpolator.h get_start_indicies for an order-3 kernel. Cast is
    # deliberately truncation toward zero rather than floor for negative PE.
    nearest = (coordinate + 0.5).to(tl.int64)
    return tl.where(nearest.to(tl.float64) < coordinate, nearest - 1, nearest - 2)


@triton.jit
def _official_weight(distance):
    absolute = tl.abs(distance)
    remainder = 2.0 - absolute
    inner = tl.full((), 2.0 / 3.0, tl.float64) + 0.5 * absolute * absolute * (absolute - 2.0)
    outer = tl.full((), 1.0 / 6.0, tl.float64) * (remainder * remainder * remainder)
    return tl.where(absolute < 1.0, inner, tl.where(absolute < 2.0, outer, 0.0))


@triton.jit
def _official_weight_derivative(distance):
    absolute = tl.abs(distance)
    sign = tl.where(distance < 0.0, -1.0, 1.0).to(tl.float64)
    remainder = 2.0 - absolute
    inner = sign * (1.5 * absolute * absolute - 2.0 * absolute)
    outer = sign * -0.5 * remainder * remainder
    return tl.where(absolute < 1.0, inner, tl.where(absolute < 2.0, outer, 0.0))


@triton.jit
def _official_cubic_sample_kernel(
    coefficients, coordinates, values, derivatives, validity,
    N: tl.constexpr, SIZE_X: tl.constexpr, SIZE_Y: tl.constexpr,
    SIZE_Z: tl.constexpr, PE_AXIS: tl.constexpr,
    WITH_DERIVATIVES: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    present = index < N
    x = tl.load(coordinates + index, present, 0.0).to(tl.float64)
    y = tl.load(coordinates + N + index, present, 0.0).to(tl.float64)
    z = tl.load(coordinates + 2 * N + index, present, 0.0).to(tl.float64)
    start_x, start_y, start_z = _official_start(x), _official_start(y), _official_start(z)
    valid = present
    if PE_AXIS != 0:
        valid = valid & (x + 1.0e-8 >= 0.0) & (x <= SIZE_X - 1 + 1.0e-8)
    if PE_AXIS != 1:
        valid = valid & (y + 1.0e-8 >= 0.0) & (y <= SIZE_Y - 1 + 1.0e-8)
    if PE_AXIS != 2:
        valid = valid & (z + 1.0e-8 >= 0.0) & (z <= SIZE_Z - 1 + 1.0e-8)
    value = tl.full((BLOCK,), 0.0, tl.float64)
    if WITH_DERIVATIVES:
        gradient_x = tl.full((BLOCK,), 0.0, tl.float64)
        gradient_y = tl.full((BLOCK,), 0.0, tl.float64)
        gradient_z = tl.full((BLOCK,), 0.0, tl.float64)
    # FSL's 3D value_and_derivatives_at loop is k(z), j(y), i(x).
    # The source groups the tap product as coefficient * wx * (wz * wy).
    for iz in tl.static_range(4):
        address_z = _periodic_index(start_z + 1, iz, SIZE_Z)
        distance_z = z - (start_z + iz).to(tl.float64)
        wz = _official_weight(distance_z)
        if WITH_DERIVATIVES:
            dwz = _official_weight_derivative(distance_z)
        for iy in tl.static_range(4):
            address_y = _periodic_index(start_y + 1, iy, SIZE_Y)
            distance_y = y - (start_y + iy).to(tl.float64)
            wy = _official_weight(distance_y)
            wzy = wz * wy
            if WITH_DERIVATIVES:
                dwy = _official_weight_derivative(distance_y)
                dwzy = dwz * wy
                wzdy = wz * dwy
            for ix in tl.static_range(4):
                address_x = _periodic_index(start_x + 1, ix, SIZE_X)
                distance_x = x - (start_x + ix).to(tl.float64)
                wx = _official_weight(distance_x)
                address = address_x * (SIZE_Y * SIZE_Z) + address_y * SIZE_Z + address_z
                coefficient = tl.load(coefficients + address, present, 0.0).to(tl.float64)
                coefficient_x = coefficient * wx
                value = value + coefficient_x * wzy
                if WITH_DERIVATIVES:
                    dwx = _official_weight_derivative(distance_x)
                    gradient_x = gradient_x + coefficient * dwx * wzy
                    gradient_y = gradient_y + coefficient_x * wzdy
                    gradient_z = gradient_z + coefficient_x * dwzy
    tl.store(values + index, value.to(tl.float32), present)
    tl.store(validity + index, valid, present)
    if WITH_DERIVATIVES:
        tl.store(derivatives + index, gradient_x.to(tl.float32), present)
        tl.store(derivatives + N + index, gradient_y.to(tl.float32), present)
        tl.store(derivatives + 2 * N + index, gradient_z.to(tl.float32), present)


def _validate(coefficients, coordinates, phase_encode_axis):
    if (coefficients.device.type != "cuda" or coordinates.device != coefficients.device):
        raise ValueError("TOPUP cubic sampling requires tensors on the same CUDA device")
    if coefficients.dtype != torch.float32 or coordinates.dtype != torch.float32:
        raise ValueError("TOPUP cubic sampling requires float32 coefficients and coordinates")
    if coefficients.ndim != 3 or min(coefficients.shape) < 1:
        raise ValueError("image coefficients must have nonempty shape [X, Y, Z]")
    if coordinates.ndim < 2 or coordinates.shape[0] != 3 or coordinates.numel() == 0:
        raise ValueError("coordinates must have nonempty shape [3, *grid]")
    if phase_encode_axis not in (0, 1, 2):
        raise ValueError("phase_encode_axis must be 0, 1 or 2")
    if coefficients.requires_grad:
        raise ValueError("image spline coefficients must be fixed inputs without gradients")


def _sample(coefficients, coordinates, phase_encode_axis, *, with_derivatives, official_precision=True):
    if official_precision is not True:
        raise ValueError("TOPUP sampling requires official_precision=True")
    _validate(coefficients, coordinates, phase_encode_axis)
    coefficients = coefficients.contiguous()
    coordinates = coordinates.contiguous()
    grid_shape = coordinates.shape[1:]
    count = coordinates.numel() // 3
    values = torch.empty(grid_shape, device=coordinates.device, dtype=torch.float32)
    valid = torch.empty(grid_shape, device=coordinates.device, dtype=torch.bool)
    derivative = torch.empty_like(coordinates) if with_derivatives else None
    with torch.cuda.device(coordinates.device):
        _official_cubic_sample_kernel[(triton.cdiv(count, 128),)](
            coefficients, coordinates, values,
            derivative if derivative is not None else values, valid,
            N=count, SIZE_X=coefficients.shape[0], SIZE_Y=coefficients.shape[1],
            SIZE_Z=coefficients.shape[2], PE_AXIS=int(phase_encode_axis),
            WITH_DERIVATIVES=with_derivatives, BLOCK=128, num_warps=4,
            enable_fp_fusion=False,
        )
    return values, derivative, valid


def sample_cubic_with_derivatives_cuda(coefficients, coordinates, phase_encode_axis, *, official_precision=True):
    """Return ``(values, spatial_gradient, valid)`` for fixed image coefficients.

    Inputs are CUDA float32 image coefficients ``[X,Y,Z]`` and finite voxel
    coordinates ``[3,*grid]``. Noncontiguous inputs are accepted. Outputs have
    shapes ``[*grid]``, ``[3,*grid]`` and ``[*grid]``; the last is boolean.
    Sampling wraps every axis periodically, even where ``valid`` is false.
    ``valid`` tests only non-PE axes and does not mask values or derivatives.
    Outputs expose explicit derivatives, without recording an autograd graph.
    Follows FSL Splinterpolator order-3 tap starts, piecewise weights, z/y/x
    loop order and double accumulation before final float32 casts. The
    ``official_precision=True`` keyword remains for existing callers; False
    is rejected because TOPUP uses this one official-precision algorithm.
    """
    return _sample(
        coefficients, coordinates, phase_encode_axis, with_derivatives=True,
        official_precision=official_precision,
    )


class _CoordinateCubicSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, coordinates, phase_encode_axis, official_precision):
        values, derivative, valid = _sample(
            coefficients, coordinates, phase_encode_axis, with_derivatives=True,
            official_precision=official_precision,
        )
        ctx.save_for_backward(derivative)
        ctx.mark_non_differentiable(valid)
        return values, valid

    @staticmethod
    @once_differentiable
    def backward(ctx, value_gradient, valid_gradient):
        (derivative,) = ctx.saved_tensors
        coordinate_gradient = derivative * value_gradient.unsqueeze(0)
        return None, coordinate_gradient, None, None


def sample_cubic_cuda(coefficients, coordinates, phase_encode_axis, *, official_precision=True):
    """Return ``(values, valid)`` with first-order coordinate-only autograd.

    Uses the same float32/periodic contract as the analytic interface. Image
    coefficients cannot require gradients. Higher derivatives are unavailable;
    use ``sample_cubic_with_derivatives_cuda`` for an explicit matrix-free solve.
    Uses FSL double weights/accumulation as described in the analytic interface;
    coordinates and returned tensors remain float32. ``official_precision``
    must remain True.
    """
    if coordinates.requires_grad:
        return _CoordinateCubicSample.apply(
            coefficients, coordinates, phase_encode_axis, official_precision
        )
    values, _, valid = _sample(
        coefficients, coordinates, phase_encode_axis, with_derivatives=False,
        official_precision=official_precision,
    )
    return values, valid


__all__ = ["sample_cubic_cuda", "sample_cubic_with_derivatives_cuda"]
