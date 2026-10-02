"""Optional fused float32 preparation of MCFLIRT's motion NCC inputs.

The reduction stays in :mod:`fnit.mcflirt.core`.  This sampler reuses the
rounded arithmetic helpers from TorchFLIRT, while keeping MCFLIRT's row
start and successive x-coordinate additions.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from ..flirt._batched_cuda import _add_f32, _sub_f32, _mul_f32, _div_f32


@triton.jit
def _minimum_f32(a, b):
    # torch.minimum propagates either NaN rather than choosing the finite
    # operand.  The explicit selection also applies to row bounds.
    return tl.where(a != a, a, tl.where(b != b, b, tl.minimum(a, b)))


@triton.jit
def _maximum_f32(a, b):
    return tl.where(a != a, a, tl.where(b != b, b, tl.maximum(a, b)))


@triton.jit
def _row_bounds(origin, direction, upper, xmin, xmax,
                NEAR_ZERO: tl.constexpr, REF_X: tl.constexpr):
    if NEAR_ZERO:
        outside = (origin < 0.0) | (origin > upper)
        xmin = tl.where(outside, float(REF_X), xmin)
    else:
        bound0 = _div_f32(-origin, direction)
        bound1 = _div_f32(_sub_f32(upper, origin), direction)
        lower = _minimum_f32(bound0, bound1)
        higher = _maximum_f32(bound0, bound1)
        xmin = _maximum_f32(xmin, tl.ceil(lower))
        xmax = _minimum_f32(xmax, tl.floor(higher))
    return xmin, xmax


@triton.jit
def _motion_sample_kernel(
        reference, moving, reference_values, moving_values, weights,
        c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23,
        upper_x, upper_y, upper_z, smooth_x, smooth_y, smooth_z,
        N: tl.constexpr, REF_X: tl.constexpr, REF_Y: tl.constexpr,
        MOV_X: tl.constexpr, MOV_Y: tl.constexpr, MOV_Z: tl.constexpr,
        NEAR_ZERO_X: tl.constexpr, NEAR_ZERO_Y: tl.constexpr,
        NEAR_ZERO_Z: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    present = index < N
    offset_x = index % REF_X
    reference_y = ((index // REF_X) % REF_Y).to(tl.float32)
    reference_z = (index // (REF_X * REF_Y)).to(tl.float32)

    origin_x = _add_f32(_add_f32(_mul_f32(reference_y, c01),
                                _mul_f32(reference_z, c02)), c03)
    origin_y = _add_f32(_add_f32(_mul_f32(reference_y, c11),
                                _mul_f32(reference_z, c12)), c13)
    origin_z = _add_f32(_add_f32(_mul_f32(reference_y, c21),
                                _mul_f32(reference_z, c22)), c23)
    xmin = tl.full((BLOCK,), 0.0, tl.float32)
    xmax = tl.full((BLOCK,), float(REF_X - 1), tl.float32)
    xmin, xmax = _row_bounds(origin_x, c00, upper_x, xmin, xmax,
                             NEAR_ZERO_X, REF_X)
    xmin, xmax = _row_bounds(origin_y, c10, upper_y, xmin, xmax,
                             NEAR_ZERO_Y, REF_X)
    xmin, xmax = _row_bounds(origin_z, c20, upper_z, xmin, xmax,
                             NEAR_ZERO_Z, REF_X)
    xmin = _minimum_f32(_maximum_f32(xmin, 0.0), float(REF_X))

    x = _add_f32(origin_x, _mul_f32(xmin, c00))
    y = _add_f32(origin_y, _mul_f32(xmin, c10))
    z = _add_f32(origin_z, _mul_f32(xmin, c20))
    # A direct origin + actual_x * direction expression loses the source's
    # float32 recurrence.  Each lane keeps exactly offset_x additions.
    for step in range(1, REF_X):
        x = tl.where(offset_x >= step, _add_f32(x, c00), x)
        y = tl.where(offset_x >= step, _add_f32(y, c10), y)
        z = tl.where(offset_x >= step, _add_f32(z, c20), z)
    actual_x = _add_f32(xmin, offset_x.to(tl.float32))
    valid = ((actual_x <= xmax) &
             (x >= 0.0) & (x <= upper_x) &
             (y >= 0.0) & (y <= upper_y) &
             (z >= 0.0) & (z <= upper_z))

    sx = _minimum_f32(_maximum_f32(x, 0.0), upper_x)
    sy = _minimum_f32(_maximum_f32(y, 0.0), upper_y)
    sz = _minimum_f32(_maximum_f32(z, 0.0), upper_z)
    ix = tl.maximum(tl.minimum(tl.floor(sx).to(tl.int32), MOV_X - 2), 0)
    iy = tl.maximum(tl.minimum(tl.floor(sy).to(tl.int32), MOV_Y - 2), 0)
    iz = tl.maximum(tl.minimum(tl.floor(sz).to(tl.int32), MOV_Z - 2), 0)
    dx = _sub_f32(sx, ix.to(tl.float32))
    dy = _sub_f32(sy, iy.to(tl.float32))
    dz = _sub_f32(sz, iz.to(tl.float32))
    stride_x, stride_y = MOV_Y * MOV_Z, MOV_Z
    moving_offset = ix * stride_x + iy * stride_y + iz
    v000 = tl.load(moving + moving_offset, present, 0.0)
    v001 = tl.load(moving + moving_offset + 1, present, 0.0)
    v010 = tl.load(moving + moving_offset + stride_y, present, 0.0)
    v011 = tl.load(moving + moving_offset + stride_y + 1, present, 0.0)
    v100 = tl.load(moving + moving_offset + stride_x, present, 0.0)
    v101 = tl.load(moving + moving_offset + stride_x + 1, present, 0.0)
    v110 = tl.load(moving + moving_offset + stride_x + stride_y, present, 0.0)
    v111 = tl.load(moving + moving_offset + stride_x + stride_y + 1, present, 0.0)
    t1 = _add_f32(_mul_f32(_sub_f32(v100, v000), dx), v000)
    t2 = _add_f32(_mul_f32(_sub_f32(v101, v001), dx), v001)
    t3 = _add_f32(_mul_f32(_sub_f32(v110, v010), dx), v010)
    t4 = _add_f32(_mul_f32(_sub_f32(v111, v011), dx), v011)
    t5 = _add_f32(_mul_f32(_sub_f32(t3, t1), dy), t1)
    t6 = _add_f32(_mul_f32(_sub_f32(t4, t2), dy), t2)
    value = _add_f32(_mul_f32(_sub_f32(t6, t5), dz), t5)

    far_x = _sub_f32(upper_x, x)
    far_y = _sub_f32(upper_y, y)
    far_z = _sub_f32(upper_z, z)
    wx = tl.where(x < smooth_x, _div_f32(x, smooth_x),
                  tl.where(far_x < smooth_x, _div_f32(far_x, smooth_x), 1.0))
    wy = tl.where(y < smooth_y, _div_f32(y, smooth_y),
                  tl.where(far_y < smooth_y, _div_f32(far_y, smooth_y), 1.0))
    wz = tl.where(z < smooth_z, _div_f32(z, smooth_z),
                  tl.where(far_z < smooth_z, _div_f32(far_z, smooth_z), 1.0))
    weight = _mul_f32(_mul_f32(wx, wy), wz)
    weight = tl.where(weight <= 0.0, 0.0, weight)
    weight = _mul_f32(weight, valid.to(tl.float32))

    reference_x = tl.minimum(actual_x.to(tl.int32), REF_X - 1)
    reference_offset = (index // REF_X) * REF_X + reference_x
    reference_value = tl.load(reference + reference_offset, present, 0.0)
    tl.store(reference_values + index, reference_value, present)
    tl.store(moving_values + index, value, present)
    tl.store(weights + index, weight, present)


class FusedMotionSampler:
    """Reuse three ZYX float32 buffers for one motion-cost object's calls.

    ``reference_zyx`` is the contiguous transposed reference; ``moving``
    keeps the original contiguous XYZ layout. ``prepare`` receives the
    same CPU float32 3-by-4 pull coefficients as the tensor path and returns
    ``(reference_values, moving_values, weights)`` in reference ZYX shape.
    The returned buffers are overwritten by the next call.
    """

    def __init__(self, reference_zyx, moving_contiguous, moving_sizes):
        if (reference_zyx.device.type != "cuda" or
                moving_contiguous.device != reference_zyx.device):
            raise ValueError("motion sampler tensors must share a CUDA device")
        if (reference_zyx.ndim != 3 or moving_contiguous.ndim != 3 or
                reference_zyx.dtype != torch.float32 or
                moving_contiguous.dtype != torch.float32 or
                not reference_zyx.is_contiguous() or
                not moving_contiguous.is_contiguous()):
            raise ValueError("motion sampler requires contiguous float32 3D tensors")
        if min(reference_zyx.shape) < 1 or min(moving_contiguous.shape) < 2:
            raise ValueError("motion sampler requires nonempty reference and moving axes >= 2")
        if len(moving_sizes) != 3 or any(not np.isfinite(v) or v <= 0 for v in moving_sizes):
            raise ValueError("moving_sizes must contain three positive finite voxel sizes")
        self.reference = reference_zyx
        self.moving = moving_contiguous
        self.reference_values = torch.empty_like(reference_zyx)
        self.moving_values = torch.empty_like(reference_zyx)
        self.weights = torch.empty_like(reference_zyx)
        self.upper = tuple(float(np.float32(size - 1.0001))
                           for size in moving_contiguous.shape)
        self.smooth = tuple(float(np.float32(1.0 / float(size))) for size in moving_sizes)

    def prepare(self, coefficients_np_float32):
        coefficients = np.asarray(coefficients_np_float32, dtype=np.float32)
        if coefficients.shape != (3, 4):
            raise ValueError("motion pull coefficients must have shape (3, 4)")
        near_zero = tuple(abs(float(value)) < 1e-8 for value in coefficients[:, 0])
        scalars = tuple(float(value) for value in coefficients.reshape(-1))
        ref_z, ref_y, ref_x = self.reference.shape
        mov_x, mov_y, mov_z = self.moving.shape
        count = ref_z * ref_y * ref_x
        block = 256
        _motion_sample_kernel[(triton.cdiv(count, block),)](
            self.reference, self.moving, self.reference_values, self.moving_values,
            self.weights, *scalars, *self.upper, *self.smooth,
            N=count, REF_X=ref_x, REF_Y=ref_y, MOV_X=mov_x, MOV_Y=mov_y,
            MOV_Z=mov_z, NEAR_ZERO_X=near_zero[0], NEAR_ZERO_Y=near_zero[1],
            NEAR_ZERO_Z=near_zero[2], BLOCK=block, num_warps=4,
            enable_fp_fusion=False)
        return self.reference_values, self.moving_values, self.weights
