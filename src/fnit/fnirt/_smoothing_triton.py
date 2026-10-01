"""GPU Gaussian passes with the offset-serial FSL float assignment order."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gaussian_axis(
    source, output, weights,
    elements: tl.constexpr, axis_size: tl.constexpr, axis_stride: tl.constexpr,
    radius: tl.constexpr, block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < elements
    coordinate = (index // axis_stride) % axis_size
    accumulated = tl.full((block,), 0.0, tl.float32)
    for offset in tl.static_range(-radius, radius + 1):
        inside = valid & (coordinate + offset >= 0) & (coordinate + offset < axis_size)
        value = tl.load(source + index + offset * axis_stride, mask=inside, other=0)
        weight = tl.load(weights + offset + radius)
        # newimage convolve assigns every double multiply/add back to float.
        # Skipped out-of-FOV offsets retain the previous float, including -0.
        product = value.to(tl.float64) * weight
        next_value = (accumulated.to(tl.float64) + product).to(tl.float32)
        accumulated = tl.where(inside, next_value, accumulated)
    tl.store(output + index, accumulated, mask=valid)


def gaussian_blur_cuda(volume, kernels):
    if not volume.is_cuda or volume.dtype != torch.float32:
        raise ValueError("FNIRT fused smoothing requires a CUDA float32 volume")
    result = volume.contiguous()
    for axis, kernel in enumerate(kernels):
        dimension = axis + 2
        axis_stride = result.stride(dimension)
        weights = torch.tensor(kernel, dtype=torch.float64, device=result.device)
        output = torch.empty_like(result)
        _gaussian_axis[(triton.cdiv(result.numel(), 128),)](
            result, output, weights, result.numel(), result.shape[dimension],
            axis_stride, len(kernel) // 2, 128,
            enable_fp_fusion=False,
        )
        result = output
    return result
