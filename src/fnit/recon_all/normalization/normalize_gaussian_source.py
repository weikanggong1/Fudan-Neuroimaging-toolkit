"""Experimental FreeSurfer sigma-8 gentle bias smoothing and application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import convolve1d
import torch
from numba import njit, prange


def fs_gaussian_kernel(sigma: float) -> np.ndarray:
    length = int(8 * sigma + 0.5) + 1
    if length % 2 == 0:
        length += 1
    half = length // 2
    result = np.zeros(length, dtype=np.float32)
    norm = np.float32(0)
    for index, offset in enumerate(range(-half, half + 1)):
        absolute = abs(float(offset))
        if absolute <= 2 * sigma:
            weight = np.float32(np.exp(-offset * offset / (2 * sigma * sigma)))
        elif absolute <= 4 * sigma:
            weight = np.float32((4 - absolute / sigma) ** 4 / (16 * np.e ** 2))
        else:
            weight = np.float32(0)
        result[index] = weight
        norm = np.float32(norm + weight)
    result /= norm
    return result


@njit(parallel=True)
def _convolve_axis_strict(source: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    sx, sy, sz = source.shape
    result = np.empty_like(source)
    half = len(kernel) // 2
    for z in prange(sz):
        for y in range(sy):
            for x in range(sx):
                total = np.float32(0)
                for offset in range(len(kernel)):
                    shift = offset - half
                    if axis == 0:
                        xi = min(sx - 1, max(0, x + shift))
                        value = source[xi, y, z]
                    elif axis == 1:
                        yi = min(sy - 1, max(0, y + shift))
                        value = source[x, yi, z]
                    else:
                        zi = min(sz - 1, max(0, z + shift))
                        value = source[x, y, zi]
                    total = np.float32(total + np.float32(kernel[offset] * value))
                result[x, y, z] = total
    return result


def smooth_bias(voronoi: np.ndarray, source: np.ndarray, control: np.ndarray,
                sigma: float = 8.0, strict: bool = True) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    if voronoi.shape != source.shape or source.shape != control.shape:
        raise ValueError("input shapes differ")
    bias = np.asarray(voronoi, dtype=np.float32)
    kernel = fs_gaussian_kernel(sigma)
    for axis in (0, 1, 2):
        bias = (_convolve_axis_strict(bias, kernel, axis) if strict
                else convolve1d(bias, kernel, axis=axis, mode="nearest"))
    bias[control > 0] = source[control > 0]
    return bias, {"sigma": sigma, "kernel_length": len(kernel), "strict": strict,
                  "wall_seconds": time.perf_counter() - started}


def smooth_bias_torch(voronoi: torch.Tensor, source: torch.Tensor,
                      control: torch.Tensor, sigma: float = 8.0) -> tuple[torch.Tensor, dict]:
    """三轴按固定核顺序平滑偏置，并恢复控制点原强度。

    voronoi/source/control为同shape三维张量；sigma=8，单位体素。
    返回同设备float32偏置及核长度/设备/秒数；CUDA复用两缓冲，不改算术顺序。
    nearest边界与CPU接口一致，shape不符抛ValueError；属于mri_normalize内部步骤。
    """
    if voronoi.shape != source.shape or source.shape != control.shape:
        raise ValueError("input shapes differ")
    started = time.perf_counter()
    kernel = torch.as_tensor(fs_gaussian_kernel(sigma), device=source.device)
    if source.is_cuda:
        from ._normalization_cuda import ordered_smoothing
        bias = ordered_smoothing(voronoi, kernel)
    else:
        bias = voronoi.float()
        for axis in (0, 1, 2):
            length = bias.shape[axis]
            index = torch.arange(length, device=bias.device)
            result = torch.zeros_like(bias)
            for offset in range(len(kernel)):
                neighbor = (index + offset - len(kernel) // 2).clamp(0, length - 1)
                result += bias.index_select(axis, neighbor) * kernel[offset]
            bias = result
    bias[control > 0] = source[control > 0].float()
    if source.is_cuda:
        torch.cuda.synchronize(source.device)
    return bias, {"sigma": sigma, "kernel_length": len(kernel), "device": str(source.device),
                  "wall_seconds": time.perf_counter() - started}


def apply_gentle_bias_float(source: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if source.shape != bias.shape:
        raise ValueError("source and bias shapes differ")
    values = source.trunc()
    integer_bias = bias.trunc()
    return torch.where(integer_bias != 0, values * 110.0 / integer_bias, values)


def apply_gentle_bias(source: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    corrected = apply_gentle_bias_float(source, bias)
    return torch.floor(corrected + 0.5).clamp(0, 255).to(torch.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--voronoi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = nib.load(str(args.source))
    source = np.asarray(image.dataobj).astype(np.float32, copy=True)
    control = np.asarray(nib.load(str(args.control)).dataobj)
    voronoi = np.asarray(nib.load(str(args.voronoi)).dataobj).astype(np.float32, copy=True)
    result, details = smooth_bias(voronoi, source, control)
    nib.save(nib.MGHImage(result, image.affine), str(args.output))
    print(json.dumps(details))


if __name__ == "__main__":
    main()
