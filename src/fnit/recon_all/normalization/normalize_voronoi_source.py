"""Experimental 26-neighbor Voronoi fill from FreeSurfer's gentle pass.

The control set and source image are matched first. This implements the
wavefront mean in mriBuildVoronoiDiagramFloat (d932c45), before smoothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_cdt
import torch


def voronoi_fill(source: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, dict]:
    if source.ndim != 3 or source.shape != control.shape:
        raise ValueError("source and control must have matching 3D shapes")
    started = time.perf_counter()
    source = np.asarray(source, dtype=np.float32)
    marked = np.asarray(control, dtype=bool)
    if not marked.any():
        raise ValueError("at least one control point is required")
    distance = distance_transform_cdt(~marked, metric="chessboard")
    distances = distance.ravel()
    sorted_indices = np.argsort(distances, kind="stable")
    boundaries = np.cumsum(np.bincount(distances))
    field = np.where(marked, source, np.float32(0)).astype(np.float32)
    sx, sy, sz = source.shape
    stride = sy * sz
    maximum = int(distance.max())
    for level in range(1, maximum + 1):
        linear = sorted_indices[boundaries[level - 1]:boundaries[level]]
        x = linear // stride
        y = (linear // sz) % sy
        z = linear % sz
        total = np.zeros(len(linear), dtype=np.float32)
        count = np.zeros(len(linear), dtype=np.int16)
        for dz in (-1, 0, 1):
            zi = np.clip(z + dz, 0, sz - 1)
            for dy in (-1, 0, 1):
                yi = np.clip(y + dy, 0, sy - 1)
                for dx in (-1, 0, 1):
                    xi = np.clip(x + dx, 0, sx - 1)
                    valid = distance[xi, yi, zi] < level
                    total += np.where(valid, field[xi, yi, zi], np.float32(0))
                    count += valid
        if np.any(count == 0):
            raise RuntimeError(f"unreachable voxels in wavefront level {level}")
        field[x, y, z] = total / count.astype(np.float32)
    return field, {"levels": maximum, "controls": int(marked.sum()),
                   "wall_seconds": time.perf_counter() - started}


def voronoi_fill_torch(source: torch.Tensor, control: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """在同网格按原层/邻域顺序传播float32；CUDA用融合内核减少分配。

    source/control为同shape三维张量，control非零为控制点；空控制抛ValueError。
    chessboard距离仍由既有SciPy计算，每层仅依赖之前层；保留边界重复邻居。
    返回同设备偏置图及levels/controls/wall_seconds，不改变TF32或空间。
    CPU仍用原Torch gathers；属于mri_normalize内部步骤，无独立CLI。
    """
    if source.ndim != 3 or source.shape != control.shape:
        raise ValueError("source and control must have matching 3D shapes")
    started = time.perf_counter()
    marked = control.bool()
    if not bool(marked.any()):
        raise ValueError("at least one control point is required")
    distance_cpu = distance_transform_cdt(~marked.cpu().numpy(), metric="chessboard")
    distances = distance_cpu.ravel()
    sorted_indices = np.argsort(distances, kind="stable")
    boundaries = np.cumsum(np.bincount(distances))
    distance = torch.as_tensor(distance_cpu, device=source.device)
    field = torch.where(marked, source.float(), 0)
    sx, sy, sz = source.shape
    stride = sy * sz
    maximum = int(distance_cpu.max())
    if source.is_cuda:
        from ._normalization_cuda import ordered_wavefront
        field = field.contiguous()
        ordered = torch.as_tensor(sorted_indices, device=source.device)
        field = ordered_wavefront(field, distance.contiguous(), ordered, boundaries)
    else:
        for level in range(1, maximum + 1):
            linear = torch.as_tensor(sorted_indices[boundaries[level - 1]:boundaries[level]],
                                     device=source.device)
            x, y, z = linear // stride, (linear // sz) % sy, linear % sz
            total = torch.zeros(len(linear), dtype=torch.float32, device=source.device)
            count = torch.zeros(len(linear), dtype=torch.int32, device=source.device)
            for dz in (-1, 0, 1):
                zi = (z + dz).clamp(0, sz - 1)
                for dy in (-1, 0, 1):
                    yi = (y + dy).clamp(0, sy - 1)
                    for dx in (-1, 0, 1):
                        xi = (x + dx).clamp(0, sx - 1)
                        total += field[xi, yi, zi]
                        count += (distance[xi, yi, zi] < level)
            field[x, y, z] = total / count.float()
    if source.is_cuda:
        torch.cuda.synchronize(source.device)
    return field, {"levels": maximum, "controls": int(marked.sum()),
                   "wall_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = nib.load(str(args.input))
    source = np.asarray(image.dataobj).astype(np.float32, copy=True)
    control = np.asarray(nib.load(str(args.control)).dataobj)
    result, details = voronoi_fill(source, control)
    nib.save(nib.MGHImage(result, image.affine), str(args.output))
    print(json.dumps(details))


if __name__ == "__main__":
    main()
