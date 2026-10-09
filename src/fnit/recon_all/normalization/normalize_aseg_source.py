"""Initial inputs for the ``mri_normalize -aseg -mask`` branch."""

from __future__ import annotations

import numpy as np
from numba import njit

from .normalize_gaussian_source import smooth_bias
from .normalize_tissue_peaks import _smooth
from .normalize_voronoi_source import voronoi_fill


def prepare_aseg_source(
    norm: np.ndarray, brainmask: np.ndarray, aseg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the masked float32 source and the initial 2/41 WM controls.

    Arrays must already share a voxel grid. The frozen ``fs_sub01`` inputs do;
    native ``MRIresample`` takes an identity path for its type-only mismatch.
    """
    if norm.shape != brainmask.shape or norm.shape != aseg.shape:
        raise ValueError("norm, brainmask, and aseg must share a voxel grid")
    masked = np.where(brainmask == 0, 0, norm).astype(np.float32)
    controls = np.where((aseg == 2) | (aseg == 41), aseg, 0).astype(np.int32)
    return masked, controls


@njit(cache=True)
def _filter_ridge(source: np.ndarray, controls: np.ndarray, threshold: int) -> np.ndarray:
    removed = np.zeros(controls.shape, np.uint8)
    width, height, depth = controls.shape
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if controls[x, y, z] == 0:
                    continue
                maximum = 0.0
                for xx in range(max(0, x - 5), min(width, x + 6)):
                    for yy in range(max(0, y - 5), min(height, y + 6)):
                        for zz in range(max(0, z - 5), min(depth, z + 6)):
                            if controls[xx, yy, zz] and source[xx, yy, zz] > maximum:
                                maximum = source[xx, yy, zz]
                value = source[x, y, z]
                if value + 15 < maximum and value < threshold:
                    controls[x, y, z] = 0
                    removed[x, y, z] = 128
    return removed


def filter_aseg_ridge(source: np.ndarray, ridge: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply native scan-order WM outlier removal to a medial-ridge mask.

    The distance transform and ridge selection are a separate preceding stage.
    """
    source = np.ascontiguousarray(source, dtype=np.float32)
    controls = np.ascontiguousarray(ridge != 0, dtype=np.uint8)
    if source.shape != controls.shape:
        raise ValueError("source and ridge must share a voxel grid")
    values = source[controls != 0]
    minimum, maximum = np.float32(values.min()), np.float32(values.max())
    step = np.float32((maximum - minimum) / np.float32(255))
    bins = minimum + step * np.arange(256, dtype=np.float32)
    index = np.floor((values - minimum) / step + np.float32(0.5)).astype(np.int32)
    counts = np.bincount(np.clip(index, 0, 255), minlength=256).astype(np.float32)
    peak = int(bins[int(np.argmax(_smooth(counts)[1:]) + 1)])
    # The active -mprage option sets intensity_below to 15.
    removed = _filter_ridge(source, controls, peak - 15)
    return controls, removed, peak


def apply_initial_aseg_bias(source: np.ndarray, controls: np.ndarray, *,
                            backend: str = "cpu", device: str | None = None) -> np.ndarray:
    """按 aseg 初始规则传播、sigma-8平滑并校正同网格强度。

    source为三维(x,y,z)NumPy强度，转float32；controls为同shape非零控制点图。
    网格由调用者保证，sigma单位体素，不改变RAS。backend默认cpu，完整保留
    原NumPy/Numba路径；torch复用已有Voronoi/高斯GPU实现，要求显式CUDA设备。
    平滑传全零控制图，不恢复控制点强度；最后先float64除法再乘法，转float32。
    返回同shape CPU NumPy float32图，含同步下载，不修改source/controls或TF32。
    后端/设备无效、形状不符、空控制集抛异常；CUDA错误向上传播，不自动回退。
    属于mri_normalize -aseg的内部步骤，无独立原软件CLI。
    """
    if backend not in {"cpu", "torch"}:
        raise ValueError("initial bias backend must be cpu or torch")
    source = np.asarray(source, dtype=np.float32)
    if backend == "torch":
        import torch
        from .normalize_gaussian_source import smooth_bias_torch
        from .normalize_voronoi_source import voronoi_fill_torch
        if device is None:
            raise ValueError("torch initial bias requires an explicit CUDA device")
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            raise ValueError("torch initial bias requires an explicit CUDA device")
        # 各既有内核按目标CUDA stream执行，退出后恢复父API当前设备。
        with torch.cuda.device(target):
            image = torch.as_tensor(np.ascontiguousarray(source), device=target)
            selected = torch.as_tensor(np.ascontiguousarray(controls), device=target)
            filled, _ = voronoi_fill_torch(source=image, control=selected)
            bias, _ = smooth_bias_torch(voronoi=filled, source=image,
                control=torch.zeros_like(selected), sigma=8.0)
            corrected = (image.double() * (110.0 / bias.double())).float()
            return corrected.cpu().numpy()
    voronoi, _ = voronoi_fill(source, controls)
    bias, _ = smooth_bias(voronoi, source, np.zeros_like(controls), strict=True)
    return np.float32(source.astype(np.float64) * (110.0 / bias.astype(np.float64)))
