"""PyTorch preprocessing for the non-robust SynthSeg 2.0 T1 pathway."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class PreprocessedT1:
    image: torch.Tensor
    input_affine: np.ndarray
    aligned_affine: np.ndarray
    original_shape: tuple[int, int, int]
    content_slices: tuple[slice, slice, slice]
    volume_affine: np.ndarray
    voxel_volume_mm3: float


def _smooth_downsample(image: torch.Tensor, sigmas: np.ndarray) -> torch.Tensor:
    volume = image[None, None]
    for axis, sigma in enumerate(sigmas):
        if sigma <= 0:
            continue
        radius = int(4 * sigma + 0.5)
        location = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
        kernel = torch.exp(-0.5 * (location / sigma).square())
        kernel /= kernel.sum()
        size = image.shape[axis]
        index = torch.arange(-radius, size + radius, device=image.device)
        index = index.remainder(2 * size)
        index = torch.where(index < size, index, 2 * size - index - 1)
        volume = volume.index_select(axis + 2, index)
        kernel_shape = [1, 1, 1, 1, 1]
        kernel_shape[axis + 2] = len(kernel)
        volume = F.conv3d(volume, kernel.view(kernel_shape))
    return volume[0, 0]


def _resample_1mm(image: torch.Tensor, affine: np.ndarray):
    # The official resampler derives its factor from the affine, even though
    # the preceding resampling decision uses the header's voxel sizes.
    factor = np.linalg.norm(affine[:3, :3], axis=0)
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0
    image = _smooth_downsample(image, sigmas)
    axes = []
    for length, scale in zip(image.shape, factor):
        count = int(np.ceil(length * scale))
        coordinate = -(scale - 1) / (2 * scale) + torch.arange(
            count, dtype=image.dtype, device=image.device
        ) / scale
        axes.append(coordinate.clamp(0, length - 1))
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    normalized = [2 * coordinate / (length - 1) - 1 for coordinate, length in
                  ((z, image.shape[2]), (y, image.shape[1]), (x, image.shape[0]))]
    grid = torch.stack(normalized, dim=-1)[None]
    image = F.grid_sample(image[None, None], grid, mode="bilinear",
                          padding_mode="border", align_corners=True)[0, 0]
    new_affine = affine.copy()
    new_affine[:3, :3] /= factor
    new_affine[:3, 3] -= new_affine[:3, :3] @ (0.5 * (factor - 1))
    return image, new_affine


def _ras_axes(affine: np.ndarray) -> np.ndarray:
    axes = np.argmax(np.abs(np.linalg.inv(affine)[:3, :3]), axis=0)
    for axis in range(3):
        if axis not in axes:
            unique, counts = np.unique(axes, return_counts=True)
            repeated = unique[np.argmax(counts)]
            axes[np.where(axes == repeated)[0][-1]] = axis
    return axes


def _align_ras(image: torch.Tensor, affine: np.ndarray):
    aligned_affine = affine.copy()
    ras_axes = _ras_axes(aligned_affine)
    aligned_affine[:, :3] = aligned_affine[:, ras_axes]
    for axis in range(3):
        if ras_axes[axis] != axis:
            other = int(np.where(ras_axes == axis)[0][0])
            image = image.swapaxes(int(ras_axes[axis]), axis)
            ras_axes[other], ras_axes[axis] = ras_axes[axis], ras_axes[other]
    for axis in range(3):
        if aligned_affine[axis, axis] < 0:
            image = image.flip(axis)
            aligned_affine[:, axis] *= -1
            aligned_affine[:3, 3] -= aligned_affine[:3, axis] * (image.shape[axis] - 1)
    return image, aligned_affine


def preprocess_t1(path: str | Path | nib.spatialimages.SpatialImage,
                  device="cpu", min_pad: int = 128) -> PreprocessedT1:
    """Load, resample, orient, normalize, and center-pad a T1 for SynthSeg.

    ``min_pad=128`` matches official ``mri_synthseg`` without ``--crop``.
    Smaller values are useful for local crop tests; output dimensions remain
    multiples of 32. The returned image is a 3-D float32 tensor on ``device``.
    """
    source = (nib.load(str(path)) if isinstance(path, (str, Path)) else path)
    if not isinstance(source, nib.spatialimages.SpatialImage):
        raise TypeError("path must be a path or nibabel spatial image")
    data = np.squeeze(source.get_fdata(dtype=np.float64))
    if data.ndim != 3:
        raise ValueError("SynthSeg requires a single 3-D image")
    affine = source.affine.copy()
    original_shape = tuple(data.shape)
    voxsize = np.asarray(source.header.get_zooms()[:3], dtype=float)
    image = torch.as_tensor(data, device=device)
    if np.any((voxsize > 1.05) | (voxsize < 0.95)):
        image, affine = _resample_1mm(image, affine)
        voxsize = np.ones(3)
    volume_affine = affine.copy()
    image, aligned_affine = _align_ras(image, affine)
    limits = torch.quantile(image.flatten(), torch.tensor([0.005, 0.995],
                           dtype=image.dtype, device=image.device))
    image = image.clamp(limits[0], limits[1])
    image = (image - limits[0]) / (limits[1] - limits[0]) if limits[0] != limits[1] else image * 0
    image = image.to(torch.float32)

    target = [max(min_pad, int(np.ceil(length / 32) * 32)) for length in image.shape]
    before = [(target[i] - image.shape[i]) // 2 for i in range(3)]
    after = [target[i] - image.shape[i] - before[i] for i in range(3)]
    content = tuple(slice(before[i], before[i] + image.shape[i]) for i in range(3))
    image = F.pad(image, (before[2], after[2], before[1], after[1], before[0], after[0]))
    aligned_affine = aligned_affine.copy()
    aligned_affine[:3, 3] -= aligned_affine[:3, :3] @ np.asarray(before)
    return PreprocessedT1(image, source.affine.copy(), aligned_affine,
                          original_shape, content, volume_affine,
                          float(np.prod(voxsize)))
