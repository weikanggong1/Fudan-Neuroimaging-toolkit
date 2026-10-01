"""Prepare multiresolution GEMS alpha templates without a FreeSurfer runtime."""

from __future__ import annotations

import hashlib
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import ive
import torch
from torch.nn import functional as F

from .atlas import GEMSAtlas
from .deformation import ordered_vertex_sum, prepare_vertex_reduction
from .rasterize import rasterize_priors


def _discrete_gaussian_kernel(sigma: float) -> np.ndarray:
    variance = float(sigma) ** 2
    coefficients = [float(ive(0, variance)), float(ive(1, variance))]
    mass = coefficients[0] + 2 * coefficients[1]
    order = 2
    while mass < 0.9:
        value = float(ive(order, variance))
        coefficients.append(value)
        mass += 2 * value
        order += 1
    return np.asarray(coefficients[:0:-1] + coefficients, dtype=np.float32) / mass


def _separable_convolve(volume: torch.Tensor, kernel: np.ndarray, *,
                        mode: str = "nearest", channel_chunk: int = 4) -> torch.Tensor:
    """Filter CXYZ data on its device with SciPy-compatible boundary indices.

    A small channel batch bounds convolution workspace and padded buffers.
    SciPy's ``reflect`` repeats the edge voxel (half-sample symmetry), which
    differs from PyTorch's reflection padding, so use explicit indices.
    """
    radius = len(kernel) // 2
    if radius == 0:
        return volume
    if mode not in ("nearest", "reflect"):
        raise ValueError("mode must be nearest or reflect")
    coefficients = torch.as_tensor(kernel, device=volume.device, dtype=volume.dtype)
    outputs = []
    for start in range(0, len(volume), channel_chunk):
        data = volume[start:start + channel_chunk].unsqueeze(0)
        for axis in range(3):
            length = data.shape[axis + 2]
            positions = torch.arange(-radius, length + radius, device=data.device)
            if mode == "nearest":
                positions = positions.clamp(0, length - 1)
            else:
                positions = positions.remainder(2 * length)
                positions = torch.where(positions < length, positions,
                                        2 * length - 1 - positions)
            padded = data.index_select(axis + 2, positions)
            kernel_shape = [1, 1, 1]
            kernel_shape[axis] = len(kernel)
            weight = coefficients.reshape(1, 1, *kernel_shape).expand(
                data.shape[1], 1, *kernel_shape).contiguous()
            data = F.conv3d(padded, weight, groups=data.shape[1])
        outputs.append(data.squeeze(0))
    return torch.cat(outputs, dim=0)


def _gaussian_filter3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Match scipy.ndimage.gaussian_filter's FP32, reflect, truncate=4 path."""
    if sigma <= 1e-15:
        return volume
    radius = int(4.0 * sigma + 0.5)
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 / sigma**2 * coordinates**2)
    kernel /= kernel.sum()
    return _separable_convolve(volume.unsqueeze(0), kernel.astype(np.float32),
                                mode="reflect").squeeze(0)


def _smoothing_key(atlas: GEMSAtlas, classes: np.ndarray, device: torch.device) -> tuple:
    """Fingerprint all inputs to the fixed reference-grid rasterization."""
    identity = hashlib.sha256()
    for values in (atlas.reference_vertices, atlas.tetrahedra, atlas.alphas, classes):
        identity.update(np.ascontiguousarray(values).view(np.uint8))
        identity.update(str((values.shape, values.dtype)).encode())
    return str(device), identity.digest()


def smooth_atlas_alphas(
    atlas: GEMSAtlas,
    label_classes: np.ndarray,
    sigma: float,
    *,
    device: str | torch.device = "cpu",
    cache: dict | None = None,
    stable_vertex_statistics: bool = False,
) -> np.ndarray:
    """Rasterize, smooth and refit grouped alphas on the reference mesh.

    This mirrors the reference atlas preparation: spatial Gaussian filtering
    of voxel priors followed by ten vertex-alpha EM updates. The result is
    defined in the supplied reference mesh's coordinates; a transformed
    subject mesh therefore needs its own smoothing.
    """
    classes = np.asarray(label_classes, dtype=np.int64)
    if classes.shape != (atlas.n_labels,):
        raise ValueError("label_classes must match the atlas label channels")
    grouped = np.zeros((len(atlas.vertices), int(classes.max()) + 1), dtype=np.float32)
    for channel, group in enumerate(classes):
        grouped[:, group] += atlas.alphas[:, channel]
    if sigma == 0:
        return grouped

    device = torch.device(device)
    alphas = torch.as_tensor(grouped, device=device)
    key = _smoothing_key(atlas, classes, device) if cache is not None else None
    entry = None if cache is None else cache.get("reference_raster")
    if entry is not None and entry[0] == key:
        priors, covered, cells, weights = entry[1]
    else:
        vertices = torch.as_tensor(atlas.reference_vertices, device=device, dtype=torch.float32)
        tetrahedra = torch.as_tensor(atlas.tetrahedra, device=device, dtype=torch.long)
        shape = tuple(np.floor(atlas.reference_vertices.max(0)).astype(int) + 1)
        priors, covered, cells, weights = rasterize_priors(
            vertices, tetrahedra, alphas, shape, background_channel=None,
            return_assignment=True)
        cells = cells[covered]
        weights = weights[covered]
        if cache is not None:
            # Replace rather than accumulate reference grids/classes in GPU memory.
            cache["reference_raster"] = (key, (priors, covered, cells, weights))
    kernel = _discrete_gaussian_kernel(sigma)
    if device.type == "cuda":
        smoothed = _separable_convolve(priors, kernel)
        target = smoothed[:, covered].T.contiguous()
    else:
        smoothed = priors.numpy()
        for axis in (1, 2, 3):
            smoothed = convolve1d(smoothed, kernel, axis=axis, mode="nearest")
        target = torch.as_tensor(smoothed[:, covered.numpy()].T.copy())
    del smoothed
    fitted = torch.full_like(alphas, 1.0 / grouped.shape[1])
    chunk_size = max(1, 131_072 if device.type == "cuda" else len(cells))
    layouts = None
    if stable_vertex_statistics:
        layout_key = (key, len(fitted), chunk_size)
        saved = None if cache is None else cache.get("stable_vertex_statistics")
        if saved is not None and saved[0] == layout_key:
            layouts = saved[1]
        else:
            layouts = []
            for start in range(0, len(cells), chunk_size):
                local_cells = cells[start:start + chunk_size]
                # Retain the original corner-then-point accumulation order.
                ids = local_cells.T.contiguous().reshape(-1)
                layouts.append((ids, prepare_vertex_reduction(ids, len(fitted))))
            if cache is not None:
                cache["stable_vertex_statistics"] = (layout_key, layouts)
    for _ in range(10):
        statistics = torch.zeros_like(fitted)
        # Each point contributes independently; chunk only the temporary Cx4
        # responsibilities, retaining all ten alpha-EM updates.
        for block, start in enumerate(range(0, len(cells), chunk_size)):
            local_cells = cells[start:start + chunk_size]
            contribution = fitted[local_cells] * weights[start:start + chunk_size, ..., None]
            prediction = contribution.sum(1).clamp_min(1e-15)
            contribution = contribution / prediction[:, None] * target[start:start + chunk_size, None]
            if stable_vertex_statistics:
                ids, layout = layouts[block]
                values = contribution.permute(1, 0, 2).reshape(-1, contribution.shape[-1])
                statistics.add_(ordered_vertex_sum(ids, values, len(fitted), layout))
            else:
                for corner in range(4):
                    statistics.index_add_(0, local_cells[:, corner], contribution[:, corner])
        fitted = statistics / statistics.sum(1, keepdim=True).clamp_min(1e-12)
    return fitted.cpu().numpy()
