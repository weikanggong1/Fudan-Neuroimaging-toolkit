"""Prepare multiresolution GEMS alpha templates without a FreeSurfer runtime."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import ive
import torch

from .atlas import GEMSAtlas
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


def smooth_atlas_alphas(
    atlas: GEMSAtlas,
    label_classes: np.ndarray,
    sigma: float,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Rasterize, smooth and refit grouped alphas on the reference mesh.

    This mirrors the reference atlas preparation: spatial Gaussian filtering
    of voxel priors followed by ten vertex-alpha EM updates. The result is
    subject independent and can be stored alongside the atlas.
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
    vertices = torch.as_tensor(atlas.reference_vertices, device=device, dtype=torch.float32)
    tetrahedra = torch.as_tensor(atlas.tetrahedra, device=device, dtype=torch.long)
    alphas = torch.as_tensor(grouped, device=device)
    shape = tuple(np.floor(atlas.reference_vertices.max(0)).astype(int) + 1)
    priors, covered, cells, weights = rasterize_priors(
        vertices, tetrahedra, alphas, shape, background_channel=None,
        return_assignment=True)
    kernel = _discrete_gaussian_kernel(sigma)
    smoothed = priors.cpu().numpy()
    for axis in (1, 2, 3):
        smoothed = convolve1d(smoothed, kernel, axis=axis, mode="nearest")
    target = torch.as_tensor(smoothed[:, covered.cpu().numpy()].T.copy(), device=device)
    cells = cells[covered]
    weights = weights[covered]
    fitted = torch.full_like(alphas, 1.0 / grouped.shape[1])
    for _ in range(10):
        contribution = fitted[cells] * weights[..., None]
        prediction = contribution.sum(1).clamp_min(1e-15)
        contribution = contribution / prediction[:, None] * target[:, None]
        statistics = torch.zeros_like(fitted)
        for corner in range(4):
            statistics.index_add_(0, cells[:, corner], contribution[:, corner])
        fitted = statistics / statistics.sum(1, keepdim=True).clamp_min(1e-12)
    return fitted.cpu().numpy()
