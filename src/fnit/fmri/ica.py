"""Spatial ICA of a masked 4D BOLD image without an external executable.

The returned components are independent estimates, not MELODIC's mixture-
model maps. Component number is explicit because matching MELODIC's automatic
model order would require a separate, validated model-selection procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


@dataclass(frozen=True)
class ICAResult:
    """Paths and diagnostics for one spatial ICA fit.

    ``final_decorrelation_change`` is the last maximum sign-invariant
    diagonal change between ICA updates. ``component_maps`` holds
    unthresholded spatial Z scores, while
    ``thresholded_maps`` applies a simple absolute-Z cutoff. The latter is an
    approximation for ICA-AROMA's spatial features, not a MELODIC mixture-
    model threshold. ``mixing`` has T rows and K columns; ``frequency_power``
    has floor(T/2) positive-frequency rows and K columns.
    """

    component_maps: Path
    thresholded_maps: Path
    mixing: Path
    frequency_power: Path
    n_components: int
    n_voxels: int
    n_iterations: int
    converged: bool
    final_decorrelation_change: float
    pca_variance_explained: float


def _symmetric_decorrelation(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix @ matrix.T)
    return (vectors * values.clamp_min(1e-12).rsqrt()) @ vectors.T @ matrix


def _device(device: str | torch.device | None) -> torch.device:
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def decompose_spatial_ica(
    input_bold: str | Path,
    brain_mask: str | Path,
    output_dir: str | Path,
    *,
    n_components: int,
    device: str | torch.device | None = None,
    voxel_batch_size: int = 8192,
    max_iter: int = 500,
    tolerance: float = 1e-3,
    random_state: int = 0,
    z_threshold: float = 2.3,
) -> ICAResult:
    """Fit temporal PCA followed by symmetric FastICA across brain voxels.

    Each voxel's time course is demeaned, then divided by the temporal
    standard deviation left after thresholding its first 30 PCA scores at
    absolute Z=2.3, following MELODIC's variance-normalization procedure.
    The symmetric ICA update uses MELODIC's default pow3 contrast. PCA and
    ICA operate in float32 on the selected device;
    voxel chunks and the T-by-K whitened data bound GPU memory. The output
    NIfTI files inherit the input BOLD affine and header geometry. K must be
    supplied explicitly; this routine does not reproduce MELODIC's automatic
    model order, mixture-model thresholding, or component identities.
    """
    if n_components < 1 or voxel_batch_size < 1 or max_iter < 1:
        raise ValueError("n_components, voxel_batch_size, and max_iter must be positive")
    if not 0 < tolerance < 1 or z_threshold <= 0:
        raise ValueError("tolerance must be in (0, 1) and z_threshold must be positive")
    image = nib.load(str(input_bold))
    mask_image = nib.load(str(brain_mask))
    if len(image.shape) != 4 or image.shape[3] < 3:
        raise ValueError("input_bold must be a 4D NIfTI with at least three time points")
    if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, atol=1e-4):
        raise ValueError("brain_mask must be aligned with the BOLD spatial grid")
    nt = image.shape[3]
    if n_components >= nt:
        raise ValueError("n_components must be smaller than the number of time points")
    selected = _device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    data = np.asarray(image.dataobj, dtype=np.float32).reshape((-1, nt))
    mask = np.asarray(mask_image.dataobj) > 0
    indices = np.flatnonzero(mask.reshape(-1))
    if len(indices) <= n_components:
        raise ValueError("brain_mask contains too few voxels for n_components")

    # First PCA estimates residual variance rather than dividing by the raw
    # voxelwise standard deviation. Its covariance centers across voxels.
    covariance = torch.zeros((nt, nt), dtype=torch.float32, device=selected)
    sum_time = torch.zeros(nt, dtype=torch.float32, device=selected)
    accepted: list[np.ndarray] = []
    for start in range(0, len(indices), voxel_batch_size):
        batch_indices = indices[start:start + voxel_batch_size]
        series = torch.as_tensor(data[batch_indices].copy(), device=selected)
        series -= series.mean(dim=1, keepdim=True)
        valid = torch.isfinite(series).all(dim=1) & (series.square().mean(dim=1) > 1e-12)
        batch_indices = batch_indices[valid.cpu().numpy()]
        if not len(batch_indices):
            continue
        series = series[valid]
        accepted.append(batch_indices)
        covariance.addmm_(series.T, series)
        sum_time += series.sum(dim=0)
    if not accepted:
        raise ValueError("brain_mask contains no finite, temporally varying voxels")
    indices = np.concatenate(accepted)
    n_voxels = len(indices)
    if n_voxels <= n_components:
        raise ValueError("brain_mask contains too few varying voxels for n_components")
    spatial_mean = sum_time / n_voxels
    covariance = covariance / n_voxels - torch.outer(spatial_mean, spatial_mean)
    first_values, first_vectors = torch.linalg.eigh(covariance)
    first_count = min(30, nt - 1, n_voxels - 1)
    first_values = first_values.flip(0)[:first_count].clamp_min(1e-12)
    first_vectors = first_vectors.flip(1)[:, :first_count]
    first_scale = first_values.sqrt()

    covariance.zero_()
    sum_time.zero_()
    accepted = []
    scales: list[np.ndarray] = []
    for start in range(0, len(indices), voxel_batch_size):
        batch_indices = indices[start:start + voxel_batch_size]
        series = torch.as_tensor(data[batch_indices].copy(), device=selected)
        series -= series.mean(dim=1, keepdim=True)
        scores = (series @ first_vectors) / first_scale
        scores *= (scores.abs() >= 2.3)
        residual = series - (scores * first_scale) @ first_vectors.T
        scale = residual.std(dim=1, correction=1)
        valid = torch.isfinite(scale) & (scale >= 0.01)
        if not valid.any():
            continue
        accepted.append(batch_indices[valid.cpu().numpy()])
        scales.append(scale[valid].cpu().numpy())
        normalized = series[valid] / scale[valid, None]
        covariance.addmm_(normalized.T, normalized)
        sum_time += normalized.sum(dim=0)
    if not accepted:
        raise ValueError("brain_mask contains no voxels after variance normalization")
    indices = np.concatenate(accepted)
    scales_np = np.concatenate(scales)
    n_voxels = len(indices)
    if n_voxels <= n_components:
        raise ValueError("brain_mask contains too few varying voxels for n_components")
    spatial_mean = sum_time / n_voxels
    covariance = covariance / n_voxels - torch.outer(spatial_mean, spatial_mean)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)[:n_components]
    eigenvectors = eigenvectors.flip(1)[:, :n_components]
    if eigenvalues[-1] <= eigenvalues[0] * 1e-7:
        raise ValueError("n_components exceeds the numerical rank of the BOLD data")
    variance_explained = (eigenvalues.sum() / covariance.diagonal().sum()).item()

    whitened = torch.empty((n_voxels, n_components), dtype=torch.float32, device=selected)
    for start in range(0, n_voxels, voxel_batch_size):
        stop = min(start + voxel_batch_size, n_voxels)
        series = torch.as_tensor(data[indices[start:stop]].copy(), device=selected)
        series -= series.mean(dim=1, keepdim=True)
        scale = torch.as_tensor(scales_np[start:stop], device=selected)
        whitened[start:stop] = ((series / scale[:, None] - spatial_mean) @ eigenvectors) / eigenvalues.sqrt()

    generator = torch.Generator(device=selected).manual_seed(random_state)
    initial_time = torch.rand((nt, n_components), generator=generator, device=selected)
    unmixing = _symmetric_decorrelation(
        (initial_time.T @ eigenvectors) / eigenvalues.sqrt()
    )
    converged = False
    for iteration in range(1, max_iter + 1):
        projection = whitened @ unmixing.T
        # MELODIC calls this skewness contrast "pow3"; its score is quadratic.
        updated = (3 * (projection.square().T @ whitened)) / n_voxels
        updated -= projection.mean(dim=0)[:, None] * unmixing
        updated = _symmetric_decorrelation(updated)
        difference = (updated @ unmixing.T).diagonal().abs().sub(1).abs().max().item()
        unmixing = updated
        if difference < tolerance:
            converged = True
            break

    maps = whitened @ unmixing.T
    mixing = (eigenvectors * eigenvalues.sqrt()) @ unmixing.T
    signs = torch.sign(maps[maps.abs().argmax(dim=0), torch.arange(n_components, device=selected)])
    maps *= signs
    mixing *= signs
    maps_np = maps.cpu().numpy()
    mixing_np = mixing.cpu().numpy()
    all_maps = np.zeros((data.shape[0], n_components), dtype=np.float32)
    all_maps[indices] = maps_np
    all_maps = all_maps.reshape((*image.shape[:3], n_components))
    thresholded = np.where(np.abs(all_maps) >= z_threshold, all_maps, 0).astype(np.float32)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    component_maps = output / "ica_components_z.nii.gz"
    thresholded_maps = output / "ica_components_abs_z_thresholded.nii.gz"
    mixing_path = output / "ica_mixing.tsv"
    spectrum_path = output / "ica_frequency_power.tsv"
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_zooms((*image.header.get_zooms()[:3], 1.0))
    header.set_xyzt_units(xyz=image.header.get_xyzt_units()[0], t="unknown")
    nib.save(nib.Nifti1Image(all_maps, image.affine, header), str(component_maps))
    nib.save(nib.Nifti1Image(thresholded, image.affine, header), str(thresholded_maps))
    np.savetxt(mixing_path, mixing_np, fmt="%.9g", delimiter="\t")
    np.savetxt(spectrum_path, np.abs(np.fft.rfft(mixing_np, axis=0)[1:]) ** 2,
               fmt="%.9g", delimiter="\t")
    return ICAResult(
        component_maps=component_maps,
        thresholded_maps=thresholded_maps,
        mixing=mixing_path,
        frequency_power=spectrum_path,
        n_components=n_components,
        n_voxels=n_voxels,
        n_iterations=iteration,
        converged=converged,
        final_decorrelation_change=difference,
        pca_variance_explained=variance_explained,
    )
