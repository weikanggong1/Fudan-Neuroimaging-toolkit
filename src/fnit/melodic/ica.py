"""Single-session probabilistic spatial ICA on a masked 4D BOLD image."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


@dataclass(frozen=True)
class ICAResult:
    """Paths and diagnostics for one spatial ICA fit.

    ``component_maps`` are null-standardised spatial scores. ``posterior_maps``
    contains the fitted signal-class probability; ``thresholded_maps`` keeps
    scores where that probability reaches ``mm_threshold``. ``mixing`` is T×K
    and ``frequency_power`` is ceil(T/2)×K, with odd time series zero-padded.
    """

    component_maps: Path
    thresholded_maps: Path
    posterior_maps: Path
    mixing: Path
    frequency_power: Path
    n_components: int
    n_voxels: int
    n_iterations: int
    converged: bool
    final_decorrelation_change: float
    pca_variance_explained: float
    model_order_method: str
    estimated_resels: float | None


def _symmetric_decorrelation(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix @ matrix.T)
    return (vectors * values.clamp_min(1e-12).rsqrt()) @ vectors.T @ matrix


def _fsl_uniform(rows: int, columns: int, seed: int) -> np.ndarray:
    """Linux FSL ``srand``/``unifrnd`` sequence without changing process RNGs.

    FSL consumes glibc's additive random generator in column order. A seed of
    zero is treated as one by glibc. Keeping this stream in Python also makes
    CPU and GPU initialisations identical, independently of PyTorch's RNG.
    """
    initial = int(seed) & 0xFFFFFFFF
    initial = initial or 1
    state = [initial]
    previous = initial if initial < 0x80000000 else initial - 0x100000000
    for _ in range(1, 31):
        previous = (16807 * previous) % 2147483647
        state.append(previous)
    state.extend(state[:3])
    for index in range(34, 344):
        state.append((state[index - 31] + state[index - 3]) & 0xFFFFFFFF)
    values = np.empty(rows * columns, dtype=np.float64)
    for position in range(len(values)):
        index = len(state)
        value = (state[index - 31] + state[index - 3]) & 0xFFFFFFFF
        state.append(value)
        values[position] = ((value >> 1) + 1) / 2147483649.0
    return values.reshape(columns, rows).T.copy()


def _circle_law(dimension: int, samples: int) -> np.ndarray:
    """Discrete Marchenko-Pastur inversion used by FSL's PPCA estimator."""
    aspect = np.float32(dimension / samples)
    lower = np.float32((1 - math.sqrt(float(aspect))) ** 2)
    upper = np.float32((1 + math.sqrt(float(aspect))) ** 2)
    start, end = np.float32(0.9 * lower), np.float32(1.1 * upper)
    interval = np.float32((end - start) / (30 * dimension))
    grid = (start + interval * np.arange(1, 30 * dimension + 1, dtype=np.float32)).astype(np.float64)
    step = np.float32((upper - lower) / (10 * dimension))
    offsets = (step * np.arange(1, 10 * dimension + 1, dtype=np.float32)).astype(np.float64)
    density = np.sqrt(np.maximum(offsets * (float(upper - lower) - offsets), 0))
    density /= 2 * math.pi * float(aspect) * (offsets + float(lower))
    positions = offsets + float(lower)
    partial = np.r_[0.0, np.cumsum(density)]
    integral = partial[np.searchsorted(positions, grid, side="left")]
    counts = np.maximum(dimension * (1 - float(step) * integral), 0)
    result = np.zeros(dimension, dtype=np.float64)
    falling = np.flatnonzero(np.floor(counts[:-1]) > np.floor(counts[1:]))
    for index in falling:
        order = int(math.floor(counts[index]))
        if 1 <= order <= dimension:
            result[order - 1] = grid[index]
    return result.clip(min=5e-9)


def _device(device: str | torch.device | None) -> torch.device:
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def _estimate_resels(data: np.ndarray, mask: np.ndarray, batch_size: int) -> float:
    """Estimate the product of directional FWHM values on interior voxels."""
    axes = 3 if mask.shape[2] > 1 else 2
    linear = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    interior = [slice(1, None), slice(1, None), slice(1, None) if axes == 3 else slice(None)]
    common = mask[tuple(interior)].copy()
    neighbours = []
    for axis in range(axes):
        side = interior.copy()
        side[axis] = slice(None, -1)
        common &= mask[tuple(side)]
        neighbours.append(linear[tuple(side)])
    centres = linear[tuple(interior)][common]
    numerator = np.zeros(axes, dtype=np.float64)
    denominator = np.zeros(axes, dtype=np.float64)
    for start in range(0, len(centres), batch_size):
        stop = start + batch_size
        center = data[centres[start:stop]].astype(np.float64)
        center -= center.mean(axis=1, keepdims=True)
        center /= np.sqrt(np.maximum(np.sum(center * center, axis=1, keepdims=True), 1e-12))
        center_energy = np.sum(center * center)
        for axis in range(axes):
            adjacent = data[neighbours[axis][common][start:stop]].astype(np.float64)
            adjacent -= adjacent.mean(axis=1, keepdims=True)
            adjacent /= np.sqrt(np.maximum(np.sum(adjacent * adjacent, axis=1, keepdims=True), 1e-12))
            numerator[axis] += np.sum(center * adjacent)
            denominator[axis] += 0.5 * (center_energy + np.sum(adjacent * adjacent))
    correlations = np.clip(numerator / np.maximum(denominator, 1e-12), 1e-4, 0.99999)
    return float(np.prod(np.sqrt(-2 * math.log(2) / np.log(correlations))))


def _ppca_order(eigenvalues: torch.Tensor, n_voxels: int, resels: float) -> int:
    """FSL Laplace PPCA order with its discrete spectrum and tail indexing."""
    values = eigenvalues.detach().cpu().double().numpy()[2:][::-1].copy()
    values = np.maximum(values, max(values[0], 1.0) * 1e-12)
    dimension = len(values)
    if dimension < 4:
        raise ValueError("at least six time points are required for automatic model order")
    samples = max(int(n_voxels / (2.5 * np.float32(resels))), dimension + 1)
    expected = _circle_law(dimension, samples)
    adjusted = np.sort(values / expected)[::-1]
    cumulative = np.cumsum(values) / values.sum()
    cap = int(np.searchsorted(cumulative, 0.98))
    if cap < 3:
        cap = dimension // 2
    cap = min(cap, dimension - 2)
    spectrum = np.maximum(adjusted[:cap], adjusted[0] * 1e-12)
    suffix_sum = np.cumsum(spectrum[::-1])[::-1]
    remainder = np.r_[suffix_sum[1:], 0.95 * spectrum[-1]]
    tail_mean = (remainder / np.maximum(cap - np.arange(1, cap + 1), 1)).clip(min=0.01)
    row_hessian = np.zeros(cap)
    for i in range(cap - 1):
        differences = spectrum[i] - spectrum[i + 1:]
        differences = np.where(differences > 0, differences, 1.0)
        inverse = 1 / tail_mean[i + 1:] - 1 / spectrum[i]
        inverse = np.where(inverse > 0, inverse, 1.0)
        row_hessian[i] = np.log(differences).sum() + np.log(inverse).sum()
    cumulative_hessian = np.cumsum(row_hessian)
    cumulative_log = np.cumsum(np.log(spectrum))
    scores = []
    log_volume = 0.0
    for count in range(1, cap - 1):
        log_volume += (-math.log(2) + math.lgamma((cap - count + 1) / 2)
                       - 0.5 * math.log(math.pi) * (cap - count + 1))
        noise = tail_mean[count - 1]
        m = cap * count - count * (count + 1) / 2
        score = (log_volume - (samples // 2) * cumulative_log[count - 1]
                 - (samples // 2) * (cap - count) * math.log(max(noise, 1e-12))
                 + cumulative_hessian[count - 1]
                 + 0.5 * math.log(2 * math.pi) * (m + count)
                 - 0.5 * math.log(samples) * count)
        scores.append(score)
    scores = np.asarray(scores)
    peaks = np.flatnonzero(scores[:-1] >= scores[1:])
    return int(peaks[0] + 1 if len(peaks) else np.argmax(scores) + 1)


def _gamma_pdf(x: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    # FSL's scalar Gamma parameters and log normaliser are floats, while the
    # histogram coordinates and density calculation use doubles.
    mean32, variance32 = mean.float(), variance.float()
    shape = (mean32.double().square() / variance32.double()).float()
    rate = mean32 / variance32
    normaliser = torch.lgamma(shape.double()).float()
    log_pdf = ((shape * rate.log()).double()
               + (shape - 1).double() * x.clamp_min(1e-6).log()
               - rate.double() * x - normaliser.double())
    valid = (x > 1e-6) & (mean32 > 0) & (variance32 > 1e-5) & (normaliser.abs() < 150)
    return torch.where(valid, log_pdf.exp(), torch.zeros_like(x))


def _normal_pdf(x: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    mean, variance = mean.float().double(), variance.float().double()
    return torch.exp(-0.5 * (x - mean).square() / variance) / torch.sqrt(2 * math.pi * variance)


def _mixture_posterior(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit Gaussian null and positive/negative Gamma alternatives by EM."""
    x = (z.double() - z.double().mean()) / z.double().std(correction=1).clamp_min(1e-6)
    variance = x.square().mean()
    means = torch.stack((-2 * x.mean(), x.mean() + 2 * variance.sqrt(), x.mean() - 2 * variance.sqrt()))
    variances = variance.expand(3).clone()
    weights = torch.full((3,), 1 / 3, dtype=x.dtype, device=x.device)
    previous = 1.0
    likelihoods = None
    for iteration in range(1, 501):
        variances.clamp_(min=1e-4)
        noise_width = variances[0].sqrt()
        positive_floor = ((2.6 - weights[0]) * noise_width + means[0]).float().double()
        negative_floor = ((2.6 - weights[0]) * noise_width - means[0]).float().double()
        positive_floor = 0.5 * (positive_floor + torch.sqrt(positive_floor.square() + 4 * variances[1]))
        negative_floor = 0.5 * (negative_floor + torch.sqrt(negative_floor.square() + 4 * variances[2]))
        means[1] = torch.maximum(means[1], positive_floor.clamp_min(1e-3))
        means[2] = -torch.maximum(-means[2], negative_floor.clamp_min(1e-3))
        variances[1] = torch.minimum(variances[1], 0.5 * means[1].square()).clamp_min(1e-4)
        variances[2] = torch.minimum(variances[2], 0.5 * means[2].square()).clamp_min(1e-4)
        null = _normal_pdf(x, means[0], variances[0].clamp_min(1e-4))
        positive = _gamma_pdf(x, means[1].clamp_min(1e-3), variances[1].clamp_min(1e-4))
        negative = _gamma_pdf(-x, (-means[2]).clamp_min(1e-3), variances[2].clamp_min(1e-4))
        likelihoods = torch.stack((null, positive, negative)) * weights[:, None]
        # Unmodelled extreme voxels have a fixed 1e-4 likelihood in FSL.
        # Consequently responsibilities need not sum to one there, and the
        # component masses are not renormalised on each EM step.
        total = likelihoods.sum(dim=0).clamp_min(1e-4)
        responsibilities = likelihoods / total
        score = total.log().sum().float().item()
        change = abs(score - previous)
        previous = score
        update = iteration < 30 or (change > 1e-6 and iteration < 300)
        if not update:
            break
        masses = responsibilities.sum(dim=1).clamp_min(1e-7)
        for component in range(3):
            if masses[component] < 1e-4 * len(x):
                masses += 1e-4
        new_means = (responsibilities @ x) / masses
        new_variances = (responsibilities * (x[None, :] - new_means[:, None]).square()).sum(dim=1) / masses
        means = new_means
        variances = new_variances.clamp_min(1e-4)
        weights = masses / len(x)
    null, positive, negative = likelihoods.unbind(0)
    adjusted = (x - means[0]) / variances[0].sqrt()
    posterior = ((positive + negative) /
                 (null + positive + negative).clamp_min(torch.finfo(x.dtype).tiny)).clamp(0, 1)
    return adjusted.float(), posterior.float()


def decompose_spatial_ica(
    input_bold: str | Path,
    brain_mask: str | Path,
    output_dir: str | Path,
    *,
    n_components: int | None = None,
    device: str | torch.device | None = None,
    voxel_batch_size: int = 8192,
    max_iter: int = 500,
    tolerance: float = 1e-3,
    random_state: int = 0,
    mm_threshold: float = 0.5,
) -> ICAResult:
    """Fit MELODIC-style spatial PICA with PPCA order and mixture inference.

    ``n_components=None`` estimates order by Laplace PPCA using spatial
    smoothness to adjust the effective number of voxels. An explicit integer
    matches MELODIC ``-d K``. The non-Gaussian contrast is ``pow3``. Spatial
    maps are divided by estimated residual noise before fitting the Gaussian
    null and positive/negative Gamma mixture; ``mm_threshold`` is the signal
    posterior cutoff. ICA source identity and mixture parameters are not
    guaranteed to equal FSL.
    """
    if (n_components is not None and n_components < 1) or voxel_batch_size < 1 or max_iter < 1:
        raise ValueError("n_components, voxel_batch_size, and max_iter must be positive")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be in (0, 1)")
    if not 0 < mm_threshold < 1:
        raise ValueError("mm_threshold must be in (0, 1)")
    image = nib.load(str(input_bold))
    mask_image = nib.load(str(brain_mask))
    if len(image.shape) != 4 or image.shape[3] < 3:
        raise ValueError("input_bold must be a 4D NIfTI with at least three time points")
    if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, atol=1e-4):
        raise ValueError("brain_mask must be aligned with the BOLD spatial grid")
    nt = image.shape[3]
    if n_components is not None and n_components >= nt:
        raise ValueError("n_components must be smaller than the number of time points")
    selected = _device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    data = np.asarray(image.dataobj, dtype=np.float32).reshape((-1, nt))
    mask = np.asarray(mask_image.dataobj) > 0
    indices = np.flatnonzero(mask.reshape(-1))
    if n_components is not None and len(indices) <= n_components:
        raise ValueError("brain_mask contains too few voxels for n_components")

    # First PCA estimates residual variance rather than dividing by the raw
    # voxelwise standard deviation. Its covariance centers across voxels.
    # MELODIC's matrix arithmetic is double precision. These PCA/ICA tensors
    # are small relative to the input 4D image, and avoid TF32-dependent source
    # changes while leaving the package's global TF32 setting enabled.
    covariance = torch.zeros((nt, nt), dtype=torch.float64, device=selected)
    sum_time = torch.zeros(nt, dtype=torch.float64, device=selected)
    accepted: list[np.ndarray] = []
    for start in range(0, len(indices), voxel_batch_size):
        batch_indices = indices[start:start + voxel_batch_size]
        series = torch.as_tensor(data[batch_indices].copy(), dtype=torch.float64, device=selected)
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
    if n_components is not None and n_voxels <= n_components:
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
        series = torch.as_tensor(data[batch_indices].copy(), dtype=torch.float64, device=selected)
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
    if n_components is not None and n_voxels <= n_components:
        raise ValueError("brain_mask contains too few varying voxels for n_components")
    spatial_mean = sum_time / n_voxels
    covariance = covariance / n_voxels - torch.outer(spatial_mean, spatial_mean)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    estimated_resels = None
    if n_components is None:
        estimated_resels = _estimate_resels(data, mask, voxel_batch_size)
        n_components = _ppca_order(eigenvalues, n_voxels, estimated_resels)
    eigenvalues = eigenvalues.flip(0)[:n_components]
    eigenvectors = eigenvectors.flip(1)[:, :n_components]
    if eigenvalues[-1] <= eigenvalues[0] * 1e-7:
        raise ValueError("n_components exceeds the numerical rank of the BOLD data")
    variance_explained = (eigenvalues.sum() / covariance.diagonal().sum()).item()

    whitened = torch.empty((n_voxels, n_components), dtype=torch.float64, device=selected)
    for start in range(0, n_voxels, voxel_batch_size):
        stop = min(start + voxel_batch_size, n_voxels)
        series = torch.as_tensor(data[indices[start:stop]].copy(), dtype=torch.float64, device=selected)
        series -= series.mean(dim=1, keepdim=True)
        scale = torch.as_tensor(scales_np[start:stop], device=selected)
        # Spatial centering belongs to the PCA covariance, not to the input
        # used by spatial FastICA. The global time course remains in Data.
        whitened[start:stop] = ((series / scale[:, None]) @ eigenvectors) / eigenvalues.sqrt()

    initial_time = torch.as_tensor(_fsl_uniform(nt, n_components, random_state), device=selected)
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
    temporal_scale = mixing.std(dim=0, correction=1)
    mixing /= temporal_scale
    maps *= temporal_scale
    unmixing *= temporal_scale[:, None]
    signs = torch.sign(maps[maps.abs().argmax(dim=0), torch.arange(n_components, device=selected)])
    maps *= signs
    mixing *= signs
    order = maps.std(dim=0, correction=1).argsort(descending=True)
    maps = maps[:, order]
    mixing = mixing[:, order]
    unmixing = unmixing[order]
    mixing_np = mixing.cpu().numpy()
    # PICA divides spatial sources by the voxelwise residual-noise standard
    # deviation and by the norm of the full temporal unmixing row.
    # The row norm is evaluated without forming a time-by-time inverse.
    unmixing_full = (unmixing / eigenvalues.sqrt()[None, :]) @ eigenvectors.T
    component_scale = unmixing_full.square().sum(dim=1).rsqrt()
    corrected = torch.empty_like(maps)
    degree_factor = math.sqrt((nt - n_components) / (nt - 1))
    for start in range(0, n_voxels, voxel_batch_size):
        stop = min(start + voxel_batch_size, n_voxels)
        series = torch.as_tensor(data[indices[start:stop]].copy(), dtype=torch.float64, device=selected)
        series -= series.mean(dim=1, keepdim=True)
        scale = torch.as_tensor(scales_np[start:stop], device=selected)
        normalized = series / scale[:, None]
        residual = normalized - maps[start:stop] @ mixing.T
        noise = residual.std(dim=1, correction=1)
        noise = torch.where(noise >= 0.05, noise, torch.ones_like(noise))
        corrected[start:stop] = maps[start:stop] * (degree_factor / noise)[:, None] * component_scale

    corrected_np = np.empty((n_voxels, n_components), dtype=np.float32)
    posterior_np = np.empty_like(corrected_np)
    for component in range(n_components):
        adjusted, posterior = _mixture_posterior(corrected[:, component])
        corrected_np[:, component] = adjusted.cpu().numpy()
        posterior_np[:, component] = posterior.cpu().numpy()
    all_maps = np.zeros((data.shape[0], n_components), dtype=np.float32)
    all_posteriors = np.zeros_like(all_maps)
    all_maps[indices] = corrected_np
    all_posteriors[indices] = posterior_np
    all_maps = all_maps.reshape((*image.shape[:3], n_components))
    all_posteriors = all_posteriors.reshape(all_maps.shape)
    thresholded = np.where(all_posteriors >= mm_threshold, all_maps, 0)
    thresholded = thresholded.astype(np.float32)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    component_maps = output / "ica_components_z.nii.gz"
    thresholded_maps = output / "ica_components_pica_thresholded.nii.gz"
    posterior_maps = output / "ica_components_signal_probability.nii.gz"
    mixing_path = output / "ica_mixing.tsv"
    spectrum_path = output / "ica_frequency_power.tsv"
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_zooms((*image.header.get_zooms()[:3], 1.0))
    header.set_xyzt_units(xyz=image.header.get_xyzt_units()[0], t="unknown")
    nib.save(nib.Nifti1Image(all_maps, image.affine, header), str(component_maps))
    nib.save(nib.Nifti1Image(thresholded, image.affine, header), str(thresholded_maps))
    nib.save(nib.Nifti1Image(all_posteriors, image.affine, header), str(posterior_maps))
    np.savetxt(mixing_path, mixing_np, fmt="%.9g", delimiter="\t")
    np.savetxt(spectrum_path, np.abs(np.fft.rfft(mixing_np, n=nt + nt % 2, axis=0)[1:]) ** 2,
               fmt="%.9g", delimiter="\t")
    return ICAResult(
        component_maps=component_maps,
        thresholded_maps=thresholded_maps,
        posterior_maps=posterior_maps,
        mixing=mixing_path,
        frequency_power=spectrum_path,
        n_components=n_components,
        n_voxels=n_voxels,
        n_iterations=iteration,
        converged=converged,
        final_decorrelation_change=difference,
        pca_variance_explained=variance_explained,
        model_order_method="laplace_ppca" if estimated_resels is not None else "fixed",
        estimated_resels=estimated_resels,
    )
