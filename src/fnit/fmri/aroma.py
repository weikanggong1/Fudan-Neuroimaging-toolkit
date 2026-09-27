"""Independent ICA-AROMA feature extraction, classification, and cleanup."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .confounds import _device, _load_bold, _load_matrix, _mask_on_bold_grid, _save_bold


def _maximum_absolute_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    norm = np.linalg.norm(left, axis=0)[:, None] * np.linalg.norm(right, axis=0)[None, :]
    correlations = np.divide(left.T @ right, norm, out=np.zeros_like(norm), where=norm > 0)
    return np.max(np.abs(correlations), axis=1)


def motion_correlation_feature(
    mixing: str | Path | np.ndarray,
    motion: str | Path | np.ndarray,
    *,
    n_splits: int = 1000,
    random_state: int = 0,
) -> np.ndarray:
    """Mean maximum IC–motion correlation over 90% time-point subsamples.

    The motion design matches ICA-AROMA's current/derivative and one-frame
    forward/backward shifted design; both raw and squared correlations count.
    """
    mix = _load_matrix(mixing, "mixing")
    rp6 = _load_matrix(motion, "motion")
    if rp6.shape != (mix.shape[0], 6) or mix.shape[0] < 4:
        raise ValueError("motion must be T x 6, matching at least four IC time points")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    derivative = np.vstack((np.zeros((1, 6)), np.diff(rp6, axis=0)))
    rp12 = np.column_stack((rp6, derivative))
    forward = np.vstack((np.zeros((1, 12)), rp12[:-1]))
    backward = np.vstack((rp12[1:], np.zeros((1, 12))))
    rp_model = np.column_stack((rp12, forward, backward))

    count = round(0.9 * mix.shape[0])
    rng = random.Random(random_state)
    scores = np.zeros(mix.shape[1])
    for _ in range(n_splits):
        rows = rng.sample(range(mix.shape[0]), k=count)
        direct = _maximum_absolute_correlation(mix[rows], rp_model[rows])
        squared = _maximum_absolute_correlation(mix[rows] ** 2, rp_model[rows] ** 2)
        scores += np.maximum(direct, squared)
    return scores / n_splits


def frequency_feature(ftmix: str | Path | np.ndarray, tr: float) -> np.ndarray:
    """Normalized half-power frequency from MELODIC's Fourier mixture matrix."""
    spectrum = _load_matrix(ftmix, "ftmix")
    if tr <= 0 or spectrum.shape[0] < 2 or (spectrum < 0).any():
        raise ValueError("ftmix must have nonnegative powers and tr must be positive")
    nyquist = 0.5 / tr
    frequencies = nyquist * np.arange(1, spectrum.shape[0] + 1) / spectrum.shape[0]
    selected = frequencies > 0.01
    if not selected.any():
        raise ValueError("no spectral bin above 0.01 Hz")
    power = spectrum[selected]
    cumulative = np.cumsum(power, axis=0)
    total = cumulative[-1]
    fraction = np.divide(cumulative, total, out=np.zeros_like(cumulative), where=total > 0)
    index = np.argmin(np.abs(fraction - 0.5), axis=0)
    normalized = (frequencies[selected] - 0.01) / (nyquist - 0.01)
    return np.where(total > 0, normalized[index], 0.0)


def spatial_features(
    thresholded_ic_maps: str | Path,
    csf_mask: str | Path,
    edge_mask: str | Path,
    outside_mask: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge and CSF fractions of absolute thresholded IC Z scores.

    All four images must share one grid. Official-equivalence comparisons
    require the original ICA-AROMA masks and thresholded maps on MNI 2 mm;
    aligned native-space masks give an approximate classification. The caller
    supplies mixture-model thresholded maps, not raw ICA maps.
    """
    image = nib.load(str(thresholded_ic_maps))
    if len(image.shape) != 4:
        raise ValueError("thresholded_ic_maps must be a 4D NIfTI")
    csf = _mask_on_bold_grid(csf_mask, image)
    edge = _mask_on_bold_grid(edge_mask, image)
    outside = _mask_on_bold_grid(outside_mask, image)
    maps = np.asarray(image.dataobj, dtype=np.float32)
    edge_fraction = np.zeros(image.shape[3])
    csf_fraction = np.zeros(image.shape[3])
    for component in range(image.shape[3]):
        z = np.abs(maps[..., component])
        total = z.sum(dtype=np.float64)
        if total == 0:
            continue
        csf_sum = z[csf].sum(dtype=np.float64)
        denominator = total - csf_sum
        edge_fraction[component] = (z[edge].sum(dtype=np.float64) + z[outside].sum(dtype=np.float64)) / denominator if denominator > 0 else 0
        csf_fraction[component] = csf_sum / total
    return edge_fraction, csf_fraction


def classify_components(
    max_rp_corr: np.ndarray,
    edge_fraction: np.ndarray,
    high_freq_content: np.ndarray,
    csf_fraction: np.ndarray,
) -> np.ndarray:
    """Return zero-based noise IC indices from ICA-AROMA's fixed classifier."""
    features = [np.asarray(x, dtype=np.float64).reshape(-1) for x in
                (max_rp_corr, edge_fraction, high_freq_content, csf_fraction)]
    if len({len(x) for x in features}) != 1 or not all(np.isfinite(x).all() for x in features):
        raise ValueError("all four feature arrays must be finite and have equal length")
    motion, edge, frequency, csf = features
    hyperplane = -19.9751070082159 + 9.95127547670627 * motion + 24.8333160239175 * edge
    return np.flatnonzero((hyperplane > 0) | (csf > 0.10) | (frequency > 0.35))


def classify_aroma(
    thresholded_ic_maps: str | Path,
    mixing: str | Path | np.ndarray,
    ftmix: str | Path | np.ndarray,
    motion: str | Path | np.ndarray,
    csf_mask: str | Path,
    edge_mask: str | Path,
    outside_mask: str | Path,
    tr: float,
    *,
    n_splits: int = 1000,
    random_state: int = 0,
) -> dict[str, np.ndarray]:
    """Extract four features from aligned masks, thresholded maps, and FTmix.

    This function does not run ICA or infer masks; all inputs must come from
    the same run and image-space transform. Returned noise indices are zero-based.
    """
    max_rp_corr = motion_correlation_feature(mixing, motion, n_splits=n_splits, random_state=random_state)
    high_freq_content = frequency_feature(ftmix, tr)
    edge_fraction, csf_fraction = spatial_features(thresholded_ic_maps, csf_mask, edge_mask, outside_mask)
    lengths = {len(x) for x in (max_rp_corr, high_freq_content, edge_fraction, csf_fraction)}
    if len(lengths) != 1:
        raise ValueError("component count differs across maps, mixing matrix, and FTmix")
    noise_indices = classify_components(max_rp_corr, edge_fraction, high_freq_content, csf_fraction)
    return {
        "max_rp_corr": max_rp_corr,
        "edge_fraction": edge_fraction,
        "high_freq_content": high_freq_content,
        "csf_fraction": csf_fraction,
        "noise_indices": noise_indices,
    }


def denoise_aroma(
    input_bold: str | Path,
    mixing: str | Path | np.ndarray,
    noise_indices: np.ndarray | list[int],
    output_bold: str | Path,
    *,
    mode: str = "nonaggr",
    device: str | torch.device | None = None,
    chunk_size: int = 4096,
) -> Path:
    """Remove noise ICs with partial (nonaggr) or full (aggr) regression.

    `noise_indices` are zero-based. Nonaggressive cleanup fits all IC time
    courses and subtracts only noise-IC partial contributions. Aggressive
    cleanup fits only noise time courses. Both preserve the temporal mean
    inside the FSL default mask: voxels with mean intensity at least 1%
    above the image mean-intensity minimum; other voxels are set to zero.
    Regression arithmetic uses float64 for FSL-equivalent float32 output.
    """
    if mode not in {"nonaggr", "aggr"}:
        raise ValueError("mode must be 'nonaggr' or 'aggr'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    mix = _load_matrix(mixing, "mixing")
    indices = np.asarray(noise_indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= mix.shape[1]):
        raise ValueError("noise_indices contains an out-of-range component")
    indices = np.unique(indices)
    output = Path(output_bold)
    if not len(indices):
        if Path(input_bold).resolve() != output.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_bold, output)
        return output

    image, data = _load_bold(input_bold)
    nt = data.shape[3]
    if mix.shape[0] != nt:
        raise ValueError("mixing row count must equal the number of BOLD volumes")
    centered = mix - mix.mean(axis=0)
    fit_design = centered if mode == "nonaggr" else centered[:, indices]
    subtract_design = centered[:, indices]
    pseudoinverse = np.linalg.pinv(fit_design, rcond=1e-8)
    if mode == "nonaggr":
        pseudoinverse = pseudoinverse[indices]

    selected = _device(device)
    subtract_tensor = torch.as_tensor(subtract_design, dtype=torch.float64, device=selected)
    inverse_tensor = torch.as_tensor(pseudoinverse, dtype=torch.float64, device=selected)
    flat = data.reshape((-1, nt))
    mean = data.mean(axis=3, dtype=np.float32)
    threshold = float(mean.min() + np.float32(0.01) * (mean.max() - mean.min()))
    included = np.flatnonzero((mean >= threshold).reshape(-1))
    cleaned = np.zeros_like(flat)
    for start in range(0, len(included), chunk_size):
        voxels = included[start:start + chunk_size]
        series = torch.as_tensor(flat[voxels].T.copy(), dtype=torch.float64, device=selected)
        cleaned[voxels] = (series - subtract_tensor @ (inverse_tensor @ series)).T.cpu().numpy()
    return _save_bold(output, cleaned.reshape(data.shape), image)
