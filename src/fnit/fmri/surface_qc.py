"""UKB-style cortical ribbon and good-voxel mask on an MNI fMRI grid."""

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


@dataclass(frozen=True)
class SurfaceQCResult:
    ribbon: Path
    goodvoxels: Path
    report: Path


def _same_grid(image, reference):
    return image.shape[:3] == reference.shape[:3] and np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    )


def make_ribbon_goodvoxels(
    clean_bold,
    reference,
    left_white,
    left_pial,
    right_white,
    right_pial,
    output_dir,
    *,
    wb_command="wb_command",
    neighborhood_sigma_mm=5.0,
    threshold_factor=0.5,
):
    """Make the ribbon and COV-based goodvoxels used by UKB's surface mapping.

    Surfaces must already be in the RAS world coordinates of the MNI reference.
    Workbench computes signed distances; NumPy/SciPy replace the original FSL
    threshold, temporal statistics, Gaussian smoothing and mean dilation steps.
    """
    source = nib.load(str(clean_bold))
    target = nib.load(str(reference))
    if source.ndim != 4 or target.ndim != 3 or not _same_grid(source, target):
        raise ValueError("clean_bold and 3D reference must have the same MNI grid")
    if source.shape[3] < 2 or neighborhood_sigma_mm <= 0 or threshold_factor < 0:
        raise ValueError("invalid time length or goodvoxels parameters")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    distances = {}
    for side, white, pial in (
        ("L", left_white, left_pial), ("R", right_white, right_pial)
    ):
        for layer, surface in (("white", white), ("pial", pial)):
            path = output / f"{side}.{layer}.signed_distance.nii.gz"
            subprocess.run(
                [str(wb_command), "-create-signed-distance-volume", str(surface),
                 str(reference), str(path)],
                check=True, capture_output=True, text=True,
            )
            image = nib.load(str(path))
            if image.ndim != 3 or not _same_grid(image, target):
                raise ValueError("Workbench signed distance must match the reference grid")
            distances[(side, layer)] = np.asarray(image.dataobj, dtype=np.float32)
    ribbon = np.zeros(target.shape, dtype=bool)
    for side in ("L", "R"):
        ribbon |= (distances[(side, "white")] > 0) & (
            distances[(side, "pial")] < 0
        )
    if not ribbon.any():
        raise ValueError("white/pial surfaces produced an empty cortical ribbon")
    ribbon_path = output / "ribbon_only.nii.gz"
    nib.save(nib.Nifti1Image(ribbon.astype(np.uint8), target.affine), str(ribbon_path))
    ribbon_seconds = time.perf_counter() - started

    data = np.asarray(source.dataobj, dtype=np.float32)
    mean = np.mean(data, axis=3, dtype=np.float64)
    second_moment = np.einsum("xyzt,xyzt->xyz", data, data, dtype=np.float64) / data.shape[3]
    std = np.sqrt(np.maximum(second_moment - mean * mean, 0.0)
                  * data.shape[3] / (data.shape[3] - 1))
    cov = np.divide(std, mean, out=np.zeros_like(std), where=mean != 0)
    cov_ribbon = np.where(ribbon, cov, 0.0)
    nonzero_cov = cov_ribbon[cov_ribbon != 0]
    if nonzero_cov.size == 0:
        raise ValueError("ribbon contains no nonzero fMRI coefficient of variation")
    ribbon_cov_mean = float(nonzero_cov.mean())
    normalized = cov_ribbon / ribbon_cov_mean
    sigma_voxels = np.asarray(
        [neighborhood_sigma_mm / zoom for zoom in target.header.get_zooms()[:3]],
        dtype=np.float64,
    )
    numerator = gaussian_filter(normalized, sigma_voxels, mode="constant")
    denominator = gaussian_filter((normalized != 0).astype(np.float64), sigma_voxels, mode="constant")
    trend = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    # FSL -dilD fills zero-valued neighbors with their nonzero-neighbor mean.
    support = convolve((trend != 0).astype(np.float64), np.ones((3, 3, 3)), mode="constant")
    local_sum = convolve(trend, np.ones((3, 3, 3)), mode="constant")
    fill = np.divide(local_sum, support, out=np.zeros_like(local_sum), where=support > 0)
    trend = np.where(trend != 0, trend, fill)
    modulated = np.divide(
        cov / ribbon_cov_mean, trend,
        out=np.zeros_like(cov), where=trend != 0,
    )
    inside = modulated[ribbon & (modulated != 0)]
    if inside.size == 0:
        raise ValueError("ribbon contains no valid normalized fMRI variation")
    upper = float(inside.mean() + threshold_factor * inside.std(ddof=1))
    good = (mean != 0) & np.isfinite(modulated) & (modulated < upper)
    good_path = output / "goodvoxels.nii.gz"
    nib.save(nib.Nifti1Image(good.astype(np.uint8), target.affine), str(good_path))
    total_seconds = time.perf_counter() - started
    report = output / "goodvoxels_report.json"
    report.write_text(json.dumps({
        "ribbon_seconds": ribbon_seconds,
        "statistics_seconds": total_seconds - ribbon_seconds,
        "total_seconds": total_seconds,
        "ribbon_voxels": int(ribbon.sum()),
        "ribbon_cov_mean": ribbon_cov_mean,
        "neighborhood_sigma_mm": neighborhood_sigma_mm,
        "threshold_factor": threshold_factor,
        "normalized_upper_threshold": upper,
        "goodvoxels": int(good.sum()),
    }, indent=2) + "\n", encoding="utf-8")
    return SurfaceQCResult(ribbon_path, good_path, report)
