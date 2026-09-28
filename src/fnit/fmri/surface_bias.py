"""Bias correction for the T1w and T2 FLAIR pair used by UKB myelin mapping."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit
from scipy import ndimage


@dataclass(frozen=True)
class SurfaceBiasResult:
    bias: Path
    t1_restored: Path
    t1_brain_restored: Path
    t2_restored: Path
    t2_brain_restored: Path


@njit(cache=True)
def _mean_dilate_fov(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Fill zero voxels in scan order from the current 26-neighbour mean."""
    filled = values != 0
    queue = np.empty(values.size * 27, dtype=np.int32)
    head = tail = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                index = x + nx * (y + ny * z)
                if filled[index]:
                    continue
                adjacent = False
                for zz in range(max(0, z - 1), min(nz, z + 2)):
                    for yy in range(max(0, y - 1), min(ny, y + 2)):
                        for xx in range(max(0, x - 1), min(nx, x + 2)):
                            if filled[xx + nx * (yy + ny * zz)]:
                                adjacent = True
                if adjacent:
                    queue[tail] = index
                    tail += 1
    while head < tail:
        end = tail
        for position in range(head, end):
            index = queue[position]
            if filled[index]:
                continue
            z = index // (nx * ny)
            y = (index // nx) % ny
            x = index % nx
            total = 0.0
            count = 0
            for zz in range(max(0, z - 1), min(nz, z + 2)):
                for yy in range(max(0, y - 1), min(ny, y + 2)):
                    for xx in range(max(0, x - 1), min(nx, x + 2)):
                        neighbour = xx + nx * (yy + ny * zz)
                        if neighbour != index and filled[neighbour]:
                            total += values[neighbour]
                            count += 1
            values[index] = total / max(count, 1)
            filled[index] = True
            for zz in range(max(0, z - 1), min(nz, z + 2)):
                for yy in range(max(0, y - 1), min(ny, y + 2)):
                    for xx in range(max(0, x - 1), min(nx, x + 2)):
                        neighbour = xx + nx * (yy + ny * zz)
                        if not filled[neighbour]:
                            queue[tail] = neighbour
                            tail += 1
        head = end
    return values


def correct_surface_structural_bias(
    t1w: str | Path,
    t2_flair: str | Path,
    t1_brain: str | Path,
    output_dir: str | Path,
    *,
    sigma_mm: float = 5.0,
) -> SurfaceBiasResult:
    """Estimate a common multiplicative bias field from aligned T1w and FLAIR.

    The images must occupy the same scanner grid. The procedure follows UKB's
    square-root product, within-mask Gaussian smoothing, thresholded modulation,
    26-neighbour mean dilation and final smoothing. FSL commands are not run.
    """
    images = [nib.load(str(Path(p).expanduser().resolve())) for p in (t1w, t2_flair, t1_brain)]
    if any(image.ndim != 3 for image in images):
        raise ValueError("t1w, t2_flair and t1_brain must be 3D images")
    if any(image.shape != images[0].shape or not np.allclose(
        image.affine, images[0].affine, atol=1e-4, rtol=0,
    ) for image in images[1:]):
        raise ValueError("t1w, t2_flair and t1_brain must share one grid")
    if sigma_mm <= 0:
        raise ValueError("sigma_mm must be positive")
    axes = nib.aff2axcodes(images[0].affine)
    if axes not in (("R", "A", "S"), ("L", "A", "S")):
        raise ValueError("structural bias correction requires RAS or LAS voxel axes")
    t1, t2, brain = (np.asarray(image.dataobj, dtype=np.float32) for image in images)
    if not all(np.isfinite(data).all() for data in (t1, t2, brain)):
        raise ValueError("structural inputs contain nonfinite values")
    mask = brain > 0
    product = np.sqrt(np.abs(t1 * t2), dtype=np.float32)
    product[~mask] = 0
    positive = product > 0
    if not np.any(positive):
        raise ValueError("T1w and T2 FLAIR have no positive product within the brain")
    normalized = product / float(product[positive].mean())
    sigma = np.asarray([sigma_mm / zoom for zoom in images[0].header.get_zooms()[:3]])
    smoothed_mask = ndimage.gaussian_filter(positive.astype(np.float32), sigma)
    smoothed = ndimage.gaussian_filter(normalized, sigma)
    np.divide(smoothed, smoothed_mask, out=smoothed, where=smoothed_mask > 0)
    modulation = np.divide(normalized, smoothed, out=np.zeros_like(normalized),
                           where=smoothed > 0)
    nonzero = modulation[modulation != 0]
    lower = float(nonzero.mean() - 0.5 * nonzero.std(ddof=1))
    thresholded = ndimage.binary_erosion(modulation >= lower, structure=np.ones((3, 3, 3)))
    labels, count = ndimage.label(thresholded, structure=ndimage.generate_binary_structure(3, 1))
    if not count:
        raise ValueError("structural bias mask is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = labels == np.argmax(sizes)
    seeded = np.where(selected, normalized, 0)
    radiological = seeded[::-1] if axes[0] == "R" else seeded
    flat = np.asfortranarray(radiological).ravel(order="F").copy()
    dilated = _mean_dilate_fov(flat, *seeded.shape).reshape(seeded.shape, order="F")
    if axes[0] == "R":
        dilated = dilated[::-1]
    bias = ndimage.gaussian_filter(dilated, sigma)
    if np.any(bias <= 0) or not np.isfinite(bias).all():
        raise ValueError("estimated structural bias field is not positive and finite")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = SurfaceBiasResult(*(output / name for name in (
        "bias.nii.gz", "T1w_restored.nii.gz", "T1w_restored_brain.nii.gz",
        "T2_FLAIR_restored.nii.gz", "T2_FLAIR_restored_brain.nii.gz",
    )))
    for path, data in (
        (paths.bias, bias),
        (paths.t1_restored, t1 / bias),
        (paths.t1_brain_restored, np.where(mask, t1 / bias, 0)),
        (paths.t2_restored, t2 / bias),
        (paths.t2_brain_restored, np.where(mask, t2 / bias, 0)),
    ):
        nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), images[0].affine,
                                 images[0].header), str(path))
    return paths
