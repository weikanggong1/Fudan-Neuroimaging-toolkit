"""Prepare automatic raw T1 inputs with native PyTorch bias correction."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from time import monotonic

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage
import torch

from .context import SubregionContext


def image_geometry(image) -> dict:
    return {"shape": [int(size) for size in image.shape],
            "affine": np.asarray(image.affine).tolist(),
            "voxel_sizes_mm": np.linalg.norm(image.affine[:3, :3], axis=0).tolist()}


def _coronal_grid(image, data) -> tuple[tuple[int, int, int], np.ndarray]:
    from ..recon_all.conform_gpu import _coronal_affine
    sizes = np.linalg.norm(image.affine[:3, :3], axis=0)
    width = max(256, math.ceil(float(np.max(np.asarray(image.shape) * sizes))))
    if width > 256 and (width - 256) / 256 < .1:
        width = 256
    # MGH derives its world center from the input's shape and affine. Reuse
    # only the established conform geometry helper; intensities stay float32.
    header_image = nib.MGHImage(np.asarray(data, dtype=np.float32), image.affine)
    return (width, width, width), _coronal_affine(header_image, width)


def _resample(array, affine, grid, *, order, dtype):
    source = nib.Nifti1Image(np.asarray(array, dtype=dtype), affine)
    return np.asarray(resample_from_to(source, grid, order=order).dataobj, dtype=dtype)


def prepare_automatic_raw_input(context: SubregionContext, *, device, threads: int) -> SubregionContext:
    """Bias-correct automatic coarse-label inputs and choose their fitting grid.

    Sparse positive eroded WM retains the original input and records the
    reason. The public pipeline calls this only when coarse labels were
    inferred from the raw T1 rather than supplied by the caller.
    """
    started = monotonic()
    original = context.image
    data = np.asarray(context.data, dtype=np.float32)
    metadata = dict(context.metadata)
    record = {
        "method": "TorchFAST_default_tensor_restored_then_eroded_WM_median_110",
        "mask_rule": "coarse > 0; includes CSF",
        "wm_rule": "labels 2/41; one voxel binary erosion; finite positive intensities",
        "minimum_wm_samples": 100, "intensity_dtype": "float32",
        "original_geometry": image_geometry(original),
        "processing_geometry": image_geometry(original),
        "bias_correction_seconds": 0.0, "intensity_scale": 1.0,
        "device": str(device), "threads": threads, "applied": False,
    }
    metadata["intensity_preprocessing"] = record

    def fallback(reason):
        record.update(fallback_reason=reason, grid_rule="preserve_original_input",
                      seconds=monotonic() - started)
        return replace(context, metadata=metadata, native_image=original)

    if context.coarse_segmentation is None:
        return fallback("automatic_coarse_labels_unavailable")
    coarse = context.coarse_segmentation
    wm_mask = ndimage.binary_erosion(np.isin(coarse, (2, 41)), iterations=1)
    valid = wm_mask & np.isfinite(data) & (data > 0)
    record["original_wm_samples"] = int(np.count_nonzero(valid))
    if record["original_wm_samples"] < 100:
        return fallback("fewer_than_100_positive_eroded_WM_samples")

    from ..fast import TorchFAST
    device = torch.device(device)
    if device.type == "cuda":
        # Shared SynthSeg outputs are already native NumPy arrays. Release
        # its unused allocator blocks before the next full-volume estimator.
        torch.cuda.empty_cache()
    estimator = TorchFAST(device=device, threads=threads)
    record["fast_config"] = asdict(estimator.config)
    brain_mask = coarse > 0
    record["brain_mask_voxels"] = int(np.count_nonzero(brain_mask))
    input_image = nib.Nifti1Image(data, original.affine)
    mask_image = nib.Nifti1Image(brain_mask.astype(np.uint8), original.affine)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bias_started = monotonic()
    result = estimator(input_image, mask_image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    record["bias_correction_seconds"] = monotonic() - bias_started
    if result.restored.shape != original.shape or not np.allclose(
            result.restored.affine, original.affine, atol=1e-5, rtol=0):
        raise ValueError("TorchFAST restored image changed input geometry")
    restored = np.asarray(result.restored.dataobj, dtype=np.float32)
    samples = restored[wm_mask & np.isfinite(restored) & (restored > 0)]
    record["restored_wm_samples"] = int(samples.size)
    record["tissue_means"] = [float(value) for value in result.tissue_means]
    if samples.size < 100:
        return fallback("fewer_than_100_positive_restored_eroded_WM_samples")
    median = float(np.median(samples))
    scale = 110.0 / median
    corrected = restored * np.float32(scale)
    record.update(wm_median_before_scale=median, intensity_scale=scale, applied=True)
    del result, estimator

    sizes = np.linalg.norm(original.affine[:3, :3], axis=0)
    if float(sizes.mean()) < .99:
        image = nib.Nifti1Image(corrected, original.affine)
        record["grid_rule"] = "preserve_high_resolution_native_grid"
        parc, wm = context.cortical_parcellation, context.wmparc_proxy
    else:
        grid = _coronal_grid(original, data)
        corrected = _resample(corrected, original.affine, grid, order=1, dtype=np.float32)
        coarse = _resample(coarse, original.affine, grid, order=0, dtype=np.int32)
        parc = (None if context.cortical_parcellation is None else
                _resample(context.cortical_parcellation, original.affine, grid, order=0, dtype=np.int32))
        wm = (None if context.wmparc_proxy is None else
              _resample(context.wmparc_proxy, original.affine, grid, order=0, dtype=np.int32))
        image = nib.Nifti1Image(corrected, grid[1])
        record["grid_rule"] = "raw_header_derived_coronal_1mm_float32_linear_intensity_nearest_labels"
    record.update(processing_geometry=image_geometry(image), seconds=monotonic() - started)
    return replace(context, image=image, data=corrected, coarse_segmentation=coarse,
                   cortical_parcellation=parc, wmparc_proxy=wm, metadata=metadata,
                   native_image=original)
