"""BIDS entry point for UKB-style FEAT preprocessing before ICA denoising."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..feat.temporal import gaussian_highpass, grand_mean_scale
from .bids import locate_bids_inputs
from .mask import epi_brain_mask
from ..mcflirt import TorchMCFLIRT
from .spatial import apply_motion_warp


@dataclass(frozen=True)
class FeatCoreResult:
    output_dir: Path
    filtered_func_data: Path
    mask: Path
    mean_func: Path
    motion_parameters: Path
    motion_matrices: Path
    intensity_factor: float
    unwarp_applied: bool


def _save(data, reference, output):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header)
    image.set_qform(reference.affine, code=int(reference.header["qform_code"]))
    image.set_sform(reference.affine, code=int(reference.header["sform_code"]))
    nib.save(image, str(output))
    return Path(output)


def run_feat_core(
    bids_root,
    output_dir,
    *,
    subject,
    session=None,
    task="rest",
    run=None,
    acquisition=None,
    direction=None,
    reconstruction=None,
    echo=None,
    brain_mask=None,
    brain_extraction="synthstrip",
    synthstrip_weights=None,
    spatial_warp=None,
    postmat=None,
    highpass_cutoff_seconds=100.0,
    device=None,
    batch_size=32,
    motion_iterations=(1, 1, 1),
    overwrite=False,
):
    """Run pre-ICA volumetric FEAT core on one BIDS BOLD run.

    Outputs follow FEAT's names. ``spatial_warp`` is an optional FSL dense
    field or FNIRT cubic coefficient image, and ``postmat`` is an optional
    warp-reference-to-BOLD FLIRT matrix. If neither is given, this stage
    performs motion-only resampling. It never manufactures a missing B0
    fieldmap or GDC warp. ``motion_iterations`` counts coordinate-optimizer
    sweeps at 8/4/4 mm, one per stage as in MCFLIRT; it no longer counts
    the removed Adam steps. The optional `brain_mask` must already be in the
    reference grid. Without one, SynthStrip extracts the corrected EPI mean
    by default; ``brain_extraction='otsu'`` selects the older independent mask.
    """
    inputs = locate_bids_inputs(
        bids_root, subject=subject, session=session, task=task, run=run,
        acquisition=acquisition, direction=direction,
        reconstruction=reconstruction, echo=echo,
    )
    if inputs.fieldmaps and spatial_warp is None:
        raise NotImplementedError(
            "BIDS fieldmap inputs were found, but B0 warp estimation is not yet "
            "implemented; supply a precomputed spatial_warp"
        )
    if highpass_cutoff_seconds <= 0:
        raise ValueError("highpass_cutoff_seconds must be positive")
    if brain_extraction not in ("synthstrip", "otsu"):
        raise ValueError("brain_extraction must be 'synthstrip' or 'otsu'")
    output = Path(output_dir).expanduser().resolve()
    filtered_path = output / "filtered_func_data.nii.gz"
    if filtered_path.exists() and not overwrite:
        raise FileExistsError(filtered_path)
    output.mkdir(parents=True, exist_ok=True)
    mc_dir = output / "mc"
    matrices_dir = mc_dir / "prefiltered_func_data_mcf.mat"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    raw = nib.load(str(inputs.bold))
    if inputs.sbref is not None:
        reference = nib.load(str(inputs.sbref))
    else:
        midpoint = np.asarray(raw.dataobj[..., raw.shape[3] // 2], dtype=np.float32)
        reference = nib.Nifti1Image(midpoint, raw.affine, raw.header.copy())
    example_path = output / "example_func.nii.gz"
    _save(np.asarray(reference.dataobj), reference, example_path)
    reference = nib.load(str(example_path))
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    motion_only = spatial_warp is None and postmat is None
    fit = TorchMCFLIRT(device=selected_device).run(
        raw, reference, stage_iterations=motion_iterations,
        resample=motion_only, interpolation="spline",
    )
    matrices = fit.matrices
    for frame, matrix in enumerate(matrices):
        np.savetxt(matrices_dir / f"MAT_{frame:04d}", matrix, fmt="%.12g")
    par_path = mc_dir / "prefiltered_func_data_mcf.par"
    np.savetxt(par_path, fit.parameters, fmt="%.9g")
    corrected = fit.corrected if motion_only else apply_motion_warp(
        raw, reference, matrices, warp=spatial_warp, postmat=postmat,
        interpolation="spline", batch_size=batch_size, device=selected_device,
    )
    corrected.header.set_zooms((*corrected.header.get_zooms()[:3], float(inputs.tr)))
    corrected.header.set_xyzt_units(
        xyz=corrected.header.get_xyzt_units()[0], t="sec"
    )
    corrected_data = np.asarray(corrected.dataobj, dtype=np.float32)
    mean = corrected_data.mean(axis=3)
    mean_before_mask = output / "prefiltered_func_data_unwarp_mean.nii.gz"
    _save(mean, reference, mean_before_mask)
    if brain_mask is None:
        if brain_extraction == "synthstrip":
            from ..synthstrip import SynthStrip

            stripped = SynthStrip(
                weights=synthstrip_weights, device=selected_device
            )(mean_before_mask)
            mask_image = nib.Nifti1Image(
                np.asarray(stripped.mask.data, dtype=np.uint8), reference.affine
            )
        else:
            mask_image = epi_brain_mask(nib.load(str(mean_before_mask)))
    else:
        mask_image = nib.load(str(brain_mask))
        if mask_image.shape != reference.shape or not np.allclose(
            mask_image.affine, reference.affine, atol=1e-4
        ):
            raise ValueError("brain_mask must match the BOLD reference grid")
    mask = np.asarray(mask_image.dataobj) > 0
    mask_path = output / "mask.nii.gz"
    mask_header = reference.header.copy()
    mask_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, mask_header), str(mask_path))
    corrected_data *= mask[..., None]
    scaled, factor = grand_mean_scale(corrected_data, mask)
    filtered = gaussian_highpass(
        scaled, sigma_volumes=highpass_cutoff_seconds / (2 * inputs.tr),
        device=selected_device, preserve_mean=True,
    )
    _save(filtered, corrected, filtered_path)
    mean_path = output / "mean_func.nii.gz"
    _save(filtered.mean(axis=3), reference, mean_path)
    return FeatCoreResult(
        output, filtered_path, mask_path, mean_path, par_path, matrices_dir,
        factor, spatial_warp is not None,
    )
