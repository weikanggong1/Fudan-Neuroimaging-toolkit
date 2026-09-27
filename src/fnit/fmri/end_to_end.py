"""One-run raw BIDS to ICA-AROMA cleaned BOLD on a 2-mm MNI152 grid."""

from dataclasses import dataclass
from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np
import torch

from ..fast import TorchFAST
from ..synthstrip import SynthStrip
from .aroma_pipeline import AromaResult, run_aroma_pipeline
from .bids import locate_bids_inputs
from .normalization import T1MNIResult, register_t1_to_mni, resample_world
from .pipeline import FeatCoreResult, run_feat_core
from .surface_pipeline import SurfacePipelineInputs, SurfacePipelineResult, run_surface_from_mni
from .surface_prepare import prepare_fs_sphere_projection_inputs


@dataclass(frozen=True)
class FMRIPipelineResult:
    """Paths for the native FEAT stage, registration, ICA and final MNI BOLD."""

    clean_mni: Path
    mask_mni: Path
    report: Path
    feat: FeatCoreResult
    aroma: AromaResult
    bbr_matrix: Path
    t1_to_mni: T1MNIResult
    timing_seconds: dict[str, float]
    surface: SurfacePipelineResult | None = None


def _save_mask(data, reference, path):
    image = nib.load(str(reference))
    array = np.asarray(data > 0, dtype=np.uint8)
    if array.shape != image.shape[:3] or not array.any():
        raise ValueError("mask must be nonempty and match its reference")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, image.affine), str(output))
    return output


def _reference_image(inputs, output):
    if inputs.sbref is not None:
        return inputs.sbref
    source = nib.load(str(inputs.bold))
    middle = np.asarray(source.dataobj[..., source.shape[3] // 2], dtype=np.float32)
    nib.save(nib.Nifti1Image(middle, source.affine), str(output))
    return output


def _select_t1(inputs, requested):
    candidates = inputs.t1w_images
    if requested is not None:
        selected = Path(requested).expanduser().resolve()
        if selected not in (path.resolve() for path in candidates):
            raise ValueError("t1w_image must be one of this BIDS run's T1w images")
        return selected
    if len(candidates) != 1:
        raise ValueError("multiple BIDS T1w images found; supply t1w_image")
    return candidates[0]


def run_fmri_pipeline(
    bids_root,
    output_dir,
    *,
    subject,
    mni_template,
    session=None,
    task="rest",
    run=None,
    acquisition=None,
    direction=None,
    reconstruction=None,
    echo=None,
    t1w_image=None,
    mni_brain_mask=None,
    registration_backend="synthmorph",
    surface_inputs: SurfacePipelineInputs | None = None,
    surface_subject_dir=None,
    surface_assets_dir=None,
    wb_command="wb_command",
    synthstrip_weights=None,
    synthmorph_weights=None,
    ica_n_components=None,
    aroma_mode="nonaggr",
    regress_wm=False,
    regress_csf=False,
    regress_motion=False,
    motion_model=24,
    bandpass=None,
    global_signal=False,
    highpass_cutoff_seconds=100.0,
    device=None,
    batch_size=8,
    motion_iterations=(35, 25, 15),
    ica_max_iter=500,
    n_splits=1000,
    random_state=0,
    overwrite=False,
):
    """Run motion/FEAT, SynthStrip/FAST, BBR, PICA/AROMA and MNI resampling.

    B0 fieldmap and GDC estimation are deliberately absent because no raw
    fieldmaps or GDC warp are available in the specified UKB example. Any
    associated BIDS fieldmaps currently cause an explicit error in FEAT core.
    """
    if surface_inputs is not None and surface_subject_dir is not None:
        raise ValueError("surface_inputs and surface_subject_dir are mutually exclusive")
    if (surface_subject_dir is None) != (surface_assets_dir is None):
        raise ValueError("surface_subject_dir and surface_assets_dir must be provided together")
    output = Path(output_dir).expanduser().resolve()
    clean_mni = output / "filtered_func_data_clean_MNI152_2mm.nii.gz"
    if clean_mni.exists() and not overwrite:
        raise FileExistsError(clean_mni)
    output.mkdir(parents=True, exist_ok=True)
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(selected).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(selected))
    template = nib.load(str(mni_template))
    if template.ndim != 3 or not np.allclose(template.header.get_zooms()[:3], 2, atol=0.01):
        raise ValueError("mni_template must be a 3D MNI152 2-mm image")
    asset_dir = Path(__file__).parent / "assets"
    classification_masks = tuple(asset_dir / f"mask_{name}.nii.gz" for name in ("csf", "edge", "out"))
    for mask_path in classification_masks:
        mask_image = nib.load(str(mask_path))
        if mask_image.shape != template.shape or not np.allclose(
            mask_image.affine, template.affine, atol=1e-4
        ):
            raise ValueError("mni_template must match the ICA-AROMA MNI152 2-mm mask grid")
    inputs = locate_bids_inputs(
        bids_root, subject=subject, session=session, task=task, run=run,
        acquisition=acquisition, direction=direction,
        reconstruction=reconstruction, echo=echo,
    )
    t1w = _select_t1(inputs, t1w_image)
    if surface_subject_dir is not None:
        scanner_orig = Path(surface_subject_dir).expanduser().resolve() / "mri/orig/001.mgz"
        structural = nib.load(str(scanner_orig))
        selected_t1 = nib.load(str(t1w))
        if structural.shape != selected_t1.shape or not np.allclose(
            structural.affine, selected_t1.affine, rtol=0, atol=1e-4
        ):
            raise ValueError("surface subject scanner T1 and selected BIDS T1 have different grids")
    mask_dir = output / "masks"
    mask_dir.mkdir(exist_ok=True)
    reference = _reference_image(inputs, output / "reference_epi.nii.gz")
    strip = SynthStrip(weights=synthstrip_weights, device=selected)
    timing = {}

    started = time.perf_counter()
    epi_mask = _save_mask(
        strip(reference).mask.data, reference, mask_dir / "epi_synthstrip.nii.gz"
    )
    t1_extracted = strip(t1w)
    t1_brain = output / "T1_brain.nii.gz"
    t1_mask = mask_dir / "T1_synthstrip.nii.gz"
    t1_extracted.image.save(t1_brain)
    _save_mask(t1_extracted.mask.data, t1w, t1_mask)
    if mni_brain_mask is None:
        template_extracted = strip(mni_template)
        template_brain_data = np.asarray(template_extracted.image.data, dtype=np.float32)
        template_mask = _save_mask(
            template_extracted.mask.data, mni_template,
            mask_dir / "MNI152_synthstrip.nii.gz",
        )
    else:
        supplied = nib.load(str(mni_brain_mask))
        if supplied.shape != template.shape or not np.allclose(supplied.affine, template.affine, atol=1e-4):
            raise ValueError("mni_brain_mask must match mni_template")
        template_mask = _save_mask(
            np.asarray(supplied.dataobj), mni_template,
            mask_dir / "MNI152_brain_mask.nii.gz",
        )
        template_brain_data = np.asarray(template.dataobj, dtype=np.float32) * (
            np.asarray(supplied.dataobj) > 0
        )
    mni_brain = output / "MNI152_2mm_brain.nii.gz"
    nib.save(nib.Nifti1Image(template_brain_data, template.affine), str(mni_brain))
    timing["synthstrip"] = time.perf_counter() - started

    started = time.perf_counter()
    feat = run_feat_core(
        bids_root=bids_root, output_dir=output / "feat", subject=subject,
        session=session, task=task, run=run, acquisition=acquisition,
        direction=direction, reconstruction=reconstruction, echo=echo,
        brain_mask=epi_mask, highpass_cutoff_seconds=highpass_cutoff_seconds,
        device=selected, batch_size=batch_size,
        motion_iterations=motion_iterations, overwrite=overwrite,
    )
    timing["feat_core"] = time.perf_counter() - started

    started = time.perf_counter()
    tissues = TorchFAST(device=selected)(t1_brain, mask=t1_mask)
    wm_pve = mask_dir / "T1_pve_wm.nii.gz"
    csf_pve = mask_dir / "T1_pve_csf.nii.gz"
    wm_seg = mask_dir / "T1_wmseg.nii.gz"
    tissues.pve_wm.save(wm_pve)
    tissues.pve_csf.save(csf_pve)
    _save_mask(np.asarray(tissues.pve_wm.data) >= 0.5, t1_brain, wm_seg)
    timing["fast"] = time.perf_counter() - started

    started = time.perf_counter()
    from .bbr import register_bbr

    bbr = register_bbr(
        epi=feat.output_dir / "example_func.nii.gz",
        t1=t1_brain, wmseg=wm_seg, device=selected,
    )
    reg_dir = output / "reg"
    reg_dir.mkdir(exist_ok=True)
    bbr_matrix = reg_dir / "example_func2highres.mat"
    bbr.save(
        output=reg_dir / "example_func2highres.nii.gz",
        omat=bbr_matrix,
    )
    t1_to_mni = register_t1_to_mni(
        t1_brain, mni_brain, reg_dir,
        backend=registration_backend,
        synthmorph_weights=synthmorph_weights,
        reference_mask=template_mask,
        device=selected,
    )
    timing["bbr_and_t1_to_mni"] = time.perf_counter() - started

    started = time.perf_counter()
    epi_ref = feat.output_dir / "example_func.nii.gz"
    csf_epi_pve = resample_world(
        csf_pve, epi_ref, bbr.moving_to_fixed_world,
        mask_dir / "csf_pve_epi.nii.gz", device=selected,
    )
    wm_epi_pve = resample_world(
        wm_pve, epi_ref, bbr.moving_to_fixed_world,
        mask_dir / "wm_pve_epi.nii.gz", device=selected,
    )
    brain = np.asarray(nib.load(str(feat.mask)).dataobj) > 0
    csf = (np.asarray(nib.load(str(csf_epi_pve)).dataobj) >= 0.8) & brain
    wm = (np.asarray(nib.load(str(wm_epi_pve)).dataobj) >= 0.8) & brain
    csf_mask = _save_mask(csf, epi_ref, mask_dir / "csf_epi.nii.gz")
    wm_mask = _save_mask(wm, epi_ref, mask_dir / "wm_epi.nii.gz")
    timing["aroma_masks"] = time.perf_counter() - started

    started = time.perf_counter()
    aroma = run_aroma_pipeline(
        filtered_func_data=feat.filtered_func_data,
        brain_mask=feat.mask,
        motion_parameters=feat.motion_parameters,
        csf_mask=classification_masks[0],
        edge_mask=classification_masks[1],
        outside_mask=classification_masks[2],
        mni_template=mni_template,
        mni_pull_ras=t1_to_mni.pull_ras,
        epi_to_t1_world=bbr.moving_to_fixed_world,
        output_dir=output / "aroma",
        n_components=ica_n_components,
        tr=inputs.tr,
        mode=aroma_mode,
        device=selected,
        n_splits=n_splits,
        random_state=random_state,
        ica_max_iter=ica_max_iter,
        wm_mask=wm_mask if regress_wm else None,
        regression_csf_mask=csf_mask if regress_csf else None,
        regress_csf=regress_csf,
        regress_motion=regress_motion,
        motion_model=motion_model,
        bandpass=bandpass,
        global_signal=global_signal,
    )
    timing["pica_aroma_confounds"] = time.perf_counter() - started

    started = time.perf_counter()
    to_epi_world = np.linalg.inv(bbr.moving_to_fixed_world)
    clean_native = aroma.confounds_cleaned_bold or aroma.denoised_bold
    mask_mni_raw = resample_world(
        feat.mask, mni_template, to_epi_world,
        mask_dir / "brain_MNI152_2mm.nii.gz",
        pre_affine_pull_ras=t1_to_mni.pull_ras,
        interpolation="nearest", device=selected,
    )
    mask_mni = _save_mask(
        (np.asarray(nib.load(str(mask_mni_raw)).dataobj) > 0.5)
        & (np.asarray(nib.load(str(template_mask)).dataobj) > 0),
        mni_template, mask_mni_raw,
    )
    resample_world(
        clean_native, mni_template, to_epi_world, clean_mni,
        pre_affine_pull_ras=t1_to_mni.pull_ras,
        output_mask=mask_mni, batch_size=batch_size, device=selected,
    )
    timing["mni_resampling"] = time.perf_counter() - started
    surface = None
    if surface_subject_dir is not None:
        started = time.perf_counter()
        prepared = prepare_fs_sphere_projection_inputs(
            subject_dir=surface_subject_dir, pull_ras=t1_to_mni.pull_ras,
            initial_t1_to_mni_world=t1_to_mni.moving_to_fixed_world,
            mni_reference=mni_template, hcp_assets_dir=surface_assets_dir,
            output_dir=output / "surface" / "prepared",
            wb_command=wb_command, device=selected, overwrite=overwrite,
        )
        surface_inputs = SurfacePipelineInputs(
            left=prepared.left, right=prepared.right,
            subject_rois=prepared.subject_rois, atlas_rois=prepared.atlas_rois,
            wb_command=wb_command,
        )
        timing["surface_preparation"] = time.perf_counter() - started
    if surface_inputs is not None:
        started = time.perf_counter()
        surface = run_surface_from_mni(
            clean_mni=clean_mni, mni_reference=mni_template,
            inputs=surface_inputs, output_dir=output / "surface",
            overwrite=overwrite,
        )
        timing["surface_projection"] = time.perf_counter() - started
    timing["total"] = sum(timing.values())
    report = output / "pipeline_report.json"
    report.write_text(json.dumps({
        "input_bold_shape": list(nib.load(str(inputs.bold)).shape),
        "tr_seconds": inputs.tr,
        "mni_shape": list(template.shape),
        "mni_voxel_mm": [float(value) for value in template.header.get_zooms()[:3]],
        "registration_backend": registration_backend,
        "ica_components": aroma.ica.n_components,
        "ica_converged": aroma.ica.converged,
        "ica_iterations": aroma.ica.n_iterations,
        "aroma_noise_components": len(aroma.noise_components.read_text().split()),
        "aroma_mode": aroma_mode,
        "wm_csf_motion_regression": {
            "wm": regress_wm, "csf": regress_csf, "motion": regress_motion,
        },
        "timing_seconds": timing,
        "cuda_peak_allocated_gb": (
            torch.cuda.max_memory_allocated(torch.device(selected)) / 1e9
            if str(selected).startswith("cuda") else None
        ),
        "cuda_peak_reserved_gb": (
            torch.cuda.max_memory_reserved(torch.device(selected)) / 1e9
            if str(selected).startswith("cuda") else None
        ),
        "outputs": {
            "clean_mni": clean_mni.name,
            "mask_mni": str(mask_mni.relative_to(output)),
            "feat_filtered": str(feat.filtered_func_data.relative_to(output)),
            "aroma_thresholded_ic_mni": "aroma/ica_thresholded_MNI152_2mm.nii.gz",
            "aroma_clean_native": str(clean_native.relative_to(output)),
            "surface_dtseries": (
                str(surface.projection.dtseries.relative_to(output))
                if surface is not None else None
            ),
            "surface_coverage_report": (
                str(surface.projection.coverage_report.relative_to(output))
                if surface is not None else None
            ),
            "surface_goodvoxels": (
                str(surface.qc.goodvoxels.relative_to(output))
                if surface is not None and surface.qc is not None else None
            ),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return FMRIPipelineResult(
        clean_mni=clean_mni, mask_mni=mask_mni, report=report,
        feat=feat, aroma=aroma, bbr_matrix=bbr_matrix,
        t1_to_mni=t1_to_mni, timing_seconds=timing, surface=surface,
    )
