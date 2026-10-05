"""One-run BIDS to T1w/MNI preproc and native/MNI ICA-AROMA clean BOLD."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import shutil
from tempfile import TemporaryDirectory
import time

import nibabel as nib
import numpy as np
import scipy
import torch

from ..synthstrip import SynthStrip
from ..fast import FASTConfig
from ..flirt import TorchFLIRT
from .. import __version__
from ..weights import resolve_weights
from .aroma_pipeline import run_aroma_pipeline
from .bids import locate_bids_inputs
from .derivatives import ensure_derivative_dataset, fmri_derivative_paths, sidecar, write_json, publish_derivatives
from ..applywarp import TorchApplyWarp, WorldTransformChain
from ..synthmorph import apply_transform
from ._anatomical import prepare_anatomical
from ..flirt.coordinates import flirt_to_world_affine
from .pipeline import run_feat_core
from .sampling_reference import native_bold_sampling_reference
from .timing import prepare_timing_parameters


@dataclass(frozen=True)
class FMRIVolumeResult:
    """Persistent preproc/clean BOLD, anatomy, transforms and run metadata."""

    clean_native: Path
    clean_mni: Path
    mask_mni: Path
    t1_brain: Path
    bbr_matrix: Path
    metadata: Path
    timing_seconds: dict[str, float]
    preproc_t1w: Path | None = None
    preproc_mni: Path | None = None
    motion_pull: Path | None = None
    mni_pull: Path | None = None


def _resample_final_volume(
    source, reference, reference_to_source_world, output, *, backend,
    pre_affine_pull_ras=None, output_mask=None, interpolation="linear",
    boundary="grid-constant", motion_pull_world=None,
    coordinate_precision="float64", spatial_chunk_size=262144,
    batch_size=8, device=None,
):
    """Route each final volume output through its public warp interface."""
    chain = WorldTransformChain(
        reference=reference,
        reference_to_source_world=reference_to_source_world,
        pre_affine_pull_ras=pre_affine_pull_ras,
        motion_pull_world=motion_pull_world,
        coordinate_precision=coordinate_precision,
    )
    if backend == "fnirt":
        return TorchApplyWarp(device=device).run_world(
            source, chain, output, interpolation=interpolation,
            boundary=boundary, output_mask=output_mask,
            batch_size=batch_size, spatial_chunk_size=spatial_chunk_size,
        )
    if backend != "synthmorph":
        raise ValueError("backend must be synthmorph or fnirt")
    image = apply_transform(
        source, chain, method=interpolation, fill=0, dtype="float32",
        device=device, frame_chunk_size=batch_size, boundary=boundary,
        output_mask=output_mask, spatial_chunk_size=spatial_chunk_size,
    )
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return output_path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standard_template_identity(path):
    image = nib.as_closest_canonical(nib.load(str(path)))
    if image.ndim != 3:
        raise ValueError("mni_template must be a 3D MNI152NLin6Asym res-02 template")
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(image.affine, dtype="<f8").tobytes())
    digest.update(np.asarray(image.dataobj, dtype="<f4").tobytes())
    if digest.hexdigest() not in {
        "63ab7291db3a91045a1ecf63e59280b747480922820911679e21f41bf2b09c4a",
        "cfef33f878a190c81bf11a407638d22e9f5b03c96c2888c959eff8d0f5a9f2de",
    }:
        raise ValueError("mni_template content is not the verified TemplateFlow "
                         "MNI152NLin6Asym res-02 full or brain-masked template")
    return {"StandardSpace": "MNI152NLin6Asym",
            "StandardTemplateSHA256": _sha256(path),
            "StandardTemplateIdentity": "TemplateFlow:MNI152NLin6Asym:res-02"}


def _classification_masks_on_template(template, mask_paths, output_dir):
    """Use exact axis permutations/flips for masks on the same physical grid.

    TemplateFlow can store the verified template in RAS while the original
    ICA-AROMA masks use LAS. Reorientation changes storage order only; it
    neither interpolates nor changes anatomical support or mask values.
    Different physical grids still fail before volume computation.
    """
    aligned = []
    target_orientation = nib.orientations.io_orientation(template.affine)
    for mask_path in mask_paths:
        image = nib.load(str(mask_path))
        if image.shape == template.shape and np.allclose(image.affine, template.affine, rtol=0, atol=1e-4):
            aligned.append(mask_path)
            continue
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), target_orientation)
        reoriented = image.as_reoriented(transform)
        if (reoriented.shape != template.shape or not np.allclose(
                reoriented.affine, template.affine, rtol=0, atol=1e-4)):
            raise ValueError("mni_template must match the ICA-AROMA MNI152 2-mm mask grid")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / Path(mask_path).name
        nib.save(reoriented, str(destination))
        aligned.append(destination)
    return tuple(aligned)


def _motion_world_pulls(raw, reference, matrices_dir):
    matrices = sorted(Path(matrices_dir).glob("MAT_*"))
    if len(matrices) != raw.shape[3]:
        raise ValueError("motion matrix count differs from the BOLD frame count")
    return np.stack([np.linalg.inv(flirt_to_world_affine(
        np.loadtxt(path), raw.affine, reference.affine, raw.shape[:3],
        reference.shape[:3], raw.header.get_zooms()[:3],
        reference.header.get_zooms()[:3],
    )) for path in matrices])


def _set_bold_tr(path, tr):
    image = nib.load(str(path))
    image.header.set_zooms((*image.header.get_zooms()[:3], float(tr)))
    image.header.set_xyzt_units(xyz="mm", t="sec")
    nib.save(image, str(path))


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
        selected = Path(requested).expanduser()
        logical = selected.parent.resolve() / selected.name
        for path in candidates:
            if path.parent.resolve() / path.name == logical:
                return path
        matching = [path for path in candidates if path.resolve() == selected.resolve()]
        if len(matching) != 1:
            raise ValueError("t1w_image must identify one of this BIDS run's T1w images; "
                             "use its BIDS path when several paths link to the same file")
        # Keep the logical BIDS path for derivative naming and Sources.
        return matching[0]
    if len(candidates) != 1:
        raise ValueError("multiple BIDS T1w images found; supply t1w_image")
    return candidates[0]


def _source_provenance(registration_backend):
    """Identify the installed runtime sources without requiring a Git checkout."""
    package = Path(__file__).resolve().parents[1]
    directories = (
        "fmri", "feat", "melodic", "fast", "synthstrip", "flirt", "mcflirt",
        "applywarp", "eddy", registration_backend,
    )
    files = {path for directory in directories
             for path in (package / directory).rglob("*.py")}
    files.update(package.glob("*.py"))
    hashes = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    return {
        "Version": __version__,
        "SourceSHA256": hashes,
        "SourceManifestSHA256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "Dependencies": {
            "torch": str(torch.__version__), "numpy": np.__version__,
            "nibabel": nib.__version__, "scipy": scipy.__version__,
        },
    }


def _weight_location(filename, requested):
    path = Path(resolve_weights(filename, explicit=requested)).expanduser().resolve()
    return {"Model": filename, "Path": str(path), "SizeBytes": path.stat().st_size}


def fMRIVolume_pipeline(
    bids_root,
    derivatives_root,
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
    fnirt_config=None,
    synthstrip_weights=None,
    synthmorph_weights=None,
    ica_n_components=None,
    aroma_mode="nonaggr",
    regress_wm=False,
    regress_csf=False,
    regress_motion=False,
    motion_model=24,
    bandpass=None,
    confound_projection="orthogonal",
    global_signal=False,
    highpass_cutoff_seconds=100.0,
    slice_timing=False,
    slice_time_reference=0.5,
    device=None,
    batch_size=8,
    motion_iterations=(1, 1, 1),
    ica_max_iter=500,
    n_splits=1000,
    random_state=0,
    overwrite=False,
    reuse_anatomical=True,
    bbr_execution="batched",
    fnirt_execution="optimized",
):
    """Generate both preproc and clean derivatives for one selected BIDS run.

    Preproc preserves raw intensity with one motion/spatial interpolation to
    T1w at BOLD resolution and MNI at 2 mm. Slice timing is off by default.
    Clean adds FEAT scaling/highpass, PICA/AROMA and optional confound
    regression, and is saved on the native EPI and MNI grids.

    Motion and anatomical estimation reuse TorchMCFLIRT, SynthStrip,
    source-ordered TorchFAST, BBR and the selected SynthMorph/FNIRT backend.
    Associated fieldmaps require a supplied B0 warp in the FEAT subfunction;
    this full entry does not estimate or accept B0/GDC warps.
    """
    pipeline_started = time.perf_counter()
    if registration_backend not in ("synthmorph", "fnirt"):
        raise ValueError("registration_backend must be 'synthmorph' or 'fnirt'")
    if bbr_execution not in ("reference", "batched"):
        raise ValueError("bbr_execution must be 'reference' or 'batched'")
    if fnirt_execution not in ("reference", "optimized"):
        raise ValueError("fnirt_execution must be 'reference' or 'optimized'")
    if registration_backend == "fnirt":
        from ..fnirt import resolve_fnirt_config
        fnirt_config = resolve_fnirt_config(fnirt_config, default="t1")
    elif fnirt_config is not None:
        raise ValueError("fnirt_config requires registration_backend='fnirt'")
    if registration_backend != "fnirt" and fnirt_execution != "optimized":
        raise ValueError("fnirt_execution requires registration_backend='fnirt'")
    inputs = locate_bids_inputs(
        bids_root, subject=subject, session=session, task=task, run=run,
        acquisition=acquisition, direction=direction,
        reconstruction=reconstruction, echo=echo,
    )
    t1w = _select_t1(inputs, t1w_image)
    paths = fmri_derivative_paths(inputs, t1w, derivatives_root)
    destinations = (paths.clean_native, paths.clean_mni, paths.mask_mni,
                    paths.t1_brain, paths.bbr_matrix, paths.preproc_t1w,
                    paths.preproc_mni, paths.motion_pull, paths.mni_pull)
    for destination in destinations:
        if destination == paths.t1_brain:
            # Anatomical derivatives are shared by runs. Validate identity
            # after obtaining the selected anatomy, before publishing.
            if not overwrite and any(path.is_symlink() and not path.exists()
                                     for path in (destination, sidecar(destination))):
                raise FileExistsError(destination)
            continue
        if (destination.exists() or destination.is_symlink()
                or (destination.name.endswith(".nii.gz")
                    and (sidecar(destination).exists() or sidecar(destination).is_symlink()))) and not overwrite:
            raise FileExistsError(destination)
    for destination in (paths.bbr_matrix.with_suffix(".json"), paths.motion_pull.with_suffix(".json")):
        if (destination.exists() or destination.is_symlink()) and not overwrite:
            raise FileExistsError(destination)
    template_identity = _standard_template_identity(mni_template)
    ensure_derivative_dataset(paths.root, inputs.bids_root)
    work = TemporaryDirectory(prefix="fnit-volume-")
    output = Path(work.name)
    clean_mni = output / "filtered_func_data_clean_MNI152_2mm.nii.gz"
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
    classification_masks = _classification_masks_on_template(
        template, classification_masks, output / "classification_masks")
    mask_dir = output / "masks"
    mask_dir.mkdir(exist_ok=True)
    reference = _reference_image(inputs, output / "reference_epi.nii.gz")
    strip = SynthStrip(weights=synthstrip_weights, device=selected)
    timing = {}

    started = time.perf_counter()
    epi_mask = _save_mask(
        strip(reference).mask.data, reference, mask_dir / "epi_synthstrip.nii.gz"
    )
    timing["epi_synthstrip"] = time.perf_counter() - started
    anatomical = prepare_anatomical(
        t1w, mni_template, template_mask=mni_brain_mask, strip=strip,
        backend=registration_backend, morph_weights=synthmorph_weights,
        fnirt_config=fnirt_config, device=selected, work_dir=output,
        cache_dir=paths.anat_dir / ".fnit_anatomical", reuse=reuse_anatomical,
        fnirt_execution=fnirt_execution,
    )
    timing.update(anatomical.timing_seconds)
    t1_brain = anatomical.path("T1_brain.nii.gz")
    t1_mask = anatomical.path("T1_mask.nii.gz")
    template_mask = anatomical.path("MNI_mask.nii.gz")
    wm_pve = anatomical.path("T1_pve_wm.nii.gz")
    csf_pve = anatomical.path("T1_pve_csf.nii.gz")
    wm_seg = anatomical.path("T1_wmseg.nii.gz")
    t1_to_mni = anatomical.registration

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

    from .bbr import register_bbr

    bbr = register_bbr(
        epi=feat.output_dir / "example_func.nii.gz",
        t1=t1_brain, wmseg=wm_seg, device=selected,
        execution=bbr_execution,
    )
    reg_dir = output / "reg"
    reg_dir.mkdir(exist_ok=True)
    bbr_matrix = reg_dir / "example_func2highres.mat"
    bbr.save(
        output=reg_dir / "example_func2highres.nii.gz",
        omat=bbr_matrix,
    )
    phases = bbr.phase_timings
    timing["bbr_initial_flirt"] = phases["initial_flirt"]
    timing["bbr_refinement"] = sum(phases[name] for name in (
        "boundary_preparation", "coarse_bbr", "local_bbr",
    ))
    timing["bbr_final_resampling"] = phases["final_resampling"]

    started = time.perf_counter()
    epi_ref = feat.output_dir / "example_func.nii.gz"
    brain = np.asarray(nib.load(str(feat.mask)).dataobj) > 0
    tissue_masks = {}
    tissue_resampler = TorchFLIRT(device=str(selected))
    t1_to_epi_matrix = np.linalg.inv(bbr.matrix)
    for name, enabled, pve in (("csf", regress_csf, csf_pve), ("wm", regress_wm, wm_pve)):
        if not enabled:
            continue
        # FLIRT 在较粗的 EPI 网格采样前预滤波，防止概率图降采样混叠。
        epi_pve = tissue_resampler.applyxfm(pve, epi_ref, init=t1_to_epi_matrix).moved
        nib.save(epi_pve, str(mask_dir / f"{name}_pve_epi.nii.gz"))
        tissue = (np.asarray(epi_pve.dataobj) >= 0.8) & brain
        if not tissue.any():
            raise ValueError(f"regress_{name}=True requires a nonempty EPI {name.upper()} mask")
        tissue_masks[name] = _save_mask(tissue, epi_ref, mask_dir / f"{name}_epi.nii.gz")
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
        wm_mask=tissue_masks.get("wm"),
        regression_csf_mask=tissue_masks.get("csf"),
        regress_csf=regress_csf,
        regress_motion=regress_motion,
        motion_model=motion_model,
        bandpass=bandpass,
        confound_projection=confound_projection,
        global_signal=global_signal,
    )
    timing["pica_aroma_confounds"] = time.perf_counter() - started

    started = time.perf_counter()
    to_epi_world = np.linalg.inv(bbr.moving_to_fixed_world)
    clean_native = aroma.confounds_cleaned_bold or aroma.denoised_bold
    mask_mni_raw = _resample_final_volume(
        feat.mask, mni_template, to_epi_world,
        mask_dir / "brain_MNI152_2mm.nii.gz",
        pre_affine_pull_ras=t1_to_mni.pull_ras,
        interpolation="nearest", device=selected, backend=registration_backend,
    )
    mask_mni = _save_mask(
        (np.asarray(nib.load(str(mask_mni_raw)).dataobj) > 0.5)
        & (np.asarray(nib.load(str(template_mask)).dataobj) > 0),
        mni_template, mask_mni_raw,
    )
    _resample_final_volume(
        clean_native, mni_template, to_epi_world, clean_mni,
        pre_affine_pull_ras=t1_to_mni.pull_ras,
        output_mask=mask_mni, interpolation="spline",
        boundary="periodic", batch_size=batch_size, device=selected,
        backend=registration_backend,
    )
    timing["mni_resampling"] = time.perf_counter() - started
    configuration = {
        "registration_backend": registration_backend,
        "fnirt_config": asdict(fnirt_config) if fnirt_config is not None else None,
        "ica_n_components": ica_n_components,
        "ica_max_iter": ica_max_iter,
        "aroma_mode": aroma_mode,
        "regress_wm": regress_wm, "regress_csf": regress_csf,
        "regress_motion": regress_motion, "motion_model": motion_model,
        "bandpass": list(bandpass) if bandpass is not None else None,
        "confound_projection": confound_projection,
        "global_signal": global_signal,
        "highpass_cutoff_seconds": highpass_cutoff_seconds,
        "slice_timing": slice_timing,
        "slice_time_reference": slice_time_reference,
        "device": str(selected), "batch_size": batch_size,
        "motion_iterations": list(motion_iterations),
        "motion_algorithm": "MCFLIRT-2111.0-8/4/4mm-coordinate-Brent",
        "motion_output": "NEWIMAGE-float32-Constant-spline-source-dtype-cast",
        "n_splits": n_splits, "random_state": random_state,
        "brain_extraction": "synthstrip",
        "mni_template": str(Path(mni_template).expanduser().resolve()),
        "mni_brain_mask": (
            str(Path(mni_brain_mask).expanduser().resolve()) if mni_brain_mask is not None else None
        ),
        "fast_config": asdict(FASTConfig(execution="fsl")),
        "reuse_anatomical": reuse_anatomical,
        "anatomical_cache": {"reused": anatomical.reused, "fingerprint": anatomical.fingerprint},
        "bbr_execution": bbr_execution,
        "fnirt_execution": fnirt_execution,
        "mni_interpolation": "cubic-bspline-periodic",
        "preproc_interpolation": "cubic-bspline-grid-constant",
        "preproc_coordinate_precision": "fmriprep",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "weights": {"synthstrip": _weight_location("synthstrip.1.pt", strip.model_path)},
    }
    if registration_backend == "synthmorph":
        configuration["weights"]["synthmorph"] = _weight_location(
            "synthmorph.deform.3.h5",
            synthmorph_weights.get("deform") if isinstance(synthmorph_weights, dict) else synthmorph_weights,
        )
    provenance = _source_provenance(registration_backend)

    started = time.perf_counter()
    raw_image = nib.load(str(inputs.bold))
    reference_image = nib.load(str(epi_ref))
    motion_pull = _motion_world_pulls(raw_image, reference_image, feat.motion_matrices)
    motion_pull_file = output / "motion_pull_world.npy"
    np.save(motion_pull_file, motion_pull)
    minimal_bold = inputs.bold
    stc_metadata = {"SliceTimingCorrected": False}
    if slice_timing and len(inputs.bold_metadata.get("SliceTiming", [])) > 1:
        from .slice_timing import slice_timing_correct
        minimal_bold, stc_metadata = slice_timing_correct(
            inputs.bold, output / "stc_bold.nii.gz", inputs.bold_metadata,
            reference_fraction=slice_time_reference, device=selected,
        )
    t1_reference = native_bold_sampling_reference(
        t1_brain, epi_ref, t1_mask, output / "T1w_native_bold_reference.nii.gz",
    )
    preproc_t1w = _resample_final_volume(
        minimal_bold, t1_reference, to_epi_world, output / "preproc_T1w.nii.gz",
        motion_pull_world=motion_pull, interpolation="spline",
        boundary="grid-constant", coordinate_precision="fmriprep",
        batch_size=batch_size, device=selected, backend=registration_backend,
    )
    preproc_mni = _resample_final_volume(
        minimal_bold, mni_template, to_epi_world, output / "preproc_MNI.nii.gz",
        pre_affine_pull_ras=t1_to_mni.pull_ras, motion_pull_world=motion_pull,
        interpolation="spline", boundary="grid-constant",
        coordinate_precision="fmriprep",
        batch_size=batch_size, device=selected, backend=registration_backend,
    )
    _set_bold_tr(preproc_t1w, inputs.tr)
    _set_bold_tr(preproc_mni, inputs.tr)
    timing["single_pass_preproc"] = time.perf_counter() - started
    timing["total"] = time.perf_counter() - pipeline_started
    report = output / "pipeline_report.json"
    report.write_text(json.dumps({
        "input_bold_shape": list(nib.load(str(inputs.bold)).shape),
        "tr_seconds": inputs.tr,
        "mni_shape": list(template.shape),
        "mni_voxel_mm": [float(value) for value in template.header.get_zooms()[:3]],
        "registration_backend": registration_backend,
        "mni_interpolation": "cubic-bspline-periodic",
        "t1_to_mni_qc": t1_to_mni.qc,
        "anatomical_cache": {"reused": anatomical.reused,
                             "fingerprint": anatomical.fingerprint},
        "bbr_phase_timings": bbr.phase_timings,
        "ica_components": aroma.ica.n_components,
        "ica_converged": aroma.ica.converged,
        "ica_iterations": aroma.ica.n_iterations,
        "aroma_noise_components": len(aroma.noise_components.read_text().split()),
        "aroma_mode": aroma_mode,
        "aroma_completed": True,
        "configuration": configuration,
        "source": provenance,
        "preproc": {
            "SliceTimingCorrected": stc_metadata["SliceTimingCorrected"],
            "SusceptibilityCorrection": False,
            "Interpolation": "cubic-bspline-grid-constant",
            "IntensityNormalization": None,
            "TemporalFiltering": None,
            "ConfoundRegression": False,
        },
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
            "preproc_t1w": preproc_t1w.name,
            "preproc_mni": preproc_mni.name,
            "mask_mni": str(mask_mni.relative_to(output)),
            "feat_filtered": str(feat.filtered_func_data.relative_to(output)),
            "aroma_thresholded_ic_mni": "aroma/ica_thresholded_MNI152_2mm.nii.gz",
            "aroma_clean_native": str(clean_native.relative_to(output)),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    publish_work = TemporaryDirectory(prefix=".fnit-volume-publish-", dir=paths.root)
    publish_files = []

    def staged_path(destination):
        staged = Path(publish_work.name) / destination.relative_to(paths.root)
        staged.parent.mkdir(parents=True, exist_ok=True)
        publish_files.append((staged, destination))
        return staged

    def write_details(destination, details):
        write_json(staged_path(destination), details)

    raw_bold = inputs.bold.relative_to(inputs.bids_root).as_posix()
    raw_t1w = t1w.relative_to(inputs.bids_root).as_posix()
    t1_details = {"Sources": [f"bids:raw:{raw_t1w}"], "SkullStripped": True}
    reuse_t1 = paths.t1_brain.is_file() and not overwrite
    if reuse_t1:
        if _sha256(paths.t1_brain) != _sha256(t1_brain):
            raise FileExistsError("shared T1w derivative differs from the selected anatomy; use overwrite=True")
        if sidecar(paths.t1_brain).exists() and json.loads(
            sidecar(paths.t1_brain).read_text(encoding="utf-8")
        ) != t1_details:
            raise ValueError("shared T1w derivative metadata differs from the selected source")
    for source, destination in (
        (clean_native, paths.clean_native), (clean_mni, paths.clean_mni),
        (mask_mni, paths.mask_mni), (t1_brain, paths.t1_brain),
        (bbr_matrix, paths.bbr_matrix),
        (preproc_t1w, paths.preproc_t1w), (preproc_mni, paths.preproc_mni),
        (motion_pull_file, paths.motion_pull), (t1_to_mni.pull_ras, paths.mni_pull),
    ):
        if destination == paths.t1_brain and reuse_t1:
            continue
        shutil.copyfile(source, staged_path(destination))
    metadata = {
        **{key: inputs.bold_metadata[key] for key in (
            "EchoTime", "FlipAngle", "MagneticFieldStrength", "Manufacturer",
            "PhaseEncodingDirection", "Units",
        ) if key in inputs.bold_metadata},
        "TaskName": inputs.bold_metadata["TaskName"], "RepetitionTime": inputs.tr,
        "SkullStripped": True,
        "Sources": [f"bids:raw:{raw_bold}", f"bids:raw:{raw_t1w}"],
        "FNIT": {
            "Signal": "clean",
            **template_identity,
            "SourceT1w": raw_t1w,
            "ConfoundRegression": {"wm": regress_wm, "csf": regress_csf,
                                  "motion": regress_motion},
            "RegistrationBackend": registration_backend,
            "Denoising": {"Method": "ICA-AROMA", "Mode": aroma_mode, "Completed": True},
            "Configuration": configuration,
            "Source": provenance,
            "MNIInterpolation": "cubic-bspline-periodic",
            "TissueInterpolation": "flirt-trilinear-prefilter-float32-coordinates",
            "TimingSeconds": timing,
            "Report": json.loads(report.read_text(encoding="utf-8")),
        },
    }
    for destination in (paths.clean_native, paths.clean_mni):
        details = dict(metadata)
        details.update(prepare_timing_parameters(inputs.bold_metadata, slice_timing_corrected=False))
        if destination == paths.clean_mni:
            details["Resolution"] = "2 mm isotropic"
        else:
            reference = inputs.sbref or inputs.bold
            details["SpatialReference"] = (
                f"bids:raw:{reference.relative_to(inputs.bids_root).as_posix()}"
            )
        write_details(sidecar(destination), details)
    for destination in (paths.preproc_t1w, paths.preproc_mni):
        details = dict(metadata)
        details.update(prepare_timing_parameters(
            inputs.bold_metadata, slice_timing_corrected=stc_metadata["SliceTimingCorrected"],
            reference_fraction=slice_time_reference,
        ))
        details.update(stc_metadata)
        details["SkullStripped"] = False
        details["FNIT"] = {
            **metadata["FNIT"], "Signal": "preproc",
            "Denoising": {"Method": None, "Mode": None, "Completed": False},
            "MNIInterpolation": "cubic-bspline-grid-constant",
            "ConfoundRegression": {"wm": False, "csf": False, "motion": False},
            "TemporalFiltering": None, "IntensityNormalization": None,
            "Interpolation": "cubic-bspline-grid-constant",
            "MotionPull": paths.motion_pull.name,
            "MNIToT1wPull": paths.mni_pull.name,
            "SliceTimingCorrection": stc_metadata["SliceTimingCorrected"],
            "SusceptibilityCorrection": False,
        }
        details["SpatialReference"] = ("MNI152NLin6Asym" if destination == paths.preproc_mni
                                       else f"bids:raw:{raw_t1w}")
        details["Resolution"] = ("2 mm isotropic" if destination == paths.preproc_mni
                                 else "native BOLD resolution")
        write_details(sidecar(destination), details)
    write_details(sidecar(paths.mask_mni), {"Sources": [f"bids:raw:{raw_bold}"],
                                                "Resolution": "2 mm isotropic",
                                                "Type": "Brain"})
    if not reuse_t1 or not sidecar(paths.t1_brain).is_file():
        write_details(sidecar(paths.t1_brain), t1_details)
    write_details(paths.bbr_matrix.with_suffix(".json"), {
        "Sources": [f"bids:raw:{raw_bold}", f"bids:raw:{raw_t1w}"],
        "Description": "EPI reference to T1w affine in FSL FLIRT matrix convention",
    })
    write_details(paths.motion_pull.with_suffix(".json"), {
        "Sources": [f"bids:raw:{raw_bold}"],
        "Description": "Per-frame boldref-world-RAS to original-frame-world-RAS pull matrices",
        "Shape": list(motion_pull.shape), "Units": "mm", "SHA256": _sha256(motion_pull_file),
    })
    publish_derivatives(publish_files, paths.root, overwrite=overwrite)
    publish_work.cleanup()
    work.cleanup()
    return FMRIVolumeResult(
        clean_native=paths.clean_native, clean_mni=paths.clean_mni,
        mask_mni=paths.mask_mni, t1_brain=paths.t1_brain,
        bbr_matrix=paths.bbr_matrix, metadata=sidecar(paths.clean_mni),
        timing_seconds=timing,
        preproc_t1w=paths.preproc_t1w, preproc_mni=paths.preproc_mni,
        motion_pull=paths.motion_pull, mni_pull=paths.mni_pull,
    )
