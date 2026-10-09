"""Map a verified BIDS Derivatives volume run to fsLR32k with MSMSulc."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import inspect
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np

from .._hemisphere_parallel import (
    hemisphere_items, map_hemispheres, resolve_cpu_threads, workbench_environment,
)
from ..flirt.coordinates import flirt_to_world_affine
from ..msm import MSMAllConfig, MSMAllInputs, prepare_msmsulc_inputs, run_msmsulc
from ..msm.config import MSMSulcConfig
from .assets_setup import BASE_URL, MESH, _sha256
from .bids import locate_bids_inputs
from .derivatives import ensure_derivative_dataset, fmri_derivative_paths, sidecar, write_json
from .normalization import resample_world
from .surface import SurfaceHemisphere
from .surface_fmriprep import (
    _cifti_assets, _las_grid, _millimeter_affine, _mni_grid, _publish_projection, _tr_seconds,
    fmriprep_cifti_metadata, run_fmriprep_surface_projection,
)
from .surface_prepare import load_fsnative_to_t1w, prepare_fmriprep_surface_inputs
from .surface_volume import inspect_surface_volume
from .surface_reconstruction import prepare_surface_reconstruction


@dataclass(frozen=True)
class FMRISurfaceResult:
    """Persistent BIDS surface data, metadata, QC and registered spheres."""

    left: Path
    right: Path
    dtseries: Path
    metadata: Path
    timing_seconds: dict[str, float]
    qc_report: Path | None = None
    registered_spheres: tuple[Path, Path] | None = None
    recon_all: Path | None = None
    volume_executed: bool = False


def _ensure_surface_volume(inputs, derivatives_root, *, signal, t1w_image,
                           hcp_assets_dir, device, auto_volume, volume_options):
    """Reuse verified inputs or run the mature volume entry once from raw BIDS."""
    from .end_to_end import fMRIVolume_pipeline

    options = dict(volume_options or {})
    locked = {"bids_root", "derivatives_root", "subject", "session", "task", "run",
              "acquisition", "direction", "reconstruction", "echo", "device", "t1w_image"}
    unexpected = set(options) - (set(inspect.signature(fMRIVolume_pipeline).parameters) - locked)
    if unexpected:
        raise ValueError(f"volume_options cannot override run identity or unknown options: {sorted(unexpected)}")
    template = options.get("mni_template")
    if template is None:
        candidate = Path(hcp_assets_dir).expanduser() / "fmriprep/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"
        if candidate.is_file():
            template = candidate
            options["mni_template"] = candidate
    status = inspect_surface_volume(
        inputs, derivatives_root, signal=signal, t1w_image=t1w_image,
        mni_template=template, hcp_assets_dir=hcp_assets_dir,
    )
    initial_state = status.state
    elapsed = 0.0
    executed = False
    if status.state == "missing":
        if not auto_volume:
            raise FileNotFoundError("FNIT volume is missing and auto_volume=False: " + "; ".join(status.reasons))
        if template is None:
            raise FileNotFoundError("automatic volume requires volume_options['mni_template'] or the installed surface-assets MNI T1w template")
        # Reuse the exact selected BIDS entities; a filename reconstruction is
        # never used to guess a different run.
        from .bids import _entities
        parsed = _entities(inputs.bold)
        if parsed is None:
            raise ValueError("selected BIDS BOLD filename lacks entities")
        entities, _ = parsed
        started = time.perf_counter()
        fMRIVolume_pipeline(
            inputs.bids_root, derivatives_root, subject=inputs.subject,
            session=inputs.session, task=entities.get("task", "rest"),
            run=entities.get("run"), acquisition=entities.get("acq"),
            direction=entities.get("dir"), reconstruction=entities.get("rec"),
            echo=entities.get("echo"), t1w_image=status.source_t1w, device=device, **options,
        )
        elapsed = time.perf_counter() - started
        executed = True
        status = inspect_surface_volume(
            inputs, derivatives_root, signal=signal, t1w_image=status.source_t1w,
            mni_template=template, hcp_assets_dir=hcp_assets_dir,
        )
    if status.state != "ready":
        if status.state == "partial":
            hint = "; rerun volume or explicitly select signal='clean'" if signal == "preproc" else ""
            raise FileNotFoundError("FNIT volume is partial: " + "; ".join(status.reasons) + hint)
        raise ValueError(f"FNIT volume is {status.state}: " + "; ".join(status.reasons))
    return status, {"InitialState": initial_state, "Executed": executed,
                    "Reused": not executed, "Seconds": elapsed}


def _world_affine(value):
    matrix = load_fsnative_to_t1w(value)
    if not np.isfinite(np.linalg.det(matrix[:3, :3])):
        raise ValueError("fsnative_to_t1w must have a finite invertible world affine")
    return matrix


def _matching_original_t1(subject_dir, source_t1, world_affine):
    scanner = nib.as_closest_canonical(nib.load(str(subject_dir / "mri/orig/001.mgz")))
    raw = nib.as_closest_canonical(nib.load(str(source_t1)))
    if scanner.ndim != 3 or raw.ndim != 3:
        raise ValueError("recon-all original T1 and source BIDS T1 must be 3D")
    if world_affine is not None:
        return {"OriginalT1Identity": "explicit fsnative-to-T1w world affine",
                "FsnativeToT1wWorldAffine": world_affine.tolist()}
    if scanner.shape != raw.shape or not np.allclose(scanner.affine, raw.affine, rtol=0, atol=1e-4):
        raise ValueError("recon-all original T1 and source BIDS T1 have different grids; provide fsnative_to_t1w")
    scanner_data = np.asarray(scanner.dataobj, dtype=np.float32)
    raw_data = np.asarray(raw.dataobj, dtype=np.float32)
    if (not np.isfinite(scanner_data).all() or not np.isfinite(raw_data).all()
            or not np.allclose(scanner_data, raw_data, rtol=1e-5, atol=1e-3)):
        raise ValueError("recon-all original T1 does not match the source BIDS T1 image content")
    return {"OriginalT1Identity": "matching source image grid and voxel content",
            "FsnativeToT1wWorldAffine": np.eye(4).tolist()}


def _surface_extra_paths(paths, signal):
    report = sidecar(paths.dtseries).with_name(
        sidecar(paths.dtseries).name.removesuffix("_bold.json") + "_report.json"
    )
    spheres = tuple(path.with_name(
        path.name.split("_space-fsLR_")[0] + f"_space-fsLR_desc-{signal}Reg_sphere.surf.gii"
    ) for path in (paths.left, paths.right))
    sphere_json = tuple(path.with_name(path.name.removesuffix(".surf.gii") + ".json") for path in spheres)
    return report, spheres, sphere_json


def _check_final_outputs(destinations, overwrite):
    for path in destinations:
        if path.is_dir():
            raise ValueError(f"output file is a directory: {path}")
        if (path.exists() or path.is_symlink()) and not overwrite:
            raise FileExistsError(path)


def _validate_run_metadata(metadata, inputs, source_t1):
    """Require the selected BOLD/T1w sources and recorded repetition time."""
    required_sources = {
        f"bids:raw:{path.relative_to(inputs.bids_root).as_posix()}"
        for path in (inputs.bold, source_t1)
    }
    sources = metadata.get("Sources", [])
    if not isinstance(sources, list) or not required_sources.issubset(sources):
        raise ValueError("volume derivative sources do not match the selected BIDS BOLD and T1w")
    recorded_tr = metadata.get("RepetitionTime")
    if (isinstance(recorded_tr, bool) or not isinstance(recorded_tr, (int, float))
            or not np.isfinite(recorded_tr) or not np.isclose(recorded_tr, inputs.tr, rtol=1e-5, atol=1e-6)):
        raise ValueError("volume derivative RepetitionTime differs from the selected BIDS run")


def _validate_volume_metadata(metadata, inputs, source_t1):
    """Require completed AROMA, without imposing extra regression."""
    _validate_run_metadata(metadata, inputs, source_t1)
    details = metadata.get("FNIT", {})
    if details.get("Signal") not in (None, "clean"):
        raise ValueError("clean volume metadata explicitly identifies a different signal")
    if details.get("SourceT1w") != source_t1.relative_to(inputs.bids_root).as_posix():
        raise ValueError("clean volume derivative does not identify the selected source T1w")
    report = details.get("Report", {})
    denoising = details.get("Denoising")
    if denoising is None:
        completed = report.get("aroma_mode") in ("nonaggr", "aggr") and report.get("ica_converged") is True
    else:
        completed = (isinstance(denoising, dict) and denoising.get("Method") == "ICA-AROMA"
                     and denoising.get("Mode") in ("nonaggr", "aggr")
                     and denoising.get("Completed") is True)
        if "ica_converged" in report and report["ica_converged"] is not True:
            completed = False
    if not completed:
        raise ValueError("volume derivative lacks completed ICA-AROMA denoising")


def _validate_preproc_metadata(metadata, inputs, source_t1):
    _validate_run_metadata(metadata, inputs, source_t1)
    details = metadata.get("FNIT", {})
    denoising = details.get("Denoising") or {}
    if (details.get("Signal") != "preproc"
            or not isinstance(denoising, dict)
            or denoising.get("Completed") is True
            or any(details.get("ConfoundRegression", {}).values())
            or details.get("TemporalFiltering") is not None
            or details.get("IntensityNormalization") is not None):
        raise ValueError("preprocessed volume metadata describes cleaned or scaled data; rerun volume")


def _validate_native_bold(image, inputs):
    raw = nib.load(str(inputs.bold))
    reference = nib.load(str(inputs.sbref or inputs.bold))
    if (image.ndim != 4 or image.shape[:3] != reference.shape[:3]
            or image.shape[3] != raw.shape[3]
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4)):
        raise ValueError("native volume derivative does not match the selected BIDS BOLD reference grid")
    if not _tr_matches(image, inputs.tr):
        raise ValueError("native volume derivative TR differs from the selected BIDS run")


def _tr_matches(image, tr):
    unit = image.header.get_xyzt_units()[1]
    scale = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}.get(unit)
    return image.ndim == 4 and scale is not None and np.isclose(
        float(image.header.get_zooms()[3]) * scale, tr, rtol=1e-5, atol=1e-6
    )


def _saved_native_sphere_qc(sphere, reference, *, baseline):
    """Check the actual saved native mesh, separately from a 32k solver mesh."""
    from ..msm._affine import _surface
    from ..msm.msmsulc import _native_output_qc

    points, faces = _surface(sphere)
    original, original_faces = _surface(reference)
    if points.shape != original.shape or not np.array_equal(faces, original_faces):
        raise ValueError("final native sphere must preserve the reference vertex order and topology")
    qc = _native_output_qc(points, faces, original)
    # A saved-sphere reread cannot recover pre-save solver coordinates.
    qc.pop("folded_solver_faces")
    qc.pop("minimum_solver_orientation_ratio")
    qc.pop("absolute_folded_solver_faces")
    qc.pop("minimum_absolute_solver_orientation_ratio")
    qc.pop("new_relative_folded_solver_faces")

    def determinants(vertices):
        triangles = vertices[faces]
        return (np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0]) * triangles[:, 0]).sum(1)

    before, after = determinants(original), determinants(points)
    # Use the reference mesh's majority winding, while also retaining the
    # raw signed counts. A relative ratio alone can conceal an existing fold
    # when its reference face already points inward.
    sign = 1 if np.count_nonzero(before > 0) >= np.count_nonzero(before < 0) else -1
    qc.update(
        baseline=baseline,
        coordinates="saved GIFTI coordinates",
        reference_sha256=_sha256(Path(reference)),
        output_sha256=_sha256(Path(sphere)),
        vertex_count=len(points), face_count=len(faces),
        absolute_orientation_reference_sign=sign,
        absolute_folded_input_faces=int(np.count_nonzero(before * sign <= 0)),
        absolute_folded_output_faces=int(np.count_nonzero(after * sign <= 0)),
        negative_input_faces=int(np.count_nonzero(before < 0)),
        negative_output_faces=int(np.count_nonzero(after < 0)),
        degenerate_output_faces=int(np.count_nonzero(after == 0)),
    )
    qc["orientation_qc"] = "warning" if (
        qc["absolute_folded_output_faces"] or qc["degenerate_input_faces"]
    ) else "pass"
    return qc


def _orientation_stage(report):
    """Summarize orientation only; never promote an unavailable check to pass."""
    hemispheres = {}
    for hemi in ("L", "R"):
        item = report.get(hemi, {}) if isinstance(report, dict) else {}
        folded_key = ("absolute_folded_output_faces" if "absolute_folded_output_faces" in item
                      else "folded_output_faces")
        if any(item.get(key, 0) > 0 for key in (
            folded_key, "degenerate_input_faces",
        )) or item.get("orientation_qc") == "warning":
            status = "warning"
        elif "folded_output_faces" in item:
            status = "pass"
        else:
            status = "not_assessed"
        hemispheres[hemi] = status
    statuses = tuple(hemispheres.values())
    status = "warning" if "warning" in statuses else (
        "not_assessed" if "not_assessed" in statuses else "pass"
    )
    return {"status": status, "hemispheres": hemispheres}


def _orientation_chain(initial, solver, final):
    stages = {
        "initial_msmsulc": _orientation_stage(initial),
        "solver_msmall": _orientation_stage(solver) if solver is not None
        else {"status": "not_applicable"},
        "final_native": _orientation_stage(final),
    }
    statuses = tuple(item["status"] for item in stages.values())
    stages["all_stages"] = "warning" if "warning" in statuses else (
        "not_assessed" if "not_assessed" in statuses else "pass"
    )
    return stages


def _refine_msmall(inputs, native_spheres, native_geometry, assets, output,
                   configuration, device, execution, wb_command,
                   parallel=True, cpu_threads=None, native_qc_references=None,
                   qc_policy="report"):
    """Refine native features or compose a prepared fsLR32k registration.

    Native feature arrays retain recon-all vertex order. Features on the
    canonical 32k sphere describe the MSMSulc representation; their estimated
    warp is composed onto the native MSMSulc spheres before BOLD projection.
    Final saved native coordinates have their own explicit QC policy; a
    clean solver mesh does not establish that the composed native mesh is clean.
    """
    from ..msm import run_msmall
    from ..msm._affine import _surface

    if qc_policy not in ("report", "repair", "error"):
        raise ValueError("qc_policy must be 'report', 'repair' or 'error'")

    entries = {}
    topology = {}
    for hemi, native_sphere, geometry in zip(("L", "R"), native_spheres, native_geometry):
        entry = inputs[hemi]
        source_points, source_faces = _surface(entry.source_sphere)
        # The registration sphere is the coordinate mesh for MSMAll.  A
        # midthickness GIFTI is allowed to carry the same vertices with a
        # different triangle ordering (for example after a Workbench export),
        # so using it as the topology oracle can reject valid native feature
        # files before MSMAll starts.  Compare against the actual MSMSulc
        # sphere passed to this stage; the midthickness geometry is only used
        # later for area-surface projection.
        native_points, native_faces = _surface(native_sphere)
        atlas_file = assets / MESH / f"{hemi}.sphere.32k_fs_LR.surf.gii"
        atlas_points, atlas_faces = _surface(atlas_file)
        if len(source_points) == len(native_points) and np.array_equal(source_faces, native_faces):
            topology[hemi] = "native"
            if entry.initial_sphere is None:
                entry = replace(entry, initial_sphere=Path(native_sphere))
        elif (source_points.shape == atlas_points.shape
              and np.array_equal(source_faces, atlas_faces)
              and np.allclose(source_points, atlas_points, atol=1e-5, rtol=0)):
            topology[hemi] = "fsLR32k"
        else:
            raise ValueError(f"{hemi} MSMAll source features must use matching native topology or the canonical fsLR32k sphere")
        entries[hemi] = entry
    registered = run_msmall(entries, output / "msmall", device=device,
                           config=configuration, execution=execution,
                           parallel=parallel, cpu_threads=cpu_threads)
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(wb_command)
    def compose(hemi, threads):
        native_sphere = native_spheres[0 if hemi == "L" else 1]
        if topology[hemi] == "native":
            return registered[hemi]
        final = output / "msmall" / f"{hemi}.sphere.MSMAll.native.surf.gii"
        # Keep the 32k solver sphere and composed native sphere distinct.
        if final == registered[hemi]:
            final = output / "msmall" / f"{hemi}.sphere.MSMAll.composed-native.surf.gii"
        subprocess.run([
            executable, "-surface-sphere-project-unproject", str(native_sphere),
            str(entries[hemi].source_sphere), str(registered[hemi]), str(final),
        ], check=True, capture_output=True, text=True, env=workbench_environment(threads))
        return final
    final_spheres = map_hemispheres(compose, parallel=parallel, cpu_threads=cpu_threads)
    references = native_qc_references or dict(zip(("L", "R"), native_spheres))
    baseline = "undeformed native sphere" if native_qc_references is not None else "input MSMSulc native sphere"
    native_report = {}
    checked_spheres = []
    for hemi, final in zip(("L", "R"), final_spheres):
        before = _saved_native_sphere_qc(final, references[hemi], baseline=baseline)
        repair = {"policy": qc_policy, "applied": False, "moved_vertices": 0,
                  "unfold_updates": 0, "seconds": 0.0}
        if qc_policy == "repair" and before["orientation_qc"] == "warning":
            import torch
            from ..msm._native_repair import repair_native_sphere

            repair_started = time.perf_counter()
            points, faces = _surface(final)
            reference_points, _ = _surface(references[hemi])
            repaired, repair_details = repair_native_sphere(
                torch.as_tensor(points, dtype=torch.float64, device=device), faces, reference_points
            )
            vertices = repaired.detach().cpu().numpy().astype(np.float32)
            # Keep solver coordinates available for independent precision/QC
            # comparisons, including when source features use native topology.
            if final == registered[hemi]:
                final = output / "msmall" / f"{hemi}.sphere.MSMAll.repaired-native.surf.gii"
            # Preserve GIFTI metadata and face order from the composed output.
            image = nib.load(str(final_spheres[0 if hemi == "L" else 1]))
            image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data = vertices
            nib.save(image, str(final))
            repair.update(repair_details,
                          seconds=time.perf_counter() - repair_started)
        # Re-read the actual float32 GIFTI after repair; solver/double precision
        # QC cannot certify the coordinates consumed by Workbench projection.
        after = (_saved_native_sphere_qc(final, references[hemi], baseline=baseline)
                 if repair["applied"] else before)
        repair["success"] = after["orientation_qc"] == "pass"
        native_report[hemi] = {**after, "native_output_qc_before_repair": before,
                               "fold_repair": repair}
        checked_spheres.append(final)
    final_spheres = tuple(checked_spheres)
    native_report["scope"] = "final saved native sphere after MSMAll and optional fsLR32k composition"
    native_report["qc_policy"] = qc_policy
    native_report["orientation_qc"] = _orientation_stage(native_report)["status"]
    (output / "msmall" / "native_composition_report.json").write_text(
        json.dumps(native_report, indent=2) + "\n", encoding="utf-8"
    )
    if qc_policy in ("repair", "error") and native_report["orientation_qc"] != "pass":
        failures = ", ".join(
            f"{hemi}: {native_report[hemi]['absolute_folded_output_faces']} folded faces"
            for hemi in ("L", "R") if native_report[hemi]["orientation_qc"] != "pass"
        )
        action = "fold repair did not pass QC" if qc_policy == "repair" else "has failed orientation QC"
        raise RuntimeError(f"MSMAll final native sphere {action} ({failures}); BOLD projection refused")
    return final_spheres, topology


def fMRISurface_pipeline(
    bids_root: str | Path, derivatives_root: str | Path, *,
    subject: str, hcp_assets_dir: str | Path, recon_all: str | Path | None = None,
    session: str | None = None, task: str = "rest", run: str | None = None,
    acquisition: str | None = None, direction: str | None = None,
    reconstruction: str | None = None, echo: str | None = None,
    wb_command: str | Path = "wb_command", device: str = "cuda:0",
    overwrite: bool = False,
    registered_spheres: tuple[str | Path, str | Path] | None = None,
    msm_config: MSMSulcConfig | str | Path | None = None,
    msm_execution: str = "optimized",
    msmsulc_qc_policy: str = "report",
    msmall_inputs: dict[str, MSMAllInputs] | str | Path | None = None,
    msmall_config: MSMAllConfig | str | Path | None = None,
    msmall_qc_policy: str = "report",
    goodvoxels: str | Path | None = None,
    signal: str = "preproc",
    fsnative_to_t1w: str | Path | np.ndarray | None = None,
    parallel: bool = True, cpu_threads: int | None = None,
    recon_all_backend: str | None = None,
    recon_all_output_dir: str | Path | None = None,
    recon_all_options: dict | None = None,
    t1w_image: str | Path | None = None,
    auto_volume: bool = True,
    volume_options: dict | None = None,
) -> FMRISurfaceResult:
    """Run raw BIDS through verified volume, reconstruction and surface stages.

    Missing volume is computed automatically; partial or invalid volume is
    rejected without changing it. ``volume_options`` configures the existing
    volume API while the selected BIDS run, T1w and device remain fixed.
    ``recon_all_backend`` selects ``fnit``, ``freesurfer`` or ``provided``;
    omission selects provided when recon_all is passed, otherwise FNIT.
    ``recon_all`` remains a read-only input, and ``recon_all_output_dir``
    stores a generated reconstruction with validated reuse provenance.

    ``signal='preproc'`` uses the volume's retained T1w/MNI preprocessed
    series, matching fMRIPrep's projection input. ``signal='clean'`` explicitly
    selects denoised volume data. Missing preprocessed data never fall back
    to clean data. ``fsnative_to_t1w`` maps recon-all scanner RAS to the source
    T1w RAS; omitting it requires the original images to match in content.
    ``parallel=True`` runs independent L/R preparation, registration and
    projection branches. ``cpu_threads`` is the shared total budget; None
    uses OMP_NUM_THREADS or the current PyTorch thread count. A budget of one
    selects serial execution. Parent process settings are preserved.
    ``msmsulc_qc_policy`` controls the initial MSMSulc sphere: ``report``
    preserves source-compatible interpolation, ``repair`` explicitly unfolds
    it, and ``error`` refuses to write it when folded. ``msmall_qc_policy``
    independently controls MSMAll's final saved native sphere: ``report``
    retains the composed coordinates, ``repair`` unfolds and rechecks the
    saved float32 GIFTI, and ``error`` rejects folds before projection.
    MSMAll solver output and final native composition are checked separately;
    their reports do not replace a warning in the initial registration.
    """
    started = time.perf_counter()
    budget = resolve_cpu_threads(cpu_threads)
    hemisphere_items(parallel=parallel, cpu_threads=budget)
    if signal not in ("preproc", "clean"):
        raise ValueError("signal must be 'preproc' or 'clean'")
    selected_backend = recon_all_backend or ("provided" if recon_all is not None else "fnit")
    if selected_backend not in ("fnit", "freesurfer", "provided"):
        raise ValueError("recon_all_backend must be 'fnit', 'freesurfer' or 'provided'")
    if selected_backend == "provided" and recon_all is None:
        raise ValueError("provided reconstruction requires recon_all input")
    if selected_backend != "provided" and recon_all is not None:
        raise ValueError("recon_all is a provided input; use recon_all_output_dir for generated reconstruction")
    if registered_spheres is not None and len(registered_spheres) != 2:
        raise ValueError("registered_spheres must contain left and right paths")
    if registered_spheres is not None and msm_config is not None:
        raise ValueError("msm_config cannot be applied to supplied registered_spheres")
    if registered_spheres is not None and msmsulc_qc_policy != "report":
        raise ValueError("msmsulc_qc_policy cannot be applied to supplied registered_spheres")
    if registered_spheres is not None and msmall_inputs is not None:
        raise ValueError("msmall_inputs cannot be applied to supplied registered_spheres")
    if msmall_inputs is None and msmall_config is not None:
        raise ValueError("msmall_config requires prepared msmall_inputs")
    if msmall_inputs is None and msmall_qc_policy != "report":
        raise ValueError("msmall_qc_policy requires prepared msmall_inputs")
    msmall_configuration = None
    if msmall_inputs is not None:
        from ..msm.cli import load_inputs
        if isinstance(msmall_inputs, (str, Path)):
            msmall_inputs = load_inputs(msmall_inputs, MSMAllInputs)
        if (not isinstance(msmall_inputs, dict) or set(msmall_inputs) != {"L", "R"}
                or not all(isinstance(value, MSMAllInputs) for value in msmall_inputs.values())):
            raise ValueError("msmall_inputs must contain prepared L/R MSMAllInputs")
        if msmall_config is None:
            msmall_configuration = MSMAllConfig()
        elif isinstance(msmall_config, (str, Path)):
            msmall_configuration = MSMAllConfig.from_file(msmall_config)
        elif isinstance(msmall_config, MSMAllConfig):
            msmall_configuration = msmall_config
        else:
            raise TypeError("msmall_config must be MSMAllConfig or a config path")
    if msm_execution not in ("optimized", "reference"):
        raise ValueError("msm_execution must be 'optimized' or 'reference'")
    if msmsulc_qc_policy not in ("report", "repair", "error"):
        raise ValueError("msmsulc_qc_policy must be 'report', 'repair' or 'error'")
    if msmall_qc_policy not in ("report", "repair", "error"):
        raise ValueError("msmall_qc_policy must be 'report', 'repair' or 'error'")
    if msm_config is None:
        configuration = MSMSulcConfig()
    elif isinstance(msm_config, MSMSulcConfig):
        configuration = msm_config
    elif isinstance(msm_config, (str, Path)):
        configuration = MSMSulcConfig.from_file(msm_config)
    else:
        raise TypeError("msm_config must be MSMSulcConfig or a config path")
    registration_details = {"Method": "provided spheres"}
    registration_seconds = None
    world_affine = None if fsnative_to_t1w is None else _world_affine(fsnative_to_t1w)
    inputs = locate_bids_inputs(
        bids_root, subject=subject, session=session, task=task, run=run,
        acquisition=acquisition, direction=direction, reconstruction=reconstruction, echo=echo,
    )
    # Functional output names do not depend on T1 selection. Fail before
    # expensive automatic prerequisites if their destination is protected.
    preliminary = fmri_derivative_paths(inputs, inputs.t1w_images[0], derivatives_root, signal=signal)
    preliminary_signal = signal
    if msmall_inputs is not None:
        preliminary_signal = f"MSMAll{signal}"
        preliminary = replace(preliminary, **{
            name: getattr(preliminary, name).with_name(getattr(preliminary, name).name.replace(
                f"_desc-{signal}_bold", f"_desc-{preliminary_signal}_bold"))
            for name in ("left", "right", "dtseries")
        })
    preliminary_data = (preliminary.left, preliminary.right, preliminary.dtseries)
    report_path, sphere_files, sphere_json = _surface_extra_paths(preliminary, preliminary_signal)
    _check_final_outputs((*preliminary_data, *(sidecar(path) for path in preliminary_data),
                          report_path, *sphere_files, *sphere_json), overwrite)
    volume_started = time.perf_counter()
    volume_status, volume_details = _ensure_surface_volume(
        inputs, derivatives_root, signal=signal, t1w_image=t1w_image,
        hcp_assets_dir=hcp_assets_dir, device=device, auto_volume=auto_volume,
        volume_options=volume_options,
    )
    volume_check_seconds = time.perf_counter() - volume_started - volume_details["Seconds"]
    # The run's functional paths are independent of which T1 was selected.
    # Resolve SourceT1w before checking any T1-specific derivative.
    probe = fmri_derivative_paths(inputs, inputs.t1w_images[0], derivatives_root, signal=signal)
    mni_bold = probe.preproc_mni if signal == "preproc" else probe.clean_mni
    mni_sidecar = sidecar(mni_bold)
    if not mni_bold.is_file() or not mni_sidecar.is_file():
        if signal == "preproc":
            raise FileNotFoundError("preprocessed FNIT volume derivatives are missing; rerun volume or explicitly select signal='clean'")
        raise FileNotFoundError("completed FNIT clean volume BIDS derivative not found")
    metadata = json.loads(mni_sidecar.read_text(encoding="utf-8"))
    source_label = metadata.get("FNIT", {}).get("SourceT1w")
    if not isinstance(source_label, str) or not source_label:
        raise ValueError("volume derivative does not identify its source T1w; rerun volume")
    source_relative = Path(source_label)
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise ValueError("SourceT1w must be a relative BIDS source path without '..'")
    recorded_source = inputs.bids_root / source_relative
    logical_matches = [path for path in inputs.t1w_images if path == recorded_source]
    identity_matches = logical_matches or [
        path for path in inputs.t1w_images if path.resolve() == recorded_source.resolve()
    ]
    if not identity_matches:
        raise ValueError("volume derivative T1w does not belong to this BIDS subject")
    if len(identity_matches) != 1:
        raise ValueError("volume derivative SourceT1w is ambiguous; record its exact BIDS path")
    # A BIDS image may link to external storage. Use the listed logical
    # candidate for derivative naming and Sources; resolve only for identity.
    source_t1 = identity_matches[0]
    source_label = source_t1.relative_to(inputs.bids_root).as_posix()
    expected_bold_source = f"bids:raw:{inputs.bold.relative_to(inputs.bids_root).as_posix()}"
    if expected_bold_source not in metadata.get("Sources", []):
        raise ValueError("volume derivative does not identify the requested raw BOLD source; rerun volume")
    space = metadata.get("FNIT", {})
    if signal == "preproc":
        _validate_preproc_metadata(metadata, inputs, source_t1)
    template_hash = space.get("StandardTemplateSHA256", "")
    if (space.get("StandardSpace") != "MNI152NLin6Asym"
            or space.get("StandardTemplateIdentity") != "TemplateFlow:MNI152NLin6Asym:res-02"
            or not isinstance(template_hash, str) or len(template_hash) != 64
            or any(character.lower() not in "0123456789abcdef" for character in template_hash)):
        raise ValueError("volume derivative lacks verified MNI152NLin6Asym template identity; rerun volume")
    paths = fmri_derivative_paths(inputs, source_t1, derivatives_root, signal=signal)
    output_signal = signal
    if msmall_inputs is not None:
        output_signal = f"MSMAll{signal}"
        paths = replace(paths, **{
            name: getattr(paths, name).with_name(getattr(paths, name).name.replace(
                f"_desc-{signal}_bold", f"_desc-{output_signal}_bold"))
            for name in ("left", "right", "dtseries")
        })
    qc_report, sphere_paths, sphere_sidecars = _surface_extra_paths(paths, output_signal)
    final_data = (paths.left, paths.right, paths.dtseries)
    final_sidecars = tuple(sidecar(path) for path in final_data)
    final_outputs = (*final_data, *final_sidecars, qc_report, *sphere_paths, *sphere_sidecars)
    _check_final_outputs(final_outputs, overwrite)
    selected_mni = paths.preproc_mni if signal == "preproc" else paths.clean_mni
    selected_t1w = paths.preproc_t1w if signal == "preproc" else paths.clean_native
    required = (selected_mni, selected_t1w, paths.t1_brain)
    if signal == "preproc":
        required += (sidecar(selected_t1w),)
    if signal == "clean":
        required += (paths.bbr_matrix, sidecar(selected_t1w))
        _validate_volume_metadata(metadata, inputs, source_t1)
    for path in required:
        if not path.is_file():
            if signal == "preproc":
                raise FileNotFoundError(f"preprocessed FNIT input is missing: {path}; rerun volume or select signal='clean'")
            raise FileNotFoundError(path)
    ensure_derivative_dataset(paths.root, inputs.bids_root)
    assets = Path(hcp_assets_dir).expanduser().resolve()
    mesh = assets / MESH
    left_roi = mesh / "L.atlasroi.32k_fs_LR.shape.gii"
    right_roi = mesh / "R.atlasroi.32k_fs_LR.shape.gii"
    dseg = assets / "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"
    labels, _ = _cifti_assets(left_roi, right_roi, dseg)
    mni_image = nib.load(str(selected_mni))
    input_image = nib.load(str(selected_t1w))
    t1_image = nib.load(str(paths.t1_brain))
    _tr_seconds(mni_image, "MNI BOLD", inputs.tr)
    _tr_seconds(input_image, "T1w BOLD" if signal == "preproc" else "native BOLD", inputs.tr)
    if input_image.shape[3] != mni_image.shape[3] or input_image.shape[3] != nib.load(str(inputs.bold)).shape[3]:
        raise ValueError("surface inputs and raw BOLD must have equal frame counts")
    _mni_grid(_las_grid(mni_image), labels)
    if signal == "clean":
        native_metadata = json.loads(sidecar(selected_t1w).read_text(encoding="utf-8"))
        _validate_volume_metadata(native_metadata, inputs, source_t1)
        _validate_native_bold(input_image, inputs)
    if signal == "preproc":
        t1w_metadata = json.loads(sidecar(selected_t1w).read_text(encoding="utf-8"))
        _validate_preproc_metadata(t1w_metadata, inputs, source_t1)
        if (t1w_metadata.get("Resolution") != "native BOLD resolution"
                or t1w_metadata.get("SpatialReference") != f"bids:raw:{source_label}"
                or t1w_metadata.get("FNIT", {}).get("SourceT1w") != source_label):
            raise ValueError("preprocessed T1w BOLD lacks its native-resolution source-T1w identity")
        _millimeter_affine(input_image, "preprocessed T1w BOLD")
        native_reference = nib.as_closest_canonical(nib.load(str(inputs.sbref or inputs.bold)))
        native_zooms = np.round(native_reference.header.get_zooms()[:3], 3)
        if not np.allclose(nib.affines.voxel_sizes(input_image.affine), native_zooms,
                           rtol=1e-6, atol=1e-5):
            raise ValueError("preprocessed T1w BOLD does not use the native BOLD voxel sizes")
    executable = shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(wb_command)
    paths.func_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fnit-surface-", dir=paths.func_dir.parent) as work_dir:
        work = Path(work_dir)
        t1_bold = selected_t1w
        if signal == "clean":
            epi_to_t1 = flirt_to_world_affine(
                np.loadtxt(paths.bbr_matrix), input_image.affine, t1_image.affine,
                input_image.shape[:3], t1_image.shape,
                input_image.header.get_zooms()[:3], t1_image.header.get_zooms()[:3],
            )
            t1_bold = work / "clean_T1w.nii.gz"
            resample_world(selected_t1w, paths.t1_brain, np.linalg.inv(epi_to_t1), t1_bold, device=device)
        output_dir = recon_all_output_dir
        if output_dir is None:
            stem = source_t1.name.removesuffix(".nii.gz").removesuffix(".nii").removesuffix("_T1w")
            output_dir = paths.anat_dir / f"{stem}_desc-{selected_backend}_reconall"
        recon_options = dict(recon_all_options or {})
        if world_affine is not None:
            recon_options["fsnative_to_t1w"] = world_affine
        recon_started = time.perf_counter()
        recon_result = prepare_surface_reconstruction(
            source_t1, work / "recon_all", recon_all=recon_all,
            backend=recon_all_backend, output_dir=output_dir, device=device,
            options=recon_options,
        )
        recon_seconds = time.perf_counter() - recon_started
        subject_dir = recon_result.subject_dir
        source = Path(recon_all).expanduser().resolve() if recon_all is not None else subject_dir
        identity = _matching_original_t1(subject_dir, source_t1, world_affine)
        preparation_started = time.perf_counter()
        prepared = prepare_fmriprep_surface_inputs(
            subject_dir=subject_dir, hcp_assets_dir=assets,
            output_dir=work / "prepared", wb_command=wb_command,
            fsnative_to_t1w=world_affine,
            parallel=parallel, cpu_threads=budget,
        )
        preparation_seconds = time.perf_counter() - preparation_started
        initial_msmsulc_report = None
        msmall_report = None
        if registered_spheres is None:
            registration_started = time.perf_counter()
            sulc_inputs = prepare_msmsulc_inputs(
                subject_dir=subject_dir, initial_spheres=prepared.initial_spheres,
                hcp_assets_dir=assets, output_dir=work / "msmsulc_inputs", wb_command=wb_command,
                parallel=parallel, cpu_threads=budget,
            )
            estimates = run_msmsulc(sulc_inputs, work / "msmsulc", device=device,
                                   config=configuration, execution=msm_execution,
                                   parallel=parallel, cpu_threads=budget,
                                   qc_policy=msmsulc_qc_policy)
            spheres = (Path(estimates["L"]), Path(estimates["R"]))
            registration = "MSMSulc-HOCR-FastPD"
            registration_qc = {
                "Report": json.loads((work / "msmsulc" / "registration_report.json").read_text()),
                "InputsSHA256": {
                    hemi: {name: _sha256(Path(path)) for name, path in asdict(entry).items()
                           if path is not None}
                    for hemi, entry in sulc_inputs.items()
                },
            }
            registration_seconds = time.perf_counter() - registration_started
            report = registration_qc["Report"]
            initial_msmsulc_report = report
            native_qc_references = {
                hemi: entry.rotated_sphere for hemi, entry in sulc_inputs.items()
            }
            registration_details = {
                "Method": "FNIT MSMSulc-HOCR-FastPD",
                "Configuration": configuration.to_dict(),
                "Execution": msm_execution,
                "QCPolicy": msmsulc_qc_policy,
                "Hemispheres": {
                    hemi: {key: report[hemi][key] for key in (
                        "seconds", "peak_allocated_gb", "folded_output_faces",
                        "folded_solver_faces", "minimum_output_orientation_ratio",
                        "minimum_solver_orientation_ratio", "degenerate_input_faces",
                        "absolute_orientation_reference_sign", "absolute_folded_input_faces",
                        "absolute_folded_solver_faces", "absolute_folded_output_faces",
                        "minimum_absolute_solver_orientation_ratio", "minimum_absolute_output_orientation_ratio",
                        "new_relative_folded_solver_faces", "new_relative_folded_output_faces",
                        "native_output_qc_before_repair", "fold_repair", "orientation_qc",
                    ) if key in report[hemi]} for hemi in ("L", "R")
                },
            }
        else:
            spheres = tuple(Path(path).expanduser().resolve() for path in registered_spheres)
            registration = "provided registered spheres"
            registration_qc = None
            native_qc_references = {
                hemi: work / "prepared" / f"{hemi}.sphere.FS.native.surf.gii"
                for hemi in ("L", "R")
            }
        msmall_seconds = None
        if msmall_inputs is not None:
            msmall_started = time.perf_counter()
            spheres, feature_topology = _refine_msmall(
                msmall_inputs, spheres, (prepared.geometry.left, prepared.geometry.right),
                assets, work, msmall_configuration, device, msm_execution, wb_command,
                parallel=parallel, cpu_threads=budget,
                native_qc_references=native_qc_references,
                qc_policy=msmall_qc_policy,
            )
            msmall_seconds = time.perf_counter() - msmall_started
            msmall_report = json.loads((work / "msmall/registration_report.json").read_text())
            registration_qc = {
                "InitialMSMSulc": registration_qc,
                "MSMAll": {"Report": msmall_report, "InputsSHA256": {
                    hemi: {name: _sha256(Path(path)) for name, path in asdict(entry).items()
                           if path is not None}
                    for hemi, entry in msmall_inputs.items()
                }},
            }
            registration = "MSMAll-HOCR-FastPD"
            registration_details = {
                "Method": "FNIT MSMAll-HOCR-FastPD",
                "InitialRegistration": registration_details,
                "Configuration": msmall_configuration.to_dict(), "Execution": msm_execution,
                "QCPolicy": msmall_qc_policy,
                "FeatureTopology": feature_topology,
                "FeaturePreparation": "provided multimodal feature/weight files",
                "HemispheresScope": "MSMAll solver output on the source feature topology",
                "Hemispheres": msmall_report,
            }
            final_native_report = json.loads(
                (work / "msmall/native_composition_report.json").read_text()
            )
        else:
            final_native_report = {
                hemi: _saved_native_sphere_qc(sphere, native_qc_references[hemi],
                                             baseline="undeformed native sphere")
                for hemi, sphere in zip(("L", "R"), spheres)
            }
            # A policy check inside the solver does not certify a later saved
            # mesh or reveal folds already present in its relative reference.
            # Reapply strict policies at the actual projection boundary.
            if (msmsulc_qc_policy != "report"
                    and _orientation_stage(final_native_report)["status"] != "pass"):
                raise RuntimeError("MSMSulc final saved native sphere failed orientation QC; "
                                   "BOLD projection refused")
        orientation_qc = _orientation_chain(
            initial_msmsulc_report, msmall_report, final_native_report
        )
        registration_details["FinalNativeHemispheres"] = final_native_report
        registration_details["OrientationQC"] = orientation_qc
        if registration_qc is None:
            registration_qc = {}
        registration_qc["FinalNative"] = final_native_report
        registration_qc["OrientationQC"] = orientation_qc
        def area_surface(hemi, threads):
            index = 0 if hemi == "L" else 1
            geometry = prepared.geometry.left if hemi == "L" else prepared.geometry.right
            individual_roi, sphere = prepared.individual_rois[index], spheres[index]
            middle_source = getattr(geometry, "midthickness_source", None)
            middle_info = None
            if middle_source is not None:
                middle_info = {
                    "File": Path(middle_source).relative_to(subject_dir).as_posix(),
                    "SHA256": _sha256(Path(middle_source)),
                }
            atlas_mid = work / "registered" / hemi / "midthickness.32k_fsLR.surf.gii"
            atlas_mid.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                executable, "-surface-resample", str(geometry.midthickness), str(sphere),
                str(mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii"), "BARYCENTRIC", str(atlas_mid),
            ], check=True, capture_output=True, text=True, env=workbench_environment(threads))
            hemisphere = SurfaceHemisphere(
                white=geometry.white, pial=geometry.pial, midthickness=geometry.midthickness,
                registered_sphere=sphere, native_roi=individual_roi,
                atlas_sphere=mesh / f"{hemi}.sphere.32k_fs_LR.surf.gii",
                atlas_midthickness=atlas_mid, atlas_roi=mesh / f"{hemi}.atlasroi.32k_fs_LR.shape.gii",
            )
            return hemisphere, middle_info

        area_started = time.perf_counter()
        hemisphere_results = map_hemispheres(area_surface, parallel=parallel, cpu_threads=budget)
        area_seconds = time.perf_counter() - area_started
        hemispheres = {hemi: result[0] for hemi, result in zip(("L", "R"), hemisphere_results)}
        identity["MidthicknessSource"] = {
            hemi: result[1] for hemi, result in zip(("L", "R"), hemisphere_results)
            if result[1] is not None
        }
        projection = run_fmriprep_surface_projection(
            clean_t1w=t1_bold, clean_mni=selected_mni,
            left=hemispheres["L"], right=hemispheres["R"], left_label=left_roi, right_label=right_roi,
            hcp_dseg=dseg, output_dir=work / "projection", tr_seconds=inputs.tr,
            goodvoxels=goodvoxels, wb_command=wb_command,
            parallel=parallel, cpu_threads=budget,
        )
        publish = work / "publish"
        publish.mkdir()
        for current, destination in zip(
            (projection.left_metric, projection.right_metric, projection.dtseries), final_data,
        ):
            shutil.copyfile(current, publish / destination.name)
        sphere_info = {}
        for hemisphere, current, destination in zip(("L", "R"), spheres, sphere_paths):
            shutil.copyfile(current, publish / destination.name)
            sphere_info[hemisphere] = {
                "File": f"bids::{destination.relative_to(paths.root).as_posix()}",
                "SHA256": _sha256(current),
                "EstimatedHere": registered_spheres is None,
            }
        timing = {**projection.timing_seconds, "total_before_publication": time.perf_counter() - started}
        timing["native_surface_preparation"] = preparation_seconds
        timing["volume_status_validation"] = volume_check_seconds
        timing["automatic_volume"] = volume_details["Seconds"]
        timing["reconstruction_adapter"] = recon_seconds
        timing["atlas_area_surfaces"] = area_seconds
        if registration_seconds is not None:
            timing["msmsulc_preparation_and_registration"] = registration_seconds
        if msmall_seconds is not None:
            timing["msmall_registration_and_native_composition"] = msmall_seconds
        coverage = json.loads(projection.coverage_report.read_text(encoding="utf-8"))
        details = {**{key: metadata[key] for key in (
            "TaskName", "EchoTime", "FlipAngle", "MagneticFieldStrength", "Manufacturer",
            "PhaseEncodingDirection", "Units", "SliceTimingCorrected", "StartTime", "DelayTime",
            "AcquisitionDuration", "VolumeTiming",
        ) if key in metadata},
            "Sources": [f"bids::{selected_t1w.relative_to(paths.root).as_posix()}",
                        f"bids::{selected_mni.relative_to(paths.root).as_posix()}"],
            "RepetitionTime": inputs.tr, "SkullStripped": metadata.get("SkullStripped", signal == "clean"),
            "FNIT": {"Registration": registration, "RegisteredSpheres": sphere_info,
                     "RegistrationDetails": registration_details,
                     "RegistrationQC": f"bids::{qc_report.relative_to(paths.root).as_posix()}",
                     "SourceT1w": source_label, "ReconAllSource": str(source), "Signal": signal,
                     "VolumePrerequisite": volume_details,
                     "Reconstruction": recon_result.metadata,
                     "VolumeProcessing": {key: space[key] for key in (
                         "TemporalFiltering", "IntensityNormalization", "Interpolation",
                         "SliceTimingCorrection", "SusceptibilityCorrection",
                     ) if key in space},
                     "Projection": "fMRIPrep-style T1w cortex + MNI subcortex",
                     "HemisphereExecution": {"Parallel": parallel and budget > 1,
                                             "CPUThreads": budget,
                                             "CPUThreadsPerHemisphere": dict(hemisphere_items(
                                                 parallel=parallel, cpu_threads=budget))},
                     "StandardSpace": space["StandardSpace"],
                     "StandardTemplateSHA256": template_hash,
                     "StandardTemplateIdentity": space["StandardTemplateIdentity"],
                     "TimingSeconds": timing, "Coverage": coverage, "Geometry": identity,
                     "TimingScope": "Stage times and total_before_publication stop before final sidecar/QC serialization and atomic publication; returned timing_seconds.total includes publication and temporary cleanup.",
                     "SurfaceAssetsSHA256": {"LeftROI": _sha256(left_roi), "RightROI": _sha256(right_roi),
                                              "HCPdseg": _sha256(dseg)}},
        }
        for path, json_path in zip(final_data, final_sidecars):
            item = dict(details)
            if path == paths.dtseries:
                item.update(fmriprep_cifti_metadata())
            else:
                hemi = "L" if path == paths.left else "R"
                item.update(Density="32,492 vertices per hemisphere",
                            SpatialReference=BASE_URL + MESH + f"{hemi}.sphere.32k_fs_LR.surf.gii")
            write_json(publish / json_path.name, item)
        write_json(publish / qc_report.name, {
            "Sources": details["Sources"], "Registration": registration,
            "RegisteredSpheres": sphere_info, "Geometry": identity,
            "MSM": registration_qc,
            "OrientationQC": orientation_qc,
            "VolumePrerequisite": volume_details, "Reconstruction": recon_result.metadata,
            "Coverage": coverage, "TimingSeconds": timing,
        })
        for hemisphere, destination in zip(("L", "R"), sphere_sidecars):
            write_json(publish / destination.name, {
                "Sources": [f"bids:raw:{source_label}"], "Registration": registration,
                "Geometry": identity, **sphere_info[hemisphere],
            })
        _publish_projection(publish, paths.func_dir, tuple(path.name for path in final_outputs), overwrite)
    timing["total"] = time.perf_counter() - started
    return FMRISurfaceResult(paths.left, paths.right, paths.dtseries, sidecar(paths.dtseries),
                             timing, qc_report, sphere_paths, recon_result.subject_dir,
                             volume_details["Executed"])
