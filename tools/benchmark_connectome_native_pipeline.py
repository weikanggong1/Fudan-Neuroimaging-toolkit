"""Measure a real corrected-DWI connectome call and independent native references.

This evaluator uses the caller-selected FNIT checkout. It never substitutes
official outputs into the scientific pipeline. Completed recon-all and corrected
DWI are explicit inputs; this run does not remeasure TOPUP, EDDY or reconstruction.
Reports and intermediates contain private paths and must be reviewed before
publication. Reference executables must come from an explicitly supplied build.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch


MATRIX_NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")
STAGE_FUNCTIONS = (
    "_bet_on_dwi_grid", "dwi2mask_legacy", "fit_mrtrix_dhollander_tensor",
    "freesurfer_five_tissue", "gmwmi_from_five_tissue", "_registration",
    "estimate_mrtrix_dhollander", "fit_mrtrix_msmt_csd",
    "normalise_mrtrix_three_tissue", "probabilistic_tractography",
    "estimate_sift2_weights", "sample_streamline_mean_precise",
    "fs_aparc_atlas", "fs_aparc_a2009s_atlas", "resample_labels_nearest",
    "build_connectomes",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(path, report):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def vector_metrics(candidate, reference):
    """Keep nonfinite diagnostics and avoid undefined empty correlations."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape:
        raise ValueError(f"comparison shapes differ: {candidate.shape} / {reference.shape}")
    finite = np.isfinite(candidate) & np.isfinite(reference)
    left, right = candidate[finite].astype(np.float64), reference[finite].astype(np.float64)
    delta = np.abs(left - right)
    equal = (candidate == reference) | (np.isnan(candidate) & np.isnan(reference))
    denominator = np.abs(right).sum()
    return {
        "shape": list(candidate.shape), "finite_pairs": int(finite.sum()),
        "unequal_values": int((~equal).sum()),
        "nonfinite_mismatch": int((np.isfinite(candidate) != np.isfinite(reference)).sum()),
        "max_abs": float(delta.max(initial=0)),
        "mae": float(delta.mean()) if delta.size else None,
        "relative_l1": float(delta.sum() / denominator) if denominator else None,
        "pearson": (float(np.corrcoef(left, right)[0, 1])
                    if left.size > 1 and left.std() > 0 and right.std() > 0 else None),
    }


def matrix_metrics(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("matrix comparison requires a square atlas")
    if candidate.shape != reference.shape:
        raise ValueError(f"matrix shapes differ: {candidate.shape} / {reference.shape}")
    upper = np.triu_indices(candidate.shape[0], 1)
    left, right = candidate[upper], reference[upper]
    common = (left != 0) & (right != 0)
    support_total = int((left != 0).sum() + (right != 0).sum())
    result = vector_metrics(left, right)
    result.update(
        matrix_shape=list(candidate.shape),
        full_matrix=vector_metrics(candidate, reference),
        support_dice=float(2 * common.sum() / support_total) if support_total else 1.,
        candidate_support=int((left != 0).sum()), reference_support=int((right != 0).sum()),
        common_support=vector_metrics(left[common], right[common]),
    )
    return result


def path_digest(paths):
    """Digest ordered float32 RAS points and per-track boundaries, not TCK headers."""
    points_hash, offsets_hash = hashlib.sha256(), hashlib.sha256()
    offsets_hash.update(np.asarray([0], dtype="<i8").tobytes())
    count, points = 0, 0
    for path in paths:
        if isinstance(path, torch.Tensor):
            path = path.detach().cpu().numpy()
        array = np.asarray(path, dtype="<f4", order="C")
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("streamline must be [P,3]")
        count += 1
        points += len(array)
        points_hash.update(array.tobytes())
        offsets_hash.update(np.asarray([points], dtype="<i8").tobytes())
    return dict(streamlines=count, points=points, points_sha256=points_hash.hexdigest(),
                offsets_sha256=offsets_hash.hexdigest())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measured_call(function, *, device, stage_rows, name):
    @wraps(function)
    def call(*arguments, **options):
        _sync(device)
        start = time.perf_counter()
        try:
            return function(*arguments, **options)
        finally:
            _sync(device)
            stage_rows.append(dict(stage=name, seconds=time.perf_counter() - start))
    return call


def bind_inputs(arguments):
    paths = dict(dwi=arguments.dwi, bvals=arguments.bvals, bvecs=arguments.bvecs)
    relative = ["mri/brain.mgz", "mri/aparc+aseg.mgz"]
    if "fs-aparc-a2009s" in arguments.atlas or arguments.reuse_atlas == "fs-aparc-a2009s":
        relative.append("mri/aparc.a2009s+aseg.mgz")
        relative.extend(("label/lh.aparc.a2009s.annot", "label/rh.aparc.a2009s.annot"))
    for name in relative:
        paths["recon_all/" + name] = arguments.freesurfer_subject_dir / name
    for name in ("brain_mask", "response_mask", "fod_mask", "normalise_mask", "fa_map", "transform"):
        if (value := getattr(arguments, name)) is not None:
            paths[name] = value
    return {name: dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))
            for name, path in paths.items()}


def source_binding():
    import fnit.connectome.pipeline as pipeline
    package = Path(pipeline.__file__).parent
    files = set(package.rglob("*.py")) | set(package.rglob("*.npz")) | set(package.rglob("*.tsv"))
    return {str(path.relative_to(package)): sha256(path) for path in sorted(files)}


def _export_image(value, affine, path, *, voxel_spacing_mm=None):
    """Use adapter image serialization so official tracking reads identical bytes."""
    from fnit.connectome.tracking import write_tracking_image
    write_tracking_image(value, affine, path, voxel_spacing_mm=voxel_spacing_mm)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def export_result(result, output_dir, spacing):
    """Export only after the full-call timer; no export changes scientific inputs."""
    output_dir.mkdir()
    images = {}
    for name, data, affine, zooms in (
        ("wm_fod", result.wm_sh, result.dwi_affine, None),
        ("five_tissue", result.five_tissue, result.five_tissue_affine, spacing),
        ("gmwmi", result.gmwmi, result.five_tissue_affine, spacing),
        ("fa", result.fa, result.dwi_affine, None),
    ):
        images[name] = _export_image(data, affine, output_dir / f"{name}.nii", voxel_spacing_mm=zooms)
    provenance = result.tractogram.native_provenance
    for candidate_key, exported_key in (("wm_fod", "wm_fod"), ("five_tissue", "five_tissue"),
                                        ("gmwmi", "gmwmi")):
        expected = provenance["input_images"][candidate_key]["sha256"]
        if expected != images[exported_key]["sha256"]:
            raise ValueError(f"tracking export differs from actual native input: {candidate_key}")
    step_size = float(provenance["tck_header"]["step_size"])
    tck = output_dir / "tracks.tck"
    arrays = [path.detach().cpu().numpy() for path in result.tractogram.paths]
    saved = nib.streamlines.TckFile(
        nib.streamlines.Tractogram(arrays, affine_to_rasmm=np.eye(4)),
        header={"step_size": str(step_size)},
    )
    saved.save(tck)
    track_hash = path_digest(result.tractogram.paths)
    if path_digest(nib.streamlines.load(tck, lazy_load=True).streamlines) != track_hash:
        raise ValueError("candidate TCK export changed streamline points or order")
    for name, values in (("sift2_weights", result.sift2_weights),
                         ("length_mm", result.tractogram.lengths_mm),
                         ("mean_fa", result.tractogram.mean_fa)):
        np.savetxt(output_dir / (name + ".txt"), values.detach().cpu().numpy())
    atlases = {}
    for name, atlas in result.atlas_results.items():
        atlas_dir = output_dir / name
        atlas_dir.mkdir()
        atlas_path = atlas_dir / "atlas.nii"
        image = nib.Nifti2Image(atlas.atlas.detach().cpu().numpy(), atlas.atlas_affine.cpu().numpy())
        nib.save(image, atlas_path)
        matrices = {}
        for metric, values in atlas.matrices.items():
            target = atlas_dir / (metric + ".csv")
            np.savetxt(target, values.detach().cpu().numpy(), delimiter=",")
            matrices[metric] = dict(path=str(target), sha256=sha256(target))
        atlases[name] = dict(path=str(atlas_path), sha256=sha256(atlas_path),
                            region_labels=list(atlas.region_labels), matrices=matrices)
    return dict(images=images, tracks=dict(path=str(tck), sha256=sha256(tck),
                                         ordered_path_digest=track_hash), atlases=atlases)


def run_official(command, output_dir, name, report, *, environment=None):
    """Run one explicit independent binary, keeping exact argv, version and errors."""
    command = [str(value) for value in command]
    program = Path(command[0]).resolve(strict=True)
    program_sha = sha256(program)
    version = subprocess.check_output([str(program), "-version"], stderr=subprocess.STDOUT, text=True)
    started = time.perf_counter()
    with (output_dir / (name + ".log")).open("w") as stream:
        child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                    **(environment or {})})
    row = dict(command=command, binary_sha256=program_sha, version=version,
               wall_seconds=time.perf_counter() - started, returncode=child.returncode,
               explicit_environment=environment or {})
    report[name] = row
    if sha256(program) != program_sha:
        raise RuntimeError("reference executable changed during measurement")
    if child.returncode:
        raise RuntimeError(f"official {name} failed; see preserved log")
    return row


def _native_reading_config():
    return ["-config", "RealignTransform", "false", "-config", "NIfTIUseSform", "true",
            "-config", "NIfTIAutoLoadJSON", "false", "-config", "TckgenEarlyExit", "false"]


def _reference_matrix(candidate, reference):
    """Retain trailing absent nodes without inventing reference entries."""
    array = np.atleast_2d(np.loadtxt(reference, delimiter=","))
    if array.shape == candidate.shape:
        return array
    raise ValueError(f"reference {reference.name} shape {array.shape} != {candidate.shape}; no padding")


def fixed_tck_reference(arguments, result, exported, report):
    output_dir = arguments.output_dir / "official_fixed_tck"
    output_dir.mkdir()
    binaries = arguments.official_bin_dir
    images = {name: Path(row["path"]) for name, row in exported["images"].items()}
    tracks = Path(exported["tracks"]["path"])
    threads = ["-nthreads", str(arguments.reference_threads), *_native_reading_config()]
    commands = {}
    atlas_results, vectors = {}, {}
    report.update(scope="same candidate TCK/FOD/5TT/FA/atlas; official downstream programs only",
                  commands=commands, per_track_vectors=vectors, atlas_matrices=atlas_results,
                  native_tracking_equivalence_assessed_here=False)
    weights = output_dir / "sift2_weights.txt"
    fa = output_dir / "mean_fa.txt"
    run_official([binaries / "tcksift2", tracks, images["wm_fod"], weights,
                  "-act", images["five_tissue"], *threads], output_dir, "sift2", commands,
                 environment={"MRTRIX_CONFIGFILE": "/dev/null"})
    run_official([binaries / "tcksample", tracks, images["fa"], fa,
                  "-precise", "-stat_tck", "mean", *threads], output_dir, "fa", commands,
                 environment={"MRTRIX_CONFIGFILE": "/dev/null"})
    vectors.update({
        "sift2_weights": vector_metrics(result.sift2_weights.cpu().numpy(), np.loadtxt(weights)),
        "mean_fa": vector_metrics(result.tractogram.mean_fa.cpu().numpy(), np.loadtxt(fa)),
    })
    for name, atlas in result.atlas_results.items():
        target = output_dir / name
        target.mkdir()
        common = [tracks, exported["atlases"][name]["path"], "-symmetric",
                  "-assignment_radial_search", str(arguments.assignment_radius), *threads]
        metrics = {}
        atlas_results[name] = metrics
        for metric in MATRIX_NAMES:
            csv = target / (metric + ".csv")
            extra = {
                "count": [], "sift2_fbc": ["-tck_weights_in", weights],
                "mean_length": ["-tck_weights_in", weights, "-scale_length", "-stat_edge", "mean"],
                "mean_fa": ["-tck_weights_in", weights, "-scale_file", fa, "-stat_edge", "mean"],
            }[metric]
            command = [binaries / "tck2connectome", *common[:2], csv, *common[2:], *extra]
            run_official(command, target, metric, commands.setdefault(name, {}),
                         environment={"MRTRIX_CONFIGFILE": "/dev/null"})
            candidate = atlas.matrices[metric].cpu().numpy()
            metrics[metric] = matrix_metrics(candidate, _reference_matrix(candidate, csv))
        atlas_results[name] = metrics


def _tracking_command(binaries, images, destination, n_seeds, threads):
    return [binaries / "tckgen", images["wm_fod"]["path"], destination,
            "-algorithm", "iFOD2", "-seed_gmwmi", images["gmwmi"]["path"],
            "-act", images["five_tissue"]["path"], "-seeds", str(n_seeds),
            "-select", "0", "-maxlength", "250", "-angle", "45", "-cutoff", "0.1",
            "-samples", "3", "-power", "0.5", "-nthreads", str(threads),
            *_native_reading_config()]


def tracking_reference(arguments, result, exported, report=None):
    """Separate deterministic one-thread oracle from threaded population checks."""
    from fnit.connectome.tracking import probabilistic_tractography
    output_dir = arguments.output_dir / "official_tracking"
    output_dir.mkdir()
    report = {} if report is None else report
    report.update(commands={}, scope="same actual pipeline FOD/5TT/GMWMI, independent tckgen command",
                  step_angle_minlength="official step/minlength defaults; angle45 identical to pipeline")
    spacing = nib.load(arguments.freesurfer_subject_dir / "mri/aparc+aseg.mgz").header.get_zooms()[:3]
    if arguments.strict_tracking_seeds:
        start = time.perf_counter()
        candidate = probabilistic_tractography(
            result.wm_sh, result.dwi_affine, result.five_tissue, result.five_tissue_affine,
            result.gmwmi, n_seeds=arguments.strict_tracking_seeds, seed=arguments.seed,
            tracking_threads=1, five_tissue_spacing_mm=spacing)
        _sync(result.wm_sh.device)
        candidate_seconds = time.perf_counter() - start
        destination = output_dir / "strict_one_thread.tck"
        run_official(_tracking_command(arguments.official_bin_dir, exported["images"], destination,
                                       arguments.strict_tracking_seeds, 1),
                     output_dir, "strict_one_thread", report["commands"],
                     environment={"MRTRIX_RNG_SEED": str(arguments.seed), "MRTRIX_CONFIGFILE": "/dev/null"})
        actual = path_digest(candidate.paths)
        reference = path_digest(nib.streamlines.load(destination, lazy_load=True).streamlines)
        report["strict_one_thread"] = dict(
            requested_seed_budget=arguments.strict_tracking_seeds, candidate_seconds=candidate_seconds,
            candidate_provenance=candidate.native_provenance, candidate_digest=actual,
            reference_digest=reference, ordered_points_and_offsets_equal=actual == reference)
        if actual != reference:
            raise RuntimeError("single-thread same-input native/reference tracking is not byte identical")
    repeats = report["threaded_independent_repeats"] = []
    candidate_lengths = result.tractogram.lengths_mm.detach().cpu().numpy()
    for seed in range(arguments.reference_repeats):
        destination = output_dir / f"seed_{seed}.tck"
        run_official(_tracking_command(arguments.official_bin_dir, exported["images"], destination,
                                       arguments.n_seeds, arguments.tracking_threads),
                     output_dir, f"seed_{seed}", report["commands"],
                     environment={"MRTRIX_RNG_SEED": str(seed), "MRTRIX_CONFIGFILE": "/dev/null"})
        loaded = nib.streamlines.load(destination, lazy_load=False)
        # Stored polyline lengths are explicit population diagnostics. They are
        # not silently substituted for MRtrix's fixed-step connectome lengths.
        polyline_lengths = np.asarray([
            np.linalg.norm(np.diff(np.asarray(path, dtype=np.float64), axis=0), axis=1).sum()
            for path in loaded.streamlines])
        candidate_polyline_lengths = np.asarray([
            np.linalg.norm(np.diff(path.detach().cpu().numpy().astype(np.float64), axis=0), axis=1).sum()
            for path in result.tractogram.paths])
        from scipy.stats import ks_2samp
        repeats.append(dict(seed=seed, accepted_streamlines=len(loaded.streamlines),
                            requested_seed_budget=arguments.n_seeds,
                            length_definition="sum of Euclidean stored-polyline segment lengths",
                            mean_polyline_length_mm=float(polyline_lengths.mean()) if len(polyline_lengths) else None,
                            mean_candidate_polyline_length_mm=float(candidate_polyline_lengths.mean())
                            if len(candidate_polyline_lengths) else None,
                            length_ks=(float(ks_2samp(candidate_polyline_lengths, polyline_lengths).statistic)
                                       if len(polyline_lengths) and len(candidate_polyline_lengths) else None),
                            header_total_count=str(loaded.header.get("total_count")),
                            ordered_path_digest=path_digest(loaded.streamlines),
                            reference_tck_sha256=sha256(destination)))
    report["threaded_independent_repeats"] = repeats
    report["candidate_tracking_length_definition"] = "MRtrix float32 left-to-right sum of stored-polyline segments"
    report["candidate_mean_tracking_length_mm"] = float(candidate_lengths.mean()) if len(candidate_lengths) else None
    report["repeat_envelope_acceptance_claim"] = None
    return report


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "bvals", "bvecs", "freesurfer-subject-dir", "output-dir", "official-bin-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--atlas", nargs="+", default=["fs-aparc"],
                        choices=("fs-aparc", "fs-aparc-a2009s"))
    parser.add_argument("--reuse-atlas", choices=("fs-aparc", "fs-aparc-a2009s"))
    parser.add_argument("--n-seeds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracking-threads", type=int, default=8)
    parser.add_argument("--reference-threads", type=int, default=8)
    parser.add_argument("--strict-tracking-seeds", type=int, default=100)
    parser.add_argument("--reference-repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-radius", type=float, default=4.)
    for name in ("brain-mask", "response-mask", "fod-mask", "normalise-mask", "fa-map", "transform"):
        parser.add_argument("--" + name, type=Path)
    return parser.parse_args()


def execute(arguments):
    import fnit.connectome.pipeline as pipeline
    if arguments.output_dir.exists():
        raise FileExistsError("use a new output directory; old results cannot become this run")
    if arguments.n_seeds < 1 or min(arguments.tracking_threads, arguments.reference_threads) < 1:
        raise ValueError("seed budget and thread counts must be positive")
    if arguments.strict_tracking_seeds < 0 or arguments.reference_repeats < 0:
        raise ValueError("oracle seed budget and repeat count must be nonnegative")
    arguments.output_dir.mkdir(parents=True)
    report_path = arguments.output_dir / "report.private.json"
    report = dict(schema_version=1, status="running", scope="corrected DWI + completed recon-all to SC",
                  raw_topup_eddy_recon_all_rerun=False,
                  timing_scope="real full call with synchronized per-stage diagnostics; export and reference outside timer",
                  benchmark_sha256=sha256(__file__), input_before=bind_inputs(arguments),
                  source_before=source_binding(), parameters=vars(arguments),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   nibabel=nib.__version__, numpy=np.__version__), stages=[])
    write_report(report_path, report)
    try:
        device = torch.device(arguments.device)
        engine = pipeline.UKBConnectome_pipeline(device=arguments.device)
        options = dict(freesurfer_subject_dir=arguments.freesurfer_subject_dir,
                       atlas=arguments.atlas, n_seeds=arguments.n_seeds, seed=arguments.seed,
                       tracking_threads=arguments.tracking_threads,
                       assignment_radius=arguments.assignment_radius,
                       checkpoint_dir=arguments.output_dir / "checkpoints")
        for name in ("brain_mask", "response_mask", "fod_mask", "normalise_mask", "fa_map"):
            options[name] = getattr(arguments, name)
        if arguments.transform is not None:
            options["dwi_to_t1_world"] = np.loadtxt(arguments.transform, comments="#")
        with ExitStack() as stack:
            for name in STAGE_FUNCTIONS:
                if hasattr(pipeline, name):
                    stack.enter_context(patch.object(pipeline, name, measured_call(
                        getattr(pipeline, name), device=device, stage_rows=report["stages"], name=name)))
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = engine(arguments.dwi, arguments.bvals, arguments.bvecs, **options)
            _sync(device)
            report["full_call_seconds"] = time.perf_counter() - start
        report["cache_status"] = result.cache_status
        report["native_tracking"] = result.tractogram.native_provenance
        report["accepted_streamlines"] = len(result.tractogram.paths)
        report["cuda_allocator_peak"] = dict(
            allocated_bytes=torch.cuda.max_memory_allocated(device),
            reserved_bytes=torch.cuda.max_memory_reserved(device),
            process_total_under_20_gb_assessed=False) if device.type == "cuda" else None
        report["tf32"] = dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                              cudnn=torch.backends.cudnn.allow_tf32)
        report["input_after"] = bind_inputs(arguments)
        report["source_after"] = source_binding()
        if report["input_before"] != report["input_after"] or report["source_before"] != report["source_after"]:
            raise RuntimeError("real input or production source changed during full call")
        write_report(report_path, report)
        spacing = nib.load(arguments.freesurfer_subject_dir / "mri/aparc+aseg.mgz").header.get_zooms()[:3]
        exported = export_result(result, arguments.output_dir / "candidate", spacing)
        report["exported"] = exported
        write_report(report_path, report)
        report["tracking_reference"] = {}
        tracking_reference(arguments, result, exported, report["tracking_reference"])
        write_report(report_path, report)
        reference = {}
        report["fixed_tck_reference"] = reference
        fixed_tck_reference(arguments, result, exported, reference)
        if arguments.reuse_atlas is not None:
            reuse_options = {**options, "atlas": arguments.reuse_atlas}
            _sync(device)
            start = time.perf_counter()
            reused = engine(arguments.dwi, arguments.bvals, arguments.bvecs, **reuse_options)
            _sync(device)
            report["atlas_reuse"] = dict(seconds=time.perf_counter() - start,
                                        atlas=arguments.reuse_atlas, cache_status=reused.cache_status,
                                        paths_equal=path_digest(reused.tractogram.paths) == path_digest(result.tractogram.paths),
                                        weights_equal=torch.equal(reused.sift2_weights, result.sift2_weights))
            if (reused.cache_status["core"] != "skipped"
                    or not report["atlas_reuse"]["paths_equal"] or not report["atlas_reuse"]["weights_equal"]):
                raise RuntimeError("template change failed to reuse the exact completed shared core")
        report["status"] = "completed"
        report["scientific_parity_claim"] = "read actual metrics; completion alone is not equivalence"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_report(report_path, report)
    return report


if __name__ == "__main__":
    execute(_parse())
