#!/usr/bin/env python3
"""Run real FastVBM / volume / dMRI APIs and compare frozen FNIT outputs.

The caller supplies a private JSON with API keyword arguments. No data paths,
subject labels or raw outputs enter the exported scalar report. Original
software is never invoked. Use separate source trees and output directories
for baseline and candidate; no anatomical cache is reused for volume.
``--augment-existing-report`` adds post-run input readback and corrects captured
stage timings without executing any pipeline or changing measured source hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root):
    paths = []
    for component in ("fnirt", "applywarp", "synthmorph", "synthstrip", "fast_vbm", "fast", "flirt",
                      "fmri", "feat", "melodic", "mcflirt", "dmri_pipeline", "eddy", "topup", "dtifit", "amico_noddi"):
        paths.extend((root / "src/fnit" / component).rglob("*.py"))
    paths.extend((root / "src/fnit").glob("*.py"))
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def input_file_summary(path):
    """Identify an input by content, without exporting its path or filename."""
    path = Path(path).expanduser()
    result = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    if path.name.endswith((".nii", ".nii.gz", ".mgz", ".mgh")):
        image = nib.load(str(path))
        result.update(shape=list(image.shape), dtype=str(image.get_data_dtype()))
    return result


def input_summaries(pipeline, case):
    """Resolve the same BIDS selection and report only anonymous input roles."""
    paths = {}
    metadata = {}
    if pipeline == "fast-vbm":
        paths.update({key: case[key] for key in ("image", "template", "reference_mask")})
    elif pipeline == "volume":
        from fnit.fmri.bids import locate_bids_inputs
        from fnit.fmri.end_to_end import _select_t1
        selection = {key: case[key] for key in (
            "subject", "session", "task", "run", "acquisition", "direction", "reconstruction", "echo"
        ) if key in case}
        inputs = locate_bids_inputs(case["bids_root"], **selection)
        paths.update(bold=inputs.bold, t1w=_select_t1(inputs, case.get("t1w_image")),
                     sbref=inputs.sbref, mni_template=case["mni_template"],
                     mni_brain_mask=case.get("mni_brain_mask"))
        paths.update({f"bold_sidecar_{i:03d}": path
                      for i, path in enumerate(inputs.bold_sidecars, start=1)})
        metadata["effective_bold_metadata"] = inputs.bold_metadata
    else:
        from fnit.dmri_pipeline.bids import _applicable, locate_bids_dwi
        selection = {key: case[key] for key in (
            "subject", "session", "run", "acquisition", "direction", "t1"
        ) if key in case}
        inputs = locate_bids_dwi(case["bids_root"], **selection, select_t1=False)
        paths.update(dwi=inputs.image, bval=inputs.bval, bvec=inputs.bvec,
                     reverse_pe=inputs.reverse, reverse_bval=inputs.reverse_bval,
                     fa_template=case["fa_template"], fa_skeleton=case.get("fa_skeleton"))
        for role, image in (("dwi", inputs.image), ("reverse_pe", inputs.reverse)):
            if image is not None:
                paths.update({f"{role}_sidecar_{i:03d}": path for i, path in enumerate(
                    _applicable(inputs.root, image, "json"), start=1
                )})
        metadata["effective_dwi_metadata"] = inputs.metadata
        if inputs.reverse_metadata is not None:
            metadata["effective_reverse_metadata"] = inputs.reverse_metadata
    if pipeline in ("fast-vbm", "volume"):
        from fnit.weights import resolve_weights
        paths["synthstrip_weights"] = resolve_weights(
            "synthstrip.1.pt", explicit=case.get("synthstrip_weights")
        )
    summaries = {role: input_file_summary(path) for role, path in paths.items() if path is not None}
    for role, value in metadata.items():
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        summaries[role] = {"sha256": hashlib.sha256(encoded).hexdigest(),
                           "content_bytes": len(encoded), "encoding": "canonical UTF-8 JSON"}
    return summaries


def corrected_pipeline_timings(pipeline, stages, capture_seconds):
    """Subtract only the measured capture interval from stages containing it."""
    affected = {
        "fast-vbm": ("registration_jacobian_modulation", "total"),
        "volume": ("t1_to_mni_nonlinear", "total"),
        "dmri": ("registration_and_map_propagation",),
    }[pipeline]
    corrected = dict(stages)
    attribution = {}
    for key in affected:
        if key in corrected:
            if corrected[key] < capture_seconds:
                raise ValueError(f"capture exceeds containing pipeline stage: {key}")
            corrected[key] -= capture_seconds
            attribution[key] = capture_seconds
    return corrected, attribution


def solver_trace(qc):
    """Retain all scientific QC; omit execution label and measured wall clocks."""
    if "levels" not in qc:
        raise ValueError("FNIRT QC is missing its solver trace")

    def without_wall_clocks(value):
        if isinstance(value, dict):
            return {key: without_wall_clocks(item) for key, item in value.items()
                    if key != "elapsed_seconds"}
        if isinstance(value, list):
            return [without_wall_clocks(item) for item in value]
        return value

    # Topology QC nests elapsed_seconds alongside scientific state. Timing
    # changes are measured separately; costs, counts, flags and decisions stay.
    return without_wall_clocks({key: value for key, value in qc.items()
                               if key != "execution"})


def safe_output_role(path, root, index):
    """Keep fixed capture roles; anonymize public BIDS filenames and labels."""
    relative = path.relative_to(root)
    if len(relative.parts) == 1 and relative.name.startswith(("captured_", "scientific_output_")):
        return relative.name
    if relative.parts[0] == "captured":
        return "_".join(relative.parts)
    return f"scientific_output_{index:04d}"


def image_check(path):
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values).all():
        raise AssertionError(f"nonfinite output: {path.name}")
    result = {"shape": list(image.shape), "dtype": str(image.get_data_dtype()),
              "all_finite": True, "sha256": sha256(path), "size_bytes": path.stat().st_size}
    if values.ndim == 4 and image.header.get_intent()[0] == "none":
        result.update(tr=float(image.header.get_zooms()[3]),
                      units=list(image.header.get_xyzt_units()))
    return result


def compare_images(first_path, second_path):
    first, second = [nib.load(str(p)) for p in (first_path, second_path)]
    # Numeric geometry, units, scaling and intent are scientific metadata;
    # descrip/aux_file may record execution labels rather than image values.
    header_fields = ("dim", "pixdim", "datatype", "bitpix", "scl_slope", "scl_inter",
                     "xyzt_units", "intent_code", "intent_p1", "intent_p2", "intent_p3",
                     "qform_code", "sform_code", "quatern_b", "quatern_c", "quatern_d",
                     "qoffset_x", "qoffset_y", "qoffset_z", "srow_x", "srow_y", "srow_z",
                     "slice_start", "slice_end", "slice_code", "slice_duration", "toffset", "dim_info")
    result = {"shape_equal": first.shape == second.shape,
              "dtype_equal": first.get_data_dtype() == second.get_data_dtype(),
              "affine_equal": bool(np.array_equal(first.affine, second.affine)),
              "full_header_bytes_equal_diagnostic": first.header.binaryblock == second.header.binaryblock,
              "scientific_header_equal": all(first.header[key].tobytes() == second.header[key].tobytes()
                                              for key in header_fields),
              "extensions_equal": [(e.get_code(), e.content) for e in first.header.extensions]
                                  == [(e.get_code(), e.content) for e in second.header.extensions]}
    if not result["shape_equal"]:
        return {**result, "bitwise_equal": False}
    # Inflate each gzip once; slab comparisons avoid additional full differences.
    x, y = np.asanyarray(first.dataobj), np.asanyarray(second.dataobj)
    changed = 0
    maximum = 0.0
    for index in range(first.shape[2]):
        a, b = np.ascontiguousarray(x[:, :, index]), np.ascontiguousarray(y[:, :, index])
        if a.dtype == b.dtype:
            unsigned = np.dtype(f"u{a.dtype.itemsize}")
            changed += int(np.count_nonzero(a.view(unsigned) != b.view(unsigned)))
        else:
            changed += int(np.count_nonzero(a != b))
        maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return {**result, "bitwise_equal": changed == 0, "changed_values_including_signed_zero": changed,
            "maximum_absolute_error": maximum}


def compare_trees(first, second):
    # FNIRT snapshots cover coefficients, both Jacobians, pull field and solver QC.
    def numerical_paths(root):
        return {str(p.relative_to(root)): p for p in root.rglob("*")
                if p.is_file() and p.name.endswith((
                    ".nii.gz", ".nii", ".mat", ".npy", ".tsv", ".bval", ".bvec",
                    ".eddy_parameters", ".eddy_rotated_bvecs", ".eddy_movement_rms",
                    ".eddy_restricted_movement_rms", ".eddy_outlier_map",
                ))}
    left, right = numerical_paths(first), numerical_paths(second)
    checks = {}
    for index, name in enumerate(sorted(left.keys() & right.keys()), start=1):
        role = safe_output_role(left[name], first, index)
        if name.endswith((".nii.gz", ".nii")):
            checks[role] = compare_images(left[name], right[name])
        elif name.endswith(".npy"):
            a, b = np.load(left[name]), np.load(right[name])
            checks[role] = {"bitwise_equal": a.dtype == b.dtype and a.shape == b.shape
                           and a.tobytes() == b.tobytes()}
        elif name.endswith(".mat"):
            a, b = np.loadtxt(left[name]), np.loadtxt(right[name])
            checks[role] = {"bitwise_equal": a.shape == b.shape and a.tobytes() == b.tobytes()}
        else:
            checks[role] = {"scientific_text_equal": left[name].read_bytes() == right[name].read_bytes()}
    left_qc = {str(p.relative_to(first)): p for p in first.rglob("fnirt_qc.json")}
    right_qc = {str(p.relative_to(second)): p for p in second.rglob("fnirt_qc.json")}
    for index, name in enumerate(sorted(left_qc.keys() & right_qc.keys()), start=1):
        a, b = [json.loads(path.read_text()) for path in (left_qc[name], right_qc[name])]
        checks[f"fnirt_solver_trace_{index}"] = {"solver_trace_equal":
            json.dumps(solver_trace(a), sort_keys=True, separators=(",", ":"))
            == json.dumps(solver_trace(b), sort_keys=True, separators=(",", ":"))}
    missing = [f"unmatched_scientific_output_{index:04d}" for index, _ in enumerate(
        sorted((left.keys() ^ right.keys()) | (left_qc.keys() ^ right_qc.keys())), start=1
    )]
    if not left_qc or not right_qc:
        missing.append("fnirt_solver_trace_missing")
    passed = bool(checks) and not missing and all(all(v for k, v in record.items()
                          if k.endswith("equal")) for record in checks.values())
    return {"all_equal": passed, "missing_outputs": missing, "outputs": checks}


class Phases:
    """Synchronized public boundaries; inner FNIRT work is only counted."""
    def __init__(self, capture):
        self.capture = capture
        self.phases = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        self.counts = Counter()
        self.capture_seconds = 0.0

    def sync(self):
        torch.cuda.synchronize()

    def __enter__(self):
        from fnit.fnirt import registration as reg
        from fnit.flirt import TorchFLIRT
        from fnit.applywarp import TorchApplyWarp
        self.stack = ExitStack()
        for obj, method, label in ((TorchFLIRT, "__call__", "flirt"),
                                  (TorchApplyWarp, "__call__", "applywarp"),
                                  (reg.TorchFNIRT, "__call__", "fnirt")):
            original = getattr(obj, method)
            def wrapper(*args, _original=original, _label=label, **kwargs):
                self.sync()
                started = time.perf_counter()
                result = _original(*args, **kwargs)
                self.sync()
                record = self.phases[_label]
                record["calls"] += 1
                record["seconds"] += time.perf_counter() - started
                if _label == "fnirt":
                    started = time.perf_counter()
                    root = self.capture / f"fnirt_{record['calls']}"
                    root.mkdir(parents=True)
                    for name, image in (("moved", result.moved), ("coefficient", result.coefficient_image),
                                        ("pull_ras", result.pull_transform),
                                        ("jacobian_nonlinear", result.nonlinear_jacobian),
                                        ("jacobian_full", result.full_pull_jacobian)):
                        nib.save(image, str(root / f"{name}.nii.gz"))
                    (root / "fnirt_qc.json").write_text(json.dumps(result.qc, indent=2) + "\n")
                    self.capture_seconds += time.perf_counter() - started
                return result
            self.stack.enter_context(patch.object(obj, method, wrapper))
        for name in ("expand_coefficients", "adjoint_field", "design_diagonal", "_trilinear_sample"):
            original = getattr(reg, name)
            def counted(*args, _original=original, _name=name, **kwargs):
                self.counts[_name] += 1
                if _name == "_trilinear_sample":
                    self.counts["trilinear_with_gradients" if kwargs.get("derivatives", True)
                                else "trilinear_without_gradients"] += 1
                return _original(*args, **kwargs)
            self.stack.enter_context(patch.object(reg, name, counted))
        return self

    def __exit__(self, *args):
        self.stack.__exit__(*args)


def augment_existing_report(output_dir, case_json, requested_pipeline=None):
    """Repair metadata only; retain the original measured source and memory."""
    path = output_dir / "report.safe.json"
    report = json.loads(path.read_text())
    pipeline = report["pipeline"]
    if requested_pipeline is not None and requested_pipeline != pipeline:
        raise ValueError("requested pipeline differs from completed report")
    if not report.get("fnirt_actually_executed"):
        raise ValueError("a completed real FNIRT report is required")
    case = json.loads(case_json.read_text())[pipeline]
    inputs = input_summaries(pipeline, case)
    timing = report["timing_seconds"]
    original_stages = timing.get("pipeline_stages_including_validation_capture", timing["pipeline_stages"])
    corrected, attribution = corrected_pipeline_timings(
        pipeline, original_stages, timing["validation_capture"]
    )
    timing.update(pipeline_stages_including_validation_capture=original_stages,
                  pipeline_stages=corrected, capture_seconds_by_containing_stage=attribution,
                  pipeline_stages_exclude_validation_capture=True,
                  stage_times_are_nested_and_not_additive=True,
                  scope="instrumented API including save, constructor and lazy imports; CUDA initialization and output checks excluded; phase-boundary synchronization retained")
    # Old reports used private BIDS relative filenames as dictionary keys.
    report["outputs"] = {
        safe_output_role(output_dir / role, output_dir, index): value
        for index, (role, value) in enumerate(sorted(report["outputs"].items()), start=1)
    }
    report["input_summaries"] = inputs
    report["schema_version"] = 2
    report["input_provenance_collection"] = (
        "post-run readback from the supplied private case; no input hashes were captured "
        "before the original API run, so historical input immutability is not asserted"
    )
    original_digest = report.get("metadata_amendment", {}).get("original_report_sha256", sha256(path))
    report["metadata_amendment"] = {
        "original_report_sha256": original_digest,
        "amendment_tool_sha256": sha256(Path(__file__)),
        "pipeline_rerun": False, "measured_source_sha256_preserved": True,
        "api_elapsed_memory_operation_counts_preserved": True,
    }
    backup = output_dir / "report.before-metadata.private.json"
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("fast-vbm", "volume", "dmri"))
    parser.add_argument("--case-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--augment-existing-report", action="store_true",
                        help="Add input readback and correct capture attribution; do not run an API")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-gb", type=float, default=19)
    args = parser.parse_args()
    if args.augment_existing_report:
        if args.case_json is None or args.compare_to:
            parser.error("augmentation requires case-json and cannot be combined with compare-to")
        report = augment_existing_report(args.output_dir, args.case_json, args.pipeline)
        print(json.dumps({"pipeline": report["pipeline"], "metadata_augmented": True,
                          "pipeline_rerun": False, "input_roles": len(report["input_summaries"])}))
        return
    if args.compare_to:
        report = compare_trees(args.compare_to, args.output_dir)
        (args.output_dir / "comparison.safe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"all_equal": report["all_equal"], "compared_outputs": len(report["outputs"])}))
        if not report["all_equal"]:
            raise SystemExit(1)
        return
    if args.pipeline is None or args.case_json is None or args.source_root is None:
        parser.error("pipeline, case-json and source-root are required for a run")
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory to avoid cached execution")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(
        min(1, args.memory_limit_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    config = json.loads(args.case_json.read_text())
    case = config[args.pipeline]
    inputs = input_summaries(args.pipeline, case)
    benchmark_tool_sha256 = sha256(Path(__file__))
    started = time.perf_counter()
    with Phases(args.output_dir / "captured") as phases:
        if args.pipeline == "fast-vbm":
            from fnit.fast_vbm import FastVBM
            model = FastVBM(device="cuda:0", threads=args.threads, registration_backend="fnirt",
                            synthstrip_weights=case["synthstrip_weights"])
            result = model(case["image"], case["template"], reference_mask=case["reference_mask"])
            result.save(args.output_dir / "outputs")
            timing = result.timing_sec
        elif args.pipeline == "volume":
            from fnit.fmri import fMRIVolume_pipeline
            result = fMRIVolume_pipeline(**case, derivatives_root=args.output_dir / "outputs",
                                        registration_backend="fnirt", device="cuda:0", reuse_anatomical=False)
            timing = result.timing_seconds
        else:
            from fnit.dmri_pipeline import DMRIPipeline
            model = DMRIPipeline(device="cuda:0", registration_backend="tbss", eddy_gp_seed=12345)
            result = model.run_bids(**case, output_dir=args.output_dir / "outputs")
            timing = result.qc["timings_seconds"]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if phases.phases["fnirt"]["calls"] != 1:
        raise AssertionError("the full pipeline must actually run FNIRT exactly once")
    checks = {safe_output_role(p, args.output_dir, index): image_check(p)
              for index, p in enumerate(sorted(args.output_dir.rglob("*.nii.gz")), start=1)}
    corrected_stages, capture_attribution = corrected_pipeline_timings(
        args.pipeline, timing, phases.capture_seconds
    )
    report = {"schema_version": 2, "pipeline": args.pipeline, "real_data": True,
              "full_default_schedule": True, "fnirt_actually_executed": True,
              "anatomical_cache_reused": False, "source_sha256": source_hashes(args.source_root),
              "benchmark_tool_sha256": benchmark_tool_sha256, "input_summaries": inputs,
              "input_provenance_collection": "before API timer; content SHA-256, size and image shape by anonymous role",
              "timing_seconds": {"api_including_save_excluding_validation_capture": elapsed - phases.capture_seconds,
                                  "validation_capture": phases.capture_seconds,
                                  "pipeline_stages": corrected_stages,
                                  "pipeline_stages_including_validation_capture": timing,
                                  "pipeline_stages_exclude_validation_capture": True,
                                  "capture_seconds_by_containing_stage": capture_attribution,
                                  "public_function_phases": dict(phases.phases),
                                  "scope": "instrumented API including save, constructor and lazy imports; input/source hashing, CUDA initialization, output checks and comparison excluded; phase-boundary synchronization retained",
                                  "stage_times_are_nested_and_not_additive": True},
              "operation_counts": dict(phases.counts), "outputs": checks,
              "memory": {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()},
              "conditions": {"threads": args.threads, "tf32": True, "float16": False,
                             "eddy_gp_seed": 12345 if args.pipeline == "dmri" else None,
                             "original_software_called": False, "shared_gpu": True}}
    (args.output_dir / "report.safe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pipeline": args.pipeline, "seconds": elapsed - phases.capture_seconds,
                      "peak_allocated": report["memory"]["peak_cuda_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
