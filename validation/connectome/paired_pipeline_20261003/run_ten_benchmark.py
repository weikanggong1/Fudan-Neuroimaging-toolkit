"""Real paired-template CLI benchmark; GPU execution is explicitly requested.

A preflight reads real inputs and writes hashes without initializing CUDA.
The coordinator runs one fresh worker per case under the shared GPU lock.
Reports are private because they contain actual input paths; public-summary
exports only case IDs, statistics, source hashes, times and validation results.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[3]
# Import existing benchmark monitor utilities, without changing the caller's
# declared PYTHONPATH for FNIT computational source.
sys.path.insert(0, str(REPOSITORY))
from tools.benchmark_connectome_raw_bids import (
    GPUProcessMonitor, PeakLedger, atomic_json, cli_value, json_safe, sha256,
)

CASES = ("sub-CON01", "sub-CON03", "sub-CON04", "sub-CON05", "sub-CON06",
         "sub-CON07", "sub-CON08", "sub-CON09", "sub-CON10", "sub-CON11")
MATRICES = ("count", "sift2_fbc", "mean_length", "mean_fa")
PHASES = ("cold", "repeat", "new_template", "radius_changed")


def source_identity(commit):
    import fnit
    package = Path(fnit.__file__).resolve().parent
    files = {str(path.relative_to(package)): sha256(path)
             for path in sorted(package.rglob("*")) if path.is_file()
             and path.suffix in {".py", ".tsv", ".npz", ".json"}}
    return {"declared_git_commit": commit, "fnit_import_path": str(package),
            "source_sha256": files, "source_inventory_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "harness_sha256": sha256(__file__),
            "memory_monitor_sha256": sha256(REPOSITORY / "tools/benchmark_connectome_raw_bids.py")}


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path)}


def template_definitions(subject):
    root = Path(subject)
    surface = lambda name: dict(name=name, kind="surface", space="native",
                               left_path=str(root / f"label/lh.{name}.annot"),
                               right_path=str(root / f"label/rh.{name}.annot"))
    volume = lambda name: dict(name=name, kind="volume", space="t1",
                              volume_path=str(root / f"mri/{name}.mgz"))
    a, b, c, d = surface("aparc"), surface("aparc.a2009s"), volume("aparc+aseg"), volume("aparc.a2009s+aseg")
    initial = [dict(name="SS", first=a, second=b), dict(name="VV", first=c, second=d),
               dict(name="SV", first=a, second=d)]
    new_pair = [dict(name="SV_new_aseg", first=a, second=volume("aseg"))]
    return initial, new_pair


def preflight(bindings_path, selected_cases, output, source_commit):
    import nibabel as nib
    import numpy as np
    import torch
    from fnit.connectome.recon_backend import inspect_recon_subject
    from fnit.connectome.template_inputs import TemplateSpec, template_dependency_paths
    from fnit.dmri_pipeline.bids import locate_bids_dwi
    bindings = json.loads(Path(bindings_path).read_text())
    report = {"status": "complete", "scope": "read-only ten real input preflight; no reconstruction, DWI processing or tracking",
              "source": source_identity(source_commit), "bindings": file_record(bindings_path),
              "cases": {}, "cuda_initialized": torch.cuda.is_initialized()}
    for case in selected_cases:
        entry = bindings["cases"][case]
        selected = entry["selected_inputs"]
        arguments = entry["prior_cli_arguments"]
        root = cli_value(arguments, "--bids-root")
        selector = dict(subject=case.removeprefix("sub-"), session=cli_value(arguments, "--session"),
                        acquisition=cli_value(arguments, "--acquisition"), run=cli_value(arguments, "--run"),
                        direction=cli_value(arguments, "--direction"))
        raw = locate_bids_dwi(root, **selector)
        image = nib.load(selected["dwi"])
        values = np.loadtxt(selected["bvals"]).reshape(-1)
        vectors = np.loadtxt(selected["bvecs"])
        if len(image.shape) != 4 or len(values) != image.shape[-1] or vectors.shape not in ((3, len(values)), (len(values), 3)):
            raise ValueError(f"{case}: corrected DWI and bval/rotated-bvec counts disagree")
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValueError(f"{case}: non-finite gradients")
        anatomy = inspect_recon_subject(selected["freesurfer_subject_dir"])
        for name, saved in entry["anatomy"]["files"].items():
            if sha256(Path(selected["freesurfer_subject_dir"]) / name) != saved["sha256"]:
                raise ValueError(f"{case}: supplied anatomy changed: {name}")
        initial, extra = template_definitions(selected["freesurfer_subject_dir"])
        paths = set()
        for pair in (*initial, *extra):
            for side in ("first", "second"):
                paths.update(template_dependency_paths(TemplateSpec(**pair[side]), selected["freesurfer_subject_dir"]))
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"{case}: missing actual pair templates")
        records = {str(path): file_record(path) for path in sorted(paths)}
        raw_paths = [raw.image, raw.bval, raw.bvec, raw.reverse, raw.reverse_bval, raw.t1w,
                     Path(root) / "dataset_description.json"]
        report["cases"][case] = {
            "bids_root": str(raw.root), "selector": selector,
            "corrected": {name: file_record(selected[name]) for name in ("dwi", "bvals", "bvecs")},
            "corrected_shape": [int(v) for v in image.shape], "corrected_affine": image.affine.tolist(),
            "raw_input_provenance": [file_record(path) for path in raw_paths if path is not None],
            "raw_t1w": str(raw.t1w) if raw.t1w else None,
            "freesurfer_subject_dir": selected["freesurfer_subject_dir"],
            "anatomy_content_sha256": anatomy["content_sha256"],
            "template_inputs": records, "initial_pairs": initial, "replacement_pairs": extra,
            "preprocessing_scope": "corrected DWI and rotated bvec supplied from previous bound FNIT raw run; raw inputs retained as provenance",
        }
    report["cuda_initialized"] = torch.cuda.is_initialized()
    if report["cuda_initialized"]:
        raise RuntimeError("preflight unexpectedly initialized CUDA")
    atomic_json(output, report)
    return report


def array_hash(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def result_fingerprints(result):
    fields = {name: array_hash(getattr(result, name)) for name in (
        "five_tissue", "five_tissue_affine", "gmwmi", "wm_sh", "fa", "brain_mask",
        "sift2_weights", "dwi_affine", "dwi_to_t1_world")}
    tracks = result.tractogram
    fields.update({"track_" + name: array_hash(getattr(tracks, name))
                   for name in ("endpoints", "lengths_mm", "mean_fa", "accepted_seeds")})
    digest = hashlib.sha256()
    offsets = [0]
    for path in tracks.paths:
        values = path.detach().cpu().numpy()
        digest.update(values.tobytes())
        offsets.append(offsets[-1] + len(values))
    fields["track_points"] = dict(sha256=digest.hexdigest(), offsets_sha256=array_hash(offsets)["sha256"],
                                  tracks=len(tracks.paths), points=offsets[-1], seeds_attempted=tracks.seeds_attempted)
    matrices = {name: {kind: array_hash(value) for kind, value in pair.matrices.items()}
                for name, pair in result.pair_results.items()}
    return {"core": fields, "matrices": matrices}


def pair_cpu_snapshot(pair, result):
    # Small actual post-timing inputs; no FOD copy and no change to production.
    return {"first": pair.first.labels.detach().cpu(), "second": pair.second.labels.detach().cpu(),
            "first_affine": pair.first.affine.detach().cpu(), "second_affine": pair.second.affine.detach().cpu(),
            "first_count": len(pair.first.nodes), "second_count": len(pair.second.nodes),
            "endpoints": result.tractogram.endpoints.detach().cpu(),
            "weights": result.sift2_weights.detach().cpu(), "lengths": result.tractogram.lengths_mm.detach().cpu(),
            "fa": result.tractogram.mean_fa.detach().cpu(),
            "matrices": {name: value.detach().cpu().numpy() for name, value in pair.matrices.items()}}


def cpu_pair_oracle(snapshot, radius):
    """Existing endpoint spatial rule plus independently ordered Python reduction."""
    import numpy as np
    from fnit.connectome.assignment import assign_endpoint_labels
    positions = snapshot["endpoints"].reshape(-1, 3)
    a = assign_endpoint_labels(positions, snapshot["first"], snapshot["first_affine"], radius=radius).numpy().reshape(-1, 2)
    b = assign_endpoint_labels(positions, snapshot["second"], snapshot["second_affine"], radius=radius).numpy().reshape(-1, 2)
    shape = snapshot["first_count"], snapshot["second_count"]
    count = np.zeros(shape, np.int64)
    weight_sum, length_sum, fa_sum = (np.zeros(shape, np.float64) for _ in range(3))
    weights, lengths, fa = (snapshot[name].numpy().astype(np.float32) for name in ("weights", "lengths", "fa"))
    matched = 0
    for i in range(len(a)):
        forward, reverse = (int(a[i, 0]), int(b[i, 1])), (int(a[i, 1]), int(b[i, 0]))
        cells = [forward] if forward == reverse else [forward, reverse]
        valid = [(row - 1, column - 1) for row, column in cells if row > 0 and column > 0]
        matched += bool(valid)
        for row, column in valid:
            weight = float(weights[i])
            count[row, column] += 1
            weight_sum[row, column] += weight
            length_sum[row, column] += weight * float(lengths[i])
            fa_sum[row, column] += weight * float(fa[i])
    denominator = np.maximum(weight_sum, np.finfo(np.float64).tiny)
    oracle = {"count": count, "sift2_fbc": weight_sum.astype(np.float32),
              "mean_length": (length_sum / denominator).astype(np.float32),
              "mean_fa": (fa_sum / denominator).astype(np.float32)}
    comparisons = {}
    for name, expected in oracle.items():
        actual = snapshot["matrices"][name]
        equal_nan = np.isnan(actual) & np.isnan(expected)
        finite = np.isfinite(actual) & np.isfinite(expected)
        delta = np.abs(actual[finite].astype(np.float64)
                       - expected[finite].astype(np.float64))
        comparisons[name] = {"arrays_exact": bool(np.array_equal(actual, expected, equal_nan=True)),
                             "max_absolute_error": float(delta.max(initial=0)),
                             "finite_comparison_elements": int(np.count_nonzero(finite)),
                             "nan_mask_equal": bool(np.array_equal(np.isnan(actual), np.isnan(expected))),
                             "different_elements": int(np.count_nonzero(~((actual == expected) | equal_nan)))}
    return {"matched_unique_streamlines": matched, "matrix_count_sum": int(count.sum()),
            "comparisons": comparisons,
            "passed": all(value["arrays_exact"] for value in comparisons.values()),
            "scope": "same FNIT tractogram, independent per-track cell aggregation; endpoint spatial rule shared with existing FNIT; not an independent MRtrix raw SC comparison"}


class Observe:
    """Capture the actual returned result and host stage durations without extra sync."""
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.stack = ExitStack()
        self.stages, self.result = {}, None
        self.ledger = PeakLedger(torch, device)

    def __enter__(self):
        import fnit.connectome.pipeline as core
        import fnit.connectome.paired_pipeline as pairs
        self.stack.enter_context(patch.object(self.torch.cuda, "reset_peak_memory_stats", self.ledger.reset))
        for module, names in ((core, ("_compute_shared_core", "_registration", "fit_mrtrix_dhollander_tensor",
                "estimate_mrtrix_dhollander", "fit_mrtrix_msmt_csd", "normalise_mrtrix_three_tissue",
                "probabilistic_tractography", "estimate_sift2_weights", "sample_streamline_mean_precise")),
                              (pairs, ("prepare_template", "build_pair_connectomes"))):
            for name in names:
                original = getattr(module, name)
                @functools.wraps(original)
                def timed(*args, _name=name, _original=original, **kwargs):
                    start = time.perf_counter()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        self.stages.setdefault(_name, []).append(time.perf_counter() - start)
                self.stack.enter_context(patch.object(module, name, timed))
        original = core.UKBConnectome_pipeline.__call__
        @functools.wraps(original)
        def capture(*args, **kwargs):
            self.result = original(*args, **kwargs)
            return self.result
        self.stack.enter_context(patch.object(core.UKBConnectome_pipeline, "__call__", capture))
        return self

    def __exit__(self, *exc):
        self.ledger.capture("phase_exit")
        self.stack.close()


def phase_arguments(case, info, output, pairs_json, options, radius):
    result = ["UKBConnectome_pipeline", "--bids-root", info["bids_root"],
              "--subject", case.removeprefix("sub-"), "--freesurfer-subject-dir", info["freesurfer_subject_dir"],
              "--recon-backend", "provided", "--template-pairs", str(pairs_json),
              "--output-dir", str(output), "--device", options.device,
              "--n-seeds", str(options.n_seeds), "--seed", str(options.seed),
              "--assignment-radius", str(radius), "--eddy-gp-seed", str(options.eddy_gp_seed)]
    for name in ("session", "run", "acquisition", "direction"):
        if info["selector"].get(name):
            result += ["--" + name, info["selector"][name]]
    if options.input_mode == "corrected":
        result += ["--corrected-dwi", info["corrected"]["dwi"]["path"],
                   "--rotated-bvecs", info["corrected"]["bvecs"]["path"]]
    return result


def worker(options):
    import gc
    import torch
    import fnit.cli as cli
    from fnit.connectome.assignment import build_connectomes
    from fnit.connectome.paired_assignment import build_pair_connectomes
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(options.threads)
    case_root = Path(options.run_root) / options.case
    if case_root.exists():
        raise FileExistsError(f"fresh worker requires a new case directory: {case_root}")
    case_root.mkdir(parents=True)
    report_path = case_root / "benchmark.private.json"
    report = {"case_id": options.case, "status": "running", "start_utc": datetime.now(timezone.utc).isoformat(),
              "source": source_identity(options.source_commit), "input_mode": options.input_mode,
              "seed": options.seed, "n_seeds": options.n_seeds, "device": options.device,
              "threads": options.threads, "phases": {}, "scope": "actual paired-template CLI including normal outputs; supplied official anatomy; preprocessing supplied when input_mode=corrected",
              "official_sc_parity": "not_assessed_in_this_new_template_benchmark"}
    atomic_json(report_path, report)
    lock_started = time.perf_counter()
    with open(options.gpu_lock, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report["gpu_lock_queue_seconds"] = time.perf_counter() - lock_started
        try:
            pre = json.loads(Path(options.preflight).read_text())
            info = pre["cases"][options.case]
            if report["source"]["source_inventory_sha256"] != pre["source"]["source_inventory_sha256"]:
                raise RuntimeError("worker source differs from preflight freeze")
            for saved in (*info["corrected"].values(), *info["template_inputs"].values()):
                if sha256(saved["path"]) != saved["sha256"]:
                    raise RuntimeError("declared corrected input/template changed after preflight")
            output = case_root / "connectome"
            cold_fingerprints = None
            for phase in PHASES:
                pairs = info["replacement_pairs"] if phase == "new_template" else info["initial_pairs"]
                pairs_path = case_root / (phase + "_pairs.json")
                atomic_json(pairs_path, {"pairs": pairs})
                radius = options.replacement_radius if phase == "radius_changed" else options.radius
                arguments = phase_arguments(options.case, info, output, pairs_path, options, radius)
                if options.device.startswith("cuda") and torch.cuda.is_initialized():
                    torch.cuda.synchronize(options.device)
                    torch.cuda.reset_peak_memory_stats(options.device)
                monitor = GPUProcessMonitor(torch, options.device, options.memory_sample_interval, options.gpu_uuid)
                monitor.start()
                observation = Observe(torch, options.device)
                start = time.perf_counter()
                try:
                    with observation:
                        cli.main(arguments)
                        if options.device.startswith("cuda") and torch.cuda.is_initialized():
                            torch.cuda.synchronize(options.device)
                finally:
                    elapsed = time.perf_counter() - start
                    memory = monitor.finish()
                result = observation.result
                if result is None:
                    raise RuntimeError("CLI did not return a new result object")
                phase_report = {"cli_arguments": arguments, "wall_seconds": elapsed,
                    "wall_scope": "actual CLI production input checks, BIDS preparation, numeric core, checkpoint publication/restoration, label/matrix outputs, final requested-device synchronize; excludes input/source preflight hashes, oracle and returned-result hashes",
                    "stage_host_seconds": observation.stages,
                    "stage_timing_scope": "unsynchronized host function durations, inclusive/nested; GPU work can complete later; not kernel-only times and must not be summed",
                    "preparation_stages": result.preparation_stages, "cache_status": result.cache_status,
                    "allocator_memory": observation.ledger.report(), "gpu_process_memory": memory,
                    "radius_mm": radius, "seed_attempts": result.tractogram.seeds_attempted,
                    "accepted_streamlines": len(result.tractogram.paths), "pairs": {}}
                report["phases"][phase] = phase_report  # Preserve actual failed gates too.
                fingerprints = result_fingerprints(result)
                phase_report["fingerprints"] = fingerprints
                if phase == "cold":
                    cold_fingerprints = fingerprints
                else:
                    phase_report["core_arrays_exact_to_cold"] = fingerprints["core"] == cold_fingerprints["core"]
                    if not phase_report["core_arrays_exact_to_cold"] or result.cache_status["core"] != "skipped":
                        raise RuntimeError(f"{phase}: core was changed or recomputed")
                if phase == "repeat":
                    phase_report["all_pair_arrays_exact_to_cold"] = fingerprints["matrices"] == cold_fingerprints["matrices"]
                    if not phase_report["all_pair_arrays_exact_to_cold"] or any(p.cache_status != "skipped" for p in result.pair_results.values()):
                        raise RuntimeError("repeat did not reuse identical pair matrices")
                for name, pair in result.pair_results.items():
                    snapshot = pair_cpu_snapshot(pair, result)
                    started = time.perf_counter()
                    oracle = cpu_pair_oracle(snapshot, radius)
                    phase_report["pairs"][name] = {"shape": [len(pair.first.nodes), len(pair.second.nodes)],
                                                   "cache_status": pair.cache_status,
                                                   "oracle_seconds": time.perf_counter() - started, "oracle": oracle}
                    if not oracle["passed"]:
                        raise RuntimeError(f"{phase}/{name}: independent aggregation oracle mismatch")
                    if phase == "cold" and name == "SS":
                        identity = dict(weights=snapshot["weights"], lengths=snapshot["lengths"], fa=snapshot["fa"], radius=radius)
                        expected = build_connectomes(snapshot["endpoints"], snapshot["first"], snapshot["first_affine"],
                                                     node_count=snapshot["first_count"], **identity)
                        actual = build_pair_connectomes(snapshot["endpoints"], snapshot["first"], snapshot["first_affine"],
                                                        snapshot["first"], snapshot["first_affine"],
                                                        first_node_count=snapshot["first_count"], second_node_count=snapshot["first_count"],
                                                        same_template=True, **identity)
                        phase_report["identity_square_arrays_exact"] = all(
                            torch.allclose(expected[k], actual[k], rtol=0, atol=0, equal_nan=True)
                            if expected[k].is_floating_point() else torch.equal(expected[k], actual[k])
                            for k in expected)
                        if not phase_report["identity_square_arrays_exact"]:
                            raise RuntimeError("identity paired template differs from existing square builder")
                    del snapshot
                state = output / "pairs_run_state.json"
                phase_report["output_state"] = json.loads(state.read_text())
                peak = memory.get("peak_process_tree_bytes")
                issues = []
                if options.device.startswith("cuda"):
                    if memory.get("status") != "measured" or memory.get("samples", 0) < 3:
                        issues.append("insufficient_process_samples")
                    if memory.get("errors") or memory.get("failed_samples") or memory.get("unresolved_device_samples"):
                        issues.append("monitor_errors_or_unresolved_device")
                    gap = memory.get("max_observed_interval_seconds")
                    if gap is None or gap > max(2., 5 * options.memory_sample_interval):
                        issues.append("sampling_gap")
                phase_report["memory_monitor_issues"] = issues
                phase_report["process_memory_under_20gb"] = (peak < 20_000_000_000) if peak is not None and not issues else None
                report["phases"][phase] = phase_report
                atomic_json(report_path, report)
                del result, observation
                gc.collect()
                if options.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if source_identity(options.source_commit)["source_inventory_sha256"] != report["source"]["source_inventory_sha256"]:
                raise RuntimeError("source changed during worker")
            report.update(status="completed", end_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            report.update(status="failed", end_utc=datetime.now(timezone.utc).isoformat(),
                          exception={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            atomic_json(report_path, report)
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    atomic_json(report_path, report)


def public_allocator_memory(memory):
    """Keep measured allocator statistics, without device IDs or error text."""
    if memory is None:
        return None
    result = {key: memory.get(key) for key in (
        "allocated_bytes", "reserved_bytes", "allocated_gb", "allocated_gib",
        "reserved_gb", "reserved_gib", "scope")}
    intervals = memory.get("intervals", [])
    result["intervals"] = []
    for interval in intervals:
        failed = "error" in interval
        measured = "allocated_bytes" in interval and "reserved_bytes" in interval
        reason = interval.get("reason")
        result["intervals"].append({
            "reason": reason if reason in ("before_existing_reset", "phase_exit") else "unknown",
            "status": "failed" if failed else "measured" if measured else "unknown",
            "allocated_bytes": interval.get("allocated_bytes"),
            "reserved_bytes": interval.get("reserved_bytes"),
        })
    result["failed_intervals"] = sum("error" in interval for interval in intervals)
    return result


def public_summary(options):
    rows = {}
    for case in options.cases:
        path = Path(options.run_root) / case / "benchmark.private.json"
        if not path.is_file():
            rows[case] = {"status": "missing"}
            continue
        report = json.loads(path.read_text())
        rows[case] = {"status": report["status"], "input_mode": report["input_mode"],
                      "n_seeds": report["n_seeds"], "seed": report["seed"],
                      "threads": report["threads"], "device": report["device"],
                      "worker_harness_sha256": report["source"]["harness_sha256"],
                      "source_inventory_sha256": report["source"]["source_inventory_sha256"],
                      "source_commit": report["source"]["declared_git_commit"],
                      "phases": {}}
        for name, phase in report["phases"].items():
            rows[case]["phases"][name] = {key: phase.get(key) for key in (
                "wall_seconds", "wall_scope", "stage_host_seconds", "stage_timing_scope", "preparation_stages", "seed_attempts", "accepted_streamlines", "radius_mm",
                "core_arrays_exact_to_cold", "all_pair_arrays_exact_to_cold", "identity_square_arrays_exact",
                "allocator_memory", "gpu_process_memory", "memory_monitor_issues", "process_memory_under_20gb", "pairs")}
            rows[case]["phases"][name]["core_cache_status"] = phase.get("cache_status", {}).get("core")
            rows[case]["phases"][name]["allocator_memory"] = public_allocator_memory(phase.get("allocator_memory"))
            memory = phase.get("gpu_process_memory", {})
            rows[case]["phases"][name]["gpu_process_memory"] = {key: memory.get(key) for key in (
                "backend", "status", "samples", "sample_interval_seconds", "max_observed_interval_seconds",
                "failed_samples", "unresolved_device_samples", "observed_span_seconds",
                "peak_process_tree_bytes", "peak_process_tree_gb", "scope")}
        if report.get("exception"):
            rows[case]["failure"] = {"type": report["exception"]["type"], "message": "see private report; failure preserved"}
    atomic_json(options.report, {"cases": rows, "expected_cases": list(options.cases),
                                "complete_cases": sum(row["status"] == "completed" for row in rows.values()),
                                "scope": "new paired-template pipeline and cache/oracle validation; supplied anatomy/corrected preparation; not independent MRtrix raw SC equivalence"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "worker", "public-summary"))
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eddy-gp-seed", type=int, default=12345)
    parser.add_argument("--radius", type=float, default=4.)
    parser.add_argument("--replacement-radius", type=float, default=3.)
    parser.add_argument("--input-mode", choices=("corrected", "raw"), default="corrected")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--memory-sample-interval", type=float, default=.5)
    parser.add_argument("--gpu-lock", default="/tmp/fnit-recon-five-20261002-gongwk.gpu.lock")
    options = parser.parse_args()
    if options.mode == "preflight":
        if options.bindings is None or options.report is None:
            parser.error("preflight requires --bindings and --report")
        preflight(options.bindings, options.cases, options.report, options.source_commit)
    elif options.mode == "worker":
        if options.case is None or options.run_root is None or options.preflight is None:
            parser.error("worker requires --case, --run-root and --preflight")
        worker(options)
    elif options.mode == "public-summary":
        if options.run_root is None or options.report is None:
            parser.error("public-summary requires --run-root and --report")
        public_summary(options)
    else:
        if options.preflight is None or options.run_root is None:
            parser.error("run requires --preflight and --run-root")
        for case in options.cases:
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--case", case,
                       "--preflight", str(options.preflight), "--run-root", str(options.run_root),
                       "--source-commit", options.source_commit, "--device", options.device,
                       "--threads", str(options.threads), "--n-seeds", str(options.n_seeds),
                       "--seed", str(options.seed), "--eddy-gp-seed", str(options.eddy_gp_seed),
                       "--radius", str(options.radius), "--replacement-radius", str(options.replacement_radius),
                       "--input-mode", options.input_mode, "--gpu-lock", options.gpu_lock,
                       "--memory-sample-interval", str(options.memory_sample_interval)]
            if options.gpu_uuid:
                command += ["--gpu-uuid", options.gpu_uuid]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
