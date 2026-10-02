#!/usr/bin/env python3
"""Read-only live numerical audit of ten completed FNIT raw T1 pipelines.

This is completion/numerical validation of FNIT only. It is not an accuracy
comparison, not a full final cohort benchmark, and does not read official
images, launch fitting or request GPU work. The frozen analyzer is reused.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time


ANALYZER_SHA256 = "ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def distribution(values):
    values = [float(value) for value in values]
    if len(values) != 10 or not all(math.isfinite(value) for value in values):
        raise ValueError("The all-ten distribution needs ten actual finite observations")
    return {"n": 10, "mean": statistics.mean(values), "std_sample": statistics.stdev(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--analyzer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    analyzer_path = args.analyzer or root / "analyze_cohort.py"
    output = args.output or root / "raw_all_live_audit.json"
    if output.exists():
        raise ValueError("Preserve the previous live audit; use a new output path")
    if sha256(analyzer_path) != ANALYZER_SHA256:
        raise ValueError("The raw live audit must reuse the approved frozen analyzer")
    original_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if original_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, original_affinity[-2:])
    spec = importlib.util.spec_from_file_location("frozen_cohort_analyzer_readonly", analyzer_path)
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)
    manifest_path = root / "cohort_manifest.json"
    queue_path = root / "fnit_queue.json"
    manifest = analyzer.read_json(manifest_path)
    queue = analyzer.read_json(queue_path)
    expected_ids = [f"sub-{index:02d}" for index in range(1, 11)]
    if ([case["id"] for case in manifest["cases"]] != expected_ids
            or queue.get("state") != "completed"):
        raise ValueError("Require the predetermined ten subjects and completed raw queue")
    canonical = analyzer.canonical_labels(manifest, manifest_path.parent)
    runtime, source = analyzer.audit_source(manifest, manifest_path.parent)
    runtime_hash = json_hash(runtime)
    started = time.perf_counter()
    result = {"schema_version": 1, "state": "running", "host": socket.gethostname(),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "raw_fnit_completion_only": True, "cross_accuracy_evaluated": False,
        "final_full_cohort_benchmark": False, "no_official_images_read": True,
        "no_fitting_or_gpu_work_launched": True,
        "scope": "Independently re-read all saved raw FNIT native/HR maps, public T1, API/driver/observer reports and own-PID GPU monitors. Official comparison awaits complete reference outputs.",
        "audit_script": identity(Path(__file__)), "frozen_analyzer": identity(analyzer_path),
        "comparison_helper": identity(analyzer.HELPERS / "analyze_repeatability.py"),
        "cohort_manifest": identity(manifest_path), "completed_raw_queue": identity(queue_path),
        "cpu_resource_limit": {"thread_environment": {variable: os.environ[variable] for variable in THREAD_VARIABLES},
            "affinity_before": original_affinity,
            "affinity_during": sorted(os.sched_getaffinity(0)) if original_affinity else None,
            "execution": "Sequential subjects; at most two allowed CPU cores and two library threads"},
        "planned_subjects": 10, "source_audit": source,
        "runtime_module_map_sha256": runtime_hash, "canonical_label_count": len(canonical), "cases": []}
    write_json(output, result)
    try:
        for case in manifest["cases"]:
            before = time.perf_counter()
            audited = analyzer.audit_fnit(case, "raw", queue, manifest_path.parent, canonical, runtime)
            record, run, context = audited["record"], audited["run"], audited["record"]["context_identity"]
            if context is None or record["context_identity_artifact"] is None:
                raise ValueError(f"Actual raw source observer is missing: {case['id']}")
            analyzer.identity(run["context_identity"], manifest_path.parent)
            if record["context_identity_artifact"]["sha256"] != run["context_identity"]["sha256"]:
                raise ValueError("Actual observer identity differs from completed raw queue")
            if (context["runtime_source_sha256_before"] != runtime
                    or context["runtime_source_sha256_after"] != runtime):
                raise ValueError("Observer source before/after differs from the frozen source")
            if run["source_manifest_sha256"] != source["source_manifest"]["sha256"]:
                raise ValueError("Completed runner source differs from independently audited source")
            times = record["timings"]
            pairs = (("compute_seconds", "api_compute_seconds"), ("api_total_seconds", "api_total_seconds"),
                     ("output_save_seconds", "output_save_seconds"), ("context_observer_seconds", "context_observer_seconds"))
            for queue_key, actual_key in pairs:
                if not math.isclose(run[queue_key], times[actual_key], abs_tol=1e-9, rel_tol=0):
                    raise ValueError(f"Actual timing differs from raw queue: {case['id']}/{actual_key}")
            timings = {key: times[key] for key in ("api_compute_seconds", "api_total_seconds", "output_save_seconds",
                "process_wall_seconds", "context_observer_seconds", "process_wall_scope", "context_observer_scope",
                "started_unix", "finished_unix")}
            timings.update({key: run[key] for key in ("input_wait_seconds", "gpu_budget_wait_seconds", "preflight_identity_seconds")})
            result["cases"].append({"case_id": case["id"], "development_seen": case["development_seen"], "state": "passed",
                "raw_input_sha256": record["input_sha256"], "actual_input_grid": record["input_grid_and_valid_voxels"],
                "native_and_hr_label_arrays": record["arrays"], "numeric_gates": record["numeric_gates"],
                "fit_min_jacobians": record["fit_min_jacobians"], "configuration": record["configuration"],
                "api_report": record["api_report"], "driver_report": record["report"],
                "source_manifest_sha256": run["source_manifest_sha256"],
                "source_observer": {"artifact": record["context_identity_artifact"],
                    "runtime_before_map_sha256": json_hash(context["runtime_source_sha256_before"]),
                    "runtime_after_map_sha256": json_hash(context["runtime_source_sha256_after"]),
                    "runtime_before_after_exact_frozen_match": True,
                    "solver_options_changed": context.get("solver_options_changed", False),
                    "reference_used_by_observer": context.get("reference_used_by_observer", False),
                    "runtime_source_unchanged": context.get("runtime_source_unchanged", True)},
                "gpu": run["physical_gpu"], "own_pid": run["pid"], "own_gpu_monitor": record["gpu_monitor"],
                "timings": timings, "shared_preprocessing": record["shared_preprocessing"],
                "independent_cpu_read_audit_seconds": time.perf_counter() - before})
            write_json(output, result)
            print(json.dumps({"case_id": case["id"], "numeric_audit": "passed",
                "compute_seconds": times["api_compute_seconds"], "min_jacobians": record["fit_min_jacobians"],
                "own_sampled_gpu_peak_mib": record["gpu_monitor"]["sampled_peak_own_memory_mib"]}), flush=True)
        cases = result["cases"]
        input_hashes = [next(iter(case["raw_input_sha256"].values())) for case in cases]
        instances = {(case["own_pid"], case["timings"]["started_unix"]) for case in cases}
        if len(set(input_hashes)) != 10 or len(instances) != 10:
            raise ValueError("Each completed raw subject must have a distinct input hash and process instance")
        source_after, queue_after = sha256(analyzer_path), sha256(queue_path)
        if (source_after != ANALYZER_SHA256 or queue_after != result["completed_raw_queue"]["sha256"]
                or sha256(manifest_path) != result["cohort_manifest"]["sha256"]):
            raise ValueError("The frozen analyzer/cohort manifest/completed raw queue changed while auditing")
        result["summary"] = {"passed_subjects": 10, "unique_case_ids": 10, "unique_input_sha256": 10,
            "unique_fnit_process_instances": len(instances), "unique_fnit_process_pids": len({case["own_pid"] for case in cases}),
            "process_identity_scope": "PID plus actual launch timestamp; a legitimate PID reuse across sequential runs is not a failure.",
            "gpu_uuids": sorted({case["gpu"]["uuid"] for case in cases}),
            "all_native_hr_voxels_finite_integer_within_support": True,
            "all_actual_raw_inputs_and_source_observers_consistent": True,
            "all_reported_volumes_finite_nonnegative_and_native_hard_counts_consistent": True,
            "all_four_minimum_fitted_mesh_jacobians_finite_positive_per_case": True,
            "own_sampled_gpu_peak_mib": distribution([case["own_gpu_monitor"]["sampled_peak_own_memory_mib"] for case in cases]),
            "own_gpu_limit_mib": 19073, "gpu_monitor_scope": "Independent re-read of sampled own-PID memory, not instantaneous memory maximum",
            "timings_seconds": {key: distribution([case["timings"][key] for case in cases]) for key in
                ("api_compute_seconds", "api_total_seconds", "output_save_seconds", "process_wall_seconds", "context_observer_seconds")},
            "timing_scope": "Actual saved API timing and independent process watcher timing are separate; observer overhead retained, no subtraction. Input/GPU-budget wait and preflight are reported separately.",
            "jacobian_scope": "Four positive saved minima over fitted mesh tetrahedra, not an unrecorded voxelwise Jacobian image."}
        result.update(state="passed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            elapsed_independent_audit_seconds=time.perf_counter() - started,
            frozen_analyzer_unchanged=True, completed_raw_queue_unchanged=True)
        write_json(output, result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    except Exception as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        write_json(output, result)
        raise


if __name__ == "__main__":
    main()
