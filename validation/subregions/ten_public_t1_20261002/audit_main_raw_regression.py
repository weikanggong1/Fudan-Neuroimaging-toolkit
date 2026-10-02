#!/usr/bin/env python3
"""Read-only real raw10 regression: pinned main f436 versus frozen ac692bb.

Sequential CPU audit only. Reads both native/four-HR outputs and public raw T1,
never official images; no fitting, GPU, interpolation or result correction.
Preprocessing arrays were not saved: compare SHA/geometry from the verified
unchanged observer, explicitly recording that limitation. Preserve every
planned subject and every difference. Full evidence stays on the server;
summary.json and TSVs provide compact publication evidence.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

BASE_COMMIT = "ac692bb4f7868a24ea4bd67180162e81726de9b4"
MAIN_COMMIT = "f436de588647a0de80735e4a98d53df5d88e502d"
ANALYZER_SHA = "ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc"
HELPER_SHA = "beef69670191f89ebf3a0d71395c274b54b7c61d02ed096abebebd7a5172f842"
BASE_RAW_GATE_SHA = "820774cb5af76224b427f7495d8bdcbfc75f87473e3d95d3b5018c6fdb476059"
IDS = [f"sub-{index:02d}" for index in range(1, 11)]
STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
MAP_KEYS = ("labels", *[f"highres/{name}" for name in STRUCTURES])
CONTEXT_PHASES = ("prepared_context", "raw_processing_context")
CONTEXT_ARRAYS = ("data", "coarse_segmentation", "cortical_parcellation", "wmparc_proxy", "brain_mask")
TERMINAL = {"completed", "completed_with_failures", "failed"}
MEMORY_LIMIT_MIB = 19073
TIMING_FIELDS = ("api_compute_seconds", "api_total_seconds", "output_save_seconds", "process_wall_seconds", "context_observer_seconds",
                 "input_wait_seconds", "gpu_budget_wait_seconds", "preflight_identity_seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def distribution(values, planned):
    defined = [float(value) for value in values if value is not None]
    if len(defined) > planned or not all(math.isfinite(value) for value in defined):
        raise ValueError("Invalid finite observation or planned denominator")
    return {"planned_subjects": planned, "n": len(defined), "missing_or_na": planned - len(defined),
            "mean": statistics.mean(defined) if defined else None,
            "median": statistics.median(defined) if defined else None,
            "std_sample": statistics.stdev(defined) if len(defined) > 1 else None,
            "std_ddof": 1, "min": min(defined) if defined else None,
            "max": max(defined) if defined else None}


def wait_json(path, args, deadline, predicate=lambda value: True):
    while True:
        try:
            value = read_json(path)
            if predicate(value):
                return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if not args.wait or time.monotonic() >= deadline:
            raise TimeoutError(f"Required real output not ready: {path}")
        time.sleep(args.poll_seconds)


def ready_run(queue, case_id):
    rows = [run for run in queue.get("runs", []) if run["case_id"] == case_id and run["component"] == "fnit_raw"]
    successes = [run for run in rows if run.get("state") == "completed" and run.get("status") == "success"
                 and run.get("exit_code") == 0 and run.get("phase") == "verified_full_run"]
    if len(successes) > 1:
        raise ValueError("More than one final raw result; do not select among outputs")
    if len(successes) == 1:
        return successes[0]
    if any(run.get("state") in {"failed", "blocked"} for run in rows) or queue.get("state") in TERMINAL:
        raise RuntimeError(f"No successful complete raw run for {case_id}; attempts={rows}")
    return None


def validate_manifest_pair(baseline, main, canonical, analyzer, root, regression):
    if [case["id"] for case in baseline["cases"]] != IDS or [case["id"] for case in main["cases"]] != IDS:
        raise ValueError("Keep the predetermined sub-01..10 order and identity")
    for key in ("dataset", "snapshot", "license", "dataset_doi", "canonical_label_metadata", "fnit_configuration"):
        if baseline[key] != main[key]:
            raise ValueError(f"Regression changes predetermined cohort/configuration: {key}")
    if main.get("reference_used_for_raw_fnit_fitting") or baseline.get("reference_used_for_raw_fnit_fitting"):
        raise ValueError("Raw fits must be independent of official references")
    for old, new in zip(baseline["cases"], main["cases"]):
        if old["raw_t1"] != new["raw_t1"] or old["official"] != new["official"] or old["development_seen"] != new["development_seen"]:
            raise ValueError("Regression input/reference/development identity changed")
        if new["development_seen"] != (new["id"] == "sub-01"):
            raise ValueError("Only sub-01 is development-seen")
        output = analyzer.fnit_output(new, "raw", regression).resolve()
        if not output.is_relative_to(regression) or output == analyzer.fnit_output(old, "raw", root).resolve():
            raise ValueError("Regression must use a separate output tree")
    if len(canonical) != 110 or baseline["fnit_configuration"] != {
            "structures": "all", "optimization": "fast", "threads": 4,
            "precision": "float32 with default TF32 and existing deterministic accumulations"}:
        raise ValueError("Use the frozen all-structure default configuration")


def queue_header(queue, manifest_identity, source, base, analyzer):
    if queue.get("manifest", {}).get("sha256") != manifest_identity["sha256"]:
        raise ValueError("Queue manifest differs from the audited manifest")
    if (any(queue.get("source", {}).get(key) != source["source_manifest"][key] for key in ("path", "bytes", "sha256"))
            or queue.get("source_base_commit") != source["manifest_metadata"]["base_commit"]):
        raise ValueError("Queue source differs from independently read source snapshot")
    if (queue.get("optimization") != "fast" or queue.get("threads_per_process") != 4
            or queue.get("own_memory_limit_mib") != MEMORY_LIMIT_MIB
            or queue.get("observer_changes_solver_options") is not False
            or queue.get("reference_used_for_fitting") is not False):
        raise ValueError("Queue changes defaults, solver/reference policy or memory cap")
    for key in ("run_driver", "observer", "queue_script"):
        analyzer.identity(queue[key], base)
    assets = {entry["path"]: entry for entry in queue["assets"]}
    if len(assets) != len(queue["assets"]) or not assets:
        raise ValueError("Need unique actual weights/atlas asset identities")
    return assets


def run_defaults(audited, queue, source, runtime, analyzer, base):
    record, run = audited["record"], audited["run"]
    context = record["context_identity"]
    if context is None or run["source_manifest_sha256"] != source["source_manifest"]["sha256"]:
        raise ValueError("Missing or wrong actual source observer")
    analyzer.identity(run["context_identity"], base)
    if record["context_identity_artifact"]["sha256"] != run["context_identity"]["sha256"]:
        raise ValueError("Observer report changed after runner verification")
    if (context.get("observer_sha256") != queue["observer"]["sha256"]
            or context.get("driver_sha256") != queue["run_driver"]["sha256"]
            or context.get("runtime_source_sha256_before") != runtime
            or context.get("runtime_source_sha256_after") != runtime
            or context.get("runtime_source_unchanged") is not True
            or context.get("solver_options_changed") is not False
            or context.get("reference_used_by_observer") is not False):
        raise ValueError("Observer/source/solver/default identity mismatch")
    if record["gpu_monitor"]["sampling_errors"] or run.get("gpu_memory_fraction") != .23:
        raise ValueError("GPU sampling incomplete or raw allocator fraction changed")
    if any(option in run["command"] for option in ("--aseg", "--wmparc", "--quick")):
        raise ValueError("Raw benchmark must use automatic preprocessing")
    shared = record["shared_preprocessing"]
    if shared.get("model_calls") != {"SynthSeg": 0, "SynthSegPlus": 1} or shared.get("wmparc_source") != "proxy":
        raise ValueError("Default all-structure automatic preprocessing changed")
    preparation = shared["intensity_preprocessing"]
    if preparation.get("threads") != 4 or preparation.get("intensity_dtype") != "float32" or preparation["fast_config"]["execution"] != "tensor":
        raise ValueError("Actual FAST default precision/configuration changed")
    for queue_key, key in (("compute_seconds", "api_compute_seconds"), ("api_total_seconds", "api_total_seconds"),
                           ("output_save_seconds", "output_save_seconds"), ("context_observer_seconds", "context_observer_seconds")):
        if not math.isclose(run[queue_key], record["timings"][key], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Saved API/observer timing differs from queue")
    return record


def compare_map(case_id, key, old, new):
    import nibabel as nib
    import numpy as np
    old_image, new_image = nib.load(old["path"]), nib.load(new["path"])
    before, after = np.asarray(old_image.dataobj), np.asarray(new_image.dataobj)
    same_shape = before.shape == after.shape
    same_affine = np.array_equal(old_image.affine, new_image.affine)
    same_dtype = before.dtype.str == after.dtype.str and old_image.get_data_dtype().str == new_image.get_data_dtype().str
    differences = int(np.count_nonzero(before != after)) if same_shape else None
    old_data_sha = hashlib.sha256(memoryview(np.ascontiguousarray(before)).cast("B")).hexdigest()
    new_data_sha = hashlib.sha256(memoryview(np.ascontiguousarray(after)).cast("B")).hexdigest()
    if identity(old["path"])["sha256"] != old["sha256"] or identity(new["path"])["sha256"] != new["sha256"]:
        raise ValueError("Saved label map changed during independent voxel read")
    return {"case_id": case_id, "map": key, "measurement_status": "compared",
            "baseline_file_sha256": old["sha256"], "main_file_sha256": new["sha256"],
            "baseline_array_sha256": old_data_sha, "main_array_sha256": new_data_sha,
            "baseline_shape": list(before.shape), "main_shape": list(after.shape),
            "baseline_array_dtype": before.dtype.str, "main_array_dtype": after.dtype.str,
            "baseline_header_dtype": old_image.get_data_dtype().str, "main_header_dtype": new_image.get_data_dtype().str,
            "baseline_affine": old_image.affine.tolist(), "main_affine": new_image.affine.tolist(),
            "shape_exact": same_shape, "affine_exact": same_affine, "dtype_exact": same_dtype,
            "voxel_difference_scope": "Corresponding array indices when shapes match; affine equality is separately required; no interpolation.",
            "different_voxels": differences, "all_values_exact": differences == 0 if same_shape else False,
            "exact": same_shape and same_affine and same_dtype and differences == 0 and old_data_sha == new_data_sha}


def compare_volumes(case_id, old, new, canonical):
    if set(old) != set(canonical) or set(new) != set(canonical):
        raise ValueError("Both APIs must expose exactly 110 actual volume entries")
    result = []
    for label in sorted(canonical, key=int):
        before, after = old[label], new[label]
        result.append({"case_id": case_id, "label": int(label), "name": canonical[label]["name"],
            "measurement_status": "compared", "baseline_soft_mm3": before["soft_volume_mm3"],
            "main_soft_mm3": after["soft_volume_mm3"], "soft_delta_mm3": after["soft_volume_mm3"] - before["soft_volume_mm3"],
            "baseline_hard_mm3": before["hard_volume_mm3"], "main_hard_mm3": after["hard_volume_mm3"],
            "hard_delta_mm3": after["hard_volume_mm3"] - before["hard_volume_mm3"],
            "entire_dictionary_exact": before == after, "baseline_entry_sha256": json_hash(before),
            "main_entry_sha256": json_hash(after), "exact": before == after and json_hash(before) == json_hash(after)})
    return result


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if key != "seconds" and not key.endswith("_seconds") and key != "peak_gpu_gib"}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def compare_contexts(case_id, old, new):
    rows, choices = [], []
    if len(old["contexts"]) != 2 or len(new["contexts"]) != 2:
        raise ValueError("Exactly one observation is required per preprocessing phase")
    old = {entry["phase"]: entry for entry in old["contexts"]}
    new = {entry["phase"]: entry for entry in new["contexts"]}
    if set(old) != set(CONTEXT_PHASES) or set(new) != set(CONTEXT_PHASES):
        raise ValueError("Need exactly prepared/raw-processing observer phases")
    for phase in CONTEXT_PHASES:
        for key in (*CONTEXT_ARRAYS, "image_geometry/affine_identity"):
            before = old[phase]["image_geometry"]["affine_identity"] if "/" in key else old[phase][key]
            after = new[phase]["image_geometry"]["affine_identity"] if "/" in key else new[phase][key]
            if before.get("status") != "observed" or after.get("status") != "observed":
                raise ValueError("Actual default preprocessing array not observed")
            names = ("shape", "dtype", "bytes", "hash_order", "sha256")
            exact = all(before[name] == after[name] for name in names)
            rows.append({"case_id": case_id, "phase": phase, "array": key, "measurement_status": "compared",
                         "baseline": before, "main": after, "exact": exact,
                         "strides_equal": before["original_strides_bytes"] == after["original_strides_bytes"]})
        for key in ("shape", "affine"):
            before, after = old[phase]["image_geometry"][key], new[phase]["image_geometry"][key]
            rows.append({"case_id": case_id, "phase": phase, "array": "image_geometry/" + key,
                         "measurement_status": "compared", "baseline": before, "main": after, "exact": before == after})
        before = without_timing(old[phase]["metadata_and_model_choices"])
        after = without_timing(new[phase]["metadata_and_model_choices"])
        choices.append({"phase": phase, "exact_without_recorded_timers": before == after, "baseline": before, "main": after})
    return rows, choices


def actual_recipe_timers(initialization):
    result = {}
    def visit(value, path=(), timing_children=False):
        if isinstance(value, dict):
            for key, item in value.items():
                pointer = (*path, key)
                if isinstance(item, (float, int)) and not isinstance(item, bool) and (key == "seconds" or key.endswith("_seconds") or timing_children):
                    if not math.isfinite(item) or item < 0:
                        raise ValueError("Invalid saved recipe timer")
                    result["/".join(pointer)] = float(item)
                else:
                    visit(item, pointer, key == "timing_seconds")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
    visit(initialization)
    return result


def gpu_observation(audited):
    samples = [json.loads(line) for line in Path(audited["record"]["gpu_monitor"]["path"]).read_text().splitlines() if line.strip()]
    clean = [sample for sample in samples if "sampling_error" not in sample]
    if identity(audited["record"]["gpu_monitor"]["path"])["sha256"] != audited["record"]["gpu_monitor"]["sha256"]:
        raise ValueError("GPU monitor changed during independent read")
    result = {"identity": audited["record"]["gpu_monitor"], "physical_gpu": audited["run"]["physical_gpu"],
              "own_pid": audited["run"]["pid"], "launch_unix": audited["run"]["started_unix"],
              "own_sampled_peak_mib": max(sample["own_memory_mib"] for sample in clean),
              "own_limit_mib": MEMORY_LIMIT_MIB, "sampling_errors": len(samples) - len(clean),
              "torch_allocated_peak_gib": audited["report"].get("peak_gpu_gib"),
              "scope": "Actual own-PID periodic samples; shared whole-GPU occupancy is separate; not an instantaneous CUDA-process maximum.",
              "samples": samples}
    for key in ("memory_used_mib", "memory_free_mib", "utilization_percent"):
        result[key] = distribution([sample.get(key) for sample in clean], len(clean))
    result["other_pid_memory_mib"] = distribution([sum(row["used_memory_mib"] for row in sample.get("other_processes", [])) for sample in clean], len(clean))
    return result


def summarized_cases(result, new_only):
    selected = [case for case in result["cases"] if not new_only or not case["development_seen"]]
    planned = 9 if new_only else 10
    if len(selected) != planned:
        raise ValueError("Keep all ten and predeclared new nine denominators")
    rows = {}
    for version in ("baseline", "main"):
        rows[version] = {key: distribution([case.get(version, {}).get("timings", {}).get(key) for case in selected], planned) for key in TIMING_FIELDS}
        rows[version]["own_sampled_peak_mib"] = distribution([case.get(version, {}).get("gpu", {}).get("own_sampled_peak_mib") for case in selected], planned)
    rows["paired_main_over_baseline"] = {}
    for key in TIMING_FIELDS:
        values = []
        for case in selected:
            before = case.get("baseline", {}).get("timings", {}).get(key)
            after = case.get("main", {}).get("timings", {}).get(key)
            values.append(after / before if before is not None and before > 0 and after is not None else None)
        rows["paired_main_over_baseline"][key] = distribution(values, planned)
    recipe_keys = sorted({key for case in selected for version in ("baseline", "main") for key in case.get(version, {}).get("recipe_timings", {})})
    rows["recipe_timings"] = {key: {version: distribution([case.get(version, {}).get("recipe_timings", {}).get(key) for case in selected], planned)
                                    for version in ("baseline", "main")} for key in recipe_keys}
    return {"planned_subjects": planned, "case_ids": [case["case_id"] for case in selected],
            "compared_subjects": sum(case["state"] in {"zero_voxel_equivalence_passed", "completed_with_differences"} for case in selected),
            "failed_or_unavailable_subjects": [case["case_id"] for case in selected if case["state"] not in {"zero_voxel_equivalence_passed", "completed_with_differences"}],
            "measurements": rows, "interpretation": "One run per subject/version; between-subject sample SD, not within-method random variability. Recorded nested recipe timers are not additive."}


def write_tsv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def emit(output, result):
    save_json(output / "expanded.json", result)
    cases = []
    for full in result["cases"]:
        case = {key: full.get(key) for key in ("case_id", "development_seen", "state", "error", "input", "counts", "numeric_gates", "preprocessing_choices", "queue_run_sha256")}
        case["main_attempts"] = [{key: run.get(key) for key in ("case_id", "state", "status", "phase", "exit_code", "failure", "process_wall_seconds",
                                                            "started_unix", "finished_unix", "sampled_peak_own_memory_mib", "physical_gpu")}
                                 for run in full.get("main_attempts", [])]
        for version in ("baseline", "main"):
            if version in full:
                case[version] = {key: value for key, value in full[version].items() if key != "audited_record"}
                if "gpu" in case[version]:
                    case[version]["gpu"] = {key: value for key, value in case[version]["gpu"].items() if key != "samples"}
        cases.append(case)
    compact = {key: value for key, value in result.items() if key not in ("cases", "source_audits")}
    compact.update(cases=cases, source_audits={name: {key: value for key, value in entry.items() if key != "manifest_metadata"}
                                             for name, entry in result.get("source_audits", {}).items()},
                   map_pairs=[row for case in result["cases"] for row in case["map_pairs"]],
                   context_pairs=[row for case in result["cases"] for row in case["context_pairs"]])
    compact["expanded_evidence"] = identity(output / "expanded.json")
    save_json(output / "summary.json", compact)
    case_rows = []
    for case in cases:
        row = {key: case[key] for key in ("case_id", "development_seen", "state", "error")}
        row.update(case.get("counts") or {})
        for version in ("baseline", "main"):
            for key, value in case.get(version, {}).get("timings", {}).items():
                if key in TIMING_FIELDS:
                    row[f"{version}_{key}"] = value
            row[f"{version}_own_sampled_peak_mib"] = case.get(version, {}).get("gpu", {}).get("own_sampled_peak_mib")
        case_rows.append(row)
    write_tsv(output / "case.tsv", case_rows)
    write_tsv(output / "map_pairs.tsv", compact["map_pairs"])
    write_tsv(output / "volume_pairs.tsv", [row for case in result["cases"] for row in case["volume_pairs"]])
    write_tsv(output / "context_pairs.tsv", compact["context_pairs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Original frozen TASK directory")
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not 1 <= args.poll_seconds <= 60:
        raise ValueError("Positive timeout and 1–60 second polling required")
    root, regression = args.root.resolve(), args.regression_root.resolve()
    if root == regression:
        raise ValueError("Keep main regression separate from frozen TASK")
    output = args.output_dir or regression / "raw_regression_audit"
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("expanded.json", "summary.json", "case.tsv", "map_pairs.tsv", "volume_pairs.tsv", "context_pairs.tsv")):
        raise ValueError("Preserve previous audit; choose a fresh output directory")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    deadline = time.monotonic() + args.timeout_seconds
    analyzer_path = root / "analyze_cohort.py"
    helper_path = root.parent / "reproducibility_20261002/analyze_repeatability.py"
    if sha256(analyzer_path) != ANALYZER_SHA or sha256(helper_path) != HELPER_SHA:
        raise ValueError("Approved frozen analyzer/helper must remain unchanged")
    spec = importlib.util.spec_from_file_location("frozen_analyzer_main_regression_readonly", analyzer_path)
    analyzer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = analyzer; spec.loader.exec_module(analyzer)
    baseline_manifest_path, main_manifest_path = root / "cohort_manifest.json", regression / "cohort_manifest.json"
    baseline = read_json(baseline_manifest_path); main_manifest = wait_json(main_manifest_path, args, deadline)
    canonical = analyzer.canonical_labels(baseline, root)
    validate_manifest_pair(baseline, main_manifest, canonical, analyzer, root, regression)
    runtimes, sources = {}, {}
    for version, manifest, directory, commit in (("baseline", baseline, root, BASE_COMMIT), ("main", main_manifest, regression, MAIN_COMMIT)):
        if not isinstance(manifest["source"], dict) or not manifest["source"].get("manifest_sha256"):
            raise ValueError("Each separate source snapshot must be pinned by manifest SHA")
        runtimes[version], sources[version] = analyzer.audit_source(manifest, directory)
        if sources[version]["manifest_metadata"]["base_commit"] != commit:
            raise ValueError("Actual audited source is not the prescribed commit")
    base_queue_path, main_queue_path = root / "fnit_queue.json", regression / "fnit_raw_queue.json"
    base_queue, main_queue = read_json(base_queue_path), wait_json(main_queue_path, args, deadline)
    if base_queue.get("state") != "completed" or main_queue.get("modes") != ["raw"]:
        raise ValueError("Need complete baseline and separate raw-only main queue")
    manifest_ids = {"baseline": identity(baseline_manifest_path), "main": identity(main_manifest_path)}
    baseline_assets = queue_header(base_queue, manifest_ids["baseline"], sources["baseline"], root, analyzer)
    main_assets = queue_header(main_queue, manifest_ids["main"], sources["main"], regression, analyzer)
    if baseline_assets != main_assets or base_queue["observer"]["sha256"] != main_queue["observer"]["sha256"] or base_queue["run_driver"]["sha256"] != main_queue["run_driver"]["sha256"]:
        raise ValueError("Weight/atlas assets, observer or driver differ")
    for asset in baseline_assets.values():
        analyzer.identity(asset, root)
    base_gate_path = root / "raw_all_live_audit.json"
    if sha256(base_gate_path) != BASE_RAW_GATE_SHA:
        raise ValueError("Previously independent baseline raw10 gate changed")
    baseline_gate = read_json(base_gate_path)
    gates = {entry["case_id"]: entry for entry in baseline_gate["cases"]}
    if baseline_gate.get("state") != "passed" or set(gates) != set(IDS):
        raise ValueError("Baseline gate does not cover all ten subjects")
    for actual, declared in ((manifest_ids["baseline"], baseline_gate["cohort_manifest"]),
                             (identity(base_queue_path), baseline_gate["completed_raw_queue"]),
                             (sources["baseline"]["source_manifest"], baseline_gate["source_audit"]["source_manifest"])):
        if any(actual[key] != declared[key] for key in ("bytes", "sha256")):
            raise ValueError("Baseline manifest/queue/source no longer matches the independent passed gate")
    if json_hash(runtimes["baseline"]) != baseline_gate["runtime_module_map_sha256"]:
        raise ValueError("Baseline runtime differs from original independent observation")
    cases = []
    for case in baseline["cases"]:
        cases.append({"case_id": case["id"], "development_seen": case["development_seen"], "state": "pending", "error": None,
                      "input": case["raw_t1"], "map_pairs": [{"case_id": case["id"], "map": key, "measurement_status": "not_audited", "exact": None} for key in MAP_KEYS],
                      "volume_pairs": [{"case_id": case["id"], "label": int(label), "name": canonical[label]["name"], "measurement_status": "not_audited", "exact": None} for label in sorted(canonical, key=int)],
                      "context_pairs": [{"case_id": case["id"], "phase": phase, "array": key, "measurement_status": "not_audited", "exact": None}
                                        for phase in CONTEXT_PHASES for key in (*CONTEXT_ARRAYS, "image_geometry/affine_identity", "image_geometry/shape", "image_geometry/affine")]})
    result = {"schema_version": 1, "state": "auditing", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "cpu_only": True, "no_fitting": True, "no_official_images_read": True, "no_result_corrections": True,
              "raw_output_equivalence_only": True, "official_accuracy_compared": False,
              "primary_full_cohort_benchmark": False, "final_main_regression_outcomes_ready": False,
              "baseline_commit": BASE_COMMIT, "main_commit": MAIN_COMMIT,
              "audit_script": identity(__file__), "frozen_analyzer": identity(analyzer_path), "comparison_helper": identity(helper_path),
              "source_audits": sources, "runtime_map_sha256": {version: json_hash(runtime) for version, runtime in runtimes.items()},
              "manifests": manifest_ids, "baseline_raw_gate": identity(base_gate_path), "baseline_queue": identity(base_queue_path),
              "fixed_asset_inventory_sha256": json_hash(sorted(baseline_assets.values(), key=lambda item: item["path"])),
              "verified_asset_files": len(baseline_assets),
              "planned_subjects": 10, "planned_new_subjects": 9, "planned_map_pairs": 50, "planned_volume_pairs": 1100, "planned_context_pairs": 160,
              "cpu_resource_limit": {"threads": 2, "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
              "preprocessing_comparison_scope": "Verified observer C-contiguous array SHA+shape+dtype and geometry; arrays_saved=false, no new direct preprocessing voxel read. Physical strides recorded separately.",
              "default_precision_validation_scope": "Pinned source/CLI/default configurations and FP32 preprocessing outputs; the unchanged context observer did not record network forward autocast/TF32 metadata.",
              "equivalence_rule": "All ten successful audited runs; all fifty maps have identical shape/affine/dtype and zero voxel differences; all1100 volume dictionaries equal; every preproc observer array/geometry and model-choice/default config equal. No approximate tolerance used for equivalence.",
              "cases": cases}
    emit(output, result)
    begun = time.monotonic()
    for index, (old_case, new_case) in enumerate(zip(baseline["cases"], main_manifest["cases"])):
        case = cases[index]
        try:
            while True:
                main_queue = wait_json(main_queue_path, args, deadline)
                queue_header(main_queue, manifest_ids["main"], sources["main"], regression, analyzer)
                case["main_attempts"] = [run for run in main_queue.get("runs", []) if run["case_id"] == case["case_id"] and run["component"] == "fnit_raw"]
                if ready_run(main_queue, case["case_id"]) is not None:
                    break
                case["state"] = "waiting_for_verified_main_run"; emit(output, result)
                if not args.wait or time.monotonic() >= deadline:
                    raise TimeoutError("Main raw case has not completed")
                time.sleep(args.poll_seconds)
            audited = {}
            for version, specification, queue, directory in (("baseline", old_case, base_queue, root), ("main", new_case, main_queue, regression)):
                audit = analyzer.audit_fnit(specification, "raw", queue, directory, canonical, runtimes[version])
                record = audit["record"]
                audited[version] = audit
                case[version] = {"audited_record": record, "input_sha256": record["input_sha256"],
                    "source_manifest_sha256": sources[version]["source_manifest"]["sha256"],
                    "api_report": record["api_report"], "driver_report": record["report"], "observer": record["context_identity_artifact"],
                    "configuration": record["configuration"], "timings": {**record["timings"], **{key: audit["run"].get(key) for key in ("input_wait_seconds", "gpu_budget_wait_seconds", "preflight_identity_seconds")}},
                    "recipe_timings": actual_recipe_timers(audit["api"]["initialization"]), "gpu": gpu_observation(audit),
                    "fit_min_jacobians": record["fit_min_jacobians"], "numeric_gates": record["numeric_gates"]}
                run_defaults(audit, queue, sources[version], runtimes[version], analyzer, directory)
            if audited["baseline"]["record"]["configuration"] != audited["main"]["record"]["configuration"]:
                raise ValueError("Actual optimization/torch/CUDA configurations differ")
            case["numeric_gates"] = {version: audited[version]["record"]["numeric_gates"] for version in ("baseline", "main")}
            for key in MAP_KEYS:
                if audited["baseline"]["record"]["arrays"][key]["sha256"] != gates[case["case_id"]]["native_and_hr_label_arrays"][key]["sha256"]:
                    raise ValueError("Baseline saved array differs from previously independent gate")
            for map_index, key in enumerate(MAP_KEYS):
                case["map_pairs"][map_index] = compare_map(case["case_id"], key, audited["baseline"]["record"]["arrays"][key], audited["main"]["record"]["arrays"][key])
            case["volume_pairs"] = compare_volumes(case["case_id"], audited["baseline"]["api"]["volumes"], audited["main"]["api"]["volumes"], canonical)
            case["context_pairs"], case["preprocessing_choices"] = compare_contexts(case["case_id"], audited["baseline"]["record"]["context_identity"], audited["main"]["record"]["context_identity"])
            case["queue_run_sha256"] = {version: json_hash(audited[version]["run"]) for version in ("baseline", "main")}
            case["counts"] = {"compared_maps": 5, "different_maps": sum(not row["exact"] for row in case["map_pairs"]),
                              "different_voxels": sum(row["different_voxels"] for row in case["map_pairs"]) if all(row["different_voxels"] is not None for row in case["map_pairs"]) else None,
                              "sum_defined_same_index_voxel_differences": sum(row["different_voxels"] or 0 for row in case["map_pairs"]),
                              "maps_without_comparable_voxel_count": sum(row["different_voxels"] is None for row in case["map_pairs"]),
                              "compared_volume_entries": 110, "different_volume_entries": sum(not row["exact"] for row in case["volume_pairs"]),
                              "compared_context_entries": len(case["context_pairs"]), "different_context_entries": sum(not row["exact"] for row in case["context_pairs"])}
            exact = all(row["exact"] for key in ("map_pairs", "volume_pairs", "context_pairs") for row in case[key]) and all(row["exact_without_recorded_timers"] for row in case["preprocessing_choices"])
            case["state"] = "zero_voxel_equivalence_passed" if exact else "completed_with_differences"
        except Exception as error:
            case.update(state="failed_or_unavailable", error=f"{type(error).__name__}: {error}")
        emit(output, result)
        print(json.dumps({"case_id": case["case_id"], "state": case["state"], "counts": case.get("counts"), "error": case["error"]}), flush=True)
    final_error = None
    try:
        main_queue = wait_json(main_queue_path, args, deadline, lambda queue: queue.get("state") in TERMINAL)
        queue_header(main_queue, manifest_ids["main"], sources["main"], regression, analyzer)
        if main_queue.get("state") != "completed":
            raise ValueError("Main runner queue failed; do not claim complete validated equivalence")
        for version, manifest, directory in (("baseline", baseline, root), ("main", main_manifest, regression)):
            runtime_now, source_now = analyzer.audit_source(manifest, directory)
            if runtime_now != runtimes[version] or source_now != sources[version]:
                raise ValueError("Source changed while independently auditing")
        for asset in baseline_assets.values():
            analyzer.identity(asset, root)
        if identity(base_queue_path) != result["baseline_queue"] or identity(baseline_manifest_path) != manifest_ids["baseline"] or identity(main_manifest_path) != manifest_ids["main"]:
            raise ValueError("Baseline queue/cohort manifest changed during audit")
        for case in cases:
            if "queue_run_sha256" in case and json_hash(analyzer.successful_component(main_queue, case["case_id"], "fnit_raw")) != case["queue_run_sha256"]["main"]:
                raise ValueError("Verified main run record changed after comparison")
    except Exception as error:
        final_error = f"{type(error).__name__}: {error}"
    completed = all(case["state"] in {"zero_voxel_equivalence_passed", "completed_with_differences"} for case in cases)
    exact = completed and all(case["state"] == "zero_voxel_equivalence_passed" for case in cases) and final_error is None
    result.update(state="zero_voxel_equivalence_passed" if exact else "completed_with_differences" if completed and final_error is None else "completed_with_failures",
                  finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_audit_wall_seconds=time.monotonic() - begun,
                  final_queue=identity(main_queue_path), final_source_asset_metadata_error=final_error,
                  final_queue_state=main_queue.get("state"), zero_voxel_equivalence_passed=exact,
                  final_main_regression_outcomes_ready=main_queue.get("state") in TERMINAL,
                  cohort_all=summarized_cases(result, False), cohort_new_subjects=summarized_cases(result, True))
    result["summary_counts"] = {"audited_subjects": sum(case["state"] in {"zero_voxel_equivalence_passed", "completed_with_differences"} for case in cases),
        "exact_subjects": sum(case["state"] == "zero_voxel_equivalence_passed" for case in cases),
        "fully_compared_maps": sum(row["measurement_status"] == "compared" for case in cases for row in case["map_pairs"]),
        "fully_compared_volume_entries": sum(row["measurement_status"] == "compared" for case in cases for row in case["volume_pairs"]),
        "fully_compared_context_entries": sum(row["measurement_status"] == "compared" for case in cases for row in case["context_pairs"]),
        "different_maps": sum(row["exact"] is False for case in cases for row in case["map_pairs"]),
        "different_volume_entries": sum(row["exact"] is False for case in cases for row in case["volume_pairs"])}
    if exact and (result["summary_counts"]["fully_compared_maps"] != 50 or result["summary_counts"]["fully_compared_volume_entries"] != 1100
                  or result["summary_counts"]["fully_compared_context_entries"] != 160):
        raise ValueError("Incomplete planned output cannot pass equivalence")
    emit(output, result)
    print(json.dumps({"state": result["state"], "summary": identity(output / "summary.json"), "counts": result["summary_counts"]}), flush=True)
    if result["state"] == "completed_with_failures":
        raise RuntimeError("Main regression audit has failures; all planned outcomes and differences retained")


if __name__ == "__main__":
    main()
