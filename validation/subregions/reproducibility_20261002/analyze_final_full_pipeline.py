"""CPU-only final full-pipeline repeats: own noise and cross accuracy separately.

Six successful runs must all use structures=all, one reviewed immutable source,
the same actual inputs/configuration/model choices, and the same GPU accounting.
Runtime file counts come from the reviewed source manifest, not a fixed number.
No pipeline fitting, GPU work or original-software invocation is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import time
import traceback

import nibabel as nib
import numpy as np

from analyze_repeatability import _grid, audit_group
from audit_production_numeric import finite_json, valid_geometry
from build_repeatability_manifest import read_json, sha256, verify_fingerprints, verify_official_metadata
from finalize_reproducibility import verify_official_audit, write_json


def audit_snapshot(source: Path, expected: Path) -> tuple[dict, dict]:
    manifest_path = source / "source_manifest.json"
    if manifest_path.read_bytes() != expected.read_bytes():
        raise ValueError("Final snapshot manifest differs from the reviewed local manifest bytes")
    manifest = read_json(manifest_path)
    entries = manifest["files"]
    if not entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Empty source manifest or duplicate file paths")
    if any(Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts for entry in entries):
        raise ValueError("Source paths must be relative and remain inside the snapshot")
    verify_fingerprints([{**entry, "path": str(source / entry["path"])} for entry in entries], len(entries))
    runtime = {entry["path"].removeprefix("src/fnit/"): entry["sha256"] for entry in entries
               if entry["path"].startswith("src/fnit/") and entry["path"].endswith(".py")}
    if not runtime:
        raise ValueError("No FNIT runtime Python files in source snapshot")
    actual_runtime = {str(path.relative_to(source / "src/fnit"))
                      for path in (source / "src/fnit").rglob("*.py")}
    if actual_runtime != set(runtime):
        raise ValueError("Snapshot manifest must cover every current runtime Python module")
    return runtime, {"source": str(source), "source_manifest_sha256": sha256(manifest_path),
                     "reviewed_manifest_sha256": sha256(expected), "verified_files": len(entries),
                     "verified_runtime_python_files": len(runtime), "manifest_metadata": {k: v for k, v in manifest.items() if k != "files"}}


def option(command, name):
    if command.count(name) != 1:
        raise ValueError(f"Missing/duplicate final command option: {name}")
    return command[command.index(name) + 1]


def preprocessing_configuration(api: dict) -> dict:
    shared = api["initialization"]["shared_preprocessing"]
    result = {key: shared.get(key) for key in ("coarse_source", "cortical_parcellation_source", "wmparc_source", "model_calls")}
    intensity = shared.get("intensity_preprocessing", {})
    # Numerical tissue means, cropped arrays and convergence trajectories are outputs;
    # differences there are measured as end-to-end noise, not rejected as settings.
    result["intensity_configuration"] = {key: intensity.get(key) for key in
        ("method", "mask_rule", "wm_rule", "minimum_wm_samples", "intensity_dtype", "threads", "fast_config", "grid_rule")}
    return result


def verify_identity(record: dict, label: str) -> None:
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Recorded {label} identity changed: {path}")


def full_successes(queue: dict) -> list[dict]:
    successful = [run for run in queue["runs"] if run.get("exit_code") == 0]
    expected = {(mode, repeat, "all") for mode in ("stage", "raw") for repeat in (1, 2, 3)}
    if (len(successful) != 6 or
            {(run["mode"], run["repeat"], run["structure"]) for run in successful} != expected or
            any(run.get("phase") != "verified_full_run" for run in successful)):
        raise ValueError("Need stage/raw each three verified all-structure runs; partial scopes cannot enter final repeats")
    return successful


def audit_runs(args, queue, runtime, snapshot, manifest):
    if queue["state"] != "completed" or queue["source"] != str(args.source):
        raise ValueError("Final queue is incomplete or uses another snapshot")
    if (queue["source_manifest"]["sha256"] != snapshot["source_manifest_sha256"] or
            queue["verified_runtime_python_files"] != len(runtime)):
        raise ValueError("Final queue source fingerprint or dynamic runtime count differs")
    for key in ("source_manifest", "run_driver", "queue_script", "context_observer"):
        verify_identity(queue[key], key)
    if queue["queue_script"]["sha256"] != args.driver_sha256 or Path(queue["queue_script"]["path"]) != args.gpu_driver:
        raise ValueError("Actual launched final runner differs from reviewed identity")
    if queue["context_observer"]["sha256"] != args.observer_sha256:
        raise ValueError("Actual context observer differs from reviewed identity")
    expected_driver = args.source / "validation/subregions/run_unified.py"
    if Path(queue["run_driver"]["path"]) != expected_driver:
        raise ValueError("Declared production driver lies outside reviewed snapshot")
    if (queue["threads_per_process"] != 4 or queue["repeats_per_mode"] != 3 or
            queue["planned_successful_cases"] != 6 or queue["successful_full_runs"] != 6 or
            queue["reference_used_for_fitting"] or queue["observer_changes_solver_options"] or
            queue["global_deterministic_algorithms_changed"] or queue["policy_wrapper"] is not None or
            queue["cublas_workspace_config"] is not None):
        raise ValueError("Final default production protocol differs from reviewed full-run policy")
    fixed = queue["fixed_inputs_references_assets"]
    for name, record in fixed.items():
        if name != record["path"]:
            raise ValueError("Fixed-file identity key/path mismatch")
        verify_identity(record, "fixed input/reference/asset")
    gpus = queue["physical_gpu_by_mode"]
    if set(gpus) != {"stage", "raw"} or any(not gpu["uuid"] for gpu in gpus.values()):
        raise ValueError("Each mode must have a declared physical GPU UUID")
    if queue["parallel_modes"] and gpus["stage"]["uuid"] == gpus["raw"]["uuid"]:
        raise ValueError("Parallel modes cannot share the same declared physical GPU")
    successful = full_successes(queue)
    inputs = {mode: next(group for group in manifest["groups"] if group["space"] == f"{mode}_native")["fnit"][0]["provenance"]["input_sha256"] for mode in ("stage", "raw")}
    outputs, records, configs = {}, [], {}
    first_labels = None
    for run in successful:
        mode, repeat = run["mode"], run["repeat"]
        output = Path(run["output"])
        if not output.resolve().is_relative_to(args.final_root.resolve()):
            raise ValueError("Final output is outside the explicitly selected final root")
        api_path, report_path = output / "api_report.json", output / "report.json"
        api, report = read_json(api_path), read_json(report_path)
        finite_json(api, "api")
        finite_json(report, "report")
        for key, path in (("report", report_path), ("api_report", api_path),
                          ("context_identity", output / "context_identity.json")):
            if Path(run[key]["path"]) != path:
                raise ValueError("Run recorded artifact path differs from actual output")
            verify_identity(run[key], key)
        command = run["command"]
        if (Path(command[1]) != Path(queue["context_observer"]["path"]) or
                Path(option(command, "--run-driver")) != expected_driver or
                option(command, "--structures") != "all" or "--quick" in command):
            raise ValueError("Actual command differs from reviewed full-pipeline observer/driver")
        fraction = .23 if mode == "raw" else .14
        if (float(option(command, "--gpu-memory-fraction")) != fraction or
                run["gpu_memory_fraction"] != fraction or option(command, "--device") != "cuda:0" or
                run["threads"] != 4 or run["physical_gpu"] != gpus[mode] or
                run["cuda_visible_devices"] != gpus[mode]["uuid"]):
            raise ValueError("Final GPU/allocator/thread settings differ")
        if (option(command, "--output-dir") != str(output) or
                option(command, "--optimization") != report["optimization"] or
                report["optimization"] != queue["optimization"]):
            raise ValueError("Final command/report configuration differs")
        if (report["source_sha256"] != runtime or report["input_sha256"] != inputs[mode] or
                run["input_sha256"] != inputs[mode] or
                run["source_manifest_sha256"] != snapshot["source_manifest_sha256"]):
            raise ValueError("Final runtime map or original inputs differ from reviewed data")
        for path, expected_sha in report["input_sha256"].items():
            if path not in fixed or fixed[path]["sha256"] != expected_sha or sha256(Path(path)) != expected_sha:
                raise ValueError(f"Actual final input changed: {path}")
        if len(api["labels"]) != 110 or len(api["volumes"]) != 110:
            raise ValueError("Final full pipeline must report all 110 labels/volumes")
        if first_labels is not None and api["labels"] != first_labels:
            raise ValueError("Final label definitions differ among repeats")
        first_labels = api["labels"]
        if api["structures"] != ["brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right"]:
            raise ValueError("Final actual fitting scope/order differs")
        jacobians = api["fit_min_jacobians"]
        if set(jacobians) != set(api["structures"]) or any(not math.isfinite(v) or v <= 0 for v in jacobians.values()):
            raise ValueError("Every final fitted structure must have a finite positive Jacobian")
        if jacobians != report["fit_min_jacobians"]:
            raise ValueError("Final API/report Jacobian metadata differs")
        for label, values in api["volumes"].items():
            if any(not math.isfinite(values[k]) or values[k] < 0 for k in ("hard_volume_mm3", "soft_volume_mm3")):
                raise ValueError(f"Invalid final volume: {label}")
        input_image, native_image = nib.load(report["input"]), nib.load(api["files"]["labels"])
        input_grid, native_grid = valid_geometry(input_image, report["input"]), valid_geometry(native_image, api["files"]["labels"])
        if input_grid["shape"] != native_grid["shape"] or any(abs(x-y) > 1e-5 for row_a,row_b in zip(input_grid["affine"],native_grid["affine"]) for x,y in zip(row_a,row_b)):
            raise ValueError("Final native geometry does not match actual original T1")
        for key in ["labels", *(f"highres/{structure}" for structure in api["structures"])]:
            image = nib.load(api["files"][key])
            valid_geometry(image, api["files"][key])
            values = np.asarray(image.dataobj)
            if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
                raise ValueError("Original final native/HR label image has nonfinite or noninteger values")
            del values
        context = read_json(output / "context_identity.json")
        finite_json(context, "context_identity")
        phases = [item["phase"] for item in context["contexts"]]
        expected_phases = ["prepared_context"] + (["raw_processing_context"] if mode == "raw" else [])
        if (phases != expected_phases or context["solver_options_changed"] or
                context["reference_used_by_observer"] or context["arrays_saved"] or
                context["observer_sha256"] != args.observer_sha256 or
                context["driver_sha256"] != queue["run_driver"]["sha256"] or
                Path(context["runtime_source_root"]) != args.source / "src/fnit" or
                context["input_arguments"] != command[4:] or not context["runtime_source_unchanged"] or
                context["runtime_source_sha256_before"] != runtime or context["runtime_source_sha256_after"] != runtime):
            raise ValueError("Final context observation does not match actual source/configuration/phases")
        if context["observer_seconds"] != run["context_observer_seconds"]:
            raise ValueError("Observer timing record differs")
        for item in context["contexts"]:
            for name in ("data", "coarse_segmentation", "cortical_parcellation", "wmparc_proxy", "brain_mask"):
                array = item[name]
                if array["status"] == "observed":
                    if (len(array["sha256"]) != 64 or array["bytes"] <= 0 or
                            len(array["shape"]) != 3 or any(size <= 0 for size in array["shape"])):
                        raise ValueError("Incomplete observed preparation-array identity")
                elif array["status"] != "not_available" or array["sha256"] is not None:
                    raise ValueError("Invalid absent preparation-array identity")
        config = {key: report[key] for key in ("optimization", "torch_version", "cuda_version")}
        config.update(preprocessing=preprocessing_configuration(api), seed=run.get("seed"))
        if mode in configs and config != configs[mode]:
            raise ValueError("Final repeat scope/model choices/settings/recorded seed differ")
        configs[mode] = config
        monitor_path = args.queue.parent / f"{output.name}_gpu_load.jsonl"
        samples = [json.loads(line) for line in monitor_path.read_text().splitlines() if line.strip()]
        clean = [sample for sample in samples if "sampling_error" not in sample]
        if not clean:
            raise ValueError("No successfully recorded own-process GPU samples")
        for sample in clean:
            own = max([entry["used_memory_mib"] for entry in sample["processes"] if entry["pid"] == run["pid"]] or [0])
            if sample["physical_gpu"] != gpus[mode] or own != sample["own_memory_mib"]:
                raise ValueError("GPU UUID or PID-attributed memory accounting differs")
        peak = max(sample["own_memory_mib"] for sample in clean)
        declared = run["sampled_peak_own_memory_mib"]
        if run.get("memory_limit_exceeded") or run["own_memory_limit_mib"] != 19073 or peak != declared or not 0 < peak <= 19073:
            raise ValueError("Final sampled own GPU peak violates or disagrees with declared accounting")
        start, finish, preflight_start = run["started_unix"], run["finished_unix"], run["preflight_started_unix"]
        process_wall = run["process_wall_seconds"]
        if (not preflight_start <= start < finish or not math.isfinite(process_wall) or process_wall <= 0 or
                abs((finish-start)-process_wall) > .1 or run["preflight_identity_seconds"] < 0 or run["gpu_budget_wait_seconds"] < 0):
            raise ValueError("Invalid Popen-to-wait-thread process timing or separate preflight timing")
        if (run["compute_seconds_including_context_observer"] != report["wall_seconds"] or
                report["wall_seconds"] != api["timings"]["compute_seconds"] or
                run["api_total_seconds"] != report["api_total_seconds"] or
                run["output_save_seconds"] != report["output_save_seconds"]):
            raise ValueError("Final compute/API/save timings disagree")
        record = {"mode": mode, "repeat": repeat, "structure": "all", "output": str(output),
            "command": command, "pid": run["pid"], "api_report_sha256": sha256(api_path), "report_sha256": sha256(report_path),
            "input_sha256": report["input_sha256"], "source_sha256": runtime, "configuration": config,
            "seed_recorded": config["seed"] is not None, "seed": config["seed"],
            "seed_scope": "Configured seed, if recorded; otherwise identical default settings. Repeated stability is not used to infer a seed.",
            "native_geometry_matches_original_input": True, "input_geometry": input_grid,
            "fit_min_jacobians": jacobians, "monitor_sha256": sha256(monitor_path), "monitor_samples": len(samples),
            "monitor_sampling_errors": len(samples)-len(clean), "physical_gpu": gpus[mode],
            "sampled_peak_own_memory_mib": peak, "own_process_limit_mib": 19073, "allocator_fraction": fraction,
            "queue_elapsed_with_gpu_preflight_seconds": finish-preflight_start,
            "preflight_identity_seconds": run["preflight_identity_seconds"], "gpu_budget_wait_seconds": run["gpu_budget_wait_seconds"],
            "process_wall_seconds": process_wall,
            "process_wall_scope": "Monotonic immediately before Popen to independent process.wait() watcher exit; includes imports, observer source hashes, API computation/saving, validation scoring and wrapper finalization; excludes parent source/input preflight and GPU wait.",
            "api_compute_seconds": api["timings"]["compute_seconds"], "api_total_seconds": report["api_total_seconds"],
            "output_save_seconds": report["output_save_seconds"], "pytorch_peak_gpu_gib": report["peak_gpu_gib"],
            "context_observer_seconds": context["observer_seconds"],
            "context_observer_scope": "Array identity and in-hook JSON serialization/writes; initial/final source hashing and final wrapper save are additional process overhead, not included in this subtotal. Do not infer observer-free production time by subtraction.",
            "context_identity": context, "context_identity_sha256": sha256(output / "context_identity.json"),
            "shared_preprocessing": api["initialization"]["shared_preprocessing"]}
        outputs[(mode,repeat)] = {"api": api, "report": report, "record": record}
        records.append(record)
    return outputs, records

def separate_noise_and_accuracy(group: dict):
    """Official-repeat noise bounds apply to own repeats, not cross-method Dice."""
    for region in group["regions"]:
        own = region["fnit_repeat_dice"]["min"]
        official = region["official_repeat_dice"]["min"]
        own_changed = region["fnit_repeat_different_voxels"]["max"]
        official_changed = region["official_repeat_different_voxels"]["max"]
        if all(value == 0 for value in region["fnit_voxels"]):
            repeat_status = "stable_absent_hard_label" if own_changed == 0 else "undefined_hard_dice"
        elif own is None or official is None:
            repeat_status = "undefined_official_or_fnit_hard_dice"
        elif own >= official - 1e-12 and own_changed <= official_changed:
            repeat_status = "within_official_observed_repeat_noise"
        else:
            repeat_status = "outside_official_observed_repeat_noise"
        official_absent = all(value == 0 for value in region["official_voxels"])
        fnit_absent = all(value == 0 for value in region["fnit_voxels"])
        cross = region["cross_method_dice"]["min"]
        if official_absent and fnit_absent:
            accuracy_status = "both_empty_hard_label"
        elif fnit_absent:
            accuracy_status = "official_present_fnit_hard_label_absent"
        elif official_absent:
            accuracy_status = "official_hard_label_absent_fnit_present"
        else:
            accuracy_status = "measured_cross_method_bias" if cross is not None and cross < 1-1e-12 else "same_observed_hard_labels"
        for key in ("hard_geometry_status", "fnit_repeat_not_below_official", "cross_not_below_official", "soft_within_observed_official_range"):
            region.pop(key, None)
        region.update(own_repeat_status=repeat_status, cross_accuracy_status=accuracy_status,
                      accuracy_acceptance=None,
                      interpretation="Own repeatability and cross-method bias are separate. A stable implementation need not have cross Dice 1; no accuracy acceptance threshold is inferred from official repeat noise.")
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-source-manifest", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--gpu-driver", required=True, type=Path)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--before-directory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", default=10800, type=float)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = args.output_dir / "final_full_analysis_status.json"
    status = {"state": "waiting", "started_unix": started, "helper_sha256": sha256(Path(__file__)), "scope": "CPU-only full-pipeline six-run saved-output analysis."}
    write_json(status_path, status)
    try:
        while True:
            try:
                queue = read_json(args.queue)
            except (FileNotFoundError,json.JSONDecodeError):
                queue = {"state": "waiting"}
            if queue["state"] == "failed":
                raise ValueError("Final GPU queue failed; failed attempts cannot establish repeatability")
            if queue["state"] == "completed":
                break
            if time.time()-started > args.timeout_seconds:
                raise TimeoutError("Final queue did not complete before the CPU-helper deadline")
            time.sleep(10)
        status["state"] = "auditing"
        write_json(status_path,status)
        runtime, snapshot = audit_snapshot(args.source,args.expected_source_manifest)
        manifest_path = args.before_directory / "repeatability_manifest.json"
        before_path = args.before_directory / "repeatability.json"
        manifest, before = read_json(manifest_path),read_json(before_path)
        if len(manifest["groups"]) != 24 or sha256(manifest_path) != before["manifest_sha256"]:
            raise ValueError("Before full-pipeline 24-group manifest changed")
        previous_source = Path(before["source_audit"]["source"])
        previous_manifest = previous_source / "source_manifest.json"
        if sha256(previous_manifest) != before["source_audit"]["source_manifest_sha256"]:
            raise ValueError("Before 4178a48 snapshot manifest changed")
        previous_runtime, previous_snapshot = audit_snapshot(previous_source, previous_manifest)
        if (previous_snapshot["verified_files"] != before["source_audit"]["verified_frozen_files"] or
                len(previous_runtime) != before["source_audit"]["verified_runtime_python_files"]):
            raise ValueError("Before separately frozen source identity/count differs")
        old_runs = before["fnit_run_audit"]["runs"]
        if len(old_runs) != 6 or {(run["mode"],run["repeat"]) for run in old_runs} != {
                (mode,repeat) for mode in ("stage","raw") for repeat in (1,2,3)}:
            raise ValueError("Before comparison requires separately verified three all runs per mode")
        for run in old_runs:
            old_output = Path(run["output"])
            old_report,old_api = old_output/"report.json",old_output/"api_report.json"
            if (sha256(old_report) != run["validation_report_sha256"] or
                    sha256(old_api) != run["api_report_sha256"]):
                raise ValueError("Before report/API bytes changed")
            if (read_json(old_report)["source_sha256"] != previous_runtime or
                    read_json(old_report)["input_sha256"] != run["input_sha256"] or
                    read_json(old_api)["structures"] != ["brainstem","thalamus","hippo-amygdala-left","hippo-amygdala-right"]):
                raise ValueError("Before source/input/full scope no longer match separately frozen audit")
        official_metadata_path = args.before_directory / "queue.json"
        official_metadata = read_json(official_metadata_path)
        official_inputs,_ = verify_official_metadata(official_metadata)
        official_audit = verify_official_audit(args.before_directory / "official_source_audit.json")
        if official_metadata["source_audit"] != official_audit:
            raise ValueError("Fresh official audit differs from completed reference metadata")
        before_groups = {g["id"]:g for g in before["groups"]}
        for group in manifest["groups"]:
            recorded = {(run["method"],run["id"]):run for run in before_groups[group["id"]]["runs"]}
            for method in ("official","fnit"):
                for run in group[method]:
                    if sha256(Path(run["labels"])) != recorded[(method,run["id"])]["label_image_sha256"]:
                        raise ValueError("Before or fresh official image fingerprint changed")
                    for path, expected_sha in run.get("provenance",{}).get("soft_volume_files_sha256",{}).items():
                        if sha256(Path(path)) != expected_sha:
                            raise ValueError("Fresh official soft-volume fingerprint changed")
            if group["space"].startswith("stage_") and group["fnit"][0]["provenance"]["input_sha256"] != official_inputs:
                raise ValueError("Final same-stage comparison must use the fresh official norm/aseg/wmparc")
        outputs,run_audit = audit_runs(args,queue,runtime,snapshot,manifest)
        measured,previous,manifest_groups = [],[],[]
        for base in manifest["groups"]:
            group,old = copy.deepcopy(base),copy.deepcopy(base)
            mode,resolution = base["space"].split("_")
            ids = base["label_ids"]
            group["id"] = "final__"+base["id"]
            group["fnit"] = []
            for repeat in (1,2,3):
                run = outputs[(mode,repeat)]
                api = run["api"]
                structures = {api["labels"][str(label)]["source"] for label in ids}
                if len(structures) != 1:
                    raise ValueError("Family has inconsistent HR source")
                structure = next(iter(structures))
                key = "labels" if resolution == "native" else f"highres/{structure}"
                group["fnit"].append({"id":f"final_{mode}_r{repeat}","labels":api["files"][key],"label_offset":0,
                    "soft_volumes_mm3":{str(label):api["volumes"][str(label)]["soft_volume_mm3"] for label in ids},"provenance":run["record"]})
            images = [nib.load(run["labels"]) for run in old["official"]+old["fnit"]+group["fnit"]]
            shape,affine = _grid(base["grid"],images,args.before_directory)
            grid = {"shape":list(shape),"affine":affine.tolist()}
            old["grid"] = group["grid"] = grid
            old["id"] = "before__"+base["id"]
            manifest_groups.append({"original_grid_definition":base["grid"],"before":old,"final":group})
            previous.append(separate_noise_and_accuracy(audit_group(old,args.before_directory)))
            measured.append(separate_noise_and_accuracy(audit_group(group,args.before_directory)))
            status.update(completed_groups=len(measured),current_group=base["id"])
            write_json(status_path,status)
            print("audited final/before "+base["id"],flush=True)
        shared = {"schema_version":1,"analysis_script_sha256":sha256(Path(__file__)),"snapshot_audit":snapshot,
            "before_snapshot_audit":previous_snapshot,
            "reviewed_gpu_driver_sha256":args.driver_sha256,"reviewed_context_observer_sha256":args.observer_sha256,
            "queue_sha256":sha256(args.queue),"actual_launched_queue_script":queue["queue_script"],
            "actual_context_observer":queue["context_observer"],
            "before_fix_commit":before["before_fix_commit"],"before_result_sha256":sha256(before_path),
            "before_manifest_sha256":sha256(manifest_path),"fresh_official_metadata_sha256":sha256(official_metadata_path),
            "run_audit":run_audit,"excluded_failed_attempts":[run for run in queue["runs"] if run.get("exit_code") != 0],
            "methods":{"noise":"Compare FNIT own repeats with official own repeats; this does not impose cross Dice 1.",
                       "accuracy":"Cross Dice/Jaccard/changed voxel and soft volume differences describe bias separately; no acceptance cutoff derived from repeat noise.",
                       "design":"Three full stage and three full raw runs, all structures=all; old source repeats and partial-scope results are not pooled.",
                       "grid":"One fixed input native grid or official-r1 phase/axes/spacing HR union for before+final+official. No fitted registration/phase.",
                       "range":"Three-run observed ranges, not population limits or confidence intervals.","both_empty":"Hard Dice/Jaccard undefined, not one; stable absent labels and cross absence recorded separately.",
                       "timing":"Shared-GPU observations. Compute/API/save retain actual observer overhead; process elapsed uses Popen-to-wait-thread timing and includes validation scoring/wrapper overhead. Parent preflight/GPU wait are separate. Official three queues ran concurrently."}}
        prefix = args.output_dir / "final_full_repeatability"
        write_json(prefix.with_suffix(".json"),{**shared,"groups":measured,"before_groups_on_joint_grid":previous})
        write_json(prefix.with_name(prefix.name+"_manifest.json"),{**shared,"groups":manifest_groups})
        summary = []
        for current,old in zip(measured,previous):
            summary.append({"id":current["id"],"family":current["family"],"space":current["space"],"before_pair_summary":old["pair_summary"],
                "final_pair_summary":current["pair_summary"],"regions":current["regions"]})
        write_json(prefix.with_name(prefix.name+"_summary.json"),{**shared,"groups":summary})
        columns = ["group","family","space","label","name","own_repeat_status","cross_accuracy_status",
                   "official_repeat_min_dice","fnit_repeat_min_dice","cross_min_dice","official_repeat_max_different_voxels",
                   "fnit_repeat_max_different_voxels","cross_max_different_voxels","official_soft_cv","fnit_soft_cv",
                   "before_cross_min_dice","cross_min_dice_change_from_before","before_fnit_repeat_min_dice",
                   "before_fnit_repeat_max_different_voxels"]
        with prefix.with_suffix(".tsv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle,delimiter="\t",fieldnames=columns);writer.writeheader()
            row_count = 0
            for group,old in zip(measured,previous):
                old_regions = {region["label"]:region for region in old["regions"]}
                for region in group["regions"]:
                    row = {key:group[key] for key in ("family","space")};row["group"] = group["id"]
                    row.update({key:region[key] for key in ("label","name","own_repeat_status","cross_accuracy_status","official_soft_cv","fnit_soft_cv")})
                    for kind,name in (("official_repeat","official_repeat"),("fnit_repeat","fnit_repeat"),("cross_method","cross")):
                        row[name+"_min_dice"] = region[kind+"_dice"]["min"]
                        row[name+"_max_different_voxels"] = region[kind+"_different_voxels"]["max"]
                    old_region = old_regions[region["label"]]
                    row["before_cross_min_dice"] = old_region["cross_method_dice"]["min"]
                    row["before_fnit_repeat_min_dice"] = old_region["fnit_repeat_dice"]["min"]
                    row["before_fnit_repeat_max_different_voxels"] = old_region["fnit_repeat_different_voxels"]["max"]
                    row["cross_min_dice_change_from_before"] = (row["cross_min_dice"]-row["before_cross_min_dice"]
                        if row["cross_min_dice"] is not None and row["before_cross_min_dice"] is not None else None)
                    writer.writerow(row)
                    row_count += 1
        if len(measured) != 24 or row_count != 440:
            raise ValueError("Final completed analysis must contain 24 groups and 440 label-space rows")
        artifacts = [prefix.with_suffix(".json"),prefix.with_suffix(".tsv"),prefix.with_name(prefix.name+"_summary.json"),prefix.with_name(prefix.name+"_manifest.json")]
        status.update(state="completed",finished_unix=time.time(),groups=len(measured),runtime_files=len(runtime),
            artifacts={path.name:{"bytes":path.stat().st_size,"sha256":sha256(path)} for path in artifacts})
        write_json(status_path,status)
    except BaseException as error:
        status.update(state="failed",finished_unix=time.time(),error=f"{type(error).__name__}: {error}")
        write_json(status_path,status);traceback.print_exc();raise


if __name__ == "__main__":
    main()
