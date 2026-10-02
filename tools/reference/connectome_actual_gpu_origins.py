"""Explicit read-only origins for actual GPU monitor recovery runs.

Only the declared arm/case is redirected. Original successful computation with
incomplete nvidia-smi monitoring remains immutable and ineligible; replacement
must rerun all raw DWI computation in a new namespace and satisfy normal gates.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def reader():
    try:
        from . import benchmark_connectome_cohort_compare as compare
    except ImportError:
        import benchmark_connectome_cohort_compare as compare
    return compare


def bound_json(identity):
    compare = reader()
    compare.check(isinstance(identity, dict) and set(identity) == {"path", "sha256"} and Path(identity["path"]).is_absolute(), "explicit immutable origin JSON identity required")
    value, actual = compare.safe_json(identity["path"])
    compare.check(actual == identity, "original GPU recovery provenance changed")
    return value


def incomplete_monitor(GPU):
    """A measured-over-budget run is never disguised as monitor recovery."""
    compare = reader()
    process = GPU.get("gpu_memory", {}).get("process", {})
    budget = GPU.get("memory_budget", {})
    compare.check(GPU.get("status") == "completed" and GPU.get("exit_code") == 0 and
                  budget.get("status") == "not_fully_measured" and budget.get("limit_bytes") == 20_000_000_000 and
                  set(budget.get("monitor_issues", [])) <= {"failed_samples", "errors", "sampling_gap"} and
                  {"failed_samples", "errors"} <= set(budget.get("monitor_issues", [])), "original failure is not incomplete memory monitoring")
    compare.check(process.get("backend") == "nvidia-smi" and isinstance(process.get("failed_samples"), int) and process["failed_samples"] > 0 and
                  process.get("unresolved_device_samples") == 0 and len(process.get("errors", [])) == process["failed_samples"] and
                  all(error.startswith("TimeoutExpired:") and "'nvidia-smi'" in error and "timed out after 3 seconds" in error for error in process["errors"]),
                  "original monitor failure is not the known nvidia-smi timeout")
    peaks = budget.get("measurements", {})
    compare.check(all(isinstance(peaks.get(key), (int, float)) and not isinstance(peaks.get(key), bool) and math.isfinite(peaks[key]) and
                  0 <= peaks[key] < 20_000_000_000 for key in ("allocated_bytes", "reserved_bytes", "process_tree")),
                  "original run has missing or measured-over-budget peaks")


def verify_preflight(preflight, python):
    compare = reader()
    compare.check(preflight.get("gpu_python") == python and preflight.get("scope") == "optional_NVML_monitor_environment" and
                  preflight.get("science_modules_unchanged") is True and preflight.get("CUDA_initialized") is False,
                  "declared optional NVML runtime preflight required")
    original, replacement = preflight.get("original_runtime", {}), preflight.get("new_runtime", {})
    compare.check(original.get("CUDA_initialized") is False and replacement.get("CUDA_initialized") is False and
                  replacement.get("gpu_python") == python and original.get("python_binary") == replacement.get("python_binary") and
                  original.get("python_binary_sha256") == replacement.get("python_binary_sha256") == compare.anatomy.sha(python) and
                  original.get("python_version") == replacement.get("python_version"), "actual replacement Python runtime changed")
    modules = original.get("modules", {})
    compare.check({"torch", "torch._C", "numpy", "numpy.core._multiarray_umath", "nibabel"}.issubset(modules) and
                  modules == replacement.get("modules"), "actual original/replacement science module identities differ or are incomplete")
    for key, record in modules.items():
        compare.check(set(record) == {"path", "sha256", "version"} and Path(record["path"]).is_absolute() and
                      compare.anatomy.sha(record["path"]) == record["sha256"], "actual science module file changed: " + key)
    NVML = preflight.get("NVML", {})
    compare.check(NVML.get("package") == "nvidia-ml-py" and NVML.get("CUDA_initialized_before") is False and NVML.get("CUDA_initialized_after") is False and
                  Path(NVML.get("module_path", "")).is_absolute() and compare.anatomy.sha(NVML["module_path"]) == NVML.get("module_sha256"),
                  "actual direct NVML module preflight is missing or changed")
    helper = NVML.get("original_wall_helper", {})
    compare.check(Path(helper.get("path", "")).is_absolute() and compare.anatomy.sha(helper["path"]) == helper.get("sha256"), "original NVML-capable wall helper changed")


def load_bindings(path, cases, options):
    compare = reader()
    value, identity = compare.safe_json(path)
    compare.check(set(value) == {"schema_version", "scope", "bindings"} and value["schema_version"] == 1 and
                  value["scope"] == "explicit_actual_GPU_monitor_recovery" and isinstance(value["bindings"], list) and value["bindings"],
                  "explicit GPU monitor recovery binding schema required")
    canonical = {case["case_id"]: case for case in cases}
    result = {}
    for declaration in value["bindings"]:
        compare.check(set(declaration) == {"arm", "case_id", "reason", "original", "replacement"} and
                      declaration["reason"] in {"monitor_incomplete", "original_not_dispatched", "original_queue_stopped_before_compute"}, "explicit GPU origin reason required")
        arm, case_id = declaration["arm"], declaration["case_id"]
        compare.check(arm in ("baseline", "candidate") and case_id in canonical and (arm, case_id) not in result,
                      "unknown or duplicate arm/case GPU origin")
        original, replacement = declaration["original"], declaration["replacement"]
        compare.check(set(original) == {"GPU_report", "wall_report", "driver_snapshot", "configuration"} and
                      set(replacement) == {"root", "driver_status", "configuration", "runtime_preflight"}, "explicit original/replacement provenance fields required")
        root = Path(str(getattr(options, arm + "_root")))
        configuration = bound_json(original["configuration"]); driver = bound_json(original["driver_snapshot"])
        compare.check(configuration["run_root"] == str(root) and driver.get("config") == configuration, "original config/driver provenance differs")
        record = driver.get("cases", {}).get(arm + "/" + case_id, {})
        if declaration["reason"] == "monitor_incomplete":
            compare.check(original["GPU_report"]["path"] == str(root / arm / case_id / "gpu_report.json") and
                          original["wall_report"]["path"] == str(root / arm / case_id / "raw_bids_wall.json"), "original GPU origin is outside its declared arm/case")
            GPU = bound_json(original["GPU_report"]); wall = bound_json(original["wall_report"])
            incomplete_monitor(GPU)
            compare.check(GPU.get("case_id") == case_id and GPU.get("version") == arm and GPU.get("raw_input_provenance") == canonical[case_id]["input_files"] and
                          record.get("status") in {"completed", "failed_gpu_eligibility"} and record.get("gpu_report", record.get("gpu_result")) == GPU and
                          GPU["source_before"] == GPU["source_after"] == configuration["frozen_sources"][arm], "original driver/source evidence is inconsistent")
            compare.check(wall.get("status") == "completed" and wall.get("exit_code") == 0 and wall.get("outputs", {}).get("status") == "complete" and
                          GPU.get("wall_report") == original["wall_report"]["path"], "original successful computation is incomplete")
            initial = wall.get("initial_output_state", {})
            compare.check(initial.get("output_directory_existed") is False and initial.get("preexisting_run_state") is False and
                          initial.get("preexisting_state_files") == [] and wall.get("preprocessing") == [{"topup": "completed", "eddy": "completed", "recon_all": "supplied"}],
                          "original computation was not a fresh complete raw-DWI run")
            output = root / arm / case_id / "connectome"
            for relative, identity in wall["outputs"]["files"].items():
                path = output / relative
                compare.check(path.resolve().is_relative_to(output.resolve()) and identity.get("path") == str(path) and identity.get("exists") is True and
                              compare.anatomy.sha(path) == identity["sha256"], "original scientific output changed after monitor failure")
            compare.validate_input_ledger(GPU["input_verification"], canonical[case_id], "original GPU before")
            compare.validate_input_ledger(GPU["input_verification_after"], canonical[case_id], "original GPU after")
        elif declaration["reason"] == "original_not_dispatched":
            compare.check(original["GPU_report"] is None and original["wall_report"] is None and driver.get("end_utc") and
                          driver.get("dispatch_paused") is True and driver.get("status") in {"failed_or_incomplete", "failed_or_incomplete_staged_raw_cohort"} and
                          record.get("status") in {"waiting_original_cpu_report", "anatomy_validated_dispatch_paused", "bound_GPU_not_dispatched", "waiting_original_preparation_report"} and
                          not record.get("gpu_report") and not record.get("gpu_result"), "original case was not verifiably stopped before GPU dispatch")
            for relative in ("gpu_report.json", "raw_bids_wall.json", "connectome"):
                path = root / arm / case_id / relative
                compare.check(not path.exists() and not path.is_symlink(), "original supposedly unstarted GPU namespace already contains execution/output")
            GPU, wall = None, None
        else:
            compare.check(original["GPU_report"]["path"] == str(root / arm / case_id / "gpu_report.json") and original["wall_report"] is None,
                          "queued-stop original report identity is wrong")
            GPU = bound_json(original["GPU_report"]); wall = None
            compare.check(driver.get("end_utc") and driver.get("dispatch_paused") is True and record.get("status") == "failed" and
                          record.get("gpu_report", record.get("gpu_result")) == GPU and GPU.get("status") == "failed" and
                          GPU.get("error") == {"type": "RuntimeError", "message": "STOP_DISPATCH prevents starting the queued raw-DWI computation"} and
                          GPU.get("case_id") == case_id and GPU.get("version") == arm and GPU.get("raw_input_provenance") == canonical[case_id]["input_files"] and
                          GPU.get("source_before") == configuration["frozen_sources"][arm] and not GPU.get("source_after") and
                          not GPU.get("command") and not GPU.get("gpu_command_wall_seconds") and not GPU.get("wall_report") and
                          not GPU.get("gpu_memory") and GPU.get("exit_code") is None, "original queued worker did not verifiably stop before science compute")
            for relative in ("raw_bids_wall.json", "connectome"):
                path = root / arm / case_id / relative
                compare.check(not path.exists() and not path.is_symlink(), "original queued-stop worker already has raw-DWI output")
        compare.verify_source(configuration["frozen_sources"][arm])
        new_root, new_driver = Path(replacement["root"]), Path(replacement["driver_status"])
        compare.check(new_root.is_absolute() and new_driver.is_absolute() and not new_root.is_symlink(), "explicit actual replacement namespace/driver required")
        compare.check_report_namespace(new_root, [root, getattr(options, "baseline_anatomy_root"),
            getattr(options, "candidate_root" if arm == "baseline" else "baseline_root"),
            configuration["frozen_sources"][arm]["directory"], *[Path(item["path"]).parent for item in canonical[case_id]["input_files"]]])
        new_config = bound_json(replacement["configuration"])
        preflight = bound_json(replacement["runtime_preflight"])
        compare.check(new_config.get("run_root") == str(new_root) and arm in new_config.get("frozen_sources", {}) and
                      new_config["frozen_sources"][arm] == configuration["frozen_sources"][arm] and
                      new_config.get("wall_script_sha256") == configuration.get("wall_script_sha256") and
                      compare.anatomy.sha(new_config["wall_script"]) == new_config["wall_script_sha256"], "replacement altered frozen science or wall evaluator bytes")
        for key in ("atlases", "atlas_options", "eddy_gp_seed", "n_seeds", "seed", "device", "gpu_lock", "gpu_host", "gpu_cpu_threads", "cuda_visible_devices"):
            compare.check(key in new_config and new_config[key] == configuration.get(key), "replacement scientific setting or shared GPU lock changed: " + key)
        compare.check(new_config.get("gpu_python") and Path(new_config["gpu_python"]).is_absolute(), "declared monitor interpreter required")
        compare.check(Path(new_config["gpu_python"]).is_file() and os.access(new_config["gpu_python"], os.X_OK) and
                      compare.anatomy.sha(new_config["gpu_python"]) == preflight["original_runtime"]["python_binary_sha256"],
                      "replacement actual Python binary changed")
        verify_preflight(preflight, new_config["gpu_python"])
        compare.check(preflight["original_runtime"]["gpu_python"] == configuration["gpu_python"] and
                      (GPU is None or preflight["original_runtime"]["python_binary_sha256"] == GPU["identity"]["python_executable_sha256"]) and
                      preflight["NVML"]["original_wall_helper"]["sha256"] == configuration["wall_script_sha256"], "monitor preflight differs from original actual runtime")
        replacement_manifest, _ = compare.safe_json(new_root / "input_manifest.json")
        compare.check(compare.manifest_cases(replacement_manifest) == cases, "replacement canonical raw cohort changed")
        result[arm, case_id] = {"declaration": declaration, "binding_file": identity, "original_GPU": GPU,
                               "original_wall": wall, "original_configuration": configuration,
                               "replacement_configuration": new_config, "runtime_preflight": preflight}
    return result, identity


def verify_replacement(binding, GPU, wall, case, subject_dir):
    compare = reader()
    declaration = binding["declaration"]
    original = declaration["original"]; replacement = declaration["replacement"]
    for identity in original.values():
        if identity is not None: bound_json(identity)
    configuration = bound_json(replacement["configuration"]); preflight = bound_json(replacement["runtime_preflight"])
    compare.check(configuration == binding["replacement_configuration"] and preflight == binding["runtime_preflight"], "frozen replacement configuration or preflight changed")
    verify_preflight(preflight, configuration["gpu_python"])
    before, old_wall = binding["original_GPU"], binding["original_wall"]
    expected_source = binding["original_configuration"]["frozen_sources"][declaration["arm"]]
    old_identity = before["identity"] if before else {
        "python_executable_sha256": preflight["original_runtime"]["python_binary_sha256"],
        "python_version": preflight["original_runtime"]["python_version"], "hostname": binding["original_configuration"]["gpu_host"]}
    compare.check(GPU.get("case_id") == case["case_id"] and GPU.get("version") == declaration["arm"] and
                  GPU.get("raw_input_provenance") == case["input_files"] and
                  GPU["source_before"] == GPU["source_after"] == expected_source, "replacement source/raw input changed")
    compare.check(GPU["identity"].get("python_executable_sha256") == old_identity.get("python_executable_sha256") and
                  GPU["identity"].get("python_version") == old_identity.get("python_version") and
                  GPU["identity"].get("hostname") == old_identity.get("hostname") and
                  GPU["identity"].get("python") == configuration["gpu_python"], "replacement actual interpreter/host differs from declared monitor environment")
    def normalized(arguments):
        values = list(arguments)
        compare.check(values.count("--output-dir") == 1, "actual CLI output namespace flag missing or repeated")
        index = values.index("--output-dir"); compare.check(index + 1 < len(values), "actual CLI output directory absent")
        return values[:index + 1] + ["<declared_fresh_output>"] + values[index + 2:]
    if old_wall:
        compare.check(normalized(wall["cli_arguments"]) == normalized(old_wall["cli_arguments"]), "replacement actual scientific CLI changed")
        for key in ("torch_version", "torch_cuda_version", "python_version", "precision_at_exit", "threads", "cuda_environment"):
            compare.check(wall["provenance"].get(key) == old_wall["provenance"].get(key), "replacement scientific runtime setting changed: " + key)
        old_inputs = {key: value for key, value in old_wall["inputs"].items() if not key.startswith("prepared/")}
        new_inputs = {key: value for key, value in wall["inputs"].items() if not key.startswith("prepared/")}
        compare.check(old_inputs == new_inputs, "replacement raw/anatomy/resource inputs changed")
    else:
        root = Path(binding["original_configuration"]["run_root"])
        absent = ("gpu_report.json", "raw_bids_wall.json", "connectome") if before is None else ("raw_bids_wall.json", "connectome")
        for relative in absent:
            path = root / declaration["arm"] / case["case_id"] / relative
            compare.check(not path.exists() and not path.is_symlink(), "original unstarted GPU namespace later contains execution/output")
        arguments = wall["cli_arguments"]
        for flag, value in (("--n-seeds", configuration["n_seeds"]), ("--seed", configuration["seed"]), ("--device", configuration["device"])):
            compare.check(arguments.count(flag) == 1 and arguments[arguments.index(flag) + 1] == str(value), "new actual raw-DWI CLI differs from frozen scientific settings")
        index = arguments.index("--atlas") if arguments.count("--atlas") == 1 else -1
        end = next((i for i in range(index + 1, len(arguments)) if arguments[i].startswith("--")), len(arguments))
        compare.check(index >= 0 and arguments[index + 1:end] == configuration["atlases"], "new actual atlas selection differs from frozen original plan")
        compare.check(wall["provenance"].get("torch_version") == preflight["new_runtime"]["modules"]["torch"]["version"], "new actual torch version differs from runtime preflight")
    compare.check(Path(wall["selected_inputs"]["freesurfer_subject_dir"]).resolve() == Path(subject_dir).resolve(), "replacement selected wrong same-round fresh FS")
    initial = wall.get("initial_output_state", {})
    output = Path(replacement["root"]) / declaration["arm"] / case["case_id"] / "connectome"
    compare.check(initial.get("output_directory_existed") is False and initial.get("preexisting_run_state") is False and initial.get("preexisting_state_files") == [] and
                  wall.get("preprocessing") == [{"topup": "completed", "eddy": "completed", "recon_all": "supplied"}] and wall.get("actual_eddy_gp_seeds") == [configuration["eddy_gp_seed"]] and
                  wall["selected_inputs"].get("dwi") == str(output / "preproc/eddy/data.nii.gz") and
                  wall["selected_inputs"].get("bvecs") == str(output / "preproc/eddy/data.eddy_rotated_bvecs"), "replacement reused prepared DWI or an output namespace")
    process = GPU.get("gpu_memory", {}).get("process", {})
    allocator = GPU.get("gpu_memory", {}).get("allocator", {})
    budget = GPU.get("memory_budget", {})
    expected_peaks = {"process_tree": process.get("peak_process_tree_bytes"), "allocated_bytes": allocator.get("allocated_bytes"), "reserved_bytes": allocator.get("reserved_bytes")}
    compare.check(process.get("backend") == "pynvml" and process.get("status") == "measured" and not process.get("errors") and
                  process.get("failed_samples") == process.get("unresolved_device_samples") == 0 and isinstance(process.get("samples"), int) and process["samples"] > 0,
                  "recovery run did not actually use complete direct NVML measurements")
    compare.check(budget.get("limit_bytes") == 20_000_000_000 and budget.get("status") == "observed_below_budget" and not budget.get("monitor_issues") and
                  budget.get("measurements") == expected_peaks and all(isinstance(value, (float, int)) and not isinstance(value, bool) and
                  math.isfinite(value) and 0 <= value < 20_000_000_000 for value in expected_peaks.values()), "actual replacement memory qualification failed")
    interval, gap = process.get("sample_interval_seconds"), process.get("max_observed_interval_seconds")
    compare.check(all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in (interval, gap)) and
                  gap <= max(5., 10 * interval), "replacement direct NVML sampling gap failed")
    return {"status": "actual_eligible_replacement_verified", "declaration": declaration, "binding_file": binding["binding_file"],
            "original_memory_budget": before.get("memory_budget") if before else None, "original_CLI_seconds": old_wall["total_runtime_seconds"] if old_wall else None,
            "original_queued_stop_timing": {key: before.get(key) for key in ("start_utc", "end_utc", "worker_wall_seconds", "gpu_lock_queue_seconds", "error")}
                if declaration["reason"] == "original_queue_stopped_before_compute" else None,
            "new_environment": {"gpu_python": configuration["gpu_python"], "preflight": replacement["runtime_preflight"]},
            "scope": ("original successful computation remains ineligible and immutable" if old_wall else
                "original queued worker stopped before science; actual failed worker/queue timing retained, original CLI/output absent" if before else
                "original case stopped before GPU dispatch; original GPU/CLI timing and outputs are absent") +
                "; actual new execution computes complete raw DWI with same frozen source/inputs/FS/scientific settings in a fresh namespace; monitor environment/timing is separate, no stable speedup inferred"}


def selected_origin(options, arm, case_id, bindings=None):
    bindings = {} if bindings is None else bindings
    if (arm, case_id) not in bindings:
        return Path(str(getattr(options, arm + "_root"))), Path(str(getattr(options, arm + "_driver"))), None
    binding = bindings[arm, case_id]
    declaration = binding["declaration"]["replacement"]
    return Path(declaration["root"]), Path(declaration["driver_status"]), binding
