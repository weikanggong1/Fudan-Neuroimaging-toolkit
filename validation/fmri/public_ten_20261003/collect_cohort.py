#!/usr/bin/env python3
"""收集十例真实 benchmark 的匿名来源、环境、分界计时，并比较已完成配对。

CLI: --config PRIVATE.json --output-root NEW_DIRECTORY [--metadata-only]
私有配置必须具名绑定 cohort_id/candidate_root、data_manifest/assets_manifest（path/sha256）、raw_root、
resources_root、source_root、python_prefix、candidate_cases/reference_cases。
病例映射为 case_id:{report,files,config?}；程序为 programs=[{id,path,build_id?}]；
编译记录为 build_records=[{id,manifest:{path,sha256},binaries:[{id,path}],
source_files:[{id,path}]?}]；其他来源文件/目录为 provenance_files/source_trees。
所有原始路径与完整编译命令保持私有；公开输出只含公共病例 ID、数值和哈希。
探针进程的 Torch 默认状态不是运行中 pipeline 的实际精度测量。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

import compare_subject as comparison_module
from compare_subject import (AXIS_ASSET_SHA256, CASES, RUN_TIMING_BOUNDARIES, checked_file,
                             read_bound_json, read_json, reference_recovery_provenance,
                             sha256, timing_boundaries, write_json)


SOURCE_SUFFIXES = {".py", ".cpp", ".c", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".cmake", ".in"}
TIME_FIELDS = ("continuous_api_wall_seconds", "driver_through_saved_output_validation_seconds",
               "container_process_wall_seconds", "wall_seconds",
               "continuous_wall_through_saved_QC_seconds", "schema_adapter_seconds",
               "recovered_QC_seconds", "recovery_gap_since_original_end_seconds",
               "failed_attempt_wall_seconds", "process_wall_seconds")
HASH_FIELDS = ("source_revision", "source_sha256", "driver_sha256", "launcher_sha256",
               "SIF_sha256", "canonical_command_sha256", "actual_host_command_sha256", "configuration_sha256")


def utc():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not isinstance(value, str) or not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError("public identifiers must be short names without paths")
    return value


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def checked_record(entry):
    path, digest = checked_file(entry)
    record, read_digest = read_bound_json(path)
    if read_digest != digest:
        raise ValueError("bound provenance JSON changed before its actual read")
    return path, record, digest


def numeric_tree(value):
    """Never copy free-form run errors/commands/private paths to public JSON."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        return {identifier(key): numeric_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numeric_tree(item) for item in value]
    raise ValueError("numeric timing/metrics records may not contain free-form text")


def allocator_launcher_statement(text):
    """Record the last explicit allocator flag statement before runner exec.

    This is shell-source evidence, not the process environment or allocator state.
    Export assignment has presence semantics even for zero or empty values.
    """
    execution = re.search(r"^\s*exec .*run_fnit_cohort\.py", text, flags=re.MULTILINE)
    events = list(re.finditer(r"^[ \t]*(?:export[ \t]+PYTORCH_NO_CUDA_MEMORY_CACHING=(.*)|unset[ \t]+PYTORCH_NO_CUDA_MEMORY_CACHING)[ \t]*$",
                             text, flags=re.MULTILINE))
    prior = [event for event in events if execution is not None and event.start() < execution.start()]
    selected = prior[-1] if prior else None
    disabled = selected is not None and selected.group(1) is not None
    enabled = selected is not None and not disabled
    return {"no_cuda_caching_export_before_runner_exec": disabled,
            "no_cuda_caching_unset_before_runner_exec": enabled,
            "selection_at_exec_statement": "disabled" if disabled else "enabled" if enabled else "not_declared",
            "flag_presence_at_exec_statement": True if disabled else False if enabled else None,
            "allocator_statement_sha256": digest_text(selected.group(0)) if selected is not None else None,
            "precision_scope": "last explicit export/unset before runner exec; actual driver flag presence and scientific source/precision are recorded separately"}


def declared_runner_precision(source_root):
    """Read explicit assignment syntax; this is source evidence, not a probe."""
    import ast
    path = source_root / "validation/fmri/public_ten_20261003/run_fnit_subject.py"
    tree = ast.parse(path.read_text())
    declarations = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            for target in node.targets:
                name = ast.unparse(target)
                if name in ("torch.backends.cuda.matmul.allow_tf32", "torch.backends.cudnn.allow_tf32"):
                    declarations[name] = node.value.value
    return {"runner_sha256": sha256(path), "explicit_assignments": declarations,
            "scope": "declared parent runner flags; per-stage model precision and subprocess flags are not directly measured by this independent collector"}


def probe_environment(prefix):
    import torch
    cpu = {"architecture": platform.machine(), "logical_cpus": os.cpu_count(),
           "collector_affinity_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "collector_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    try:
        result = subprocess.run(["lscpu", "--json"], capture_output=True, text=True, timeout=15, check=True)
        fields = {item["field"].rstrip(":"): item["data"] for item in json.loads(result.stdout)["lscpu"]}
        cpu.update({key: fields[field] for key, field in (("model", "Model name"), ("sockets", "Socket(s)"),
            ("cores_per_socket", "Core(s) per socket"), ("threads_per_core", "Thread(s) per core")) if field in fields})
    except Exception as error:
        cpu["probe_failure_type"] = type(error).__name__
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            cpu["model"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                                 if line.startswith("model name")), "unavailable")
            cpu["model_probe"] = "procfs fallback; installed lscpu has no supported JSON output"
    memory = Path("/proc/meminfo")
    if memory.is_file():
        for line in memory.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                cpu[key + "_bytes_at_collection"] = int(value.split()[0]) * 1024
    packages = []
    package_source = "conda-meta"
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        row = read_json(path)
        packages.append({key: row[key] for key in ("name", "version", "build", "build_number") if key in row})
    if not packages:
        package_source = "importlib.metadata; no conda-meta found"
        packages = sorted(({"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
                           for dist in importlib.metadata.distributions()), key=lambda row: row["name"].lower())
    gpu = []
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
        for line in result.stdout.splitlines():
            index, name, uuid_value, total, used, utilization, driver = [value.strip() for value in line.split(",")]
            gpu.append({"physical_index": int(index), "model": name, "uuid_sha256": digest_text(uuid_value),
                        "memory_total_mib": int(total), "memory_used_mib_at_collection": int(used),
                        "utilization_percent_at_collection": int(utilization), "driver_version": driver})
    except Exception as error:
        gpu = [{"probe_failure_type": type(error).__name__}]
    return {"observed_at_utc": utc(), "observation_scope": "independent collector process; instantaneous load is not benchmark load history",
            "machine_identity_sha256": digest_text(platform.node()), "cpu": cpu, "gpu": gpu,
            "python": platform.python_version(), "python_executable_sha256": sha256(sys.executable),
            "kernel": platform.release(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "package_source": package_source, "packages": packages,
            "probe_torch_defaults": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "default_dtype": str(torch.get_default_dtype()),
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
            "precision_scope": "probe defaults are not measured pipeline settings; declared runner TF32 and model-specific FP32 policies must be bound to actual source/metadata"}


def sources_and_programs(config, private_output):
    programs, builds, files, trees = [], [], [], []
    for entry in config.get("programs", []):
        path = Path(entry["path"]).resolve()
        if "license" in path.name.lower():
            raise ValueError("license files must never be hashed by the collector")
        digest = sha256(path)
        if entry.get("sha256") is not None and entry["sha256"] != digest:
            raise ValueError("installed program differs from configured source-build SHA")
        programs.append({"id": identifier(entry["id"]), "bytes": path.stat().st_size, "sha256": digest,
                         "build_id": identifier(entry["build_id"]) if entry.get("build_id") is not None else None,
                         "identity_scope": "installed file snapshot; does not by itself prove execution"})
    for entry in config.get("build_records", []):
        manifest_path, original, digest = checked_record(entry["manifest"])
        record = {"id": identifier(entry["id"]), "manifest_sha256": digest,
                  "status": original.get("status"), "upstream_commit": original.get("source_commit"),
                  "fnit_source_revision": original.get("source_revision"), "binaries": [], "source_files": []}
        for key in ("source_sha256", "compiler_sha256", "binary_sha256", "return_code", "excluded_from_pipeline_benchmark"):
            if key in original:
                record[key] = original[key]
        if "compiler_version" in original:
            record["compiler_version_first_line"] = original["compiler_version"].splitlines()[0]
        record["flags_without_private_paths"] = [flag for flag in original.get("flags", []) if isinstance(flag, str) and "/" not in flag and "\\" not in flag]
        for item in entry.get("binaries", []):
            path = Path(item["path"]).resolve(); actual = sha256(path)
            expected = item.get("sha256", original.get("binary_sha256"))
            if expected is not None and actual != expected:
                raise ValueError("actual binary differs from its bound compiler record")
            record["binaries"].append({"id": identifier(item["id"]), "sha256": actual,
                                       "matches_recorded_sha256": actual == expected if expected else None})
        for item in entry.get("source_files", []):
            path = Path(item["path"]).resolve(); actual = sha256(path)
            expected = item.get("sha256")
            if expected is not None and actual != expected:
                raise ValueError("actual native source differs from its compiler record")
            record["source_files"].append({"id": identifier(item["id"]), "sha256": actual})
        commands = original.get("commands", [])
        if commands:
            record["private_commands_sha256"] = digest_text(json.dumps(commands, sort_keys=True, separators=(",", ":")))
            record["commands"] = [{"label": identifier(item["label"]), "seconds": item.get("seconds"),
                                    "argv_sha256": digest_text(json.dumps(item["argv"], separators=(",", ":")))} for item in commands]
        libraries = original.get("static_libraries", {})
        record["static_libraries"] = []
        for path, value in sorted(libraries.items()):
            actual = sha256(path)
            if actual != value:
                raise ValueError("static library differs from its bound compiler/link record")
            record["static_libraries"].append({"name": Path(path).name, "recorded_sha256": value,
                                                "actual_sha256": actual, "matches_recorded_sha256": True})
        if original.get("compiler"):
            record["actual_compiler_sha256"] = sha256(original["compiler"])
        builds.append(record)
    for entry in config.get("provenance_files", []):
        path = Path(entry["path"]).resolve()
        if "license" in path.name.lower():
            raise ValueError("license files must never be read or hashed")
        record = {"id": identifier(entry["id"]), "sha256": sha256(path), "bytes": path.stat().st_size}
        if entry.get("sha256") is not None and record["sha256"] != entry["sha256"]:
            raise ValueError("provenance file differs from its explicitly bound SHA")
        if entry.get("parse") == "source_build":
            import re
            text = path.read_text()
            for key, expression in (("recorded_upstream_commit", r"^SOURCE_COMMIT=([0-9a-f]{40})$"),
                                    ("recorded_archive_tree_sha256", r"^SOURCE_VALIDATION=archive-tree-sha256:([0-9a-f]{64})$")):
                match = re.search(expression, text, flags=re.MULTILINE)
                record[key] = match.group(1) if match else None
        if entry.get("parse") == "allocator_launcher":
            record.update(allocator_launcher_statement(path.read_text()))
        files.append(record)
    for entry in config.get("source_trees", []):
        root = Path(entry["root"]).resolve()
        records = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and "license" not in path.name.lower() and (path.suffix in SOURCE_SUFFIXES or path.name == "CMakeLists.txt"):
                records.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
        if not records:
            raise ValueError("configured compilation source tree is empty")
        tree_digest = digest_text(json.dumps(records, sort_keys=True, separators=(",", ":")))
        private_path = private_output / (identifier(entry["id"]) + ".source_tree.private.json")
        write_json(private_path, {"root": str(root), "records": records})
        trees.append({"id": identifier(entry["id"]), "upstream_commit_declared": entry.get("upstream_commit"),
                      "source_files": len(records), "source_bytes": sum(row["bytes"] for row in records),
                      "source_tree_sha256": tree_digest, "private_source_inventory_sha256": sha256(private_path),
                      "scope": "current compile-source/header inventory; source tree fingerprint alone does not bind every installed binary"})
    return {"programs": programs, "build_records": builds, "provenance_files": files, "source_trees": trees,
            "license_policy": "No authorization license content or hashes collected; upstream resource licenses remain bound to the separately verified assets manifest"}


def summarize_run(spec, raw, *, reference, expected_source_revision=None):
    path = Path(spec["report"])
    if not path.is_file():
        return {"status": "pending"}, None
    # A running report can be atomically replaced. Bind the bytes actually read.
    payload = path.read_bytes(); report = json.loads(payload)
    if spec.get("report_sha256") is not None and hashlib.sha256(payload).hexdigest() != spec["report_sha256"]:
        raise ValueError("run report differs from its explicitly bound immutable SHA")
    if report.get("subject") != raw["subject"]:
        raise ValueError("run report subject differs from its fixed raw cohort case")
    record = {"status": report.get("status"), "report_sha256": hashlib.sha256(payload).hexdigest(),
              "execution": {key: report.get(key) for key in HASH_FIELDS if key in report},
              "timing_seconds": {key: report[key] for key in TIME_FIELDS if key in report},
              "frames": report.get("frames", report.get("input_frames")),
              "tr_seconds": report.get("repetition_time", report.get("input_TR_seconds")),
              "input_sha256": report.get("input_sha256"),
              "source_unchanged_during_run": report.get("source_unchanged_during_run"),
              "input_unchanged_during_run": report.get("input_unchanged_during_run"),
              "volume_executed": report.get("volume_executed"),
              "timing_scope": "official workflow/container and saved wrapper QC intervals remain separate" if reference else "FNIT API and driver-through-validation intervals remain separate; complete process/import/final-report boundary comes from the queue"}
    record["timing_boundaries"] = timing_boundaries(report, record["timing_seconds"])
    if "recovered_QC_seconds" in report:
        record["reference_qc_recovery"] = reference_recovery_provenance(report)
    for key in ("container_exit_code", "command_exit_code"):
        if key in report:
            record[key] = numeric_tree(report[key])
    if reference and report.get("status") == "failed" and report.get("container_exit_code") == 0:
        record["failure_scope"] = "official MRI command exited zero; original wrapper saved-output/QC failed"
    numeric_tree(record["timing_seconds"])
    for key in ("raw_inputs_unchanged", "configuration_unchanged", "driver_unchanged"):
        if key in report:
            record[key] = report[key]
    for key in ("bold_metadata_sha256_before", "bold_metadata_sha256_after"):
        if key in report:
            # Metadata filenames remain private; bind the whole named inventory.
            inventory = report[key]
            record[key + "_inventory"] = {"entries": len(inventory),
                "inventory_sha256": digest_text(json.dumps(inventory, sort_keys=True, separators=(",", ":"))),
                "sha256_values": sorted(inventory.values())}
    if "imported_pipeline" in report:
        imported = report["imported_pipeline"]
        record["imported_pipeline_source"] = {"sha256": imported["sha256"],
            "scope": "driver verified imported callable resides in the explicitly bound source root"}
    if "cuda_allocator_environment" in report:
        environment = report["cuda_allocator_environment"]
        record["cuda_allocator_environment"] = {key: environment[key] for key in
            ("cache_disabled_flag_present", "selection_at_fresh_process_entry") if key in environment}
        record["cuda_allocator_environment"]["scope"] = "driver-observed environment flag presence; not a measurement of all stage allocators or GPU occupancy"
    if "stage_seconds" in report:
        record["stage_seconds"] = numeric_tree(report["stage_seconds"])
        record["stage_scope"] = "nested/parallel intervals; never summed to reconstruct whole wall"
    for key in ("owned_tree_peak_bytes", "owned_tree_memory_measured", "owned_tree_under_20gb"):
        if key in report:
            record[key] = numeric_tree(report[key])
    if report.get("owned_tree_memory_measured") is False:
        record["owned_tree_peak_bytes"] = None
        record["owned_tree_under_20gb"] = None
    if report.get("owned_tree_memory_measured") is True:
        observed_over = report["owned_tree_peak_bytes"] > 20_000_000_000
        record["observed_peak_over_target"] = observed_over
        record["memory_over_target"] = True if observed_over else (None if report.get("memory_sampling_errors") else False)
    if "memory_sampling_errors" in report:
        record["memory_sampling_error_count"] = len(report["memory_sampling_errors"])
        if report["memory_sampling_errors"]:
            record["owned_tree_under_20gb"] = None
    if "owned_tree_memory_measured" in report:
        record["memory_observation_scope"] = "maximum successful simultaneous job-tree sample; missing/error samples do not establish a below-target continuous peak"
    if report.get("failure"):
        record["failure_type"] = identifier(report["failure"].get("type", "UnknownFailure"))
    expected = {"t1w": raw["T1w"]["sha256"], "bold": raw["BOLD"]["sha256"]}
    reported_input = report.get("input_sha256", {})
    initial_input = {"t1w": reported_input.get("t1w", reported_input.get("T1w")),
                     "bold": reported_input.get("bold", reported_input.get("BOLD"))}
    record["raw_input_identity_matches"] = initial_input == expected
    if "T1w" in reported_input:
        record["input_schema_note"] = "Legacy initial hash keys read for progress only; final comparison requires the separate verified adapter and canonical completed schema"
    for key in ("hardware_before", "hardware_after"):
        if key in report:
            record[key] = {name: report[key][name] for name in
                ("CPU_model", "logical_cpus", "load_average", "MemTotal_GiB", "MemAvailable_GiB") if name in report[key]}
    versions = report.get("software_versions", report.get("versions", {}))
    record["reported_software_versions"] = {key: versions[key] for key in
        ("fmriprep", "freesurfer", "FreeSurfer", "smriprep", "nipype", "nibabel", "python") if key in versions}
    if record["status"] == "complete":
        if not record["raw_input_identity_matches"] or record["frames"] != 180 or record["tr_seconds"] != raw["repetition_time_seconds"]:
            raise ValueError("completed report contradicts fixed raw identity or full time axis")
        if record["source_unchanged_during_run"] is not True:
            raise ValueError("completed report does not verify unchanged execution sources")
        if not reference:
            if expected_source_revision is not None and report.get("source_revision") != expected_source_revision:
                raise ValueError("completed candidate differs from the explicitly bound cohort source revision")
            for key in ("raw_inputs_unchanged", "configuration_unchanged", "driver_unchanged"):
                if key in report and report[key] is not True:
                    raise ValueError("completed candidate contradicts its named input/configuration/driver guard")
            if "configuration_sha256" in report and spec.get("config"):
                _, config_digest = read_bound_json(spec["config"])
                if config_digest != report["configuration_sha256"]:
                    raise ValueError("completed candidate configuration differs from its exact input bytes")
        record["output_sha256"] = {key: report["output_checks"][key]["sha256"] for key in ("preproc_mni", "dtseries")}
    return record, report


def reconstruction_execution(spec, raw):
    """Publish hashes of the programs actually named by completed FNIT metadata."""
    files, files_digest = read_bound_json(spec["files"])
    metadata_path = Path(files["metadata"])
    sidecar, sidecar_digest = read_bound_json(metadata_path)
    reconstruction = sidecar["FNIT"]["Reconstruction"]
    request = reconstruction["request"]
    if (reconstruction.get("status") != "complete" or reconstruction.get("reused", False) is not False
            or request.get("backend") != "fnit" or request.get("source_sha256") != raw["T1w"]["sha256"]):
        raise ValueError("completed whole run must bind a fresh FNIT reconstruction of its raw T1w")
    programs = {}
    for row in [*reconstruction.get("native_binaries", []), *reconstruction.get("commands", [])]:
        path = Path(row["binary"]).resolve()
        if "license" in path.name.lower():
            raise ValueError("authorization license files cannot be execution provenance")
        digest = sha256(path)
        if digest != row["sha256"]:
            raise ValueError("actually selected reconstruction program changed after its run")
        name = identifier(path.name)
        if name in programs and programs[name] != digest:
            raise ValueError("different actually selected programs share an ambiguous public name")
        programs[name] = digest
    if sha256(spec["files"]) != files_digest or sha256(metadata_path) != sidecar_digest:
        raise ValueError("completed reconstruction metadata changed during collection")
    return {"surface_sidecar_sha256": sidecar_digest, "files_manifest_sha256": files_digest,
            "backend": "fnit", "reused": False, "raw_t1w_sha256": request["source_sha256"],
            "reported_native_program_sha256": programs,
            "reconstruction_timing_seconds": numeric_tree(reconstruction.get("timing", {})),
            "scope": "actual selected native programs recorded by completed mature reconstruction and graymid commands; current bytes rechecked, full private arguments/paths excluded"}


def retained_run(spec, raw, output, case, label, *, reference, inspect_reconstruction=True,
                 expected_source_revision=None):
    """A bad or incomplete case remains visible without discarding other cases."""
    try:
        record, original = summarize_run(spec, raw, reference=reference,
                                         expected_source_revision=expected_source_revision)
        if record["status"] == "complete" and not reference and inspect_reconstruction:
            record["reconstruction_execution"] = reconstruction_execution(spec, raw)
        return record, original
    except Exception as error:
        (output / f"{case}.{identifier(label)}.report_check.private.txt").write_text(traceback.format_exc())
        record = {"status": "failed", "report_integrity": "failed",
                  "failure_type": type(error).__name__,
                  "failure_scope": "independent report/source/frame identity validation; no pipeline output accepted"}
        path = Path(spec["report"])
        if path.is_file():
            try:
                original, record["report_sha256"] = read_bound_json(path)
                status = original.get("status")
                if status in ("complete", "failed", "running", "starting", "initializing"):
                    record["pipeline_reported_status"] = status
            except (OSError, ValueError):
                record["report_sha256"] = sha256(path)
        return record, None


def historical_execution_status(entry, record):
    """Bind an interruption observation without changing the original run report.

    The signal event and later process-absence observation have separate times.
    Free-form reasons, process IDs and private filenames never enter this record.
    """
    if "execution_status" not in entry:
        return record
    _, observed, observed_digest = checked_record(entry["execution_status"])
    if (observed.get("subject") != entry["case_id"]
            or not str(observed.get("status", "")).startswith("interrupted")
            or observed.get("original_report_sha256") != record.get("report_sha256")
            or observed.get("original_report_status") != record.get("status")
            or observed.get("original_report_preserved") is not True
            or observed.get("confirmed_worker_process_absent") is not True):
        raise ValueError("historical interruption does not bind the preserved original run report")
    observation = {"status": identifier(observed["status"]),
                   "sha256": observed_digest,
                   "observed_at_utc": datetime.fromisoformat(observed["utc"]).isoformat(),
                   "original_report_preserved": True, "confirmed_worker_process_absent": True,
                   "time_scope": "later interruption/process-absence observation; not the signal instant"}
    if "signal_event" in entry:
        _, event, event_digest = checked_record(entry["signal_event"])
        event_time = datetime.fromisoformat(event["utc"])
        if event_time > datetime.fromisoformat(observed["utc"]):
            raise ValueError("interruption observation precedes its explicitly bound signal event")
        observation["signal_event"] = {"sha256": event_digest,
            "status": identifier(event["status"]), "at_utc": event_time.isoformat()}
    return {**record, "status": "interrupted", "pipeline_reported_status": record["status"],
            "execution_status_observation": observation}


def collect(config, output, *, metadata_only=False):
    started = time.perf_counter()
    cohort_id = identifier(config["cohort_id"])
    candidate_root = Path(config["candidate_root"]).resolve()
    formal = re.fullmatch(r"formal-v([0-9]+)", cohort_id)
    if formal and candidate_root.name != "candidate_v" + formal.group(1):
        raise ValueError("formal cohort must bind its corresponding named fresh candidate root")
    for case, entry in config["candidate_cases"].items():
        if case not in CASES or not Path(entry["report"]).resolve().is_relative_to(candidate_root):
            raise ValueError("formal candidate mappings must stay within the explicitly bound cohort root")
    if set(config["reference_cases"]) - set(CASES):
        raise ValueError("reference mappings must retain only predeclared public cases")
    data_path, data, data_digest = checked_record(config["data_manifest"])
    assets_path, assets, assets_digest = checked_record(config["assets_manifest"])
    if data.get("license") != "CC0" or tuple(data.get("selected_subjects", [])) != CASES:
        raise ValueError("collection requires the fixed ten-case CC0 raw cohort")
    if data.get("upstream_git_commit") != "359d372c5e972a161966312128adb365870df949" or assets.get("complete") is not True:
        raise ValueError("data/resource source identity is not complete and fixed")
    resources = {row["relative_path"]: row for row in assets["resources"]}
    for row in resources.values():
        path = Path(config["resources_root"]) / row["relative_path"]
        # Existing public license SHA remains bound by manifest; do not reopen license files.
        if path.stat().st_size != row["bytes"] or ("license" not in path.name.lower() and sha256(path) != row["sha256"]):
            raise ValueError("resource no longer matches its approved original-source manifest")
    raw_cases = {row["subject"]: row for row in data["subjects"]}
    if len(data["subjects"]) != len(CASES) or set(raw_cases) != set(CASES):
        raise ValueError("raw manifest must retain every predeclared case exactly once")
    hardware = probe_environment(Path(config["python_prefix"]))
    hardware["declared_runner_precision"] = declared_runner_precision(Path(config["source_root"]))
    native = sources_and_programs(config, output)
    cases, comparisons, comparison_entries = [], [], []
    for case in CASES:
        raw = raw_cases[case]
        record = {"case_id": case, "frames": raw["complete_original_frames"], "tr_seconds": raw["repetition_time_seconds"]}
        runs = {}
        for label in ("candidate", "reference"):
            spec = config[label + "_cases"].get(case)
            if spec is None:
                record[label] = {"status": "pending"}; continue
            record[label], runs[label] = retained_run(spec, raw, output, case, label, reference=label == "reference",
                expected_source_revision=config.get("source_revision") if label == "candidate" else None)
            if spec.get("config"):
                requested, configuration_digest = read_bound_json(spec["config"])
                record[label]["private_config_sha256"] = configuration_digest
                if spec.get("config_sha256") is not None:
                    matched = configuration_digest == spec["config_sha256"]
                    record[label]["private_config_sha256_matches_bound_sha"] = matched
                    if not matched:
                        record[label].update(pipeline_reported_status=record[label]["status"],
                                             status="failed", configuration_integrity="failed")
                        runs[label] = None
                if runs[label] is not None and "configuration_sha256" in runs[label]:
                    record[label]["configuration_sha256_matches_current_private_config"] = configuration_digest == runs[label]["configuration_sha256"]
                record[label]["requested_parameters"] = {key: requested[key] for key in
                    ("signal", "device", "cpu_threads", "parallel", "recon_all_backend") if key in requested}
                for section, keys in (("recon_all_options", ("threads", "hemisphere_workers", "profile_stages", "native_optimizations")),
                                      ("volume_options", ("registration_backend", "fnirt_config", "slice_timing", "reuse_anatomical"))):
                    record[label]["requested_parameters"][section] = {key: requested.get(section, {})[key] for key in keys if key in requested.get(section, {})}
            if spec.get("queue_report") and Path(spec["queue_report"]).is_file():
                payload = Path(spec["queue_report"]).read_bytes(); queue = json.loads(payload)
                queued = queue.get("cases", {}).get(case, {})
                record[label]["queue_snapshot_sha256"] = hashlib.sha256(payload).hexdigest()
                if "process_wall_seconds" in queued:
                    record[label]["queue_process_wall_seconds"] = numeric_tree(queued["process_wall_seconds"])
                    record[label]["queue_process_scope"] = "case Python child startup through exit; includes import/initialization and wrapper QC/report, excludes waiting before child startup"
                    record[label]["queue_process_wall_boundary"] = RUN_TIMING_BOUNDARIES["process_wall_seconds"]
        record["pair_status"] = "ready" if all(record[label]["status"] == "complete" for label in ("candidate", "reference")) else "pending_or_failed"
        if record["pair_status"] == "ready" and not metadata_only:
            manifest = {"cohort_id": cohort_id, "source_revision": config["source_revision"],
                "candidate_root": config["candidate_root"], "case_id": case, "frames": 180, "tr_seconds": raw["repetition_time_seconds"],
                "raw": {key: {"path": str(Path(config["raw_root"]) / raw[original]["relative_path"]), "sha256": raw[original]["sha256"]}
                    for key, original in (("t1w", "T1w"), ("bold", "BOLD"), ("bold_json", "BOLD_json"))},
                "candidate": {**config["candidate_cases"][case], "report_sha256": record["candidate"]["report_sha256"],
                              "files_sha256": sha256(config["candidate_cases"][case]["files"])},
                "reference": {**config["reference_cases"][case], "report_sha256": record["reference"]["report_sha256"],
                              "files_sha256": sha256(config["reference_cases"][case]["files"])},
                "brain_mask": {"path": str(Path(config["resources_root"]) / "fmriprep/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz"),
                               "sha256": resources["fmriprep/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz"]["sha256"]},
                "cifti_axis_assets": {key: {"path": str(Path(config["resources_root"]) / relative), "sha256": AXIS_ASSET_SHA256[key]}
                    for key, relative in (("left_roi", "global/templates/standard_mesh_atlases/L.atlasroi.32k_fs_LR.shape.gii"),
                        ("right_roi", "global/templates/standard_mesh_atlases/R.atlasroi.32k_fs_LR.shape.gii"),
                        ("dseg", "fmriprep/tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz"))}}
            manifest_path = output / (case + ".comparison_manifest.private.json")
            write_json(manifest_path, manifest)
            compare_output = output / "comparisons" / case
            command = [sys.executable, str(Path(__file__).with_name("compare_subject.py")), "--manifest", str(manifest_path), "--output-root", str(compare_output)]
            exit_code = None
            try:
                with (output / (case + ".comparison.private.log")).open("x") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                exit_code = process.returncode
                comparison = read_json(compare_output / "comparison.public.json")
                if comparison.get("case_id") != case or comparison.get("cohort_id") != cohort_id or comparison.get("status") not in ("complete", "failed"):
                    raise ValueError("comparison subprocess returned an invalid case identity/status")
                if comparison["status"] == "complete" and exit_code != 0:
                    raise ValueError("completed comparison contradicts failed subprocess exit")
            except Exception as error:
                (output / (case + ".comparison_dispatch.private.txt")).write_text(traceback.format_exc())
                compare_output.mkdir(parents=True, exist_ok=True)
                comparison = {"schema_version": 1, "cohort_id": cohort_id, "case_id": case, "status": "failed",
                              "failure": {"type": type(error).__name__, "scope": "independent comparison dispatch/output validation"}}
                failure_path = compare_output / "comparison_dispatch.public.json"
                write_json(failure_path, comparison)
            else:
                failure_path = compare_output / "comparison.public.json"
            record["comparison"] = {"status": comparison["status"], "exit_code": exit_code,
                "report_sha256": sha256(failure_path)}
            comparisons.append(comparison)
            comparison_entries.append({"case_id": case, "report": str(failure_path),
                                       "arrays": str(compare_output / "arrays.private.npz")})
        cases.append(record)
    historical_attempts = []
    for entry in config.get("earlier_attempts", []):
        case = entry["case_id"]
        if case not in CASES or entry["side"] not in ("candidate", "reference"):
            raise ValueError("historical attempt must retain its declared public case and side")
        record, _ = retained_run(entry, raw_cases[case], output, case, identifier(entry["attempt_id"]),
                                 reference=entry["side"] == "reference", inspect_reconstruction=False)
        try:
            record = historical_execution_status(entry, record)
        except Exception as error:
            (output / f"{case}.{identifier(entry['attempt_id'])}.execution_status_check.private.txt").write_text(traceback.format_exc())
            record = {**record, "historical_execution_status_integrity": "failed",
                      "historical_execution_status_failure_type": type(error).__name__}
        historical_attempts.append({"case_id": case, "side": entry["side"], "attempt_id": identifier(entry["attempt_id"]),
                                    "historical_report_only": True, "formal_benchmark_included": False, **record})
    figures = {"status": "not_requested"}
    if config.get("render") and comparison_entries:
        manifest = {**config["render"], "cohort_id": cohort_id, "comparisons": comparison_entries}
        render_manifest = output / "render_manifest.private.json"; write_json(render_manifest, manifest)
        try:
            with (output / "render.private.log").open("x") as log:
                rendered = subprocess.run([sys.executable, str(Path(__file__).with_name("render_cohort.py")), "--manifest", str(render_manifest),
                                "--output-root", str(output / "figures")], stdout=log, stderr=subprocess.STDOUT)
            figure_path = output / "figures/figures_provenance.public.json"
            figure_report = read_json(figure_path)
            figures = {"status": figure_report["status"], "exit_code": rendered.returncode,
                       "provenance_sha256": sha256(figure_path)}
        except Exception as error:
            (output / "render_dispatch.private.txt").write_text(traceback.format_exc())
            figures = {"status": "failed", "failure_type": type(error).__name__}
    successful = [row["case_id"] for row in cases if row.get("comparison", {}).get("status") == "complete"]
    if sha256(data_path) != data_digest or sha256(assets_path) != assets_digest:
        raise RuntimeError("approved data/resource manifests changed during collection")
    policy = config.get("runtime_environment_policy", {})
    return {"schema_version": 1, "status": "complete" if len(successful) == 10 else "partial",
            "cohort_id": cohort_id,
            "runtime_environment_policy_declared": {key: policy[key] for key in
                ("PYTORCH_NO_CUDA_MEMORY_CACHING", "cuda_allocator_cache", "scientific_source_unchanged", "programs_prepared_before_formal_start",
                 "parent_idle_cuda_cache_release", "memory_target_bytes", "threads", "hardware") if key in policy},
            "observed_at_utc": utc(), "collector_script_sha256": sha256(__file__),
            "dataset_manifest_sha256": data_digest, "assets_manifest_sha256": assets_digest,
            "data_license": "CC0", "source_revision_declared": config.get("source_revision"),
            "environment": hardware, "native_provenance": native, "cases": cases,
            "earlier_attempts_retained": historical_attempts, "figures": figures,
            "completed_comparisons": successful, "pending_or_failed_cases": [row["case_id"] for row in cases if row["case_id"] not in successful],
            "collection_seconds_excluded_from_pipeline_timings": time.perf_counter() - started,
            "timing_policy": "Keep FNIT API, driver-through-saved-validation, queue process, official container and saved-wrapper-QC, preflight, schema adaptation, comparison and rendering durations separately. Hardware, automatic ICA/AROMA and official internal MNI2009cAsym workloads differ; no global speedup claim",
            "publication": "Aggregate JSON and brain PNGs only; raw MRI, paths, licenses, commands and arrays.private.npz stay private"}


def analysis_tools_inventory():
    """Bind the actual imported helper and all independent executable tools."""
    source_dir = Path(__file__).resolve().parent
    imported_helper = Path(comparison_module.__file__).resolve()
    if imported_helper != source_dir / "compare_subject.py":
        raise ValueError("comparison helper was imported outside this analysis deployment")
    return {name: sha256(source_dir / name) for name in
            ("collect_cohort.py", "compare_subject.py", "render_cohort.py", "prepare_display_geometry.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true", help="只记录来源和当前状态，不运行比较/绘图")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    tools_before = None
    try:
        tools_before = analysis_tools_inventory()
        config, digest = read_bound_json(args.config)
        report = collect(config, args.output_root, metadata_only=args.metadata_only)
        if sha256(args.config) != digest:
            raise RuntimeError("private collection config changed during collection")
        tools_after = analysis_tools_inventory()
        if tools_after != tools_before:
            raise RuntimeError("independent analysis tools changed during collection")
        report["private_config_sha256"] = digest
        report["analysis_tools_sha256_before"] = tools_before
        report["analysis_tools_sha256_after"] = tools_after
        report["analysis_tools_unchanged"] = True
    except Exception as error:
        (args.output_root / "failure.private.txt").write_text(traceback.format_exc())
        report = {"schema_version": 1, "status": "failed", "collector_script_sha256": sha256(__file__),
                  "analysis_tools_sha256_before": tools_before,
                  "failure": {"type": type(error).__name__, "details": "failure.private.txt"}}
    write_json(args.output_root / "cohort.public.json", report)
    print(json.dumps({"status": report["status"], "completed_comparisons": report.get("completed_comparisons", []),
                      "collector_script_sha256": report["collector_script_sha256"]}), flush=True)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
