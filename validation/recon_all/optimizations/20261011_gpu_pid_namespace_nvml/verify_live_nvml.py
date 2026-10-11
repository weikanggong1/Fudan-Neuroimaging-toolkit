"""临时安装声明的 NVML wheel 并查询物理 GPU；不创建 CUDA/MRI 工作负载。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _category(error: BaseException) -> str:
    """安全报告只保存异常类型，不保存可能含机器路径的消息。"""
    return type(error).__name__


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(*, source_directory: Path, source_manifest_sha256: str, wheel_path: Path,
        installation_directory: Path, output_directory: Path, physical_device: int = 0) -> dict:
    """冻结模块、清单、已有 wheel、新安装目标、新报告目录、物理卡号六个具名参数。"""
    started = time.perf_counter()
    source, wheel = source_directory.resolve(), wheel_path.resolve()
    target, output = installation_directory.resolve(), output_directory.resolve()
    target_negative_cache_at_entry = (str(target) in sys.path_importer_cache
                                     and sys.path_importer_cache[str(target)] is None)
    manifest_path = source / "SOURCE_MANIFEST.json"
    if digest(manifest_path) != source_manifest_sha256:
        raise ValueError("source manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    names = {"modules/profiling.py", "modules/nvml_process_memory.py", "verify_live_nvml.py"}
    if set(manifest["payloads"]) != names:
        raise ValueError("exact two modules and verifier required")
    for name, expected in manifest["payloads"].items():
        actual = source / name
        if digest(actual) != expected["sha256"] or actual.stat().st_size != expected["bytes"]:
            raise ValueError("source payload mismatch")
    wheel_binding = {"sha256": digest(wheel), "bytes": wheel.stat().st_size}
    if wheel_binding != manifest["dependency_wheel"]:
        raise ValueError("dependency wheel mismatch")
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith("/METADATA"))).decode()
    if "Name: nvidia-ml-py\n" not in metadata or "Version: 13.580.82\n" not in metadata:
        raise ValueError("declared NVML version required")
    if type(physical_device) is not int or physical_device < 0:
        raise ValueError("physical_device must be a nonnegative integer")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_device):
        raise ValueError("launch must expose exactly the declared physical device")
    if target.exists() or output.exists() or target == output or output.is_relative_to(source):
        raise ValueError("fresh independent target and output required")
    environment_at_entry = dict(os.environ)
    target.mkdir(mode=0o700, parents=True)
    output.mkdir(mode=0o700, parents=True)
    install_started = time.perf_counter()
    installed_process = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile",
         "--target", str(target), str(wheel)], capture_output=True, text=True, check=False)
    installation_seconds = time.perf_counter() - install_started
    if installed_process.returncode:
        _save(output / "install_failure.public.json", {
            "status": "failed", "phase": "offline_wheel_install", "exit_code": installed_process.returncode,
            "dependency_wheel": wheel_binding, "source_manifest_sha256": source_manifest_sha256})
        raise RuntimeError("offline NVML installation failed; typed receipt retained")
    package = target / "fnit_live_nvml_probe"
    package.mkdir(mode=0o700)
    (package / "__init__.py").write_text('"""Isolated sampler query namespace."""\n')
    installed_modules = {}
    for name in ("profiling.py", "nvml_process_memory.py"):
        destination = package / name
        destination.write_bytes((source / "modules" / name).read_bytes())
        installed_modules[name] = {"sha256": digest(destination), "bytes": destination.stat().st_size}
        if installed_modules[name] != manifest["payloads"]["modules/" + name]:
            raise ValueError("installed module mismatch")
    sys.path.insert(0, str(target))
    target_negative_cache_before_invalidation = (str(target) in sys.path_importer_cache
                                                 and sys.path_importer_cache[str(target)] is None)
    # PYTHONPATH 可在目录创建前被 Python 查询并缓存为不存在；创建后复核。
    importlib.invalidate_caches()
    import_started = time.perf_counter()
    import torch
    module = importlib.import_module("fnit_live_nvml_probe.profiling")
    import_seconds = time.perf_counter() - import_started
    if Path(module.__file__).resolve() != package / "profiling.py":
        raise ValueError("sampler imported outside isolated target")

    def controls():
        return {"cuda_initialized": bool(torch.cuda.is_initialized()),
                "default_dtype": str(torch.get_default_dtype()),
                "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cuda_autocast": module.autocast_state("cuda"),
                "cpu_autocast": module.autocast_state("cpu"),
                "torch_intraop_threads": int(torch.get_num_threads()),
                "torch_interop_threads": int(torch.get_num_interop_threads())}

    before = controls()
    sampler_class = module.ProcessTreeDeviceSampler
    own_pid = os.getpid()
    identity = sampler_class._process_identity(own_pid)
    cpu = sampler_class(device="cpu", parent_pid=own_pid, query_backend="auto")
    cpu.sample_if_due(force=True)
    cpu_report = cpu.report()
    cpu_created_nvml = cpu._nvml is not None
    mapping = cpu._pid_mapping({own_pid})
    cpu.close()
    gpu = sampler_class(device="cuda:0", parent_pid=own_pid, query_backend="nvml")
    gpu_query_started = time.perf_counter()
    gpu.sample_if_due(force=True)
    gpu_query_finished = time.perf_counter()
    raw_gpu = gpu.report()
    selected_device_confirmed = False
    if gpu._nvml is not None and gpu.uuid is not None:
        # UUID 仅在进程内确认卡号，不进入报告、日志或返回值。
        selected_device_confirmed = gpu._nvml.devices().get(str(physical_device)) == gpu.uuid
    gpu.close()
    gpu_report = {key: raw_gpu[key] for key in (
        "status", "query_backend_requested", "query_backend", "peak_tree_total_bytes",
        "observed_resolved_tree_peak_bytes", "peak_target_compute_process_sum_bytes",
        "peak_target_device_used_bytes", "scope")}
    gpu_report.update({"physical_device": physical_device, "logical_device": "cuda:0",
                       "physical_device_confirmed": selected_device_confirmed,
                       "sample_count": len(raw_gpu["samples"]),
                       "failed_sample_count": len(raw_gpu["failed_samples"]),
                       "backend_error_count": len(raw_gpu["query_backend_errors"]),
                       "failed_sample_categories": sorted({str(r["error"]).split("(", 1)[0]
                                                           for r in raw_gpu["failed_samples"]}),
                       "samples": [{key: row[key] for key in (
                           "ownership", "tree_total_bytes", "known_tree_bytes",
                           "target_compute_process_sum_bytes", "target_device_used_bytes",
                           "sample_query_seconds")} | {
                               "compute_process_count": len(row["processes"]) + len(row["external_processes"])
                                                        + len(row["unresolved_processes"]),
                               "resolved_process_count": len(row["processes"]),
                               "unresolved_process_count": len(row["unresolved_processes"]),
                               "device_query_succeeded": row["device_query_error"] is None,
                               "mapping_status": row["pid_mapping"]["status"],
                               "mapping_aliases_count": len(row["pid_mapping"]["aliases"]),
                               "mapping_unavailable_reasons": row["pid_mapping"]["unavailable_reasons"]}
                           for row in raw_gpu["samples"]]})
    smi_started = time.perf_counter()
    smi = {"status": "failed", "physical_device": physical_device,
           "device_used_bytes": None, "raw_memory_mib": None, "error_category": None}
    try:
        result = subprocess.run(["nvidia-smi", "--id=" + str(physical_device),
            "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        rows = [line.split(",") for line in result.stdout.strip().splitlines() if line.strip()]
        if len(rows) != 1 or int(rows[0][0]) != physical_device:
            raise ValueError("SMI physical device mismatch")
        memory = int(rows[0][1])
        if memory < 0:
            raise ValueError("SMI memory unavailable")
        smi.update(status="passed", raw_memory_mib=memory, device_used_bytes=memory * 1024 * 1024)
    except Exception as error:
        smi["error_category"] = _category(error)
    smi_finished = time.perf_counter()
    after = controls()
    changed_environment_keys = sorted(key for key in set(environment_at_entry) | set(os.environ)
        if environment_at_entry.get(key) != os.environ.get(key))
    dependency_file = target / "pynvml.py"
    loaded_dependency = importlib.import_module("pynvml")
    dependency_loaded_from_target = Path(loaded_dependency.__file__).resolve() == dependency_file
    nvml_success = (gpu_report["query_backend"] == "nvml" and gpu_report["sample_count"] == 1
                    and gpu_report["failed_sample_count"] == 0 and selected_device_confirmed
                    and gpu_report["samples"][0]["device_query_succeeded"])
    checks = {"installed_source_bytes_equal": True,
              "declared_nvml_version": importlib.metadata.version("nvidia-ml-py") == "13.580.82",
              "dependency_loaded_from_isolated_target": dependency_loaded_from_target,
              "no_cuda_initialization": not before["cuda_initialized"] and not after["cuda_initialized"],
              "controls_unchanged": before == after, "environment_unchanged": not changed_environment_keys,
              "cpu_lazy_without_nvml": cpu_report["query_backend"] is None and not cpu_created_nvml,
              "no_unproven_pid_aliases": mapping["aliases"] == {},
              "live_nvml_query_succeeded": nvml_success, "near_time_smi_query_succeeded": smi["status"] == "passed"}
    report = {"schema": "fnit.gpu_pid_namespace.live_nvml.v1",
              "status": "passed" if all(checks.values()) else "failed",
              "source_manifest_sha256": source_manifest_sha256,
              "git_head_at_bundle": manifest["git_head_at_bundle"],
              "module_source_commit": manifest["module_source_commit"],
              "installed_modules": installed_modules, "dependency_wheel": wheel_binding,
              "installed_dependency": {"version": importlib.metadata.version("nvidia-ml-py"),
                                       "pynvml_py_sha256": digest(dependency_file),
                                       "pynvml_py_bytes": dependency_file.stat().st_size},
              "import_target_negative_cache": {"at_entry": target_negative_cache_at_entry,
                  "before_invalidation": target_negative_cache_before_invalidation,
                  "caches_invalidated_after_installation": True},
              "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                           "kernel": platform.release()},
              "process_identity_availability": {
                  "nspid_available": identity["nspid_chain"] is not None,
                  "nspid_chain_length": len(identity["nspid_chain"] or []),
                  "namespace_available": identity["pid_namespace"] is not None,
                  "starttime_available": identity["starttime_ticks"] is not None,
                  "identity_error_category": identity["identity_error"]},
              "pid_mapping": {key: mapping[key] for key in ("status", "source", "unavailable_reasons")},
              "controls_before": before, "controls_after": after,
              "changed_environment_keys": changed_environment_keys,
              "cpu_sampler": {key: cpu_report[key] for key in ("status", "query_backend",
                              "peak_tree_total_bytes", "peak_target_device_used_bytes")},
              "nvml_gpu_query": gpu_report, "near_time_nvidia_smi_query": smi,
              "query_timing_seconds": {"nvml_started_after_probe_start": gpu_query_started-started,
                  "nvml_query_wall": gpu_query_finished-gpu_query_started,
                  "smi_started_after_probe_start": smi_started-started,
                  "smi_query_wall": smi_finished-smi_started,
                  "smi_start_minus_nvml_finish": smi_started-gpu_query_finished},
              "checks": checks,
              "seconds": {"offline_wheel_installation": installation_seconds,
                          "torch_and_actual_module_import": import_seconds,
                          "complete_probe": time.perf_counter()-started},
              "new_cuda_workload": False, "new_mri_computation": False,
              "old_environment_prefix_modified": False, "frozen_raw_sources_modified": False,
              "gpu_uuid_exported": False, "cmdline_read": False, "private_logs_read": False,
              "trusted_host_pid_mapping_supplied": False,
              "scope": "isolated declared wheel plus two installed modules; actual live NVML and near-time SMI observations; sequential queries differ in time, no exact FNIT memory ownership, no MRI/whole/OS-isolation validation"}
    path = output / "report.public.json"
    _save(path, report)
    completion = {"schema": "fnit.gpu_pid_namespace.live_nvml_completion.v1", "status": report["status"],
                  "report_file": path.name, "report_sha256": digest(path),
                  "source_manifest_sha256": source_manifest_sha256}
    _save(output / "COMPLETE.public.json", completion)
    if report["status"] != "passed":
        raise RuntimeError("live query gate failed; safe report retained")
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source-directory", "wheel-path", "installation-directory", "output-directory"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--physical-device", type=int, default=0)
    arguments = vars(parser.parse_args())
    entry_time = datetime.now(timezone.utc).isoformat()
    output = arguments["output_directory"].resolve()
    entry = {"schema": "fnit.gpu_pid_namespace.live_nvml_launch.v1",
             "status": "started", "entry_started_utc": entry_time,
             "verifier_sha256": digest(Path(__file__)),
             "source_manifest_sha256": arguments["source_manifest_sha256"],
             "physical_device": arguments["physical_device"],
             "cpu_affinity": sorted(os.sched_getaffinity(0)),
             "requested_cpu_threads": 4, "query_only": True}
    exit_code, completion = 0, None
    try:
        completion = run(**arguments)
    except Exception as error:
        exit_code = 1
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        import traceback
        _save(output / "failure.public.json", {
            "schema": "fnit.gpu_pid_namespace.live_nvml_failure.v2", "status": "failed",
            "exit_code": exit_code, "error_category": _category(error),
            "verifier_traceback_lines": [frame.lineno for frame in traceback.extract_tb(error.__traceback__)
                                          if Path(frame.filename).name == Path(__file__).name],
            "verifier_sha256": entry["verifier_sha256"],
            "source_manifest_sha256": arguments["source_manifest_sha256"]})
    entry["entry_finished_utc"] = datetime.now(timezone.utc).isoformat()
    _save(output / "launch.public.json", entry)
    report_path, complete_path = output / "report.public.json", output / "COMPLETE.public.json"
    _save(output / "EXIT.public.json", {
        "schema": "fnit.gpu_pid_namespace.live_nvml_process.v2",
        "status": "complete" if exit_code == 0 else "failed", "exit_code": exit_code,
        "verifier_sha256": entry["verifier_sha256"],
        "launch_sha256": digest(output / "launch.public.json"),
        "source_manifest_sha256": arguments["source_manifest_sha256"],
        "report_sha256": digest(report_path) if report_path.exists() else None,
        "completion_sha256": digest(complete_path) if complete_path.exists() else None})
    print(json.dumps({"exit_code": exit_code, "completion": completion}, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
