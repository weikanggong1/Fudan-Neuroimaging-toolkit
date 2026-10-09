"""运行具名 placement benchmark，并同步采样目标物理 GPU 与子进程显存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def load_sampler_module(profiling_source):
    """复用生产显存归属修复，并返回实际加载模块用于SHA绑定。"""
    if profiling_source is None:
        from fnit.recon_all import profiling
        return profiling
    import importlib.util
    spec = importlib.util.spec_from_file_location("placement_benchmark_profiling", profiling_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query(arguments):
    result = subprocess.run(["nvidia-smi", *arguments], check=True,
                            text=True, capture_output=True)
    return [line.strip().split(", ") for line in result.stdout.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    parser.add_argument("--profiling-source", type=Path, help="显式冻结profiling.py；未指定时使用已安装FNIT")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval_seconds <= 0 or args.output.exists():
        raise ValueError("需要非空命令、正采样间隔和不存在的报告路径")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_monotonic = time.monotonic()
    gpu = query([f"--id={args.physical_gpu}", "--query-gpu=index,uuid,name",
                 "--format=csv,noheader,nounits"])[0]
    profiling = load_sampler_module(args.profiling_source)
    process = subprocess.Popen(command)
    sampler = profiling.ProcessTreeDeviceSampler(
        device="cuda:0", parent_pid=process.pid, interval=args.interval_seconds)
    # 外部监测器不建立CUDA context；UUID来自明确物理卡查询。
    sampler.uuid = gpu[1]
    report = {
        "scope": "placement_stage_external_process_memory_not_whole_recon_all",
        "hostname": platform.node(), "physical_gpu_index": args.physical_gpu,
        "gpu_uuid": gpu[1], "gpu_name": gpu[2], "benchmark_pid": process.pid,
        "command": command, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_interval_seconds": args.interval_seconds,
        "precision": "controlled and recorded by wrapped benchmark; no override here",
        "units": "nvidia-smi integer MiB; byte values multiply by 1048576",
        "peak_interpretation": "sampled simultaneous occupancy; unsampled peaks may be higher",
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
            "PYTORCH_NO_CUDA_MEMORY_CACHING", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "profiling_sha256": hashlib.sha256(Path(profiling.__file__).read_bytes()).hexdigest(),
        "samples": [], "sampling_errors": [],
    }
    while process.poll() is None:
        tick = time.perf_counter()
        sampler.sample_if_due(force=True)
        time.sleep(max(0, args.interval_seconds - (time.perf_counter() - tick)))
    sampled = sampler.report()
    report["shared_process_sampler"] = sampled
    report["sampling_errors"] = sampled["failed_samples"]
    report["samples"] = [{
        "seconds": row["monotonic"] - started_monotonic,
        "device_used_mib": row["target_device_used_bytes"] / 1048576
                           if row["target_device_used_bytes"] is not None else None,
        "device_compute_process_used_mib": row["target_compute_process_sum_bytes"] / 1048576,
        "benchmark_tree_used_mib": row["tree_total_bytes"] / 1048576
                                  if row["tree_total_bytes"] is not None else None,
        "process_ownership": row["ownership"],
        "benchmark_tree_gpu_processes": row["processes"],
        "sampling_seconds": row["sample_query_seconds"],
    } for row in sampled["samples"]]
    report["returncode"] = process.wait()
    report["wall_seconds_including_monitor_setup_and_sampling"] = time.perf_counter() - started
    for field in ("device_used_mib", "device_compute_process_used_mib", "benchmark_tree_used_mib"):
        values = [row[field] for row in report["samples"] if row[field] is not None]
        report[f"sampled_peak_{field}"] = max(values, default=None)
    if not report["samples"]:
        report["process_tree_memory_status"] = "unavailable; no completed memory samples"
    elif any(row["benchmark_tree_used_mib"] is None for row in report["samples"]):
        report["sampled_peak_benchmark_tree_used_mib"] = None
        report["process_tree_memory_status"] = "incomplete_pid_attribution; use device occupancy as upper bound"
    else:
        report["process_tree_memory_status"] = "mapped_for_all_samples"
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(report["returncode"])


if __name__ == "__main__":
    main()
