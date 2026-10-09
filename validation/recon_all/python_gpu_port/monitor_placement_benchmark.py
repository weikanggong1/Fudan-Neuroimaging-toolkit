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


def descendants(process_id):
    """返回仍存活的本次 benchmark 进程树，不包含其他人的任务。"""
    found, pending = set(), [process_id]
    while pending:
        process = pending.pop()
        if process in found:
            continue
        found.add(process)
        try:
            pending.extend(int(value) for value in Path(
                f"/proc/{process}/task/{process}/children").read_text().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return found


def query(arguments):
    result = subprocess.run(["nvidia-smi", *arguments], check=True,
                            text=True, capture_output=True)
    return [line.strip().split(", ") for line in result.stdout.splitlines() if line.strip()]


def namespace_aliases(family):
    """读取本次进程的公开PID映射；旧内核可能不提供NSpid。"""
    aliases = set(family)
    for process in family:
        try:
            for line in Path(f"/proc/{process}/status").read_text().splitlines():
                if line.startswith("NSpid:"):
                    aliases.update(int(value) for value in line.split()[1:])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval_seconds <= 0 or args.output.exists():
        raise ValueError("需要非空命令、正采样间隔和不存在的报告路径")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gpu = query([f"--id={args.physical_gpu}", "--query-gpu=index,uuid,name",
                 "--format=csv,noheader,nounits"])[0]
    process = subprocess.Popen(command)
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
        "samples": [], "sampling_errors": [],
    }
    while process.poll() is None:
        tick = time.perf_counter()
        try:
            family = namespace_aliases(descendants(process.pid))
            processes = query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                               "--format=csv,noheader,nounits"])
            device_mib = int(query([f"--id={args.physical_gpu}", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"])[0][0])
            target = [row for row in processes if row[0] == gpu[1]]
            local = [row for row in target if int(row[1]) in family]
            unresolved = len(local) != len(target) or (not target and device_mib > 4)
            report["samples"].append({
                "seconds": tick - started, "device_used_mib": device_mib,
                "device_compute_process_used_mib": sum(int(row[2]) for row in target),
                "benchmark_tree_used_mib": None if unresolved else sum(int(row[2]) for row in local),
                "process_ownership": "unresolved_driver_pid_namespace_or_other_process"
                                     if unresolved else "all_target_compute_pids_mapped_to_benchmark",
                "benchmark_tree_gpu_processes": [{"pid": int(row[1]), "used_mib": int(row[2])}
                                                 for row in local],
                "sampling_seconds": time.perf_counter() - tick,
            })
        except (subprocess.CalledProcessError, ValueError, IndexError) as error:
            report["sampling_errors"].append({"seconds": tick - started, "error": str(error)})
        time.sleep(max(0, args.interval_seconds - (time.perf_counter() - tick)))
    report["returncode"] = process.wait()
    report["wall_seconds_including_monitor_setup_and_sampling"] = time.perf_counter() - started
    for field in ("device_used_mib", "device_compute_process_used_mib", "benchmark_tree_used_mib"):
        values = [row[field] for row in report["samples"] if row[field] is not None]
        report[f"sampled_peak_{field}"] = max(values, default=None)
    if any(row["benchmark_tree_used_mib"] is None for row in report["samples"]):
        report["sampled_peak_benchmark_tree_used_mib"] = None
        report["process_tree_memory_status"] = "incomplete_pid_attribution; use device occupancy as upper bound"
    else:
        report["process_tree_memory_status"] = "mapped_for_all_samples"
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(report["returncode"])


if __name__ == "__main__":
    main()
