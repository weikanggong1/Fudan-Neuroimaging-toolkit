"""启动候选命令，采样指定 GPU 上同一查询中父子进程的合计显存。

--gpu-uuid 为物理 GPU UUID；--output 为新诊断目录；--interval 默认 2 s，
--query-timeout 默认 5 s。-- 后为完整具名命令。输出 command.log、
gpu_samples.csv、monitor.json，记录查询耗时、缺失样本和命令全程墙钟。
NVML compute-apps 查询中的父子占用相加；whole-GPU 查询另有自己的时间，
不宣称两个查询同时。超时写缺失状态，不填零；本脚本不改变 CUDA 可见设备、
线程或 allocator，均由命令环境显式指定。仅周期采样，无连续峰值保证。
属于 benchmark 包装器，没有独立官方等价命令。失败返回被测命令退出码。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time


def descendants(pid: int) -> set[int]:
    """读取 /proc 的线程子进程表，返回当次可见的递归后代 PID。"""
    found = set()
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = set()
            for task in (Path("/proc") / str(current) / "task").iterdir():
                try:
                    children.update(int(p) for p in (task / "children").read_text().split())
                except (OSError, ValueError):
                    pass
        except OSError:
            continue
        for child in children - found:
            found.add(child)
            pending.append(child)
    return found


def query(arguments: list[str], timeout: float) -> tuple[str | None, float, str]:
    """有界运行 NVML CLI，返回文本/秒数/状态；失败不推断显存为零。"""
    tick = time.perf_counter()
    try:
        result = subprocess.run(["nvidia-smi", *arguments], capture_output=True,
                                text=True, timeout=timeout, check=True)
        return result.stdout, time.perf_counter() - tick, "ok"
    except (subprocess.SubprocessError, OSError) as error:
        return None, time.perf_counter() - tick, type(error).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--query-timeout", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0 or args.query_timeout <= 0:
        raise ValueError("a command and positive sampling intervals are required")
    args.output.mkdir(parents=True, exist_ok=False)
    samples = []
    stop = threading.Event()
    start = time.perf_counter()
    with (args.output / "command.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def monitor():
            with (args.output / "gpu_samples.csv").open("w", newline="") as stream:
                fields = ["time_utc", "monotonic_seconds", "apps_query_seconds", "apps_status",
                          "parent_bytes", "children_bytes", "total_process_bytes",
                          "gpu_query_seconds", "gpu_status", "gpu_total_mib", "gpu_utilization_percent"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                while not stop.is_set():
                    pids = descendants(process.pid)
                    apps, seconds, status = query(
                        ["--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                         "--format=csv,noheader,nounits"], args.query_timeout)
                    row = {"time_utc": datetime.now(timezone.utc).isoformat(),
                           "monotonic_seconds": time.perf_counter() - start,
                           "apps_query_seconds": seconds, "apps_status": status}
                    if apps is not None:
                        parent = children = 0
                        valid = True
                        for line in apps.splitlines():
                            parts = [item.strip() for item in line.split(",")]
                            if len(parts) != 3 or parts[1] != args.gpu_uuid:
                                continue
                            try:
                                pid, size = int(parts[0]), int(parts[2]) * 1048576
                            except ValueError:
                                valid = False
                                break
                            if pid == process.pid:
                                parent += size
                            elif pid in pids:
                                children += size
                        if valid:
                            row.update(parent_bytes=parent, children_bytes=children,
                                       total_process_bytes=parent + children)
                        else:
                            row["apps_status"] = "unparseable"
                    gpu, seconds, status = query(
                        ["--id=" + args.gpu_uuid, "--query-gpu=memory.used,utilization.gpu",
                         "--format=csv,noheader,nounits"], args.query_timeout)
                    row.update(gpu_query_seconds=seconds, gpu_status=status)
                    if gpu is not None:
                        parts = [p.strip() for p in gpu.strip().split(",")]
                        if len(parts) == 2:
                            row.update(gpu_total_mib=parts[0], gpu_utilization_percent=parts[1])
                    writer.writerow(row)
                    stream.flush()
                    samples.append(row)
                    stop.wait(args.interval)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        status = process.wait()
        command_wall = time.perf_counter() - start
        stop.set()
        thread.join(timeout=2 * args.query_timeout + 1)
    times = [row["monotonic_seconds"] for row in samples]
    gaps = [b - a for a, b in zip(times, times[1:])]
    report = {"command": command, "pid": process.pid, "gpu_uuid": args.gpu_uuid,
              "exit_code": status, "command_wall_seconds": command_wall,
              "sampling_interval_requested_seconds": args.interval,
              "query_timeout_seconds": args.query_timeout, "samples": len(samples),
              "failed_app_queries": sum(row["apps_status"] != "ok" for row in samples),
              "maximum_sampling_gap_seconds": max(gaps, default=None),
              "peak_sampled_process_bytes": max((row["total_process_bytes"] for row in samples
                                                  if "total_process_bytes" in row), default=None),
              "continuous_peak_verified": False,
              "monitor_thread_finished": not thread.is_alive(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "monitor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
