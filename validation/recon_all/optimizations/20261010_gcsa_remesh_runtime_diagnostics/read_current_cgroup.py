"""读取当前v1/v2 CPU控制器的配额与统计；不推断过去整例的限流原因。

--output为新JSON路径，--interval-seconds默认1秒（0.1至5秒）。两次快照
只测诊断当时的变化；所有旧benchmark、生产与cgroup配置只读。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def snapshot(*, root: Path) -> dict:
    paths = [root / name for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective")]
    paths += [root / "cpu,cpuacct" / name for name in
              ("cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us", "cpuacct.usage")]
    paths += [root / "cpuset" / name for name in ("cpuset.cpus", "cpuset.effective_cpus")]
    result = {}
    for path in paths:
        try:
            result[str(path.relative_to(root))] = path.read_text().strip()
        except OSError as error:
            result[str(path.relative_to(root))] = {"unavailable": type(error).__name__}
    return {"monotonic": time.perf_counter(), "wall_unix_seconds": time.time(), "files": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists() or not 0.1 <= args.interval_seconds <= 5:
        raise ValueError("new output and interval 0.1..5 seconds required")
    root = Path("/sys/fs/cgroup")
    first = snapshot(root=root)
    time.sleep(args.interval_seconds)
    second = snapshot(root=root)
    values = second["files"]
    quota = values.get("cpu,cpuacct/cpu.cfs_quota_us")
    period = values.get("cpu,cpuacct/cpu.cfs_period_us")
    budget = int(quota)/int(period) if isinstance(quota,str) and isinstance(period,str) and int(quota)>0 else None
    counters = {}
    for label, row in (("first", first), ("second", second)):
        text = row["files"].get("cpu,cpuacct/cpu.stat")
        if isinstance(text, str):
            counters[label] = {key:int(value) for key,value in (line.split() for line in text.splitlines())}
    delta = {key:counters["second"][key]-counters["first"][key] for key in counters.get("first",{})} if len(counters)==2 else None
    report = {"scope":"post-run diagnostic snapshot; not historical run telemetry", "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshots":[first,second], "requested_interval_seconds":args.interval_seconds,
        "actual_interval_seconds":second["monotonic"]-first["monotonic"], "v1_quota_cpu_equivalents":budget,
        "current_window_cpu_stat_delta":delta,
        "cpu_stat_throttled_time_unit":"nanoseconds for cgroup v1",
        "self_cgroup_descriptor_sha256":hashlib.sha256(Path('/proc/self/cgroup').read_bytes()).hexdigest(),
        "historical_causal_attribution":"unavailable_without_time-aligned_previous_snapshots"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:report[key]for key in ("v1_quota_cpu_equivalents","actual_interval_seconds","current_window_cpu_stat_delta")}),flush=True)


if __name__ == "__main__":
    main()
