"""Measure driver memory of one benchmark process, including CUDA context."""

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter, sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("supply a benchmark command and a positive interval")
    start = perf_counter()
    child = subprocess.Popen(command)
    peak_mib = samples = 0
    exceeded = False
    devices = set()
    other_memory_samples = []
    while child.poll() is None:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        usage = {}
        for row in result.stdout.splitlines():
            uuid, pid, memory = row.split(",")
            uuid = uuid.strip()
            usage.setdefault(uuid, {})[int(pid)] = int(memory)
            if int(pid) == child.pid:
                devices.add(uuid)
                peak_mib = max(peak_mib, int(memory))
                samples += 1
        if devices:
            other_memory_samples.append(sum(
                memory for uuid in devices for pid, memory in usage.get(uuid, {}).items()
                if pid != child.pid)*1024**2)
        exceeded = peak_mib*1024**2 >= args.memory_limit_bytes
        if exceeded or len(devices) > 1:
            child.terminate()
        args.output_json.write_text(json.dumps({
            "sampling_interval_seconds": args.interval, "samples": samples,
            "peak_driver_memory_bytes": peak_mib*1024**2,
            "observed_gpu_count": len(devices),
            "other_process_memory_bytes": {
                "minimum": min(other_memory_samples, default=0),
                "mean": sum(other_memory_samples)/max(1, len(other_memory_samples)),
                "maximum": max(other_memory_samples, default=0)},
            "memory_limit_exceeded": exceeded,
            "monitor_wall_seconds": perf_counter()-start})+"\n")
        if exceeded or len(devices) > 1:
            break
        sleep(args.interval)
    code = child.wait()
    raise SystemExit(1 if exceeded or len(devices) > 1 else code)


if __name__ == "__main__":
    main()
