"""Execute explicitly authorized full-input GPU requests under one GPU lock.

Requests contain private paths and the coordinator's assigned affinity. This
program reports process and allocator measurements; it never infers accuracy
from a successful exit or launches itself during benchmark preparation.
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import threading
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def descendants(parent):
    parents = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rpartition(") ")[2].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                pass
    owned = {parent}
    while True:
        expanded = owned | {pid for pid, ppid in parents.items() if ppid in owned}
        if expanded == owned:
            return owned
        owned = expanded


def monitor(child, uuid, stopped, samples, start, violation):
    while not stopped.is_set():
        row = {"seconds": time.perf_counter() - start, "memory_bytes": None}
        try:
            owned = descendants(child.pid)
            query = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            memory, matched, unavailable = 0, [], query.returncode != 0
            for line in query.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3:
                    continue
                pid = int(fields[1])
                if pid in owned:
                    if fields[0] != uuid:
                        violation.set()
                        row["wrong_owned_gpu_uuid"] = fields[0]
                        continue
                    matched.append(pid)
                    try:
                        memory += int(fields[2]) * 1024**2
                    except ValueError:
                        unavailable = True
            row["owned_gpu_pids"] = matched
            if matched and not unavailable:
                row["memory_bytes"] = memory
                if memory > 20_000_000_000:
                    violation.set()
            activity = subprocess.run(
                ["nvidia-smi", "-i", uuid, "--query-gpu=uuid,utilization.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            fields = [field.strip() for field in activity.stdout.strip().split(",")]
            if activity.returncode == 0 and len(fields) == 2 and fields[0] == uuid:
                row["utilization_percent"] = float(fields[1])
        except (OSError, ValueError, subprocess.TimeoutExpired):
            row["query_failed"] = True
        samples.append(row)
        if violation.is_set() and child.poll() is None:
            # Each request owns its new process group; other jobs are untouched.
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stopped.wait(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.requests.read_text())
    frozen = config.get("frozen_sha256")
    if not frozen:
        raise ValueError("Prepared requests must declare frozen source and adapter hashes")
    def verify_frozen():
        for filename, expected in frozen.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Actual frozen source or benchmark adapter differs")
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipts = []
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verify_frozen()
        for request in config["requests"]:
            name = request["name"]
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ValueError("Request name must be a single nonempty path component")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                       PYTHONPATH=request["source_path"], OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS"):
                env[key] = str(request["threads"])
            command = ["taskset", "-c", ",".join(map(str, request["cpus"])),
                       config["python"], config["adapter"], *request["arguments"],
                       "--device", "cuda:0", "--threads", str(request["threads"]),
                       "--gpu-memory-budget-bytes", "20000000000",
                       "--output-dir", str(args.output_dir / name)]
            stopped, violation, samples = threading.Event(), threading.Event(), []
            start = time.perf_counter()
            with (args.output_dir / (name + ".private.log")).open("wb") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                sampler = threading.Thread(target=monitor,
                    args=(child, args.gpu_uuid, stopped, samples, start, violation), daemon=True)
                sampler.start()
                code = child.wait()
                process_seconds = time.perf_counter() - start
                stopped.set()
                sampler.join(timeout=11)
            write_json(args.output_dir / (name + ".samples.private.json"), samples)
            observed = [row["memory_bytes"] for row in samples if row["memory_bytes"] is not None]
            utilization = [row["utilization_percent"] for row in samples if "utilization_percent" in row]
            gaps = [b["seconds"] - a["seconds"] for a, b in zip(samples, samples[1:])]
            peak = max(observed) if observed else None
            receipts.append({"name": name, "returncode": code, "process_seconds": process_seconds,
                "simultaneous_process_tree_peak_bytes": peak,
                "maximum_sampling_gap_seconds": max(gaps) if gaps else None,
                "sample_count": len(samples), "utilization_percent_range":
                    [min(utilization), max(utilization)] if utilization else None,
                "resource_violation": violation.is_set(), "physical_gpu_uuid": args.gpu_uuid})
            write_json(args.output_dir / "execution.public.json", {"records": receipts})
            if code:
                raise RuntimeError("Complete GPU request failed; preserve this attempt")
            if violation.is_set():
                raise RuntimeError("Owned request exceeded its GPU budget or used the wrong GPU")
            verify_frozen()
            report = json.loads((args.output_dir / name / "report.public.json").read_text())
            if max(report["peak_cuda_allocated_bytes"], report["peak_cuda_reserved_bytes"],
                   peak or 0) > 20_000_000_000:
                raise RuntimeError("Observed GPU memory exceeds the approved budget")


if __name__ == "__main__":
    main()
