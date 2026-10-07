"""Dispatch complete original-T1 arms with a common lock held per fresh process."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

UUID = "GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_tree(pid):
    pending, found = [pid], set()
    while pending and len(found) < 256:
        item = pending.pop()
        if item in found:
            continue
        found.add(item)
        try:
            pending.extend(int(value) for value in Path(f"/proc/{item}/task/{item}/children").read_text().split())
        except (OSError, ValueError):
            pass
    return found


def sample(pid):
    # No other processes' IDs or commands are returned or persisted.
    pids = current_tree(pid)
    result = {"monotonic": time.monotonic(), "own_process_count": None, "own_process_bytes": None}
    try:
        process = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
        if process.returncode:
            result["process_sample_status"] = "command_failed"
        else:
            rows = []
            for line in process.stdout.splitlines():
                fields = [value.strip() for value in line.split(",")]
                if len(fields) == 3 and fields[0].lower() == UUID.lower() and int(fields[1]) in pids:
                    rows.append(int(fields[2]) * 1024 ** 2)
            result.update(process_sample_status="ok", own_process_count=len(rows), own_process_bytes=sum(rows))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        result["process_sample_status"] = "unavailable"
    try:
        gpu = subprocess.run(["nvidia-smi", "--id=" + UUID, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        used, load = [int(value.strip()) for value in gpu.stdout.strip().split(",")]
        assert gpu.returncode == 0
        result.update(whole_gpu_sample_status="ok", whole_gpu_used_bytes=used * 1024 ** 2, whole_gpu_utilization_percent=load)
    except (OSError, ValueError, AssertionError, subprocess.TimeoutExpired):
        result["whole_gpu_sample_status"] = "unavailable"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "python", "worker", "binding", "baseline", "candidate", "run", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--affinity", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.run.mkdir(mode=0o700)
    report = {"schema": "fnit_parc_tf32_original_T1_queue/v1", "status": "waiting_common_lock",
        "hostname": os.uname().nodename, "device": args.device, "affinity": args.affinity,
        "controller_sha256": sha(__file__), "worker_sha256": sha(args.worker), "binding_sha256": sha(args.binding),
        "lock_scope": "one full arm; release between arms", "jobs": []}
    record = args.run / "queue.private.json"
    with record.open("x") as stream:
        json.dump(report, stream, indent=2)
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
           "NUMBA_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "1" if args.device != "cpu" else ""}
    report["configured_environment"] = {name: env[name] for name in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    plan = [("parc", "baseline", "true"), ("parc", "candidate", "true"),
            ("parc-fast", "candidate", "true"), ("parc-fast", "baseline", "true")]
    if args.device != "cpu":
        plan += [("parc", "candidate", "false"), ("parc", "candidate", "none"),
                 ("parc-fast", "candidate", "none"), ("parc-fast", "candidate", "false")]
    report["plan"] = [{"mode": mode, "arm": arm, "policy": policy} for mode, arm, policy in plan]
    for index, (mode, arm, policy) in enumerate(plan):
        name = mode + "_" + arm + "_" + policy
        report.update(status="waiting_common_lock", next_job=name)
        record.write_text(json.dumps(report, indent=2) + "\n")
        source = args.baseline if arm == "baseline" else args.candidate
        with args.lock.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            report.update(status="running", next_job=name)
            record.write_text(json.dumps(report, indent=2) + "\n")
            argv = ["taskset", "-c", args.affinity, "/usr/bin/time", "-v", "-o", str(args.run / (name + ".time.txt")),
                "timeout", "--kill-after=10", "600", str(args.python), str(args.worker),
                "--root", str(args.root), "--source", str(source), "--binding", str(args.binding), "--arm", arm,
                "--mode", mode, "--policy", policy, "--device", args.device, "--output", str(args.run / name)]
            start = time.perf_counter()
            with (args.run / (name + ".log")).open("x") as log:
                process = subprocess.Popen(argv, env={**env, "PYTHONPATH": str(source)}, stdout=log, stderr=subprocess.STDOUT)
                samples = []
                if args.device == "cpu":
                    rc = process.wait(timeout=625)
                else:
                    while process.poll() is None:
                        if time.perf_counter() - start > 625:
                            process.terminate()
                            raise TimeoutError("controller arm wall exceeded 625 seconds")
                        samples.append(sample(process.pid))
                        time.sleep(0.5)
                    rc = process.returncode
            row = {"name": name, "arm": arm, "mode": mode, "policy": policy, "returncode": rc,
                   "wall_seconds": time.perf_counter() - start, "load_after": list(os.getloadavg())}
            if args.device != "cpu":
                nonzero = [value for value in samples if value.get("own_process_count")]
                row["gpu_monitor"] = {"target_uuid": UUID, "requested_interval_seconds": .5,
                    "samples": [{name: value for name, value in item.items() if name != "monotonic"} for item in samples],
                    "sample_count": len(samples), "nonzero_own_process_samples": len(nonzero),
                    "process_sample_failures": sum(s["process_sample_status"] != "ok" for s in samples),
                    "whole_gpu_sample_failures": sum(s["whole_gpu_sample_status"] != "ok" for s in samples),
                    "maximum_gap_seconds": max((b["monotonic"] - a["monotonic"] for a, b in zip(samples, samples[1:])), default=None),
                    "observed_own_process_peak_bytes": max((s["own_process_bytes"] for s in nonzero), default=None),
                    "physical_budget_status": ("observed_samples_within_budget" if nonzero and all(s["own_process_bytes"] <= 20_000_000_000 for s in nonzero)
                                               else "exceeded" if nonzero else "not_assessed"),
                    "limitation": "Sampled driver process-tree aggregate; unsampled peaks remain possible. Zero samples do not establish zero usage."}
            report["jobs"].append(row)
        if rc:
            report["status"] = "failed"
            record.write_text(json.dumps(report, indent=2) + "\n")
            return rc
        report["status"] = "between_arms_lock_released"
        record.write_text(json.dumps(report, indent=2) + "\n")
        if index + 1 < len(plan):
            time.sleep(2)
    report["status"] = "complete"
    record.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
