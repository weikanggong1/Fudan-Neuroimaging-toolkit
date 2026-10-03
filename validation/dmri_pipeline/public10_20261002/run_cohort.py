#!/usr/bin/env python3
"""Execute a fixed real-data benchmark plan, with durable per-job records.

The private plan supplies explicit argv and environments for every full pipeline.
Jobs run serially on one GPU; the scheduler never substitutes subjects, retries
a failed pipeline in place, or reuses intermediate images. Completed whole jobs
may be resumed only when their exact execution signature and output report agree.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def signature(job):
    selected = {key: job[key] for key in (
        "case_id", "backend", "implementation", "argv", "environment", "report")}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def descendants(parent):
    relations = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            relations[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    result = {parent}
    while True:
        updated = result | {pid for pid, ppid in relations.items() if ppid in result}
        if updated == result:
            return result
        result = updated


def sample_gpu(job_pid, gpu_uuid):
    family = descendants(job_pid)
    output = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits"], check=True, text=True,
        capture_output=True, timeout=8).stdout
    own, other = [], []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3 or values[1] != gpu_uuid:
            continue
        try:
            item = {"pid": int(values[0]), "memory_mib": int(values[2])}
        except ValueError:
            continue
        (own if item["pid"] in family else other).append(item)
    state = subprocess.run([
        "nvidia-smi", "-i", gpu_uuid,
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"], check=True, text=True,
        capture_output=True, timeout=8).stdout.strip().split(",")
    return {"own_gpu_memory_mib": sum(item["memory_mib"] for item in own),
            "other_gpu_memory_mib": sum(item["memory_mib"] for item in other),
            "own_gpu_process_count": len(own), "other_gpu_process_count": len(other),
            "device_memory_used_mib": int(state[0]),
            "device_utilization_percent": int(state[1]),
            "device_temperature_celsius": int(state[2]),
            "device_power_watts": float(state[3])}


def execute(job, gpu_uuid):
    directory = Path(job["job_dir"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    metrics_path = directory / "process_metrics.json"
    report_path = Path(job["report"])
    digest = signature(job)
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text())
        if previous.get("execution_signature") != digest:
            raise RuntimeError("existing job has a different execution signature")
        if previous.get("exit_code") == 0 and report_path.is_file():
            result = json.loads(report_path.read_text())
            if result.get("status") == "complete":
                print(json.dumps({"event": "already_complete", "case_id": job["case_id"],
                                  "backend": job["backend"],
                                  "implementation": job["implementation"]}), flush=True)
                return previous
        # Preserve a failure rather than silently replacing its result.
        raise RuntimeError("existing failed/incomplete job requires a separately labelled run")
    if report_path.exists():
        raise RuntimeError("report already exists without a matching durable execution record")
    environment = os.environ.copy()
    environment.update(job["environment"])
    command = ["/usr/bin/time", "-v", "-o", str(directory / "time.txt"), *job["argv"]]
    start = time.perf_counter()
    metrics = {"case_id": job["case_id"], "backend": job["backend"],
               "implementation": job["implementation"], "status": "running",
               "execution_signature": digest, "started_utc": utc_now(),
               "gpu_uuid": gpu_uuid, "gpu_samples": [], "telemetry_errors": 0}
    save_json(metrics_path, metrics)
    stop = threading.Event()
    with (directory / "log.txt").open("wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=directory)

        def monitor():
            while not stop.is_set():
                try:
                    value = sample_gpu(process.pid, gpu_uuid)
                    value["seconds"] = time.perf_counter() - start
                    metrics["gpu_samples"].append(value)
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    metrics["telemetry_errors"] += 1
                save_json(metrics_path, metrics)
                stop.wait(5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            code = process.wait()
        finally:
            stop.set()
            thread.join(timeout=20)
    metrics.update({"exit_code": code, "status": "complete" if code == 0 else "failed",
                    "finished_utc": utc_now(), "wall_seconds": time.perf_counter() - start,
                    "timing_scope": "full subprocess including Python startup and post-run audits; excludes scheduler queue and GPU lock wait",
                    "sampled_own_gpu_memory_peak_mib": max(
                        (item["own_gpu_memory_mib"] for item in metrics["gpu_samples"]), default=None),
                    "sampled_other_gpu_memory_peak_mib": max(
                        (item["other_gpu_memory_mib"] for item in metrics["gpu_samples"]), default=None),
                    "gpu_memory_scope": "sum of this job's descendant native/PyTorch GPU processes at 5-second samples, includes CUDA context; does not replace allocator peak"})
    save_json(metrics_path, metrics)
    (directory / "exitcode").write_text(str(code) + "\n")
    print(json.dumps({key: metrics[key] for key in (
        "case_id", "backend", "implementation", "status", "exit_code", "wall_seconds")}), flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    arguments = parser.parse_args(argv)
    os.umask(0o077)
    plan = json.loads(arguments.plan.read_text())
    arguments.gpu_lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    outcomes = []
    for job in plan["jobs"]:
        # Timing starts after admission; own benchmark jobs cannot overlap.
        with arguments.gpu_lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                outcome = execute(job, plan["gpu_uuid"])
            except Exception as error:
                outcome = {"case_id": job["case_id"], "backend": job["backend"],
                           "implementation": job["implementation"], "status": "scheduler_failed",
                           "exception_type": type(error).__name__, "reason": str(error)}
                print(json.dumps(outcome), flush=True)
            outcomes.append(outcome)
            save_json(arguments.plan.with_name("cohort_status.json"), {
                "plan_sha256": hashlib.sha256(arguments.plan.read_bytes()).hexdigest(),
                "outcomes": outcomes, "planned_jobs": len(plan["jobs"])})
        # Admission pacing is outside every job clock. Allow another fixed queue
        # or an output comparison to acquire the shared lock at this boundary.
        time.sleep(0.25)
    return 0 if all(item["status"] == "complete" for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
