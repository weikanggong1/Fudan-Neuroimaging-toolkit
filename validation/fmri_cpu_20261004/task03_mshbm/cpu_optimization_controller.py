#!/usr/bin/env python3
"""Run full real old/new MS-HBM control under the assigned CPU resource lock."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from acceptance import compare_histories
from gpu_regression_worker import compare_outputs


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    os.umask(0o077)
    root = Path(cfg["run_root"])
    if root.exists():
        raise FileExistsError("Old/new acceptance requires a new output directory")
    root.mkdir(parents=True)
    status = {"state": "waiting_lock", "controller_pid": os.getpid(), "jobs": []}
    status_path = root / "queue_status.public.json"
    write(status_path, status)
    tool = Path(__file__).resolve().parent / "benchmark_fnit.py"
    with open(cfg["lock"], "a") as resource_lock:
        fcntl.flock(resource_lock, fcntl.LOCK_EX)
        status["state"] = "running"
        write(status_path, status)
        jobs = cfg.get("jobs", [
            {"name": "baseline_cpu1_profile", "version": "baseline", "threads": 1, "trace": True},
            {"name": "candidate_cpu1", "version": "candidate", "threads": 1},
            {"name": "baseline_cpu1", "version": "baseline", "threads": 1},
            {"name": "baseline_cpu8", "version": "baseline", "threads": 8},
            {"name": "candidate_cpu8", "version": "candidate", "threads": 8},
        ])
        for job in jobs:
            binding = dict(cfg["binding"])
            binding["source_root"] = cfg[job["version"] + "_source"]
            binding["frozen_baseline_commit"] = cfg["baseline_commit"]
            binding.update(job.get("binding_overrides", {}))
            binding_path = root / (job["name"] + ".binding.private.json")
            write(binding_path, binding)
            output = root / job["name"]
            env = os.environ.copy()
            for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env[variable] = str(job["threads"])
            affinity = cfg["cpu1"] if job["threads"] == 1 else cfg["cpu8"]
            command = ["taskset", "-c", ",".join(str(i) for i in affinity),
                       "/usr/bin/time", "-v", "-o", str(root / (job["name"] + ".clock.private.txt")),
                       cfg["python"], str(tool), "--binding", str(binding_path),
                       "--threads", str(job["threads"]), "--device", job.get("device", "cpu"),
                       "--kind", job.get("kind", "surface"), "--output-dir", str(output)]
            if job.get("trace"):
                command.append("--trace-core-lines")
            item = {"name": job["name"], "version": job["version"], "threads": job["threads"],
                    "state": "running", "affinity": affinity, "start_unix": time.time()}
            status["jobs"].append(item)
            with (root / (job["name"] + ".log")).open("wb") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = child.pid
                write(status_path, status)
                code = child.wait()
            item.update({"returncode": code, "state": "complete" if code == 0 else "failed",
                         "end_unix": time.time()})
            write(status_path, status)
            if code:
                status["state"] = "failed"
                write(status_path, status)
                sys.exit(code)
        import numpy as np
        controls = []
        pairs = cfg.get("pairs", [{"baseline": "baseline_cpu" + str(threads),
                                    "candidate": "candidate_cpu" + str(threads),
                                    "threads": threads} for threads in (1, 8)])
        for pair in pairs:
            threads = pair["threads"]
            left = root / pair["baseline"]
            right = root / pair["candidate"]
            if not left.exists() or not right.exists():
                continue
            old = json.loads((left / "report.public.json").read_text())
            new = json.loads((right / "report.public.json").read_text())
            labels_old = np.load(left / "labels_fslr32k_64984.npy", allow_pickle=False)
            labels_new = np.load(right / "labels_fslr32k_64984.npy", allow_pickle=False)
            output_checks = {}
            for filename in old["outputs_sha256"]:
                a, b = left / filename, right / filename
                if a.is_file() and b.is_file():
                    output_checks[filename] = hashlib.sha256(a.read_bytes()).hexdigest() == \
                                              hashlib.sha256(b.read_bytes()).hexdigest()
            control = {"threads": threads,
                "different_labels": int(np.count_nonzero(labels_old != labels_new)),
                "baseline": pair["baseline"], "candidate": pair["candidate"],
                "profiles_exact": (old["profiles_sha256"] == new["profiles_sha256"]
                                   if "profiles_sha256" in old and "profiles_sha256" in new else None),
                **compare_histories(old["history"], new["history"]),
                "outputs_exact": output_checks,
                "baseline_chain_seconds": old["function_chain_seconds"],
                "candidate_chain_seconds": new["function_chain_seconds"],
                "chain_speedup": old["function_chain_seconds"] / new["function_chain_seconds"],
                "baseline_core_seconds": old["stages_seconds"].get("parcellate"),
                "candidate_core_seconds": new["stages_seconds"].get("parcellate"),
                "baseline_maximum_rss_kib": old["maximum_rss_kib"],
                "candidate_maximum_rss_kib": new["maximum_rss_kib"]}
            if (left / "labels_mni.nii.gz").is_file() and (right / "labels_mni.nii.gz").is_file():
                control["volume_output_values"] = compare_outputs(left, right)
            controls.append(control)
        write(root / "old_new_control.public.json", {"controls": controls})
        status["state"] = "complete"
        write(status_path, status)


if __name__ == "__main__":
    main()
