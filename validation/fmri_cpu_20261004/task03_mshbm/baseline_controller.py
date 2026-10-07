#!/usr/bin/env python3
"""Prepare or execute the complete CPU1/CPU8 FNIT/CBIG paired baseline.

Preparation exports pristine upstream sources, validates complete real inputs
and writes bindings. Execution alone acquires the assigned CPU lock. The parent
coordinator authorizes and starts execution after preparation is ready.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import socket

from export_reference import verify_export

BASELINE_COMMIT = "cc9402734faeba93b3a13c29932fa1392eaccf62"
CBIG_COMMIT = "b69b822a15e2a94f1e439606552fc44b6858cf3c"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--start", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["run_root"])
    tools = Path(__file__).resolve().parent
    os.umask(0o077)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue_status.public.json"
    clean = Path(cfg["clean_reference"])
    if args.prepare:
        if status_path.exists():
            raise FileExistsError("Preparation status already exists; preserve this attempt")
        if not clean.exists():
            subprocess.run([cfg["python"], str(tools / "export_reference.py"),
                            "--checkout", cfg["cbig_checkout"], "--output", str(clean)], check=True)
        manifest = verify_export(clean)
        if manifest["commit"] != CBIG_COMMIT:
            raise ValueError("Reference export commit mismatch")
        relative = ("stable_projects/brain_parcellation/Kong2019_MSHBM/"
                    "step3_generate_ind_parcellations/CBIG_MSHBM_generate_individual_parcellation.m")
        if digest(clean / relative) != "ce5c5a8596994c19c88e1a0ddb65dd28b4e3c31ded5887a89dacd667f4d0b3bb":
            raise ValueError("Reference inference function is not pristine")
        import nibabel as nib
        image = nib.load(cfg["timeseries"])
        if tuple(image.shape) != (490, 91282):
            raise ValueError("Main baseline requires the complete 490-frame real CIFTI")
        actual = digest(cfg["timeseries"])
        if actual != cfg["input_sha256"]:
            raise ValueError("Complete input SHA mismatch")
        baseline = {"source_root": cfg["source_root"], "case": "CASE01",
                    "frozen_baseline_commit": BASELINE_COMMIT,
                    "assets": cfg["assets"], "timeseries": [cfg["timeseries"]],
                    "expected_frames": 490, "w": 200.0, "c": 50.0}
        official = {"cbig_clean_dir": str(clean), "cbig_sd_dir": cfg["cbig_sd_dir"],
                    "cbig_dependency_dir": cfg["cbig_checkout"],
                    "matlab_startup": cfg.get("matlab_startup"),
                    "dependency_paths": cfg.get("dependency_paths", []),
                    "timeseries": [cfg["timeseries"]], "w": 200.0, "c": 50.0}
        write_json(root / "fnit_binding.private.json", baseline)
        if cfg.get("candidate_source"):
            candidate = dict(baseline, source_root=cfg["candidate_source"])
            write_json(root / "fnit_candidate_binding.private.json", candidate)
        write_json(root / "matlab_binding.private.json", official)
        prepared = {"state": "prepared", "baseline_commit": BASELINE_COMMIT,
                    "reference_commit": CBIG_COMMIT, "input_sha256": actual,
                    "complete_frames": 490, "complete_fsLR_vertices": 64984,
                    "config_sha256": digest(args.config), "jobs": [],
                    "prepare_hostname": socket.gethostname(), "load_at_prepare": list(os.getloadavg()),
                    "cpu1": cfg["cpu1"], "cpu8": cfg["cpu8"]}
        write_json(status_path, prepared)
        print(json.dumps({"state": "prepared", "formal_benchmarks_started": False}))
        return
    status = json.loads(status_path.read_text())
    if status["state"] != "prepared" or digest(args.config) != status["config_sha256"]:
        raise ValueError("Start requires the unchanged prepared configuration")
    Path(cfg["lock"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["lock"], "a") as resource_lock:
        fcntl.flock(resource_lock, fcntl.LOCK_EX)
        status["state"] = "running"
        status["controller_pid"] = os.getpid()
        status["execution_hostname"] = socket.gethostname()
        write_json(status_path, status)
        # One complete observation per implementation/budget for this baseline.
        # Later optimization acceptance decides which comparisons merit repeats.
        order = cfg.get("jobs", [{"threads": threads, "implementation": implementation}
                    for threads, implementation in ((1, "fnit"), (1, "cbig"), (8, "cbig"), (8, "fnit"))])
        for job in order:
            threads, implementation = job["threads"], job["implementation"]
            affinity = cfg["cpu1"] if threads == 1 else cfg["cpu8"]
            name = implementation + "_cpu" + str(threads)
            output = root / name
            log = root / (name + ".log")
            clock = root / (name + ".clock.private.txt")
            env = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env[key] = str(threads)
            env["PYTHONPATH"] = str(Path(cfg["source_root"]) / "src")
            if implementation.startswith("fnit"):
                binding_name = ("fnit_candidate_binding.private.json" if implementation == "fnit_candidate"
                                else "fnit_binding.private.json")
                command = [cfg["python"], str(tools / "benchmark_fnit.py"),
                    "--binding", str(root / binding_name),
                    "--kind", "surface", "--threads", str(threads), "--device", "cpu",
                    "--output-dir", str(output)]
            else:
                invocation = "try, addpath(" + matlab_string(tools) + "); run_reference(" + \
                    matlab_string(root / "matlab_binding.private.json") + "," + \
                    matlab_string(output) + "," + str(threads) + ",'full'); exit(0); " + \
                    "catch exception, disp(getReport(exception,'extended')); exit(1); end;"
                command = [cfg["matlab"], "-nodisplay", "-nosplash", "-nodesktop", "-r", invocation]
                if cfg.get("matlab_nojvm"):
                    command.insert(1, "-nojvm")
            command = ["taskset", "-c", ",".join(str(x) for x in affinity),
                       "/usr/bin/time", "-v", "-o", str(clock), *command]
            item = {"name": name, "threads": threads, "affinity": affinity,
                    "state": "running", "start_unix": time.time(),
                    "load_before": list(os.getloadavg())}
            status["jobs"].append(item)
            write_json(status_path, status)
            with log.open("wb") as logfile:
                child = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT)
                item["pid"] = child.pid
                write_json(status_path, status)
                returncode = child.wait()
            item.update({"state": "complete" if returncode == 0 else "failed",
                         "returncode": returncode, "end_unix": time.time(),
                         "load_after": list(os.getloadavg())})
            write_json(status_path, status)
            print(json.dumps({"job": name, "state": item["state"]}), flush=True)
            if returncode:
                status["state"] = "failed"
                write_json(status_path, status)
                sys.exit(returncode)
        status["state"] = "complete"
        write_json(status_path, status)


if __name__ == "__main__":
    main()
