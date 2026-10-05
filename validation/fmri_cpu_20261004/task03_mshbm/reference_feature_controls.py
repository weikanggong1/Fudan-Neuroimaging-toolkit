#!/usr/bin/env python3
"""Run additional real-run CBIG/FNIT censor and weight controls on CPU8.

The complete uncensored CPU1/CPU8 primary queue must finish first. Both
implementations consume the same complete real CIFTI and fixed prior. Censor
is a separately prepared complete-run DVARS P95 interface control. These
observations remain separate from the primary speed benchmark.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from export_reference import verify_export


BASELINE_CORE = "8cd2f6fb706a13c60be5070c575cdb77898e2d309cf3f958c4fdbfee355a27ea"
CANDIDATE_CORE = "eabb4d62c810fc180d71f41c0e783bcf2dfd9c1bb6ae42fcc699e44859a9aa9e"
TOOLS_SHA256 = {
    "run_reference.m": "8a3e0d9c5e5c50e304fb6fce92aea55625c574656c1544f6c978be6f86d04759",
    "benchmark_fnit.py": "f997172c82eaf25d514fa2dba1f2dee232dd9a93eb573f5177b4062c465b47ba",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Unchanged v5 primary private configuration")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New private root; existing attempts are preserved")
    args = parser.parse_args()
    os.umask(0o077)
    if socket.gethostname() != "nodecw10":
        raise ValueError("Original and candidate must run on licensed nodecw10")
    if args.output_root.exists():
        raise FileExistsError("Inspect the existing attempt before any resubmission")
    cfg = json.loads(args.config.read_text())
    primary_root = Path(cfg["run_root"])
    tools = args.config.parent / "tools"
    primary = json.loads((primary_root / "queue_status.public.json").read_text())
    reader = json.loads((primary_root / "reader_probe_values.public.json").read_text())
    if primary.get("state") != "complete" or any(
            job.get("state") != "complete" or job.get("returncode") != 0
            for job in primary["jobs"]):
        raise ValueError("All six complete uncensored primary jobs must succeed first")
    if len(primary["jobs"]) != 6 or primary["config_sha256"] != sha256(args.config):
        raise ValueError("The fixed primary job/configuration scope must match")
    expected_jobs = [name + "_cpu" + str(threads) for threads in (1, 8)
                     for name in ("cbig", "fnit", "fnit_candidate")]
    if [job["name"] for job in primary["jobs"]] != expected_jobs:
        raise ValueError("The complete original/frozen/candidate CPU1/CPU8 scope must match")
    if (reader.get("compared_values") != 29111880 or reader.get("different_values") != 0
            or reader.get("input_sha256") != cfg["input_sha256"]):
        raise ValueError("The original complete reader-value gate must pass")
    if (primary.get("input_sha256") != cfg["input_sha256"]
            or primary.get("reference_commit") != "b69b822a15e2a94f1e439606552fc44b6858cf3c"
            or not reader.get("cortical_vertex_identity_exact")):
        raise ValueError("The primary original input, commit and full vertex identity must match")
    for name, expected in TOOLS_SHA256.items():
        if sha256(tools / name) != expected:
            raise ValueError("An actual primary MATLAB/FNIT benchmark adapter changed")
    if sha256(cfg["timeseries"]) != cfg["input_sha256"]:
        raise ValueError("The complete real input changed after the primary benchmark")
    manifest = verify_export(cfg["clean_reference"])
    for source, expected in ((cfg["source_root"], BASELINE_CORE),
                             (cfg["candidate_source"], CANDIDATE_CORE)):
        if sha256(Path(source) / "src/fnit/mshbm/core.py") != expected:
            raise ValueError("Actual frozen FNIT core changed")
    affinity = cfg["cpu8"]
    topology = []
    for cpu in affinity:
        root = Path("/sys/devices/system/cpu") / ("cpu" + str(cpu)) / "topology"
        topology.append((int((root / "physical_package_id").read_text()),
                         int((root / "core_id").read_text())))
    if len(affinity) != 8 or len(set(topology)) != 8 or len({x[0] for x in topology}) != 1:
        raise ValueError("Eight distinct physical cores on the same socket are required")
    censor_root = primary_root / "optional_censor_dvars_p95_v1"
    censor_report = json.loads((censor_root / "report.public.json").read_text())
    censor_path = censor_root / "censor.private.txt"
    if (censor_report.get("input_sha256") != cfg["input_sha256"]
            or censor_report.get("input_frames") != 490
            or censor_report.get("censor_sha256") != sha256(censor_path)
            or not censor_report.get("primary_benchmark_remains_uncensored")):
        raise ValueError("Censor must derive from the same predefined complete-run rule")
    import numpy as np
    censor = np.loadtxt(censor_path)
    if (censor.shape != (490,) or not np.isin(censor, [0, 1]).all()
            or int(censor.sum()) != censor_report["retained_frames"]):
        raise ValueError("The same complete 490-row censor vector is required")
    args.output_root.mkdir(parents=True)
    write_json(args.output_root / "launch.private.json", {
        "pid": os.getpid(), "state": "running", "input_sha256": cfg["input_sha256"],
        "original_commit": manifest["commit"], "original_files": len(manifest["files"]),
        "primary_complete": True, "gpu_regression_resubmitted": False})
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "8"
    environment["PYTHONPATH"] = str(Path(cfg["candidate_source"]) / "src")
    native_binding = json.loads((primary_root / "matlab_binding.private.json").read_text())
    fnit_binding = json.loads((primary_root / "fnit_candidate_binding.private.json").read_text())
    if (native_binding.get("timeseries") != [cfg["timeseries"]]
            or fnit_binding.get("timeseries") != [cfg["timeseries"]]
            or Path(native_binding["cbig_clean_dir"]).resolve() != Path(cfg["clean_reference"]).resolve()
            or Path(fnit_binding["source_root"]).resolve() != Path(cfg["candidate_source"]).resolve()
            or Path(fnit_binding["assets"]).resolve() != Path(cfg["assets"]).resolve()
            or sha256(fnit_binding["assets"]) != reader["assets_sha256"]
            or fnit_binding.get("expected_frames") != 490):
        raise ValueError("Actual primary bindings differ from the gated input, assets or frozen source")
    cases = (("censor_dvars_p95", 200.0, 50.0, [str(censor_path)]),
             ("weights_w100_c25", 100.0, 25.0, None))
    with open(cfg["lock"], "a") as resource_lock:
        fcntl.flock(resource_lock, fcntl.LOCK_EX)
        for case, w, c, censor_files in cases:
            directory = args.output_root / case
            directory.mkdir()
            native = dict(native_binding, w=w, c=c)
            candidate = dict(fnit_binding, w=w, c=c)
            if censor_files:
                native["censor"] = candidate["censor"] = censor_files
            elif native.get("censor") or candidate.get("censor"):
                raise ValueError("The weight control must retain the uncensored primary input")
            for key in ("timeseries", "w", "c", "censor"):
                if native.get(key) != candidate.get(key):
                    raise ValueError("Original and FNIT input/parameter bindings differ")
            for name, binding in (("matlab", native), ("fnit_candidate", candidate)):
                write_json(directory / (name + "_binding.private.json"), binding)
            status = {"state": "running", "scope": "additional real-run functional control",
                "case": case, "input_sha256": cfg["input_sha256"],
                "reference_commit": manifest["commit"], "complete_read_frames": 490,
                "retained_frames": int(censor.sum()) if censor_files else 490,
                "censor_sha256": censor_report["censor_sha256"] if censor_files else None,
                "w": w, "c": c, "threads": 8, "affinity": affinity,
                "paired_bindings_exact": True,
                "binding_sha256": {name: sha256(directory / (name + "_binding.private.json"))
                                   for name in ("matlab", "fnit_candidate")}, "jobs": []}
            status_path = directory / "queue_status.public.json"
            write_json(status_path, status)
            for implementation in ("cbig", "fnit_candidate"):
                name = implementation + "_cpu8"
                output = directory / name
                if implementation == "cbig":
                    expression = ("try, addpath(" + matlab_string(tools) + "); run_reference(" +
                        matlab_string(directory / "matlab_binding.private.json") + "," +
                        matlab_string(output) + ",8,'full'); exit(0); catch exception, " +
                        "disp(getReport(exception,'extended')); exit(1); end;")
                    command = [cfg["matlab"], "-nojvm", "-nodisplay", "-nosplash", "-nodesktop", "-r", expression]
                else:
                    command = [cfg["python"], str(tools / "benchmark_fnit.py"),
                        "--binding", str(directory / "fnit_candidate_binding.private.json"),
                        "--kind", "surface", "--threads", "8", "--device", "cpu",
                        "--output-dir", str(output)]
                item = {"name": name, "threads": 8, "affinity": affinity, "state": "running",
                        "start_unix": time.time(), "load_before": list(os.getloadavg())}
                status["jobs"].append(item)
                write_json(status_path, status)
                command = ["taskset", "-c", ",".join(map(str, affinity)), "/usr/bin/time", "-v", "-o",
                           str(directory / (name + ".clock.private.txt")), *command]
                with (directory / (name + ".private.log")).open("xb") as log:
                    child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    item["pid"] = child.pid
                    write_json(status_path, status)
                    returncode = child.wait()
                item.update({"state": "complete" if returncode == 0 else "failed",
                    "returncode": returncode, "end_unix": time.time(), "load_after": list(os.getloadavg())})
                status["state"] = "running" if returncode == 0 else "failed"
                write_json(status_path, status)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            status["state"] = "complete"
            write_json(status_path, status)
            print(json.dumps({"case": case, "state": "complete", "threads": 8}), flush=True)
    write_json(args.output_root / "complete.public.json", {
        "state": "complete", "cases": [x[0] for x in cases], "threads": 8,
        "gpu_regression_resubmitted": False, "primary_results_unchanged": True})


if __name__ == "__main__":
    main()
