"""Run a prepared, full-input CPU queue under its assigned physical-core lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = {"status": "waiting_for_cpu_lock", "pid": os.getpid(), "records": []}
    status_path = args.output_dir / "controller_status.private.json"
    publish(status_path, state)
    with Path(config["lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename, expected in config["frozen_sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen source or adapter differs from the prepared manifest")
        state["status"] = "running"
        for request in config["requests"]:
            name = request["name"]
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Each request needs one unique output directory name")
            env = dict(os.environ, PYTHONPATH=request["source_path"],
                       OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS"):
                env[key] = str(request["threads"])
            command = ["taskset", "-c", ",".join(map(str, request["cpus"])),
                       config["python"], config["adapter"], *request["arguments"],
                       "--device", "cpu", "--threads", str(request["threads"]),
                       "--output-dir", str(args.output_dir / name)]
            row = {"name": name, "status": "running", "threads": request["threads"]}
            state["records"].append(row)
            started = time.perf_counter()
            with (args.output_dir / (name + ".private.log")).open("wb") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                row["pid"] = child.pid
                publish(status_path, state)
                code = child.wait()
            row.update(status="completed" if code == 0 else "failed", returncode=code,
                       process_seconds=time.perf_counter() - started)
            if code:
                state["status"] = "failed"
                publish(status_path, state)
                raise RuntimeError("Full-input request failed; preserve this attempt")
            report = json.loads((args.output_dir / name / "report.public.json").read_text())
            row["api_wall_seconds"] = report["api_wall_seconds"]
            publish(status_path, state)
        state["status"] = "completed"
        publish(status_path, state)
        publish(args.output_dir / "execution.public.json", {
            "status": "completed", "records": [{k: v for k, v in row.items() if k != "pid"}
                                                  for row in state["records"]]})


if __name__ == "__main__":
    main()
