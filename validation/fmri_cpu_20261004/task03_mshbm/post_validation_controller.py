"""Run complete independent geometry/output checks and focused tests serially."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    os.umask(0o077)
    root = Path(cfg["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    status = {"state": "waiting_lock", "controller_pid": os.getpid(), "jobs": []}

    def write():
        path = root / "status.public.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(path)

    write()
    with open(cfg["lock"], "a") as resource_lock:
        fcntl.flock(resource_lock, fcntl.LOCK_EX)
        status["state"] = "running"
        write()
        for job in cfg["jobs"]:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(cfg["source_root"]) / "src")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env[key] = "8"
            item = {"name": job["name"], "state": "running", "start_unix": time.time()}
            status["jobs"].append(item)
            with (root / (job["name"] + ".private.log")).open("wb") as log:
                child = subprocess.Popen(["taskset", "-c", ",".join(map(str, cfg["cpu8"])),
                                          *job["argv"]], env=env, stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = child.pid
                write()
                code = child.wait()
            item.update({"returncode": code, "state": "complete" if code == 0 else "failed",
                         "end_unix": time.time()})
            write()
            if code:
                status["state"] = "failed"
                write()
                return
        status["state"] = "complete"
        write()


if __name__ == "__main__":
    main()
