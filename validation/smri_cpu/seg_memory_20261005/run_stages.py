"""Run a bounded CPU diagnostic queue under the common sMRI eight-core lock."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "worker", "source", "helper", "run", "lock", "array", "input", "weights"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--affinity", required=True)
    p.add_argument("--existing-checkpoint", type=Path)
    args = p.parse_args()
    os.umask(0o077)
    args.run.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8", NUMBA_NUM_THREADS="8", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="8",
               PYTHONPATH=str(args.source))
    identity = {"queue_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "worker_sha256": hashlib.sha256(args.worker.read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(args.helper.read_bytes()).hexdigest(),
                "hostname": os.uname().nodename, "affinity": args.affinity,
                "configured_environment": {key: env[key] for key in
                   ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTHONPATH")},
                "status": "waiting_common_cpu_lock"}
    launch = args.run / "queue.private.json"
    with launch.open("x") as stream:
        json.dump(identity, stream, indent=2)
    with args.lock.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity["status"] = "running"
        identity["jobs"] = []
        launch.write_text(json.dumps(identity, indent=2) + "\n")
        plan = [("A1", "baseline"), ("B1", "candidate"), ("B2", "candidate"), ("A2", "baseline")]
        if args.existing_checkpoint is None:
            plan.insert(0, ("capture", "capture"))
        checkpoint = args.existing_checkpoint or args.run / "checkpoint"
        identity["reused_checkpoint"] = str(checkpoint) if args.existing_checkpoint else None
        for name, mode in plan:
            report = args.run / (name + ".private.json")
            assert not report.exists()
            argv = ["taskset", "-c", args.affinity, "/usr/bin/time", "-v", "-o", str(args.run / (name + ".time.txt")),
                    "timeout", "--signal=TERM", "--kill-after=10", "600", str(args.python), str(args.worker),
                    "--mode", mode, "--source", str(args.source), "--checkpoint", str(checkpoint),
                    "--report", str(report), "--helper", str(args.helper), "--array", str(args.array),
                    "--input", str(args.input), "--weights", str(args.weights)]
            start = time.perf_counter()
            with (args.run / (name + ".log")).open("x") as log:
                done = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=625)
            row = {"name": name, "mode": mode, "argv": argv, "returncode": done.returncode,
                   "wall_seconds": time.perf_counter() - start, "load_after": list(os.getloadavg())}
            identity["jobs"].append(row)
            if done.returncode:
                identity["status"] = "failed"
                launch.write_text(json.dumps(identity, indent=2) + "\n")
                return done.returncode
            launch.write_text(json.dumps(identity, indent=2) + "\n")
        identity["status"] = "complete"
        launch.write_text(json.dumps(identity, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
