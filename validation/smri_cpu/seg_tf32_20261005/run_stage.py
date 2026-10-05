"""Bounded real-prefix old/new controller; common lock is held one arm at a time."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "python", "worker", "binding", "baseline", "candidate", "run", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--affinity", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.run.mkdir(mode=0o700)
    queue = {"schema": "fnit_parc_tf32_real_prefix_queue/v1", "status": "waiting",
        "hostname": os.uname().nodename, "device": args.device, "affinity": args.affinity,
        "controller_sha256": sha(__file__), "worker_sha256": sha(args.worker),
        "binding_sha256": sha(args.binding), "lock_scope": "one arm", "jobs": []}
    report_path = args.run / "queue.private.json"
    with report_path.open("x") as stream:
        json.dump(queue, stream, indent=2)
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
           "NUMBA_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "1" if args.device == "cuda:0" else ""}
    for arm, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        queue["status"] = "waiting_common_lock"
        queue["next_arm"] = arm
        report_path.write_text(json.dumps(queue, indent=2) + "\n")
        with args.lock.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue["status"] = "running"
            report_path.write_text(json.dumps(queue, indent=2) + "\n")
            start = time.perf_counter()
            command = ["taskset", "-c", args.affinity, "/usr/bin/time", "-v", "-o", str(args.run / (arm + ".time.txt")),
                "timeout", "--kill-after=10", "240", str(args.python), str(args.worker),
                "--root", str(args.root), "--source", str(source), "--binding", str(args.binding),
                "--arm", arm, "--device", args.device, "--output", str(args.run / arm)]
            with (args.run / (arm + ".log")).open("x") as log:
                process = subprocess.run(command, env={**env, "PYTHONPATH": str(source)},
                                         stdout=log, stderr=subprocess.STDOUT, timeout=260)
            queue["jobs"].append({"arm": arm, "returncode": process.returncode,
                "wall_seconds": time.perf_counter() - start, "load_after": list(os.getloadavg())})
        if process.returncode:
            queue["status"] = "failed"
            report_path.write_text(json.dumps(queue, indent=2) + "\n")
            return process.returncode
        queue["status"] = "between_arms_lock_released"
        report_path.write_text(json.dumps(queue, indent=2) + "\n")
        time.sleep(1)
    queue["status"] = "complete"
    report_path.write_text(json.dumps(queue, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
