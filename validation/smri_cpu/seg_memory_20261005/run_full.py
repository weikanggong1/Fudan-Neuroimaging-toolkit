"""Same-node frozen-source CPU/GPU pair controller; acquire a common lock per arm."""
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


def gpu_snapshot():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=8)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "worker", "baseline", "candidate", "run", "lock", "input", "weights"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p.add_argument("--affinity", required=True)
    p.add_argument("--visible-gpu", default="1")
    p.add_argument("--cudnn-tf32", choices=("default", "false"), default="default")
    p.add_argument("--candidate-first", action="store_true")
    p.add_argument("--modes", nargs="+", choices=("seg33", "parc", "parc-fast"), default=("seg33", "parc", "parc-fast"))
    args = p.parse_args()
    os.umask(0o077)
    args.run.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = args.run / "queue.private.json"
    if record.exists():
        raise RuntimeError("immutable run already dispatched")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="" if args.device == "cpu" else args.visible_gpu,
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               NUMBA_NUM_THREADS="8", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="8")
    report = {"schema": "fnit_seg_decoder_full_queue/v1", "status": "waiting_common_lock",
              "device": args.device, "hostname": os.uname().nodename,
              "controller_sha256": sha(__file__), "worker_sha256": sha(args.worker),
              "configured_environment": {key: env[key] for key in
                  ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
              "affinity": args.affinity, "cudnn_tf32": args.cudnn_tf32,
              "lock_scope": "each complete arm; released between arms", "jobs": []}
    with record.open("x") as stream:
        json.dump(report, stream, indent=2)
    arms = ("candidate", "baseline") if args.candidate_first else ("baseline", "candidate")
    plan = [(mode, arm) for mode in args.modes for arm in arms]
    report["arm_order"] = list(arms)
    for index, (mode, arm) in enumerate(plan):
        name = mode + "_" + arm
        output = args.run / name
        assert not output.exists()
        source = args.baseline if arm == "baseline" else args.candidate
        arm_env = {**env, "PYTHONPATH": str(source)}
        argv = ["taskset", "-c", args.affinity, "/usr/bin/time", "-v", "-o", str(args.run / (name + ".time.txt")),
                "timeout", "--signal=TERM", "--kill-after=10", "600", str(args.python), str(args.worker),
                "--source", str(source), "--arm", arm, "--mode", mode, "--device", args.device,
                "--input", str(args.input), "--weights", str(args.weights), "--output", str(output),
                "--cudnn-tf32", args.cudnn_tf32]
        report.update(status="waiting_common_lock", next_job=name)
        record.write_text(json.dumps(report, indent=2) + "\n")
        with args.lock.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            report.update(status="running", next_job=name)
            record.write_text(json.dumps(report, indent=2) + "\n")
            before = gpu_snapshot() if args.device != "cpu" else None
            start = time.perf_counter()
            with (args.run / (name + ".log")).open("x") as log:
                proc = subprocess.Popen(argv, env=arm_env, stdout=log, stderr=subprocess.STDOUT)
                samples = []
                if args.device == "cpu":
                    rc = proc.wait(timeout=625)
                else:
                    while proc.poll() is None:
                        timestamp = time.monotonic()
                        snap = gpu_snapshot()
                        samples.append({"elapsed": time.perf_counter() - start, "monotonic": timestamp, **snap})
                        time.sleep(0.5)
                    rc = proc.returncode
            job = {"name": name, "arm": arm, "mode": mode, "argv": argv, "returncode": rc,
                   "wall_seconds": time.perf_counter() - start, "load_after": list(os.getloadavg()),
                   "source": str(source)}
            if args.device != "cpu":
                job.update(gpu_before=before, gpu_after=gpu_snapshot(), gpu_samples=samples)
            report["jobs"].append(job)
        if rc:
            report["status"] = "failed"
            record.write_text(json.dumps(report, indent=2) + "\n")
            return rc
        report["status"] = "between_arms_lock_released"
        record.write_text(json.dumps(report, indent=2) + "\n")
        # Gives queued diagnostics a visible common-lock boundary.
        if index + 1 < len(plan):
            time.sleep(2)
    report["status"] = "complete"
    record.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
