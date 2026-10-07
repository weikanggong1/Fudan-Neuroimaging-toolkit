"""Prepared single bounded ABBA queue. No whole model/native/GPU arm exists."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from trial_bindings import check_bindings, identity, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-real-layer", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan_identity = identity(args.workspace / "PLAN.json")
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    require(args.run.resolve() == (args.root / plan["runs"]).resolve(), "new canonical run required")
    before = check_bindings(args.root, args.workspace, plan)
    require(not args.run.exists(), "one dispatch only; never overwrite existing receipts")
    args.run.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    deadline = started + 23000
    queue = {"schema": "fnit_columns_real_layer_ABBA_queue/v1", "status": "waiting_common_CPU_lock",
             "controller_PID": os.getpid(), "PLAN": plan_identity,
             "bindings_before": before, "outer_total_seconds": 23000, "jobs": [],
             "new_compilation_calls": 0, "whole_model_native_GPU_arms": 0, "whole_map_CSV_assessed": False}
    path = args.run / "QUEUE.json"
    def save():
        path.write_text(json.dumps(queue, indent=2, allow_nan=False) + "\n")
    save()
    fd = os.open(args.root / plan["common_CPU_lock"], os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline - 750:
                    raise TimeoutError("bounded queue wait expired; no worker retried")
                time.sleep(1)
        queue["status"] = "running_four_bounded_real_layer_arms"
        queue["wait_seconds"] = time.monotonic() - started
        save()
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                            "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1"})
        for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "OPENBLAS_CORETYPE"):
            environment.pop(name, None)
        interpreter = args.root / "envs/default/bin/python"
        for name, mode in (("A1_baseline", "baseline"), ("B1_candidate", "candidate"),
                           ("B2_candidate", "candidate"), ("A2_baseline", "baseline")):
            command = [str(interpreter), str(args.workspace / "layer_trial.py"), "--root", str(args.root),
                       "--workspace", str(args.workspace), "--output", str(args.run / name), "--mode", mode,
                       "--approved-real-layer"]
            if name != "A1_baseline":
                command += ["--reference", str(args.run / "A1_baseline/baseline_preELU.private.npy")]
            tick = time.monotonic()
            job = {"name": name, "mode": mode, "status": "running", "timeout_seconds": 180}
            queue["jobs"].append(job)
            save()
            with (args.run / (name + ".log")).open("xb") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment, preexec_fn=lambda: os.sched_setaffinity(0, plan["affinity"]))
                job["PID"] = process.pid
                save()
                try:
                    code = process.wait(timeout=min(180, deadline - tick - 10))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    code = -9
                    job["timeout"] = True
            job.update({"exit_code": code, "worker_wall_seconds": time.monotonic() - tick,
                        "status": "completed" if code == 0 else "failed"})
            if code:
                queue["status"] = "first_real_layer_failure_stopped_remaining_arms"
                save()
                raise RuntimeError("first arm failed; no later arm or retry: " + name)
            report = json.loads((args.run / name / "report.json").read_text())
            require(report["valid_real_layer_arm"] and report["PLAN"] == queue["PLAN"], "worker receipt not accepted")
            require(report["comparison_executed"] == (name != "A1_baseline"), "A1 alone generates reference")
            if name != "A1_baseline":
                require(report["bit_gate"]["different_bits"] == 0, "first real layer bit difference; stop")
            job["report"] = identity(args.run / name / "report.json")
            save()
        queue["status"] = "four_real_layer_arms_complete_three_actual_bit_gates"
    except Exception as error:
        queue.setdefault("first_failure_status", queue["status"])
        queue["status"] = "bounded_real_layer_queue_failed_stopped"
        queue["error_type"], queue["error"] = type(error).__name__, str(error)
        raise
    finally:
        try:
            queue["bindings_after"] = check_bindings(args.root, args.workspace, plan)
            queue["bindings_unchanged"] = queue["bindings_after"] == before
            queue["PLAN_after"] = identity(args.workspace / "PLAN.json")
            queue["PLAN_unchanged"] = queue["PLAN_after"] == plan_identity
        except Exception as error:
            queue["bindings_unchanged"] = False
            queue["postcondition_error"] = {"error_type": type(error).__name__, "error": str(error)}
        queue["controller_observation_seconds"] = time.monotonic() - started
        save()
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    require(queue["bindings_unchanged"] and queue.get("PLAN_unchanged", False), "controller post-source/input/PLAN gate failed")
    print(json.dumps({"status": queue["status"], "jobs": len(queue["jobs"])}))


if __name__ == "__main__":
    main()
