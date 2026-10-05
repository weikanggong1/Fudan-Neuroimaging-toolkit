"""One locked, finite official/FNIT target-preparation comparison.

This controller must be explicitly authorized and dispatched separately.
The plan names interpreters and a source-bound worker, never a registration
binary. It is not an end-to-end GEMS or affine benchmark.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    assert plan["scope"] == "one_real_target_preparation_only"
    assert plan["registration_calls"] == 0 and plan["whole_GEMS_calls"] == 0
    assert plan["host"] == os.uname().nodename
    assert sha(plan["worker"]["path"]) == plan["worker"]["sha256"]
    assert sha(__file__) == plan["controller"]["sha256"]
    assert plan["threads"] == 8 and len(plan["affinity"]) == 8
    os.sched_setaffinity(0, set(plan["affinity"]))
    directory = Path(plan["run_directory"])
    assert directory.is_dir()
    receipt = {"status": "waiting_for_CPU8_lock", "plan_sha256": sha(args.plan),
               "controller_sha256": sha(__file__), "pid": os.getpid(),
               "worker_count": 0, "registration_calls": 0, "whole_GEMS_calls": 0}
    receipt_path = directory / "controller.private.json"
    child = None

    def save():
        temporary = directory / "controller.private.tmp"
        with temporary.open("w") as file:
            json.dump(receipt, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, receipt_path)

    def cancelled(number, frame):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            except ProcessLookupError:
                pass
        receipt["status"] = "outer_deadline_or_signal_own_child_stopped"
        receipt["signal"] = number
        save()
        raise SystemExit(124 if number == signal.SIGALRM else 128 + number)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(number, cancelled)
    signal.alarm(plan["outer_deadline_seconds"])

    save()
    with open(plan["CPU8_lock"], "a+") as lock:
        wait_started = time.monotonic()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - wait_started > plan["lock_wait_seconds"]:
                    receipt["status"] = "CPU_lock_wait_timeout_zero_worker"
                    save()
                    return 124
                time.sleep(1)
        receipt["lock_wait_seconds"] = time.monotonic() - wait_started
        receipt["status"] = "running_target_preparation_only"
        receipt["arms"] = []
        save()
        env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                   NUMEXPR_NUM_THREADS="8", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE",
                   CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OPENBLAS_CORETYPE"):
            env.pop(name, None)

        def child_limits():
            resource.setrlimit(resource.RLIMIT_AS, (plan["address_space_cap_bytes"],) * 2)
        for arm in ("official", "fnit", "score"):
            executable = plan["official_python"] if arm == "official" else plan["fnit_python"]
            started = time.monotonic()
            with (directory / (arm + ".stdout")).open("x") as out, (directory / (arm + ".stderr")).open("x") as error:
                child = subprocess.Popen(
                    [executable, plan["worker"]["path"], "--plan", str(args.plan), "--arm", arm],
                    env=env, stdout=out, stderr=error, start_new_session=True,
                    preexec_fn=child_limits,
                )
                receipt["worker_count"] += 1
                receipt["current_arm"] = arm
                receipt["current_child_pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=plan["child_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    code = 124
            receipt["arms"].append({"arm": arm, "exit": code, "new_process_wall_seconds": time.monotonic() - started})
            save()
            if code:
                receipt["status"] = "failed_stopped_before_remaining_arms"
                save()
                return code
        receipt["status"] = "completed_target_preparation_gates_pass"
        receipt.pop("current_child_pid", None)
        save()
        signal.alarm(0)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
