"""One finite CPU8 saved-geometry diagnostic after explicit PLAN approval."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

from common import (check_bindings, digest, owned_regular_lock,
                    save_exclusive, validate_plan)


def stop_child(child, limits):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The child can exit between poll() and the group signal.
        pass
    try:
        child.wait(timeout=limits["terminate_wait_seconds"])
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=limits["kill_wait_seconds"])


def outer_timeout(signum, frame):
    raise TimeoutError("the finite 300s outer-controller limit expired")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approved-plan-sha", required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.approved_plan_sha:
        raise ValueError("explicit approved PLAN SHA differs")
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    cap = int(plan["limits"]["address_space_bytes"])
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    os.sched_setaffinity(0, plan["physical_cores"])
    run = Path(plan["run_directory"])
    if not run.is_dir() or (run / "controller.private.json").exists() or (run / "worker").exists():
        raise ValueError("a new registered private diagnostic directory is required")
    os.umask(0o077)
    env = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH",
                 "OPENBLAS_CORETYPE", "FS_SetVoxToRasXform_Change_VoxSize"):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               NUMEXPR_NUM_THREADS="8")
    report = {"status": "waiting_for_common_lock", "PID": int(os.getpid()),
              "PGID": int(os.getpgrp()), "UID": int(os.getuid()),
              "plan_sha256": args.approved_plan_sha,
              "new_registration_calls": 0, "new_official_commands": 0,
              "new_GEMS_calls": 0, "new_GPU_calls": 0}
    started, code, child, descriptor = time.monotonic(), 1, None, None
    previous_handler = signal.signal(signal.SIGALRM, outer_timeout)
    signal.alarm(int(plan["limits"]["outer_seconds"]))
    try:
        descriptor = owned_regular_lock(plan["common_CPU_lock"])
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= plan["limits"]["lock_wait_seconds"]:
                    raise TimeoutError("common CPU lock exceeded 120s; zero numerical work")
                time.sleep(.1)
        report["lock_wait_seconds"] = float(time.monotonic() - started)
        report["bindings_before"] = check_bindings(plan["bindings"])
        report["harness_before"] = check_bindings(plan["harness_bindings"])
        if Path(__file__).resolve() != (Path(plan["workspace_directory"]) / "run_prepared.py").resolve():
            raise ValueError("the invoked controller is not the frozen controller")
        command = [plan["python"], plan["worker"], "--plan", str(args.plan),
                   "--approved-plan-sha", args.approved_plan_sha,
                   "--cpu-lock-fd", str(descriptor)]
        with (run / "worker.stdout.private.txt").open("xb") as stdout, \
                (run / "worker.stderr.private.txt").open("xb") as stderr:
            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=True, pass_fds=(descriptor,))
            report["child_PID"] = int(child.pid)
            child_started = time.monotonic()
            try:
                code = int(child.wait(timeout=plan["limits"]["child_seconds"]))
            except subprocess.TimeoutExpired:
                stop_child(child, plan["limits"])
                code = 124
            report["child_wall_seconds"] = float(time.monotonic() - child_started)
        report["bindings_after"] = check_bindings(plan["bindings"])
        report["harness_after"] = check_bindings(plan["harness_bindings"])
        private_result = run / "worker/report.private.json"
        if private_result.exists():
            report["saved_private_report"] = {
                "bytes": int(private_result.stat().st_size), "sha256": digest(private_result)}
        report["status"] = "completed" if code == 0 else "stopped_first_failure"
    except BaseException as error:
        code = 1
        report["status"] = "stopped_first_failure"
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
        (run / "exception.private.txt").write_text(traceback.format_exc())
    finally:
        # Alarm remains active while an unexpected child is terminated. No retry.
        try:
            stop_child(child, plan["limits"])
        except BaseException as error:
            code = 1
            report["status"] = "stopped_first_failure"
            report["cleanup_exception"] = {
                "type": type(error).__name__, "message": str(error)}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            if descriptor is not None:
                os.close(descriptor)
        report["child_reaped"] = bool(child is None or child.poll() is not None)
        report["unreaped_child_retains_inherited_common_lock"] = bool(
            child is not None and child.poll() is None)
        report["wall_seconds"] = float(time.monotonic() - started)
        report["returncode"] = int(code)
        save_exclusive(run / "controller.private.json", report)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
