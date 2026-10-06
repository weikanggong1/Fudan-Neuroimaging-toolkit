"""ONE saved-output score recovery; never dispatch registration phases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from register_benchmark import atomic_report, check_bindings, digest
from run_benchmark import acquire, clean_environment, stop_owned_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approved-plan-sha", required=True)
    parser.add_argument("--recovery-bindings", required=True)
    parser.add_argument("--approved-recovery-bindings-sha", required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.approved_plan_sha:
        raise ValueError("original immutable plan changed")
    if digest(args.recovery_bindings) != args.approved_recovery_bindings_sha:
        raise ValueError("approved saved output/recovery source binding changed")
    plan = json.loads(Path(args.plan).read_text())
    saved = json.loads(Path(args.recovery_bindings).read_text())
    run = Path(plan["run_directory"])
    receipt_path = run / "score_recovery_controller.private.json"
    if receipt_path.exists() or (run / "score_byteorder_recovery_v2").exists():
        raise FileExistsError("recovery already dispatched; no automatic retry")
    os.umask(0o077)
    os.sched_setaffinity(0, plan["physical_cores"])
    receipt = {"scope": "ONE saved-output score; no official/FNIT registration, GEMS or GPU",
               "status": "starting", "controller_PID": os.getpid(),
               "controller_PGID": os.getpgrp(), "UID": os.getuid(),
               "controller_start_ticks": Path('/proc/self/stat').read_text().split()[21],
               "plan_sha256": args.approved_plan_sha,
               "recovery_bindings_sha256": args.approved_recovery_bindings_sha,
               "worker_started": False, "automatic_retry": False}
    code, active, started = 1, None, time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("900s finite recovery outer deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        receipt["bindings_before"] = check_bindings(plan)
        receipt["saved_output_bindings_before"] = check_bindings(saved)
        with Path(plan["common_CPU_lock"]).open('a+') as lock:
            receipt["lock_wait_seconds"] = acquire(lock, 600)
            command = [plan["python"], saved["worker_path"], "--plan", args.plan,
                       "--approved-plan-sha", args.approved_plan_sha, "--phase", "score",
                       "--recovery-bindings", args.recovery_bindings]
            with (run / "score_recovery.controller.stdout").open('xb') as stream:
                active = subprocess.Popen(command, env=clean_environment(), stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                receipt["worker_started"] = True
                receipt["worker_PID"] = active.pid
                receipt["worker_PGID"] = active.pid
                receipt["worker_start_ticks"] = Path('/proc/'+str(active.pid)+'/stat').read_text().split()[21]
                try:
                    code = active.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    stop_owned_group(active)
                    code = 124
            active = None
            receipt["status"] = "completed_gates_passed" if code == 0 else "saved_score_failed"
    except BaseException as error:
        if active is not None:
            stop_owned_group(active)
        receipt["status"] = "controller_failed"
        receipt["exception"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        signal.alarm(0)
        receipt["wall_seconds"] = time.monotonic()-started
        try:
            receipt["bindings_after"] = check_bindings(plan)
            receipt["saved_output_bindings_after"] = check_bindings(saved)
            receipt["bindings_before_after_exact"] = receipt.get("bindings_before") == receipt["bindings_after"]
            receipt["saved_outputs_before_after_exact"] = receipt.get("saved_output_bindings_before") == receipt["saved_output_bindings_after"]
        except BaseException as error:
            code = 1
            receipt["binding_error_after"] = {"type": type(error).__name__, "message": str(error)}
        receipt["returncode"] = code
        atomic_report(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "returncode": code,
                          "worker_started": receipt["worker_started"]}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
