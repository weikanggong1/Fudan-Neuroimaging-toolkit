"""Prepared finite CPU8 prefix capture then C24 ABBA; separate approval required."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time

from real_bindings import atomic_json, check_bindings, identity, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-real-layer", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    require(args.workspace.resolve() == (args.root / plan["workspace"]).resolve() and
            args.run.resolve() == (args.root / plan["runs"]).resolve(), "declared canonical paths required")
    freeze = json.loads((args.workspace / "freeze.public.json").read_text())
    for name, binding in freeze["payload"].items():
        require(identity(args.workspace / name) == binding, "frozen real-layer payload changed")
    before = check_bindings(args.root, args.workspace, plan)
    require(shutil.disk_usage(args.run.parent).free >= 6_000_000_000, "at least6GB free disk before dispatch")
    args.run.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    deadline = started + plan["limits"]["outer_total_seconds"]
    report = {"schema": "fnit_C24_real_layer_queue/v1", "status": "waiting_common_CPU_lock",
              "controller_PID": os.getpid(), "PLAN": identity(args.workspace / "PLAN.json"),
              "bindings_before": before, "arms": [], "completed": False,
              "model_forward_calls": 0, "native_calls": 0, "GPU_calls": 0, "new_compiles": 0}
    fd, process, locked = None, None, False
    def save():
        atomic_json(args.run / "QUEUE.json", report)
    def expired(signum, frame):
        raise TimeoutError("finite outer1020s expired; no retry")
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(plan["limits"]["outer_total_seconds"])
    save()
    try:
        fd = os.open(args.root / plan["CPU_lock"], os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        details = os.fstat(fd)
        require(stat.S_ISREG(details.st_mode) and details.st_uid == os.geteuid(), "owned regular common lock required")
        lock_deadline = min(deadline, time.monotonic() + 120)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= lock_deadline:
                    raise TimeoutError("common CPU lock120s; no science child launched")
                time.sleep(min(1, lock_deadline - time.monotonic()))
        report["lock_wait_seconds"] = time.monotonic() - started
        report["status"] = "common_CPU_lock_acquired"
        require(shutil.disk_usage(args.run).free >= 6_000_000_000, "disk budget no longer sufficient")
        save()
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                            "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1"})
        for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "OPENBLAS_CORETYPE"):
            environment.pop(name, None)
        phases = [("capture", "capture_prefix.py", None)] + [(name, "layer_pair.py", mode) for name, mode in
                  (("A1_baseline", "baseline"), ("B1_candidate", "candidate"),
                   ("B2_candidate", "candidate"), ("A2_baseline", "baseline"))]
        for name, script, mode in phases:
            require(check_bindings(args.root, args.workspace, plan) == before, "source changed before phase")
            command = [str(args.root / "envs/default/bin/python"), str(args.workspace / script),
                       "--root", str(args.root), "--workspace", str(args.workspace), "--run", str(args.run),
                       "--approved-real-layer"]
            if mode is not None:
                command += ["--output", str(args.run / name), "--mode", mode]
            arm = {"phase": name, "status": "starting", "timeout_seconds": 180}
            report["arms"].append(arm)
            save()
            arm_started = time.monotonic()
            with (args.run / (name + ".log")).open("xb") as log:
                process = subprocess.Popen(command, env=environment, cwd=args.workspace, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           preexec_fn=lambda: os.sched_setaffinity(0, plan["affinity"]))
                arm["PID"] = process.pid
                arm["status"] = "running"
                save()
                try:
                    code = process.wait(timeout=min(180, max(0.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                    arm["returncode"] = process.returncode
                    arm["status"] = "timeout_first_failure_stopped"
                    save()
                    raise TimeoutError("worker180s; no subsequent phase")
            arm["returncode"] = code
            arm["cold_worker_seconds"] = time.monotonic() - arm_started
            arm["status"] = "exit0_pending_receipt" if code == 0 else "nonzero_first_failure_stopped"
            receipt_path = args.run / name / "report.json"
            if receipt_path.is_file():
                arm["receipt"] = identity(receipt_path)
            arm["log"] = identity(args.run / (name + ".log"))
            save()
            require(code == 0, "first worker failure; no subsequent phases")
            receipt = json.loads(receipt_path.read_text())
            require(receipt["PLAN"] == report["PLAN"], "receipt PLAN mismatch")
            if name == "capture":
                require(receipt["valid_capture"] and receipt["conv0_calls"] == 2 and receipt["conv1_calls"] == 0,
                        "prefix receipt failed")
            else:
                require(receipt["valid_layer_arm"] and receipt["conv1_calls"] == 2, "real layer receipt failed")
                expected = 12 if mode == "candidate" else 0
                require(receipt["candidate_copy_calls"] == receipt["candidate_SGEMM_calls"] == expected, "real slab count failed")
                require(len(receipt["passes"]) == 2 and all(row["bit_gate"]["finite"] for row in receipt["passes"]), "two finite passes required")
                if name != "A1_baseline":
                    require(all(row["comparison_executed"] and row["bit_gate"]["different_bits"] == 0 for row in receipt["passes"]), "first bit difference")
            require(all(receipt.get(key) == 0 for key in ("model_forward_calls", "native_calls", "GPU_calls", "new_compile_calls")), "forbidden phase calls")
            arrays = list((args.run / "capture").glob("*.private.npy")) + list((args.run / "A1_baseline").glob("*.private.npy"))
            report["new_private_array_bytes"] = sum(path.stat().st_size for path in arrays)
            require(report["new_private_array_bytes"] <= plan["limits"]["maximum_new_private_array_bytes_including_headers"], "array disk cap failed")
            report["new_receipt_and_log_bytes"] = sum(path.stat().st_size for path in args.run.rglob("*")
                                                      if path.is_file() and not path.name.endswith(".private.npy"))
            require(report["new_receipt_and_log_bytes"] <= plan["limits"]["max_receipt_and_logs_total_bytes"],
                    "metadata/log disk cap failed")
            arm["status"] = "exit0_all_phase_gates_passed"
            save()
        report["completed"] = True
        report["status"] = "prefix_and_two_pass_ABBA_bit_exact_no_full_CNN"
    except BaseException as error:
        report["status"] = "bounded_real_queue_first_failure_stopped"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                report["child_wait_failure"] = True
        try:
            report["bindings_after"] = check_bindings(args.root, args.workspace, plan)
            report["bindings_unchanged"] = before == report["bindings_after"]
        except Exception as error:
            report["bindings_unchanged"] = False
            report["postcondition_error"] = str(error)
        report["controller_seconds"] = time.monotonic() - started
        report["valid_queue"] = bool(report["completed"] and report["bindings_unchanged"])
        try:
            save()
        finally:
            if fd is not None:
                if locked:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    require(report["valid_queue"], "real queue final gate failed")


if __name__ == "__main__":
    main()
