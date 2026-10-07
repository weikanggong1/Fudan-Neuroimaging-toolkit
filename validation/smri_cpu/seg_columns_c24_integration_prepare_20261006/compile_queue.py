"""Two fresh metadata workers: cold C24 build, then private-cache interface reuse."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from controller_safety import controller_deadline, locked_child


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "source", "plan", "bindings", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-compile-contract", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    bindings = json.loads(args.bindings.read_text())
    group = plan["compile_contract"]
    assert bindings["public_PLAN_sha256"] == sha(args.plan)
    assert sha(__file__) == plan["validation_sources"]["compile_queue.py"]
    worker = args.workspace / "compile_contract.py"
    assert sha(worker) == plan["validation_sources"]["compile_contract.py"]
    assert sha(args.workspace / "controller_safety.py") == plan["validation_sources"]["controller_safety.py"]
    args.run.mkdir(mode=0o700, exist_ok=False)
    cache = args.run / "new_C24_compile_cache"
    assert not cache.exists()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
                   "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8",
                   "PYTHONDONTWRITEBYTECODE": "1", "CXX": bindings["Conda_CXX"],
                   "FNIT_SYNTHSEG_C24_CPU_CACHE": str(cache),
                   "FNIT_SYNTHSEG_CPU_CACHE": str(args.root / bindings["C72_cache_directory_fnit_relative"])}
    for name in ("PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "OPENBLAS_CORETYPE"):
        environment.pop(name, None)
    receipt = {"schema": "fnit_C24_compile_queue/v1", "status": "prepared", "jobs": [],
               "PLAN_sha256": sha(args.plan), "validation_helper_before": sha(args.workspace / "controller_safety.py"), "private_bindings_sha256": sha(args.bindings)}
    def save():
        record = args.run / "QUEUE.json"
        temporary = record.with_suffix(".temporary")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temporary, record)
    try:
        with controller_deadline(group["outer_total_seconds"]):
            deadline = time.monotonic() + group["outer_total_seconds"]
            first = None
            for mode in ("cold", "warm"):
                started_wait = time.monotonic()
                with (args.root / group["common_lock"]).open("a+b") as lock:
                    receipt["status"] = "waiting_lock"
                    save()
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if (time.monotonic() - started_wait >= group["lock_wait_seconds"]
                                    or time.monotonic() + group["worker_timeout_seconds"] + 35 >= deadline):
                                receipt["status"] = "failed_lock_deadline_stop"
                                save()
                                return 1
                            time.sleep(1)
                    assert time.monotonic() + group["worker_timeout_seconds"] + 35 < deadline
                    assert cache.exists() == (mode == "warm")
                    command = ["/usr/bin/taskset", "-c", ",".join(map(str, group["affinity"])),
                               "/usr/bin/timeout", "--kill-after=30", str(group["worker_timeout_seconds"]),
                               str(args.root / "envs/default/bin/python"), str(worker), "--root", str(args.root),
                               "--source", str(args.source), "--plan", str(args.plan), "--bindings", str(args.bindings),
                               "--output", str(args.run / mode), "--mode", mode, "--approved-compile-contract"]
                    started = time.monotonic()
                    receipt["status"] = "running_" + mode
                    save()
                    with (args.run / (mode + ".log")).open("xb") as log:
                        child_receipt = locked_child(command, environment, log, lock.fileno(), group["worker_timeout_seconds"] + 35)
                        code = child_receipt["returncode"]
                    row = {"mode": mode, "returncode": code, "cold_worker_seconds": time.monotonic() - started, **child_receipt}
                    receipt["jobs"].append(row)
                    if code or not child_receipt["child_reaped"]:
                        receipt["status"] = "first_failure_stopped"
                        save()
                        return code
                    try:
                        assert sha(args.workspace / "controller_safety.py") == receipt["validation_helper_before"] == plan["validation_sources"]["controller_safety.py"]
                        report_path = args.run / mode / "COMPILE_CONTRACT.json"
                        report = json.loads(report_path.read_text())
                        row["report_sha256"] = sha(report_path)
                        assert report["valid_compile_contract"] and report["compile_calls"] == (1 if mode == "cold" else 0)
                        if first is None:
                            first = report
                        else:
                            assert report["artifact"] == first["artifact"] and report["library"] == first["library"]
                            assert report["cache_manifest"] == first["cache_manifest"]
                    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
                        row.update(postcondition_error_type=type(error).__name__, postcondition_error=str(error))
                        receipt["status"] = "first_postcondition_failure_stopped"
                        save()
                        return 1
                    save()
            receipt["validation_helper_after"] = sha(args.workspace / "controller_safety.py")
            assert receipt["validation_helper_after"] == receipt["validation_helper_before"] == plan["validation_sources"]["controller_safety.py"]
            receipt["status"] = "complete_metadata_only"
            save()
            return 0
    except BaseException as error:
        receipt.update(status="controller_exception_stopped", error_type=type(error).__name__, error=str(error))
        if hasattr(error, "child_receipt"):
            receipt["jobs"].append({"exceptional_child": True, **error.child_receipt})
        save()
        return 124 if isinstance(error, TimeoutError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
