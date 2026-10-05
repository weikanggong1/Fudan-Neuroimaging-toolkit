"""Prepared metadata+contracts queue; explicit approval required; no MRI arm exists."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from bindings import check_sources, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--approved-contracts", action="store_true", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    if (args.workspace.resolve() != (args.root / plan["workspace"]).resolve()
            or args.run.resolve() != (args.root / plan["runs"]).resolve()):
        raise RuntimeError("new canonical v2 workspace/run required")
    before = check_sources(args.root, args.workspace, plan)
    if args.run.exists():
        if not args.run.is_dir() or list(args.run.iterdir()):
            raise RuntimeError("new or empty private run required; never overwrite a receipt")
    else:
        args.run.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    deadline = started + 23000
    lock = args.root / "runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    receipt = {"status": "waiting_common_CPU_lock", "controller_PID": os.getpid(),
               "outer_total_seconds": 23000, "PLAN": identity(args.workspace / "PLAN.json"),
               "bindings_before": before, "MRI_authorized": False, "MRI_calls": 0,
               "new_compilation_calls": 0, "arms": []}
    queue = args.run / "QUEUE.json"
    def save():
        queue.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    save()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline - 300:
                    raise TimeoutError("one bounded queue expired before contracts; no retry")
                time.sleep(1)
        receipt["status"] = "acquired_common_CPU_lock"
        receipt["wait_seconds"] = time.monotonic() - started
        save()
        interpreter = args.root / "envs/default/bin/python"
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                            "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1"})
        for name in ("PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "OPENBLAS_CORETYPE"):
            environment.pop(name, None)
        for script, output, timeout in (("load_interface.py", "INTERFACE_LOAD.json", 60),
                                       ("check_contracts.py", "CONTRACTS.json", 180)):
            command = [str(interpreter), str(args.workspace / script), "--root", str(args.root), "--workspace", str(args.workspace),
                       "--output", str(args.run / output)]
            if script == "check_contracts.py":
                command.append("--approved-contracts")
            with (args.run / (script + ".log")).open("wb") as stream:
                arm_started = time.monotonic()
                result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=min(timeout, deadline - arm_started), check=False,
                                        preexec_fn=lambda: os.sched_setaffinity(0, plan["affinity"]))
            receipt["arms"].append({"script": script, "PID_returncode": result.returncode,
                                    "worker_wall_seconds": time.monotonic() - arm_started})
            save()
            if result.returncode:
                raise RuntimeError("first worker failure; stopped with no MRI: " + script)
        receipt["status"] = "metadata_and_bounded_contract_workers_exit0_no_MRI"
    except Exception as error:
        receipt["status"] = "bounded_queue_failed_stopped"
        receipt["error_type"], receipt["error"] = type(error).__name__, str(error)
        raise
    finally:
        receipt["outer_observation_seconds"] = time.monotonic() - started
        save()
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == "__main__":
    main()
