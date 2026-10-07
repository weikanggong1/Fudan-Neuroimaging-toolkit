"""Prepared finite CPU8 dispatcher; invoke only after root approves PLAN SHA."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from register_benchmark import atomic_report, check_bindings, digest


def clean_environment():
    environment=os.environ.copy()
    for key in ("LD_LIBRARY_PATH","LD_PRELOAD","PYTHONPATH","OPENBLAS_CORETYPE",
                "FS_SetVoxToRasXform_Change_VoxSize"):
        environment.pop(key,None)
    environment.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",
                       NUMEXPR_NUM_THREADS="8",NUMBA_NUM_THREADS="8")
    return environment


def acquire(stream, timeout):
    started=time.monotonic()
    while True:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            return time.monotonic()-started
        except BlockingIOError:
            if time.monotonic()-started>=timeout:
                raise TimeoutError("bounded common CPU8 lock wait expired; zero scientific worker")
            time.sleep(.25)


def stop_owned_group(process):
    # Popen(start_new_session=True) creates this controller's own worker
    # process group. Never kill by node/process-name/old PID guesses.
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--approved-plan-sha",required=True)
    parser.add_argument("--dispatch",action="store_true",required=True)
    args=parser.parse_args()
    if digest(args.plan)!=args.approved_plan_sha:
        raise ValueError("approved immutable plan SHA mismatch")
    plan=json.loads(Path(args.plan).read_text())
    os.umask(0o077)
    os.sched_setaffinity(0,plan["physical_cores"])
    run=Path(plan["run_directory"])
    run.mkdir(mode=0o700,parents=False,exist_ok=False)
    receipt={"status":"starting","controller_PID":os.getpid(),"controller_PGID":os.getpgrp(),
             "controller_start_ticks":Path('/proc/self/stat').read_text().split()[21],
             "UID":os.getuid(),"plan_sha256":args.approved_plan_sha,"phases":[],
             "scientific_worker_started":False,"automatic_retry":False}
    code=1;active=None;started=time.monotonic()

    def deadline(signum, frame):
        raise TimeoutError("hard outer controller deadline")

    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(plan["limits"]["outer_controller_seconds"])
    try:
        receipt["bindings_before"]=check_bindings(plan)
        with Path(plan["common_CPU_lock"]).open("a+") as lock:
            receipt["lock_wait_seconds"]=acquire(lock,plan["limits"]["lock_wait_seconds"])
            locked=time.monotonic()
            for phase in ("official","fnit","score"):
                if time.monotonic()-locked>=plan["limits"]["controller_after_lock_seconds"]:
                    raise TimeoutError("locked total deadline reached")
                command=[plan["python"],str(Path(plan["worker_directory"])/"register_benchmark.py"),
                         "--plan",args.plan,"--approved-plan-sha",args.approved_plan_sha,"--phase",phase]
                phase_started=time.monotonic()
                with (run/(phase+".controller.stdout")).open("xb") as stream:
                    active=subprocess.Popen(command,env=clean_environment(),stdout=stream,
                                            stderr=subprocess.STDOUT,start_new_session=True)
                    receipt["scientific_worker_started"]=True
                    atomic_report(run/(phase+".started.private.json"),{
                        "worker_PID":active.pid,"worker_PGID":active.pid,
                        "worker_start_ticks":Path('/proc/'+str(active.pid)+'/stat').read_text().split()[21],
                        "controller_PID":os.getpid(),"phase":phase,"plan_sha256":args.approved_plan_sha})
                    try:
                        result=active.wait(timeout=plan["limits"]["each_arm_child_seconds"])
                    except subprocess.TimeoutExpired:
                        stop_owned_group(active);result=124
                receipt["phases"].append({"phase":phase,"returncode":result,
                                           "outer_wall_seconds":time.monotonic()-phase_started})
                active=None
                if result:
                    code=result;receipt["status"]="first_failure_stopped";break
            else:
                code=0;receipt["status"]="completed_declared_gates_passed"
    except BaseException as error:
        if active is not None:
            stop_owned_group(active)
        receipt["status"]="controller_failed";receipt["exception"]={"type":type(error).__name__,"message":str(error)}
    finally:
        signal.alarm(0)
        receipt["outer_wall_seconds"]=time.monotonic()-started
        try:
            receipt["bindings_after"]=check_bindings(plan)
            receipt["bindings_before_after_exact"]=receipt.get("bindings_before")==receipt["bindings_after"]
        except BaseException as error:
            code=1;receipt["binding_error_after"]={"type":type(error).__name__,"message":str(error)}
        receipt["returncode"]=code
        atomic_report(run/"controller.receipt.private.json",receipt)
        print(json.dumps({"status":receipt["status"],"returncode":code,"worker_started":receipt["scientific_worker_started"]}))
    raise SystemExit(code)


if __name__=="__main__":
    main()
