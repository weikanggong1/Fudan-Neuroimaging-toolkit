"""Approved metadata lazy build then one finite short worker; no whole/MRI arm."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from phase1_bindings import check_sources, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root","workspace","run"):
        parser.add_argument("--" + name,type=Path,required=True)
    parser.add_argument("--approved-phase1",action="store_true")
    args = parser.parse_args()
    if not args.approved_phase1: raise RuntimeError("prepared only; separate coordinator phase1 approval required")
    os.umask(0o077)
    plan = json.loads((args.workspace / "PLAN.json").read_text())
    if args.workspace.resolve() != (args.root / plan["workspace_fnit_relative"]).resolve():
        raise RuntimeError("new canonical candidate workspace required")
    declared_run = args.root / plan["runs_fnit_relative"] / "phase1"
    if args.run.resolve() != declared_run.resolve(): raise RuntimeError("new canonical phase1 run required")
    before = check_sources(args.root,args.workspace,plan)
    args.run.mkdir(mode=0o700) # unique; no receipt/cache replacement
    cache = args.run / "private_contract_cache"
    if cache.exists(): raise RuntimeError("new private cache required")
    started = time.monotonic(); deadline = started + 23000
    receipt = {"schema":"fnit_columns_lazy_phase1_queue/v1", "status":"waiting_common_CPU_lock",
        "controller_PID":os.getpid(),"PLAN":identity(args.workspace / "PLAN.json"),
        "bindings_before":before,"outer_total_seconds":23000,"arms":[],
        "MRI_authorized":False,"MRI_calls":0,"whole_calls":0,"native_calls":0,"GPU_calls":0}
    record = args.run / "QUEUE.json"
    def save(): record.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    save()
    environment = {**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8",
        "OPENBLAS_NUM_THREADS":"8","NUMBA_NUM_THREADS":"8","PYTHONDONTWRITEBYTECODE":"1",
        "CXX":plan["Conda_CXX"],"FNIT_SYNTHSEG_CPU_CACHE":str(cache)}
    for name in ("PYTHONPATH","LD_PRELOAD","LD_LIBRARY_PATH","OPENBLAS_CORETYPE"):
        environment.pop(name,None)
    for script,output,limit in (("load_cache_interface.py","INTERFACE_LOAD.json",180),
                                ("check_cached_contracts.py","CONTRACTS.json",240)):
        try:
            receipt.update(status="waiting_common_CPU_lock",next_worker=script);save()
            lock_path = args.root / plan["whole_CPU"]["common_lock"]
            fd = os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            try:
                while True:
                    try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        if time.monotonic()+limit+35 >= deadline: raise TimeoutError("finite queue deadline before worker")
                        time.sleep(5)
                if time.monotonic()+limit+35 >= deadline: raise TimeoutError("finite queue deadline before worker")
                if script == "load_cache_interface.py" and cache.exists(): raise RuntimeError("cold contract cache already exists")
                command = ["/usr/bin/taskset","-c",",".join(map(str,plan["whole_CPU"]["affinity"])),
                    "/usr/bin/timeout","--kill-after=30",str(limit),str(args.root / "envs/default/bin/python"),
                    str(args.workspace/script),"--root",str(args.root),"--workspace",str(args.workspace),
                    "--output",str(args.run/output)]
                if script == "load_cache_interface.py":command.append("--approved-phase1")
                else:command += ["--approved-contracts","--interface",str(args.run/"INTERFACE_LOAD.json")]
                tick = time.monotonic();receipt.update(status="running",next_worker=script);save()
                with (args.run/(script+".log")).open("x") as log:
                    child = subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True)
                    try:rc=child.wait(timeout=limit+35)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                        rc=124
                row={"worker":script,"returncode":rc,"worker_wall_seconds":time.monotonic()-tick}
                receipt["arms"].append(row);save()
            finally:
                fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
            if rc:raise RuntimeError("first worker failure; stop remaining: "+script)
            actual=json.loads((args.run/output).read_text());row["receipt"]=identity(args.run/output)
            if script == "load_cache_interface.py":
                assert actual["valid_interface"] and actual["compile_calls"]==1
                assert all(actual[key]==0 for key in ("copy_calls","SGEMM_calls","MRI_calls","model_forward_calls"))
            else:
                assert actual["valid_bounded_contracts"] and len(actual["numeric_rows"])==6 and len(actual["copy_rows"])==13
                assert all(row["different_bits"]==0 for row in actual["numeric_rows"]+actual["copy_rows"])
                assert actual["candidate_copy_calls"]==actual["candidate_SGEMM_calls"]==12
                assert len(actual["guard_rows"])==actual["fallback_calls"]==23
                assert actual["new_compilation_calls"]==actual["MRI_calls"]==actual["native_calls"]==actual["model_forward_calls"]==0
            receipt.update(status="worker_gate_passed_lock_released");save()
        except BaseException as error:
            receipt.update(status="failed_stop_remaining",error_type=type(error).__name__,error=str(error),
                ended_monotonic=time.monotonic());save();return 1
    receipt["bindings_after"]=check_sources(args.root,args.workspace,plan)
    if receipt["bindings_after"] != before:
        receipt.update(status="source_after_gate_failed_stop_whole");save();return 1
    receipt.update(status="metadata_and_bounded_contracts_exit0_no_whole",ended_monotonic=time.monotonic());save();return 0


if __name__ == "__main__":raise SystemExit(main())
