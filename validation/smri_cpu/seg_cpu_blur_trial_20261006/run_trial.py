"""Wait for the shared CPU lock, then run the declared bounded blur trial."""
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


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","workspace","run","source","checkpoint"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    args.run.mkdir(mode=0o700)
    root=args.root
    plan_path=args.workspace/"PLAN.json"
    plan=json.loads(plan_path.read_text())
    for name,digest in plan["prototype_files"].items():
        assert sha(args.workspace/name)==digest
    affinity=plan["cpu_affinity"]
    queue={"schema":"fnit_seg_cpu_blur_trial_queue/v1","status":"waiting_for_common_CPU_lock",
        "controller_PID":os.getpid(),"controller_sha256":sha(__file__),
        "plan_sha256":sha(plan_path),"source_files":plan["source_files"],
        "workspace":str(args.workspace),"run":str(args.run),"source":str(args.source),
        "checkpoint":str(args.checkpoint),"jobs":[],"GEMS_full_not_interrupted":True,
        "declared_modes":["contracts","resume","A1_baseline","B1_candidate","B2_candidate","A2_baseline"],
        "updated_epoch":time.time()}
    queue_path=args.run/"queue.private.json"
    write(queue_path,queue)
    with (root/plan["lock_fnit_relative_path"]).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue.update(status="running_bounded_trial",lock_acquired_epoch=time.time(),updated_epoch=time.time())
        write(queue_path,queue)
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8",
             "OPENBLAS_NUM_THREADS":"8","NUMBA_NUM_THREADS":"8"}
        executable=root/"envs/default/bin/python"
        commands=[("contracts",[str(executable),str(args.workspace/"check_contracts.py"),
            "--source",str(args.source),"--output",str(args.run/"CONTRACTS.private.json")],60)]
        worker=args.workspace/"real_trial.py"
        for name,mode in (("resume","resume"),("A1_baseline","baseline"),("B1_candidate","candidate"),
                          ("B2_candidate","candidate"),("A2_baseline","baseline")):
            command=[str(executable),str(worker),"--root",str(root),"--source",str(args.source),
                "--plan",str(plan_path),"--checkpoint",str(args.checkpoint),"--output",str(args.run/name),
                "--mode",mode]
            if mode!="resume":
                command.extend(["--preblur",str(args.run/"resume/preblur.private.npy")])
                if name!="A1_baseline":
                    command.extend(["--reference",str(args.run/"A1_baseline/baseline_blur.private.npy")])
            commands.append((name,command,300 if mode=="resume" else 120))
        for name,command,timeout in commands:
            job={"name":name,"status":"running","start_epoch":time.time(),"hard_timeout_seconds":timeout}
            queue["jobs"].append(job)
            write(queue_path,queue)
            with (args.run/(name+".log")).open("wb") as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                         preexec_fn=lambda:os.sched_setaffinity(0,affinity))
                job["PID"]=process.pid
                write(queue_path,queue)
                try:
                    result=process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();result=-9
                    job["timeout"]=True
            job.update(status="completed" if result==0 else "failed",exit_code=result,
                       end_epoch=time.time(),wall_seconds=time.time()-job["start_epoch"])
            queue["updated_epoch"]=time.time()
            if result:
                queue["status"]="bounded_trial_stopped_at_first_failed_gate"
                write(queue_path,queue)
                raise SystemExit(1)
            write(queue_path,queue)
        queue.update(status="bounded_trial_complete",updated_epoch=time.time(),
                     whole_T1_map_CSV_status="not_assessed")
        write(queue_path,queue)
    print(json.dumps({"status":queue["status"],"jobs":len(queue["jobs"])}))


if __name__=="__main__":
    main()
