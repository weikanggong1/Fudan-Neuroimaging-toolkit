"""One private observed CPU API under the established shared eight-core lock."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan_path = args.workspace / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    worker = args.workspace / "profile_one.py"
    source = args.root / plan["source_fnit_relative_path"]
    assert sha(worker) == plan["freeze_worker_sha256"]
    assert {name: sha(source / "fnit/synthseg_parc" / name) for name in plan["source_files"]} == plan["source_files"]
    reference = args.root / plan["reference_fnit_relative_path"]
    for name, row in plan["saved_outputs"].items():
        assert {"bytes": (reference/name).stat().st_size, "sha256": sha(reference/name)} == row
    run = args.root / plan["run_fnit_relative_path"]
    run.mkdir(mode=0o700)
    logdir = args.root / "logs/smri_cpu_20261004/remaining_20261004/seg_cpu_profile_20261006_v1"
    logdir.mkdir(mode=0o700)
    python = (args.root / "envs/default").resolve() / "bin/python"
    assert python.is_file()
    lockpath = args.root / plan["lock_fnit_relative_path"]
    queue = {"schema": "fnit_synthseg_cpu_one_profile_queue/v1", "status": "waiting_common_lock",
        "controller_sha256": sha(__file__), "worker_sha256": sha(worker), "plan_sha256": sha(plan_path),
        "hostname": os.uname().nodename, "cpu_affinity": plan["cpu_affinity"], "threads": 8,
        "fresh_process": True, "actual_fnit_full_arms": 0, "actual_native_arms": 0,
        "lock_scope": "finite contracts and one observed complete ordinary33 API, then release",
        "created_utc": datetime.now(timezone.utc).isoformat(), "load_before_lock": list(os.getloadavg())}
    record = run / "queue.private.json"
    def write():
        record.write_text(json.dumps(queue, indent=2) + "\n")
    write()
    with lockpath.open("a+b") as lock:
        deadline = time.monotonic() + 300
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    queue["status"] = "lock_timeout_no_MRI_run"
                    write()
                    return 2
                time.sleep(.5)
        env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
               "OPENBLAS_NUM_THREADS": "8", "NUMBA_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": "",
               "PYTHONPATH": str(source)}
        queue.update(status="finite_contracts", environment={name:env[name] for name in (
            "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS","CUDA_VISIBLE_DEVICES")})
        write()
        affinity = ",".join(map(str, plan["cpu_affinity"]))
        contract_command = ["taskset","-c",affinity,str(python),str(args.workspace/"check_observer.py"),
            "--source",str(source),"--worker",str(worker),"--output",str(run/"observer_contract.private.json")]
        with (logdir/"contracts.log").open("x") as log:
            rc = subprocess.run(contract_command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60).returncode
        queue["contract_returncode"] = rc
        if rc:
            queue["status"] = "contract_failed_no_MRI_run"
            write()
            return rc
        command = ["taskset","-c",affinity,"/usr/bin/time","-v","-o",str(logdir/"profile.time.txt"),
            "timeout","--kill-after=10",str(plan["hard_timeout_seconds"]),str(python),str(worker),
            "--root",str(args.root),"--source",str(source),"--binding",str(plan_path),"--output",str(run/"arm")]
        queue.update(status="running_one_observed_API", load_before_API=list(os.getloadavg()))
        write()
        start = time.perf_counter()
        with (logdir/"profile.log").open("x") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            queue["worker_pid"] = process.pid
            queue["actual_fnit_full_arms"] = 1
            write()
            rc = process.wait(timeout=plan["hard_timeout_seconds"]+25)
        queue.update(worker_returncode=rc, observed_outer_seconds=time.perf_counter()-start,
                     load_after_API=list(os.getloadavg()))
        if rc:
            queue["status"] = "worker_failed"
            write()
            return rc
        queue["status"] = "API_complete_collecting_saved_outputs"
        write()
    # Collect after the worker is dead and shared compute lock is released.
    import nibabel as nib
    import numpy as np
    report = json.loads((run/"arm/profile.private.json").read_text())
    old, new = nib.load(reference/"segmentation.nii.gz"), nib.load(run/"arm/segmentation.nii.gz")
    a, b = np.asanyarray(old.dataobj), np.asanyarray(new.dataobj)
    with (reference/"volumes.csv").open() as stream:
        old_csv=list(csv.reader(stream))
    with (run/"arm/volumes.csv").open() as stream:
        new_csv=list(csv.reader(stream))
    assert len(old_csv)==len(new_csv)==2
    old_values=np.asarray(old_csv[1][1:],dtype=np.float32)
    new_values=np.asarray(new_csv[1][1:],dtype=np.float32)
    comparison={"comparison_status":"compared", "different_voxels":int(np.count_nonzero(a!=b)),
        "shape_equal":old.shape==new.shape,"dtype_equal":old.get_data_dtype()==new.get_data_dtype(),
        "affine_exact":np.array_equal(old.affine,new.affine),"header_exact":old.header.binaryblock==new.header.binaryblock,
        "extensions_exact":[(e.get_code(),e._raw) for e in old.header.extensions]==[(e.get_code(),e._raw) for e in new.header.extensions],
        "csv_header_exact":old_csv[0]==new_csv[0],"csv_subject_exact":old_csv[1][0]==new_csv[1][0],
        "csv_numeric_exact":np.array_equal(old_values,new_values),
        "csv_max_abs_mm3":float(np.max(np.abs(old_values-new_values))),
        "saved_files_SHA_exact":all(sha(run/"arm"/name)==row["sha256"] for name,row in plan["saved_outputs"].items()),
        "labels":[]}
    for label in np.union1d(np.unique(a),np.unique(b)):
        aa,bb=a==label,b==label
        counts=int(aa.sum()+bb.sum())
        comparison["labels"].append({"label":int(label),"dice":float(2*np.count_nonzero(aa&bb)/counts) if counts else 1.0})
    comparison["passed"]=comparison["different_voxels"]==0 and all(comparison[name] for name in (
        "shape_equal","dtype_equal","affine_exact","header_exact","extensions_exact","csv_header_exact",
        "csv_subject_exact","csv_numeric_exact","saved_files_SHA_exact"))
    report.pop("source_import",None)
    report["complete_saved_output_comparison"]=comparison
    report["queue"]={name:value for name,value in queue.items() if name!="worker_pid"}
    report["observer_contract"] = json.loads((run/"observer_contract.private.json").read_text())
    report["source_plan"] = plan
    passed=comparison["passed"] and all(report["diagnostic_gates"].values())
    report["status"]="complete_gates_passed" if passed else "complete_gate_failed"
    (run/"PROFILE.public.json").write_text(json.dumps(report,indent=2)+'\n')
    queue["status"]=report["status"]
    queue["report_sha256"]=sha(run/"PROFILE.public.json")
    write()
    print(json.dumps({"status":queue["status"],"report_sha256":queue["report_sha256"],
                      "observed_api_seconds":report["observed_api_seconds"],"observed_outer_seconds":queue["observed_outer_seconds"]}),flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
