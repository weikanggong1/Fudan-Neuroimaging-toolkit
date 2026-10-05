"""Non-imaging process-boundary contracts for the finite preparation runner."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    controller = Path(__file__).with_name("run_preparation.py")
    controller_sha = hashlib.sha256(controller.read_bytes()).hexdigest()
    affinity = sorted(os.sched_getaffinity(0))[:8]
    assert len(affinity) == 8
    cases = []
    with tempfile.TemporaryDirectory(prefix="fnit-prep-contract-") as temporary:
        root = Path(temporary)
        stub = root / "stub.py"
        stub.write_text("""import argparse,json,os,resource,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--arm');a=p.parse_args()
j=json.loads(Path(a.plan).read_text())
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
assert all(n not in os.environ for n in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE'))
assert all(os.environ[n]=='8' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert resource.getrlimit(resource.RLIMIT_AS)==(j['address_space_cap_bytes'],)*2
assert os.sched_getaffinity(0)==set(j['affinity'])
Path(j['run_directory'],a.arm+'.stub-only').write_text('no images loaded')
if j['stub_case'] in ('child_timeout','signal'):time.sleep(20)
if j['stub_case']=='science_failure':raise SystemExit(7)
""")
        worker_sha = hashlib.sha256(stub.read_bytes()).hexdigest()
        for name in ("success", "science_failure", "child_timeout", "outer_wait_deadline", "signal"):
            run = root / name; run.mkdir()
            plan = dict(scope="one_real_target_preparation_only", registration_calls=0, whole_GEMS_calls=0,
                        host=os.uname().nodename, threads=8, affinity=affinity, run_directory=str(run),
                        worker={"path":str(stub),"sha256":worker_sha},
                        controller={"path":str(controller),"sha256":controller_sha},
                        CPU8_lock=str(root/(name+".lock")), lock_wait_seconds=8,
                        child_timeout_seconds=1 if name=="child_timeout" else 8,
                        outer_deadline_seconds=2 if name=="outer_wait_deadline" else 15,
                        address_space_cap_bytes=20000000000,official_python=sys.executable,
                        fnit_python=sys.executable,stub_case=name)
            plan_path=run/"plan.json";plan_path.write_text(json.dumps(plan))
            lock=open(plan["CPU8_lock"],"a+")
            if name=="outer_wait_deadline":fcntl.flock(lock,fcntl.LOCK_EX)
            with (run/"controller.stdout").open("w") as out, (run/"controller.stderr").open("w") as error:
                process=subprocess.Popen([sys.executable,str(controller),"--plan",str(plan_path)],stdout=out,stderr=error)
                if name=="signal":
                    deadline=time.monotonic()+8
                    while time.monotonic()<deadline:
                        receipt=run/"controller.private.json"
                        if receipt.exists() and json.loads(receipt.read_text()).get("current_child_pid"):
                            process.send_signal(signal.SIGTERM);break
                        time.sleep(.05)
                    else:raise AssertionError("stub child never started")
                code=process.wait(timeout=20)
            lock.close()
            receipt=json.loads((run/"controller.private.json").read_text())
            expected_code={"success":0,"science_failure":7,"child_timeout":124,"outer_wait_deadline":124,"signal":143}[name]
            expected_workers={"success":3,"science_failure":1,"child_timeout":1,"outer_wait_deadline":0,"signal":1}[name]
            assert code==expected_code,(name,code,(run/"controller.stderr").read_text())
            assert receipt["worker_count"]==expected_workers,(name,receipt)
            pid=receipt.get("current_child_pid")
            if pid:
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise AssertionError("own child left running")
            cases.append(dict(case=name,exit=code,worker_count=expected_workers,own_child_reaped=True,gate=True))
    report=dict(scope="stub_process_boundaries_only_no_imaging_or_registration",cases=cases,
                controller_sha256=controller_sha,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                all_gates_pass=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":main()
