"""One finite CPU8 lock/compile/4x4 controller; no image stage exists."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--lock',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((args.workspace/'PLAN.json').read_text())
    report={'schema':1,'status':'waiting','PID':os.getpid(),'plan_sha256':hashlib.sha256((args.workspace/'PLAN.json').read_bytes()).hexdigest(),
            'scope':plan['scope'],'worker_started':False}
    started=time.monotonic();rc=1
    lock_file=args.lock.open('a')
    try:
        deadline=started+plan['resources']['lock_wait_seconds']
        while True:
            try:fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('finite lock wait exhausted')
                time.sleep(.25)
        report['lock_wait_seconds']=time.monotonic()-started
        env=dict(os.environ)
        for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','OPENBLAS_CORETYPE'):env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        result=args.run/'result.private.json'
        if result.exists():raise RuntimeError('no second contract over old result')
        report['worker_started']=True
        child_command=[args.python,str(args.workspace/'run_matrix_contract.py'),'--workspace',str(args.workspace),'--result',str(result)]
        with (args.run/'stdout.private.txt').open('xb') as out,(args.run/'stderr.private.txt').open('xb') as err:
            completed=subprocess.run(child_command,stdout=out,stderr=err,env=env,timeout=150)
        rc=completed.returncode
        report['status']='completed' if rc==0 else 'first_failure_stopped'
        if result.exists():report['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
    except BaseException as exception:
        report['exception']={'type':type(exception).__name__,'message':str(exception),'traceback':traceback.format_exc()}
    finally:
        report['returncode']=rc;report['wall_seconds']=time.monotonic()-started
        fcntl.flock(lock_file,fcntl.LOCK_UN);lock_file.close()
        raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
        fd=os.open(args.run/'controller.private.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as output:output.write(raw)
    return rc

if __name__=='__main__':raise SystemExit(main())
