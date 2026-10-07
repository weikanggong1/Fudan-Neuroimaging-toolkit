"""Bounded prepared C24 compile->contracts queue; explicit later approval required."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

from bindings import check_sources, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','workspace','run'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--approved-contracts', action='store_true', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads((args.workspace/'PLAN.json').read_text())
    if args.workspace.resolve() != (args.root/plan['workspace']).resolve() or args.run.resolve() != (args.root/plan['runs']).resolve():
        raise RuntimeError('new canonical C24 workspace/run required')
    before = check_sources(args.root,args.workspace,plan)
    args.run.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 480
    report = {'schema':'fnit_C24_prepared_queue/v1','status':'waiting_common_CPU_lock',
              'controller_PID':os.getpid(),'PLAN':identity(args.workspace/'PLAN.json'),
              'source_before':before,'arms':[],'outer_deadline_seconds':480,'lock_wait_seconds_limit':120,
              'MRI_calls':0,'native_calls':0,'GPU_calls':0,'whole_CNN_calls':0,'completed':False}
    fd = None
    process = None
    locked = False
    def save():
        destination=args.run/'QUEUE.json'
        temporary=args.run/'QUEUE.json.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(destination)
    def expired(signum,frame):
        raise TimeoutError('outer480s deadline expired; no automatic retry')
    previous_handler=signal.signal(signal.SIGALRM,expired)
    signal.alarm(480)
    save()
    try:
        fd=os.open(args.root/plan['CPU_lock'],os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        details=os.fstat(fd)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid():
            raise RuntimeError('common lock must be owned regular file')
        lock_deadline=min(deadline,time.monotonic()+120)
        while True:
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True
                break
            except BlockingIOError:
                if time.monotonic() >= lock_deadline:
                    raise TimeoutError('common lock120s timeout; no scientific worker launched')
                time.sleep(min(1,lock_deadline-time.monotonic()))
        report['status']='common_CPU_lock_acquired'
        report['wait_seconds']=time.monotonic()-started
        save()
        environment=os.environ.copy()
        environment.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
                            'OPENBLAS_NUM_THREADS':'8','NUMBA_NUM_THREADS':'8','PYTHONDONTWRITEBYTECODE':'1'})
        for name in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','OPENBLAS_CORETYPE'):
            environment.pop(name,None)
        interpreter=args.root/'envs/default/bin/python'
        for script,output,seconds in (('compile_probe.py','compile',180),('check_contracts.py','CONTRACTS.json',180)):
            command=[str(interpreter),str(args.workspace/script),'--root',str(args.root),'--workspace',str(args.workspace),
                     '--output',str(args.run/output),'--approved-contracts']
            if script=='check_contracts.py':
                command += ['--run',str(args.run)]
            arm={'script':script,'status':'starting','timeout_seconds':seconds}
            report['arms'].append(arm)
            save()
            arm_start=time.monotonic()
            with (args.run/(script+'.log')).open('wb') as stream:
                process=subprocess.Popen(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                                         start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,plan['affinity']))
                arm['PID']=process.pid
                arm['status']='running'
                save()
                try:
                    code=process.wait(timeout=min(seconds,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=5)
                    arm['status']='timeout_stopped'
                    arm['returncode']=process.returncode
                    save()
                    raise TimeoutError('first worker timeout; remaining arms not launched')
            arm['returncode']=code
            arm['worker_observation_seconds']=time.monotonic()-arm_start
            arm['status']='exit0_pending_receipt' if code==0 else 'nonzero_stopped'
            save()
            if code:
                raise RuntimeError('first worker failure; remaining arms not launched: '+script)
            receipt_path=args.run/('compile/COMPILE.json' if script=='compile_probe.py' else output)
            receipt=json.loads(receipt_path.read_text())
            arm['receipt']=identity(receipt_path)
            arm['log']=identity(args.run/(script+'.log'))
            if script=='compile_probe.py':
                if receipt.get('status')!='compiled_loaded_interface_only' or not receipt.get('valid_interface'):
                    raise RuntimeError('new C24 compile/load receipt invalid; no numerical contracts')
                if receipt.get('compile_calls')!=1 or any(receipt.get(k)!=0 for k in ('copy_calls','SGEMM_calls','MRI_calls','native_calls','model_forward_calls','GPU_calls','tensor_allocation_calls')):
                    raise RuntimeError('compile-only scope counter gate')
                if identity(args.run/'compile/columns_c24.so') != receipt['binary']:
                    raise RuntimeError('new C24 binary changed')
            else:
                if not receipt.get('valid_bounded_contracts') or receipt.get('status')!='bounded_contracts_passed_no_MRI':
                    raise RuntimeError('contract receipt invalid')
                for name,value in (('copy_oracle_calls',13),('candidate_copy_calls',12),('candidate_SGEMM_calls',12),('fallback_calls',30),('MRI_calls',0),('native_calls',0),('model_forward_calls',0),('new_compilation_calls',0)):
                    if receipt.get(name)!=value:
                        raise RuntimeError('declared scope count failed: '+name)
                if len(receipt['numeric_rows'])!=6 or any(row['different_bits']!=0 for row in receipt['numeric_rows']+receipt['copy_rows']):
                    raise RuntimeError('first numeric bit gate failed')
            arm['status']='exit0_receipt_gate_passed'
            save()
        report['completed']=True
        report['status']='compile_and_short_contracts_passed_no_MRI'
    except BaseException as error:
        report['status']='bounded_queue_failed_stopped'
        report['error_type'],report['error']=type(error).__name__,str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
        try:
            report['source_after']=check_sources(args.root,args.workspace,plan)
            report['sources_unchanged']=before==report['source_after']
        except Exception as error:
            report['sources_unchanged']=False
            report['postcondition_error']=str(error)
        report['outer_observation_seconds']=time.monotonic()-started
        report['valid_queue']=bool(report['completed'] and report['sources_unchanged'])
        try:
            save()
        finally:
            if fd is not None:
                if locked: fcntl.flock(fd,fcntl.LOCK_UN)
                os.close(fd)
            signal.alarm(0)
            signal.signal(signal.SIGALRM,previous_handler)
    if not report['valid_queue']:
        raise RuntimeError('queue final source gate failed')


if __name__=='__main__':
    main()
