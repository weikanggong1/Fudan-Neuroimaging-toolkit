"""Two separately dispatched bounded phases; no automatic MRI after contracts."""
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


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','workspace','run','source','checkpoint'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('contracts','ABBA'),required=True)
    args=p.parse_args();os.umask(0o077)
    plan_path=args.workspace/'PLAN.json';plan=json.loads(plan_path.read_text())
    for name,digest in plan['prototype_files'].items():assert sha(args.workspace/name)==digest
    before={n:sha(args.source/'fnit/synthseg_parc'/n) for n in plan['source_files']}
    assert before==plan['source_files']
    if args.phase=='contracts':
        args.run.mkdir(mode=0o700)
        bound={'schema':'fnit_seg_single_layer_total_bound/v1','initial_enqueue_epoch':time.time(),
            'outer_hard_limit_seconds':23000,'deadline_epoch':time.time()+23000}
        write(args.run/'TASK_BOUND.private.json',bound)
        queue_path=args.run/'contracts_queue.private.json'
    else:
        bound=json.loads((args.run/'TASK_BOUND.private.json').read_text())
        contracts=json.loads((args.run/'CONTRACTS.private.json').read_text())
        first=json.loads((args.run/'contracts_queue.private.json').read_text())
        assert contracts['status']=='passed' and contracts['plan_sha256']==sha(plan_path)
        assert first['status']=='contracts_passed_awaiting_review_before_ABBA'
        assert contracts['source_before']==contracts['source_after']==before
        assert contracts['worker_sha256']==plan['prototype_files']['check_contracts.py']
        queue_path=args.run/'queue.private.json';assert not queue_path.exists()
    assert bound['outer_hard_limit_seconds']==23000 and time.time()<bound['deadline_epoch']
    queue={'schema':'fnit_seg_CPU_layer_trial_queue/v1','phase':args.phase,'status':'waiting_for_common_CPU_lock',
        'controller_PID':os.getpid(),'controller_sha256':sha(__file__),'plan_sha256':sha(plan_path),
        'source_files':before,'outer_deadline_epoch':bound['deadline_epoch'],'jobs':[],
        'initial_enqueue_epoch':bound['initial_enqueue_epoch'],'updated_epoch':time.time(),
        'full_CNN_native_GPU_arms':0,'whole_T1_map_CSV_status':'not_assessed'}
    write(queue_path,queue)
    with (args.root/plan['lock_fnit_relative_path']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert time.time()<bound['deadline_epoch']
        queue.update(status='running_bounded_phase',lock_acquired_epoch=time.time());write(queue_path,queue)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
             'OPENBLAS_NUM_THREADS':'8','NUMBA_NUM_THREADS':'8','PYTHONDONTWRITEBYTECODE':'1'}
        executable=args.root/'envs/default/bin/python'
        if args.phase=='contracts':
            commands=[('contracts',[str(executable),str(args.workspace/'check_contracts.py'),
                '--root',str(args.root),'--source',str(args.source),'--plan',str(plan_path),
                '--output',str(args.run/'CONTRACTS.private.json')],120)]
        else:
            commands=[]
            for name,mode in (('A1_baseline','baseline'),('B1_candidate','candidate'),
                              ('B2_candidate','candidate'),('A2_baseline','baseline')):
                command=[str(executable),str(args.workspace/'layer_trial.py'),'--root',str(args.root),
                    '--source',str(args.source),'--plan',str(plan_path),'--checkpoint',str(args.checkpoint),
                    '--output',str(args.run/name),'--mode',mode]
                if name!='A1_baseline':command.extend(['--reference',str(args.run/'A1_baseline/conv0_baseline.private.npy')])
                commands.append((name,command,180))
        for name,command,limit in commands:
            assert time.time()<bound['deadline_epoch']
            job={'name':name,'status':'running','start_epoch':time.time(),'hard_timeout_seconds':limit}
            queue['jobs'].append(job);write(queue_path,queue)
            with (args.run/(name+'.log')).open('wb') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                    preexec_fn=lambda:os.sched_setaffinity(0,plan['cpu_affinity']))
                job['PID']=process.pid;write(queue_path,queue)
                try:rc=process.wait(timeout=min(limit,bound['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();rc=-9;job['timeout']=True
            job.update(status='completed' if rc==0 else 'failed',exit_code=rc,end_epoch=time.time(),
                       worker_wall_seconds=time.time()-job['start_epoch'])
            queue['updated_epoch']=time.time()
            if rc:
                queue['status']='stopped_at_first_failed_gate';write(queue_path,queue);raise SystemExit(1)
            write(queue_path,queue)
        queue['status']=('contracts_passed_awaiting_review_before_ABBA' if args.phase=='contracts' else 'bounded_ABBA_complete')
        queue['updated_epoch']=time.time();write(queue_path,queue)
    print(json.dumps({'status':queue['status'],'phase':args.phase,'jobs':len(queue['jobs'])}))


if __name__=='__main__':main()
