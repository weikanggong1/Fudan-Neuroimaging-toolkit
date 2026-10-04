"""独立CUDA启动边界诊断：原冻结worker与显式init各用新进程。"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import random
import signal
import subprocess
import sys
import time

GPU_UUID='GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
ENV_KEYS=('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','PYTORCH_NO_CUDA_MEMORY_CACHING',
          'PYTORCH_CUDA_ALLOC_CONF','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
          'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT','OMP_NUM_THREADS','MKL_NUM_THREADS',
          'OPENBLAS_NUM_THREADS','TORCH_SHOW_CPP_STACKTRACES','CUDA_MODULE_LOADING',
          'CUDA_FORCE_PTX_JIT','CUDA_CACHE_DISABLE','CUDA_DEVICE_ORDER','NUMBA_CUDA_DRIVER')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def host_snapshot():
    proc=Path('/proc');maps=[]
    for line in (proc/'self/maps').read_text().splitlines():
        parts=line.split()
        if len(parts)>=6 and any(x in parts[-1] for x in ('libcuda','libcudnn','libcublas','libtorch_cuda')):
            maps.append(parts[-1])
    limits={}
    for name in ('RLIMIT_AS','RLIMIT_DATA','RLIMIT_MEMLOCK','RLIMIT_NPROC','RLIMIT_NOFILE'):
        limits[name]=list(resource.getrlimit(getattr(resource,name)))
    memory={}
    for line in (proc/'meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemTotal','MemAvailable','SwapTotal','SwapFree','Mlocked','Unevictable'):
            memory[key]=value.strip()
    return {'resource_limits':limits,'loaded_cuda_libraries':sorted(set(maps)),
            'host_meminfo':memory,'resource_environment':{k:os.environ.get(k) for k in ENV_KEYS},
            'status_selected':[s for s in (proc/'self/status').read_text().splitlines()
                               if s.startswith(('VmRSS:','VmLck:','VmPin:','Threads:'))]}

def gpu_snapshot():
    r=subprocess.run(['nvidia-smi','--id='+GPU_UUID,
                      '--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
    uuid,total,used,free,util=[x.strip() for x in r.stdout.strip().split(',')]
    if uuid!=GPU_UUID:raise ValueError('wrong physical GPU')
    return {'utc':now(),'uuid':uuid,'total_bytes':int(total)*1048576,
            'used_bytes':int(used)*1048576,'free_bytes':int(free)*1048576,'utilization_percent':int(util)}

def split_child(args):
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    report={'status':'starting','pid':os.getpid(),'mode':args.child,'events':[],
            'started_utc':now(),'before_import':host_snapshot(),'scope':'CUDA boundary diagnostic, no T1'}
    def event(name):
        report['events'].append({'event':name,'utc':now(),'monotonic':time.monotonic()})
        write(root/'boundary.json',report)
    phase='import_torch'
    try:
        event('before_import_torch')
        import torch
        from fnit.recon_all.profiling import configure_cuda_allocator
        from fnit.recon_all.thread_budget import thread_budget
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.set_num_interop_threads(1)
        policy='enabled' if args.child=='split_enabled' else 'disabled'
        report['allocator']=configure_cuda_allocator('cuda:0',policy)
        report['torch_version']=torch.__version__;report['torch_cuda']=torch.version.cuda
        report['before_cuda_observation']=host_snapshot()
        report['initialized_before_explicit_init']=torch.cuda.is_initialized()
        if report['initialized_before_explicit_init']:
            raise RuntimeError('observation changed CUDA initialization; invalid diagnostic')
        phase='explicit_cuda_init';event('before_cuda_init')
        torch.cuda.init();event('after_cuda_init')
        phase='first_float32_allocation';event('before_empty_float32')
        scalar=torch.empty(1,dtype=torch.float32,device='cuda:0');event('after_empty_float32')
        phase='cuda_synchronize';torch.cuda.synchronize(torch.device('cuda:0'));event('after_synchronize')
        report['scalar_dtype']=str(scalar.dtype)
        report['free_total_bytes']=list(torch.cuda.mem_get_info(torch.device('cuda:0')))
        props=torch.cuda.get_device_properties(torch.device('cuda:0'))
        report['device']={'name':props.name,'total_memory':props.total_memory,
                          'uuid':str(getattr(props,'uuid','unavailable'))}
        report['after_cuda']=host_snapshot();del scalar
        report['status']='complete'
    except BaseException as error:
        import traceback
        report.update(status='failed',failure_phase=phase,error=repr(error),traceback=traceback.format_exc())
        print(report['traceback'],file=sys.stderr,flush=True)
    report['finished_utc']=now();write(root/'boundary.json',report)
    return 0 if report['status']=='complete' else 1

def controller(args):
    from resource_admission import snapshot_descendants,active_owned,cleanup_owned_tree
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve();worker=source/'src/fnit/recon_all/hemisphere_worker.py'
    report={'status':'waiting_lock','started_utc':now(),'host':os.uname().nodename,
            'pid':os.getpid(),'gpu_uuid':GPU_UUID,'source':str(source),'worker_sha256':digest(worker),
            'probe_sha256':digest(__file__),'target_sha256':digest(Path(__file__).with_name('cuda_bootstrap_target.py')),
            'precision':'float32; original worker TF32 true; no autocast',
            'minimum_free_bytes':args.minimum_free_bytes,'resource_scope':'scalar-only diagnostic; not whole-case admission',
            'trials':[],'matrix':'runtime_boundaries' if args.boundary_script else 'bootstrap_v2',
            'random_seed':args.seed if args.boundary_script else None,'versions_scope':'same frozen conda interpreter; library binding recorded for split/runtime children; original worker failure has no maps receipt'}
    write(root/'summary.json',report)
    lock=args.lock.open('a+');held=False;owned={};children=[];streams=[];current=None
    def cancel(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    previous={s:signal.signal(s,cancel) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        tick=time.monotonic()
        while not held:
            g=gpu_snapshot();report['waiting_gpu']=g;write(root/'summary.json',report)
            if g['free_bytes']>=args.minimum_free_bytes:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held=True
                except BlockingIOError:pass
                if held and gpu_snapshot()['free_bytes']<args.minimum_free_bytes:
                    fcntl.flock(lock,fcntl.LOCK_UN);held=False
            if not held:
                if time.monotonic()-tick>args.maximum_wait_seconds:raise TimeoutError('shared lock/resources timeout')
                time.sleep(5)
        report.update(status='running',under_lock_gpu=gpu_snapshot(),admission_wait_seconds=time.monotonic()-tick)
        # Rotate methods to avoid confounding a method with a long temporal window.
        plan=[]
        rng=random.Random(args.seed)
        for repeat in range(args.rounds):
            methods=([('original_worker',2),('direct_priority',2),('context_sync',2),
                      ('context_driver',2),('sticky_probe',2)] if args.boundary_script else
                     [('original_worker',2),('split_disabled',2),('original_worker',1),
                      ('split_disabled',1),('split_enabled',2)])
            if args.boundary_script:rng.shuffle(methods)
            for mode,workers in methods:plan.append((repeat,mode,workers))
        report['planned_trials']=[{'repeat':r,'mode':m,'workers':w} for r,m,w in plan]
        if args.boundary_script:report['boundary_script_sha256']=digest(args.boundary_script)
        for index,(repeat,mode,workers) in enumerate(plan):
            if digest(worker)!=report['worker_sha256']:raise ValueError('frozen worker changed')
            trial=root/(str(index).zfill(3)+'_'+mode+'_w'+str(workers));trial.mkdir()
            row={'index':index,'repeat':repeat,'mode':mode,'workers':workers,'threads_per_child':2,
                 'started_utc':now(),'gpu_before':gpu_snapshot(),'children':[]}
            report['trials'].append(row);children=[];owned={};streams=[]
            for slot in range(workers):
                childroot=trial/str(slot);env=dict(os.environ)
                env.update(CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONPATH=str(source/'src')+os.pathsep+str(Path(__file__).parent.resolve()),
                           TORCH_SHOW_CPP_STACKTRACES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                           MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='2',NUMBA_NUM_THREADS='2')
                if mode=='split_enabled':env.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
                else:env['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
                childreport=childroot/'boundary.json'
                if mode=='original_worker':
                    childroot.mkdir();request=childroot/'request.json';childreport=childroot/'worker.json'
                    write(request,{'callable':'cuda_bootstrap_target:noop','operation':'startup_probe','kwargs':{},'device':'cuda:0','threads':2,
                                   'precision':{'matmul_tf32':True,'cudnn_tf32':True},'profile_stages':True,'allocator_policy':'disabled'})
                    command=[sys.executable,'-m','fnit.recon_all.hemisphere_worker',str(request),str(childreport)]
                elif args.boundary_script:
                    childreport=childroot/'boundary.json'
                    command=[sys.executable,str(args.boundary_script.resolve()),'--mode',mode,'--output',str(childroot)]
                else:command=[sys.executable,str(Path(__file__).resolve()),'--child',mode,'--output',str(childroot)]
                stream=(trial/(str(slot)+'.log')).open('w');streams.append(stream)
                child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child);snapshot_descendants([child.pid],owned)
                row['children'].append({'pid':child.pid,'command':command,'report':str(childreport)})
            write(root/'summary.json',report);deadline=time.monotonic()+90
            while any(c.poll() is None for c in children):
                for c in children:snapshot_descendants([c.pid,*owned],owned)
                if time.monotonic()>deadline:raise TimeoutError('startup diagnostic child timed out')
                time.sleep(.1)
            for child,data in zip(children,row['children']):
                data['exit_code']=child.returncode;p=Path(data['report'])
                if p.exists():data['result']=json.loads(p.read_text());data['report_sha256']=digest(p)
            # Each trial must release all owned descendants before resetting identity tracking.
            for child in children:
                snapshot_descendants([child.pid,*owned],owned)
                if active_owned(owned):
                    cleanup_owned_tree(child,owned,report,lambda:write(root/'summary.json',report))
            if active_owned(owned):raise RuntimeError('trial has surviving owned processes')
            for stream in streams:stream.close()
            streams=[];row.update(finished_utc=now(),gpu_after=gpu_snapshot());write(root/'summary.json',report)
        report['failed_children']=sum(c.get('exit_code')!=0 for t in report['trials'] for c in t['children'])
        report['successful_children']=sum(c.get('exit_code')==0 for t in report['trials'] for c in t['children'])
        report['status']='matrix_complete'
        report['all_children_succeeded']=report['failed_children']==0
        report['interpretation']='completion is not root-cause confirmation or production fix'
        return 0
    except BaseException as error:
        report.update(status='failed_or_interrupted',error=repr(error));return 1
    finally:
        for child in children:
            snapshot_descendants([child.pid,*owned],owned)
            if active_owned(owned):cleanup_owned_tree(child,owned,report,lambda:write(root/'summary.json',report))
        for stream in streams:stream.close()
        if held:fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close();report['finished_utc']=now();write(root/'summary.json',report)
        for s,h in previous.items():signal.signal(s,h)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--child',choices=('split_disabled','split_enabled'))
    p.add_argument('--source',type=Path)
    p.add_argument('--lock',type=Path,default=Path('/tmp/fnit-shared-benchmark.lock'))
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--boundary-script',type=Path,help='实际CUDA runtime上下文边界脚本；只用于独立诊断')
    p.add_argument('--seed',type=int,default=20261004,help='边界矩阵交错顺序种子')
    p.add_argument('--minimum-free-bytes',type=int,default=2_000_000_000)
    p.add_argument('--maximum-wait-seconds',type=float,default=900)
    a=p.parse_args()
    if a.child:return split_child(a)
    if a.boundary_script and not a.boundary_script.is_file():p.error('boundary-script must exist')
    if not a.source or not 1<=a.rounds<=20 or a.minimum_free_bytes<2_000_000_000:p.error('source required; rounds1..20; scalar free budget>=2GB')
    return controller(a)

if __name__=='__main__':raise SystemExit(main())
