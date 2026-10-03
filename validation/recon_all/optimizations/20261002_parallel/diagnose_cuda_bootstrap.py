"""固定 worker 环境的独立 CUDA 冷启动诊断；不修改生产算法或 allocator。

--config: execute_whole_case 的 JSON；--output: 必须不存在的诊断目录；
--lock: 共用主机锁。四批 single/pair/staged-single/pair，新进程每个 CPU2，
同时最多两进程，GPU UUID 和禁用缓存策略与失败 worker 相同。
输出环境、阶段、显存快照、每个冷启动与总报告。仅测初始化，无官方 CLI。
任何探针失败总报告非零；不重试、不恢复被试结果，不构成可靠性保证。
"""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,traceback
P=pathlib.Path

def write(path,value):
    t=path.with_suffix(path.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def snapshot():
    rows={}
    for name,query in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),('apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])]:
        r=subprocess.run(query,capture_output=True,text=True,timeout=15);rows[name]={'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    return rows

def child(args):
    report={'started_utc':now(),'pid':os.getpid(),'threads':2,'staged':args.staged,'status':'failed','environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','CUDA_DEVICE_ORDER','PYTORCH_NO_CUDA_MEMORY_CACHING','PYTORCH_CUDA_ALLOC_CONF','OMP_NUM_THREADS','NUMBA_NUM_THREADS','PYTHONPATH']},'meminfo_before':P('/proc/meminfo').read_text()}
    tick=time.monotonic();code=1
    try:
        report['phase']='import';import torch
        from fnit.recon_all.profiling import configure_cuda_allocator
        report.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,cuda_initialized_before=torch.cuda.is_initialized())
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        report['phase']='allocator';report['allocator']=configure_cuda_allocator(device='cuda:0',policy='disabled')
        if args.staged:
            report['phase']='cuda_init';torch.cuda.init()
            report['phase']='set_device';torch.cuda.set_device('cuda:0')
            report['phase']='properties';report['uuid']=str(torch.cuda.get_device_properties('cuda:0').uuid)
            report['phase']='memory_info';report['free_total_before']=torch.cuda.mem_get_info('cuda:0')
        report['phase']='allocation';x=torch.empty(1,dtype=torch.float32,device='cuda:0')
        report['phase']='synchronize';torch.cuda.synchronize('cuda:0')
        report.update(status='passed',uuid=str(torch.cuda.get_device_properties('cuda:0').uuid),free_total_after=torch.cuda.mem_get_info('cuda:0'),allocated=torch.cuda.memory_allocated('cuda:0'),reserved=torch.cuda.memory_reserved('cuda:0'),tensor_bytes=x.numel()*x.element_size(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32)
        code=0
    except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
    report.update(finished_utc=now(),seconds=time.monotonic()-tick,exit_code=code,meminfo_after=P('/proc/meminfo').read_text());write(args.output,report);return code

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--config',type=P);a.add_argument('--output',type=P,required=True);a.add_argument('--lock',type=P);a.add_argument('--child',action='store_true');a.add_argument('--staged',action='store_true');args=a.parse_args()
    if args.child:return child(args)
    config=json.loads(args.config.read_text());args.output.mkdir(exist_ok=False)
    result={'status':'waiting_for_common_lock','started_utc':now(),'script_sha256':hashlib.sha256(P(__file__).read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'frozen_runtime_commit':config['code_commit'],'threads_per_worker':2,'max_concurrent_workers':2,'batches':[]};write(args.output/'summary.json',result)
    env=dict(os.environ);env.update(PYTHONPATH=str(P(config['code_root'])/'src'),CUDA_VISIBLE_DEVICES=config['gpu_uuid'],PYTORCH_NO_CUDA_MEMORY_CACHING='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
    with args.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);result['status']='running';result['lock_acquired_utc']=now()
        for name,n,staged in [('single',1,False),('pair_1',2,False),('staged_single',1,True),('pair_2',2,False)]:
            row={'name':name,'started_utc':now(),'before':snapshot(),'workers':[]};processes=[]
            for index in range(n):
                dest=args.output/(name+'_'+str(index)+'.json');log=(args.output/(name+'_'+str(index)+'.log')).open('w');argv=[config['python'],str(P(__file__).resolve()),'--child','--output',str(dest)]
                if staged:argv+=['--staged']
                proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);processes.append((proc,log,dest))
            for proc,log,dest in processes:
                try:code=proc.wait(timeout=90)
                except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
                log.close();r=json.loads(dest.read_text()) if dest.exists() else {'status':'failed','error':'no child report'};row['workers'].append({'path':str(dest),'exit_code':code,'status':r['status'],'phase':r.get('phase'),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest() if dest.exists() else None})
            row.update(after=snapshot(),finished_utc=now());result['batches'].append(row);write(args.output/'summary.json',result)
            if any(x['exit_code'] for x in row['workers']):break
    result.update(status='passed' if all(x['exit_code']==0 for b in result['batches'] for x in b['workers']) and len(result['batches'])==4 else 'failed',finished_utc=now(),interpretation='cold bootstrap observation only; no allocator or concurrency reliability inference');write(args.output/'summary.json',result);return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
