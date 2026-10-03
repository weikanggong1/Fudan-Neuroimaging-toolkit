"""相同UUID/安装/线程下比较并发、串行ACK和分阶段CUDA初始化；只做诊断。

--config: 冻结整例JSON；--output: 新目录；--lock: 共用主机flock。
ABBA四批(parallel/serial/serial/parallel)均disabled cache，再两批staged
和enabled-cache parallel。每fresh worker线程2，最大并发2，在首次CUDA前导入生产worker的thread_budget
模块（含Numba），不进入表面算法，不启用半精度。serial ACK来自保留CUDA context的真实探针进程，
两侧同步完成后同一gate放行。保存每次成功/失败/取消、SHA及同期NVML。
全部尝试保留，无静默重试；失败次数只是本次观察，不能推定驱动根因。
内部--child/--staged/--allocator/--ack/--gate仅由本脚本启动；gate限时90s。
"""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,subprocess,time,traceback
P=pathlib.Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def snapshot():
 r={}
 for name,q in [('gpu',['nvidia-smi','--query-gpu=index,uuid,name,driver_version,compute_mode,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),('apps',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])]:
  v=subprocess.run(q,capture_output=True,text=True,timeout=15);r[name]={'exit':v.returncode,'stdout':v.stdout,'stderr':v.stderr}
 return r

def child(a):
 r={'pid':os.getpid(),'started_utc':now(),'status':'failed','allocator_requested':a.allocator,'staged':a.staged,'threads':2,'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','PYTORCH_NO_CUDA_MEMORY_CACHING','OMP_NUM_THREADS','NUMBA_NUM_THREADS','PYTHONPATH']},'meminfo_before':P('/proc/meminfo').read_text()};tick=time.monotonic();code=1
 try:
  r['phase']='import';import torch
  from fnit.recon_all.profiling import configure_cuda_allocator
  r['phase']='worker_thread_budget_import';from fnit.recon_all.thread_budget import thread_budget
  r['thread_budget_import_before_cuda']=True
  torch.set_num_threads(2);torch.set_num_interop_threads(1)
  torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
  r.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,cuda_initialized_before=torch.cuda.is_initialized());r['phase']='allocator';r['allocator']=configure_cuda_allocator('cuda:0',a.allocator)
  if a.staged:
   r['phase']='cuda_init';torch.cuda.init();r['phase']='set_device';torch.cuda.set_device('cuda:0');r['phase']='properties';r['uuid']=str(torch.cuda.get_device_properties('cuda:0').uuid);r['phase']='memory_info';r['free_total_before']=torch.cuda.mem_get_info('cuda:0')
  r['phase']='allocation';x=torch.empty(1,dtype=torch.float32,device='cuda:0');r['phase']='synchronize';torch.cuda.synchronize('cuda:0')
  r.update(status='initialized_waiting',bootstrap_seconds=time.monotonic()-tick,free_total_after=torch.cuda.mem_get_info('cuda:0'),uuid=str(torch.cuda.get_device_properties('cuda:0').uuid),tensor_bytes=x.numel()*x.element_size(),cuda_context_retained_at_ack=True,fp16_or_bf16=False)
  write(a.ack,r)
  deadline=time.monotonic()+90
  while not a.gate.exists():
   if time.monotonic()>deadline:raise TimeoutError('bootstrap gate not released')
   time.sleep(.05)
  torch.cuda.synchronize('cuda:0');r.update(status='passed',gate_released_utc=now());code=0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
 r.update(finished_utc=now(),seconds=time.monotonic()-tick,exit_code=code);write(a.output,r);return code

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=P);p.add_argument('--output',type=P,required=True);p.add_argument('--lock',type=P);p.add_argument('--child',action='store_true');p.add_argument('--staged',action='store_true');p.add_argument('--allocator',choices=['enabled','disabled'],default='disabled');p.add_argument('--ack',type=P);p.add_argument('--gate',type=P);a=p.parse_args()
 if a.child:return child(a)
 cfg=json.loads(a.config.read_text());a.output.mkdir(exist_ok=False);result={'status':'waiting_for_common_lock','started_utc':now(),'script_sha256':sha(__file__),'config_sha256':sha(a.config),'runtime_commit':cfg['code_commit'],'max_concurrent_workers':2,'threads_per_worker':2,'batches':[]};write(a.output/'summary.json',result)
 env=dict(os.environ);env.update(PYTHONPATH=str(P(cfg['code_root'])/'src'),CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2')
 batches=[('parallel_A1',False,False,'disabled'),('serial_B1',True,False,'disabled'),('serial_B2',True,False,'disabled'),('parallel_A2',False,False,'disabled'),('staged_C1',False,True,'disabled'),('staged_C2',False,True,'disabled'),('cached_D1',False,False,'enabled'),('cached_D2',False,False,'enabled')]
 with a.lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);result.update(status='running',lock_acquired_utc=now())
  for name,serial,staged,allocator in batches:
   row={'name':name,'serial_actual_context_ack':serial,'staged':staged,'allocator':allocator,'started_utc':now(),'before':snapshot(),'workers':[]};procs=[];gate=a.output/(name+'.gate');e=dict(env)
   if allocator=='enabled':e.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
   else:e['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
   def await_acks(entries):
    deadline=time.monotonic()+90
    while True:
     if all(ack.exists() for proc,log,dest,ack in entries):return True
     if any(proc.poll() is not None and not ack.exists() for proc,log,dest,ack in entries):return False
     if time.monotonic()>deadline:return False
     time.sleep(.05)
   for i in range(2):
    dest=a.output/(name+'_'+str(i)+'.json');ack=a.output/(name+'_'+str(i)+'.ack.json');log=(a.output/(name+'_'+str(i)+'.log')).open('w');argv=[cfg['python'],str(P(__file__).resolve()),'--child','--output',str(dest),'--allocator',allocator,'--ack',str(ack),'--gate',str(gate)]
    if staged:argv+=['--staged']
    proc=subprocess.Popen(argv,env=e,stdout=log,stderr=subprocess.STDOUT);procs.append((proc,log,dest,ack))
    if serial and not await_acks(procs):break
   passed=len(procs)==2 and await_acks(procs)
   if passed:gate.write_text(now()+'\n')
   else:
    for proc,log,dest,ack in procs:
     if proc.poll() is None:proc.terminate()
   for proc,log,dest,ack in procs:
    try:code=proc.wait(timeout=15)
    except subprocess.TimeoutExpired:proc.kill();code=proc.wait()
    log.close();r=json.loads(dest.read_text()) if dest.exists() else {'status':'canceled','phase':json.loads(ack.read_text()).get('phase') if ack.exists() else 'unknown'};row['workers'].append({'pid':proc.pid,'path':str(dest),'status':r['status'],'phase':r.get('phase'),'exit_code':code,'ack_exists':ack.exists(),'sha256':sha(dest) if dest.exists() else None})
   row.update(finished_utc=now(),both_initialized=passed,after=snapshot());result['batches'].append(row);write(a.output/'summary.json',result)
 result.update(status='complete',finished_utc=now(),assessment='all successes and failures retained; diagnostic only; no production/cache/default change');write(a.output/'summary.json',result);return 0
if __name__=='__main__':raise SystemExit(main())
