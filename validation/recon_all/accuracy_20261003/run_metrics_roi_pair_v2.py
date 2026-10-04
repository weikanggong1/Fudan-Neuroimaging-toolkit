"""Dual-lock hidden-CUDA interface checks, then two admitted real stage arms.

Preflight operators are mocks and are never counted as benchmark results.
Any actual stage failure terminates the queue without an algorithm retry.
"""
import argparse,fcntl,json,os,pathlib,signal,subprocess,sys,time
import stage_benchmark_controller as base
from resource_admission import snapshot_descendants,active_owned,cleanup_owned_tree,digest
KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','NUMBA_NUM_THREADS')

def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True,type=pathlib.Path);a.add_argument('--controller',required=True,type=pathlib.Path);p=a.parse_args();tools=p.controller.parent
 state={'status':'waiting_cpu_preflight_resources','pid':os.getpid(),'arms':[],'order':['A8f','B3a'],'whole_case':False,'cpu_preflight':[],'algorithm_retry':False};child=None;owned={};cancelled=[]
 def write():base.write(p.run/'queue.json',state)
 def cancel(sig,frame):
  cancelled.append(sig);state['cancellation_signals']=list(cancelled)
  if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,cancel)
 configs={name:json.loads((p.run/(name+'.json')).read_text()) for name in state['order']};state['config_sha256']={name:digest(p.run/(name+'.json')) for name in configs};state['tool_manifest_sha256']=digest(tools/'tool_manifest.json')
 for c in configs.values():
  if c['threads']!=4 or c['minimum_free_bytes']<20_000_000_000 or c['lock']!='/tmp/fnit-shared-benchmark.lock' or c['gpu_lock']!='/tmp/fnit-stage-gpu1-benchmark.lock':raise ValueError('fixed resource constraints required')
 def run_owned(command,env,logpath,timeout):
  nonlocal child,owned
  if cancelled:return 130
  owned={}
  with logpath.open('x') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   snapshot_descendants([child.pid],owned);state['current_child_pid']=child.pid
   if cancelled and child.poll() is None:child.send_signal(signal.SIGTERM)
   write();deadline=time.monotonic()+timeout
   try:
    while child.poll() is None:
     snapshot_descendants([child.pid,*owned],owned)
     if cancelled or time.monotonic()>deadline:
      if not cancelled:state['timeout']=True
      cleanup_owned_tree(child,owned,state,write);break
     time.sleep(.1)
    code=child.wait()
   finally:
    snapshot_descendants([child.pid,*owned],owned)
    residual=active_owned(owned)
    if residual:state['residual_owned_at_exit']=residual
    while active_owned(owned):cleanup_owned_tree(child,owned,state,write)
    state['all_tracked_owned_exited']=not active_owned(owned);write()
   return code if code else 1 if residual or cancelled or state.get('timeout') else 0
 write();locks=[];held=[]
 try:
  c=configs['A8f'];deadline=time.monotonic()+7200
  for name in ['gpu_lock','lock']:locks.append(pathlib.Path(c[name]).open('a+'))
  while len(held)<2:
   if cancelled:raise InterruptedError('cancelled before CPU preflight')
   if time.monotonic()>deadline:raise TimeoutError('CPU preflight dual-lock admission timeout')
   for f in locks:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
    except BlockingIOError:break
   if len(held)==2:
    state['cpu_preflight_gpu_admission']=base.gpu_query(c['gpu_uuid'],5);write()
    if state['cpu_preflight_gpu_admission']['free_bytes']>=20_000_000_000:break
   for f in held:fcntl.flock(f,fcntl.LOCK_UN)
   held=[];time.sleep(1)
  state['status']='cpu_preflight_running';write()
  for arm in state['order']:
   c=configs[arm];env=dict(os.environ);env.update({k:'4' for k in KEYS});env.update(CUDA_VISIBLE_DEVICES='',PYTORCH_NO_CUDA_MEMORY_CACHING='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=c['source']+'/src'+os.pathsep+str(tools),NUMBA_CACHE_DIR=str(p.run/(arm+'.preflight_numba_cache')))
   command=[sys.executable,str(tools/'test_metrics_roi_controlflow.py'),'--output',str(p.run/(arm+'.preflight.json'))]
   code=run_owned(command,env,p.run/(arm+'.preflight.log'),120)
   state['cpu_preflight'].append({'arm':arm,'exit_code':code,'cuda_hidden':True,'numeric_operators':'mocked; not benchmark','report':str(p.run/(arm+'.preflight.json'))});write()
   if code!=0:raise RuntimeError('CPU interface preflight failed: '+arm)
  for f in held:fcntl.flock(f,fcntl.LOCK_UN)
  held=[];state['status']='cpu_preflight_passed_waiting_stages';write()
  for arm in state['order']:
   if cancelled:break
   state['current_arm']=arm;state['status']='running';write()
   code=run_owned([sys.executable,str(p.controller),'--config',str(p.run/(arm+'.json'))],None,p.run/(arm+'.controller.log'),8500)
   state['arms'].append({'name':arm,'controller_exit_code':code});write()
   if code!=0 or cancelled:break
  state['status']='cancelled' if cancelled else 'complete' if len(state['arms'])==2 and all(v['controller_exit_code']==0 for v in state['arms']) else 'failed'
 except BaseException as e:
  state.update(status='cancelled' if cancelled else 'timed_out' if isinstance(e,TimeoutError) else 'failed',error=repr(e))
 finally:
  # All child ownership cleanup above finishes before either advisory lock releases.
  if child is not None:
   snapshot_descendants([child.pid,*owned],owned)
   while active_owned(owned):cleanup_owned_tree(child,owned,state,write)
  for f in held:fcntl.flock(f,fcntl.LOCK_UN)
  for f in locks:f.close()
  state.update(current_arm=None,current_child_pid=None,all_tracked_owned_exited=not active_owned(owned));write()
 return 0 if state['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
