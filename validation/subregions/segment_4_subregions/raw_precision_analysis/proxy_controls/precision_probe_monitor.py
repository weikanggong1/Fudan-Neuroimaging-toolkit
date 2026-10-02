from pathlib import Path
import os,sys,time,json,hashlib,subprocess
root=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930');source=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930/source_precision_proxy_controls_20261002');folder=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930/precision_proxy_controls_20261002');worker=source/'validation/subregions/analyze_raw_precision.py'
def main():
 man=json.loads((source/'source_manifest.json').read_text())
 assert all((source/x['path']).stat().st_size==x['bytes'] and hashlib.sha256((source/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in man['files'])
 (folder/'source_verification.json').write_text(json.dumps({'frozen_source':str(source),'verified_files':len(man['files']),'all_size_sha256_match':True,'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest()},indent=2)+'\n')
 predecessor=root/'precision_hippo_regressions_20261002/left_raw_stable_fast_status.json'
 while not predecessor.exists() or json.loads(predecessor.read_text()).get('state')=='running':time.sleep(10)
 assert json.loads(predecessor.read_text())['state']=='completed','previous GPU queue did not complete'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONPATH=str(source/'src')+':'+str(source/'validation/subregions'),PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 jobs=[('thalamus_bias_own_grid','synthseg_fast_wm110','thalamus','fast',False,['--conform-grid']),('right_raw_stable_balanced_old_proxy','synthseg_raw','hippo-amygdala-right','balanced',True,[]),('right_raw_stable_balanced_new_proxy','synthseg_raw','hippo-amygdala-right','balanced',True,['--rebuild-wmparc']),('left_raw_stable_balanced_new_proxy','synthseg_raw','hippo-amygdala-left','balanced',True,['--rebuild-wmparc'])]
 for label,case,structure,opt,stable,extra in jobs:
  command=[sys.executable,str(worker),'--root',str(root),'--case',case,'--structure',structure,'--optimization',opt,'--cache',str(root/'raw_precision_controls_20261001/shared_cache'),'--output',str(folder/label)]
  command.extend(extra)
  if stable:command.append('--stable-fitting')
  if stable:command.append('--trace-steps')
  status={'label':label,'case':case,'structure':structure,'optimization':opt,'stable_fitting_override':stable,'state':'running','started_unix':time.time(),'command':command,'physical_gpu':1,'memory_fraction':.23,'own_limit_mib':19073,'max_own_mib':0}
  path=folder/(label+'_status.json')
  def save():
   tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(path)
  with (folder/(label+'.log')).open('w') as log,(folder/(label+'_gpu.jsonl')).open('w') as monitor_file:
   process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=folder);status['pid']=process.pid;save()
   while process.poll() is None:
    sample={'unix_time':time.time(),'own_pid':process.pid}
    try:
     sample['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()
     lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines();own=max([int(x.split(',')[1].strip()) for x in lines if x.split(',')[0].strip()==str(process.pid)] or [0]);sample['own_mib']=own;status['max_own_mib']=max(own,status['max_own_mib'])
     if own>19073:status['memory_limit_exceeded']=own;process.terminate()
    except (subprocess.SubprocessError,ValueError) as error:sample['sampling_error']=type(error).__name__
    monitor_file.write(json.dumps(sample)+'\n');monitor_file.flush();save()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
     if 'memory_limit_exceeded' in status:process.kill()
   status.update(state='completed' if process.returncode==0 else 'failed',exit_code=process.returncode,finished_unix=time.time());save()
  print(json.dumps(status),flush=True)
  if process.returncode:raise SystemExit(process.returncode)
if __name__=='__main__':main()
