"""对首次较慢的真实sub-04前段补AB/BA重复；锁每对释放。"""
import fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
OUTPUT=BASE/'accuracy_20261003/task_02/prefix_repeat_sub04'
UUID='GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
worker=ROOT/'repeat_prefix_worker.py'
original=(ROOT/'cohort_prefix.py').read_text()
old="OUTPUT=BASE/'accuracy_20261003/task_02/cohort_prefix'"
new="OUTPUT=BASE/'accuracy_20261003/task_02/prefix_repeat_sub04'/os.environ['FNIT_PREFIX_REPEAT_TRIAL']"
if old not in original:raise ValueError('unknown prefix benchmark schema')
worker.write_text(original.replace(old,new))
OUTPUT.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(BASE/'accuracy_20261003/baseline_runtime_816e5610/src'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4',PYTORCH_NO_CUDA_MEMORY_CACHING='1')
report={'case':'ds000114_sub-04','reason':'first B/A pair candidate slower; resolve reproducible regression vs run variation','worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'formal_speed_tolerance':None,'runs':[]}
for trial,order in [('AB',('baseline','candidate')),('BA',('candidate','baseline'))]:
 env['FNIT_PREFIX_REPEAT_TRIAL']=trial
 with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for backend in order:
   monitor=OUTPUT/trial/(backend+'_monitor')
   command=[sys.executable,str(BASE/'parallel_20261002/coordinator/run_monitored.py'),'--gpu-uuid',UUID,'--output',str(monitor),'--interval','1','--',sys.executable,str(worker),'--case','ds000114_sub-04','--backend',backend]
   code=subprocess.run(command,env=env).returncode
   item={'trial':trial,'backend':backend,'exit_code':code}
   path=OUTPUT/trial/'ds000114_sub-04'/backend/'scripts/prefix_run.json'
   if path.exists():item['run']=json.loads(path.read_text());item['monitor']=json.loads((monitor/'monitor.json').read_text())
   report['runs'].append(item);(OUTPUT/'report.json').write_text(json.dumps(report,indent=2))
