"""真实原始T1→filled新目录运行，CPU4/GPU串行；NVML归属未知不报零。"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import inspect
from fnit.recon_all.volume_prefix_python import run_volume_prefix
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
p.add_argument('--weights',required=True,type=Path);p.add_argument('--assets',required=True,type=Path)
p.add_argument('--n4-backend',choices=('simpleitk','torch'),required=True);p.add_argument('--device',default='cuda:0')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 return d.hexdigest()
def portable(value):
 if isinstance(value,dict):return {key:(Path(item).name if key=='model' and isinstance(item,str) and item.startswith('/') else portable(item)) for key,item in value.items()}
 if isinstance(value,list):return [portable(item) for item in value]
 return value
report={'scope':'original_T1_to_filled_prefix_only','input_sha256':sha(a.input),
 'n4_backend':a.n4_backend,'overall_equivalence':'not_assessed','strict_tolerance_declared':0,
 'cpu_affinity':sorted(os.sched_getaffinity(0)),'nvml_samples':[],
 'sample_interval_requested_seconds':1,'own_simultaneous_GPU_20GB_gate':'not_assessed',
 'nvml_attribution':'driver PIDs not mapped to container; own bytes unavailable, not zero',
 'source_sha256':{}}
source=Path(inspect.getfile(run_volume_prefix)).resolve().parent
report['benchmark_source_sha256']=sha(Path(__file__))
for name in ('volume_prefix_python.py','n4_simpleitk.py','n4_itk_torch_experimental.py','mri_em_register_complete.py','mri_em_register_python.py','mri_em_register_em_candidate.py'):
 report['source_sha256'][name]=sha(source/name)
query=None
try:
 from fnit.recon_all.nvml_process_memory import NvmlProcessQuery
 query=NvmlProcessQuery()
except Exception as e:report['nvml_init_error_type']=type(e).__name__
started=time.perf_counter()
guard="import sys, pathlib; allowed=str(pathlib.Path(sys.executable).resolve());\ndef audit(event,args):\n if event=='subprocess.Popen' and str(pathlib.Path(args[0]).resolve())!=allowed and pathlib.Path(args[0]).name not in {'lscpu','uname','ldconfig','gcc','cc','g++','c++','ptxas','nvcc','ld','nvidia-smi'}: raise RuntimeError('external stage program rejected by prefix benchmark')\nsys.addaudithook(audit)\nimport torch\ntorch.cuda.init()\ntorch.cuda.set_device(torch.device(sys.argv[sys.argv.index('--device')+1]))\ntorch.cuda.reset_peak_memory_stats()\nfrom fnit.recon_all.volume_prefix_python import main\nmain()\n"
command=[sys.executable,'-c',guard,str(a.input),str(a.output/'subject'),
 '--weights-dir',str(a.weights),'--assets-dir',str(a.assets),'--device',a.device,'--threads','4','--n4-backend',a.n4_backend]
# 调用自己的Python模块，原软件参考既不在参数中，也不向生产者传输入。
with (a.output/'producer.stdout.private.log').open('w') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 while child.poll() is None:
  sample={'elapsed_seconds':time.perf_counter()-started,'own_memory_bytes':None}
  if query is not None:
   try:
    entries,total,error=query.snapshot(os.environ['CUDA_VISIBLE_DEVICES'])
    sample.update(driver_processes=entries,device_total_bytes=total,device_query_error=error)
   except Exception as e:sample['error_type']=type(e).__name__
  report['nvml_samples'].append(sample)
  time.sleep(1)
 code=child.wait()
if query is not None:query.close()
report.update(exit_code=code,execution_complete=code==0,wall_seconds_including_spawn_import_load_IO=time.perf_counter()-started)
private=a.output/'subject/volume-prefix.private.json'
if private.exists():
 api=json.loads(private.read_text())
 report.update(status=api['status'],failed_stage=api.get('failed_stage'),error_type=api.get('error_type'),
               stages=[{'name':r['name'],'seconds':r['seconds']} for r in api['stages']],
               precision=api.get('precision'),resources=api['resources'],dependencies=api['dependencies'],
               torch_peak_allocated_bytes=api.get('torch_peak_allocated_bytes'),torch_peak_reserved_bytes=api.get('torch_peak_reserved_bytes'))
 for stage in api['stages']:
  if stage['name']=='input_talairach':
   result=stage['result'];report['input_models_actual_forwards']=result.get('actual_forwards',[])
   report['talairach_child_gpu']=result.get('talairach_child_gpu')
(a.output/'metrics.public.json').write_text(json.dumps(portable(report),indent=2,allow_nan=False)+'\n')
raise SystemExit(code)
