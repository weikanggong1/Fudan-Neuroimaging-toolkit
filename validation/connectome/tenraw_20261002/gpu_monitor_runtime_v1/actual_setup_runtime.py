import hashlib, json, os, subprocess, time, sys
from pathlib import Path
root=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/nvml_monitor_runtime_v1')
original=Path('/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/bin/python')
venv=root/'venv'
wheel=root/'nvidia_ml_py-13.580.82-py3-none-any.whl'
assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='4361db337b0c551e2d101936dae2e9a60f957af26818e8c0c3a1f32b8db8d0a7'
code='''import sys,json,hashlib,importlib,torch
from pathlib import Path
names=['torch','torch._C','numpy','numpy.core._multiarray_umath','nibabel']
modules={}
for name in names:
 m=importlib.import_module(name);p=Path(m.__file__).resolve();modules[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'version':getattr(m,'__version__',None)}
p=Path(sys.executable).resolve()
print(json.dumps({'gpu_python':sys.executable,'python_binary':str(p),'python_binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'python_version':sys.version,'modules':modules,'CUDA_initialized':torch.cuda.is_initialized()}))'''
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
env.pop('PYTHONPATH',None)
commands=[]
def run(argv):
 start=time.perf_counter();p=subprocess.run(argv,env=env,capture_output=True,text=True)
 commands.append({'argv':argv,'returncode':p.returncode,'wall_seconds':time.perf_counter()-start,'stderr':p.stderr[-3000:]})
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return p.stdout
parent=json.loads(run([str(original),'-c',code]));assert not venv.exists()
run([str(original),'-m','venv','--system-site-packages',str(venv)])
run([str(venv/'bin/python'),'-m','pip','install','--no-index','--no-deps',str(wheel)])
child=json.loads(run([str(venv/'bin/python'),'-c',code]))
for key in ['python_binary','python_binary_sha256','python_version','modules','CUDA_initialized']:assert parent[key]==child[key],key
probe='''import sys,json,time,hashlib,importlib.metadata,torch,subprocess,importlib.util
from pathlib import Path
import pynvml
p=Path(pynvml.__file__).resolve();before=torch.cuda.is_initialized();pynvml.nvmlInit()
uuid='GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba';h=pynvml.nvmlDeviceGetHandleByUUID(uuid)
start=time.perf_counter();rows=pynvml.nvmlDeviceGetComputeRunningProcesses(h);elapsed=time.perf_counter()-start
query=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
smi={}
for line in query.stdout.splitlines():
 f=[x.strip() for x in line.split(',')]
 if len(f)==3 and f[0]==uuid:smi[int(f[1])]=int(f[2])*2**20
api={int(x.pid):int(x.usedGpuMemory) for x in rows}
common=api.keys()&smi.keys()
comparison={str(pid):{'NVML_bytes':api[pid],'SMI_bytes':smi[pid],'equal':api[pid]==smi[pid]} for pid in common}
s=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/formal_harness_staged_gpu_v2/benchmark_connectome_raw_bids.py')
spec=importlib.util.spec_from_file_location('original_wall',s);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
monitor=m.GPUProcessMonitor(torch,'cuda:0',interval=.5,gpu_uuid=uuid);monitor.start();time.sleep(6);report=monitor.finish();after=torch.cuda.is_initialized()
assert not before and not after and report['backend']=='pynvml' and report['failed_samples']==0 and not report['errors'] and report['samples']>=10
assert report['max_observed_interval_seconds']<2
print(json.dumps({'package':'nvidia-ml-py','package_version':importlib.metadata.version('nvidia-ml-py'),'module_path':str(p),'module_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'torch_build_CUDA':torch.version.cuda,'actual_driver_version':pynvml.nvmlSystemGetDriverVersion(),'actual_NVML_version':pynvml.nvmlSystemGetNVMLVersion(),'NVML_query_seconds':elapsed,'SMI_NVML_common_PID_memory_comparison':comparison,'CUDA_initialized_before':before,'CUDA_initialized_after':after,'original_wall_helper':{'path':str(s),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()},'real_monitor_probe':report}))'''
probe_report=json.loads(run([str(venv/'bin/python'),'-c',probe]))
assert probe_report['original_wall_helper']['sha256']=='a34f8ee9866ff825b4ca3e3d523597cdd1d08274da8f2a92cc40f846b8ee71f8'
report={'scope':'optional_NVML_monitor_environment','gpu_python':str(venv/'bin/python'),'science_modules_unchanged':True,'CUDA_initialized':False,'original_runtime':parent,'new_runtime':child,'NVML':probe_report,'commands':commands,'download_provenance':json.loads((root/'download_provenance.json').read_text()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'production_source_and_wall_bytes_changed':False,'scientific_GPU_work_dispatched':False}
(root/'runtime_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':'actual_runtime_preflight_completed','gpu_python':report['gpu_python'],'science_modules_unchanged':True,'CUDA_initialized':False,'monitor':probe_report['real_monitor_probe']}))
