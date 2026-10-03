"""CPU-only live identity and direct NVML preflight for an existing monitor venv."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
OLD = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002')
NEW = Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_accuracy_20261003_v1')
ORIGINAL = '/cwStorage/home/gongwk/Notebook_code/fnit_conda_env_956b1a9/bin/python'
MONITOR = str(OLD/'nvml_monitor_runtime_v1/venv/bin/python')
UUID = 'GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba'

PROBE = r'''
import hashlib, importlib, importlib.metadata, json, pathlib, sys, torch
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
modules={}
for name in ('torch','torch._C','numpy','numpy.core._multiarray_umath','nibabel'):
    module=importlib.import_module(name); path=pathlib.Path(module.__file__).resolve()
    modules[name]={'path':str(path),'sha256':sha(path),'version':getattr(module,'__version__',None)}
libs={}
for directory in (pathlib.Path(torch.__file__).parent/'lib', pathlib.Path(importlib.import_module('numpy').__file__).parent.parent/'numpy.libs'):
    for path in sorted(directory.glob('*.so*')):
        if path.is_file():
            resolved=path.resolve(); libs[str(resolved)]={'size_bytes':resolved.stat().st_size,'sha256':sha(resolved)}
binary=pathlib.Path(sys.executable).resolve()
result={'executable':sys.executable,'binary':str(binary),'binary_sha256':sha(binary),'python_version':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'modules':modules,'shared_libraries':libs,'torch_cuda_build':torch.version.cuda,'CUDA_initialized':torch.cuda.is_initialized()}
try:
    import pynvml
    path=pathlib.Path(pynvml.__file__).resolve()
    result['nvml']={'version':importlib.metadata.version('nvidia-ml-py'),'path':str(path),'sha256':sha(path)}
except ImportError: result['nvml']=None
print(json.dumps(result))
'''

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None)
observed=[]
for python in (ORIGINAL,MONITOR):
    completed=subprocess.run([python,'-c',PROBE],env=env,text=True,capture_output=True,check=True)
    observed.append(json.loads(completed.stdout))
before,after=observed
fields=('binary','binary_sha256','python_version','base_prefix','modules','shared_libraries','torch_cuda_build')
assert all(before[k]==after[k] for k in fields), 'scientific runtime bytes differ'
assert before['CUDA_initialized'] is False and after['CUDA_initialized'] is False
previous=OLD/'nvml_monitor_runtime_v1/runtime_preflight.json'
prior=json.loads(previous.read_bytes())
assert sha(previous)=='a67c203afc9b7a74da52fa3a5003dc16c70cd04e3a3e13f98db09ea6ec02054e'
assert after['binary_sha256']=='c502d03a78bfafb464a218c69ba653d29bf27f3fea24324b010ddfa4d8af9153'
assert after['nvml']['sha256']=='4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea'
wall=NEW/'formal_frozen_v1/tools_source/tools/benchmark_connectome_raw_bids.py'
assert sha(wall)=='c1715067883e7e04d9b953e509572227237efd27c712cc9887956d17ad13fe75'
monitor_code=r'''
import importlib.util,json,sys,time,torch,pynvml
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('frozen_wall',sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
before=torch.cuda.is_initialized(); monitor=module.GPUProcessMonitor(torch,'cuda:0',gpu_uuid=sys.argv[2]); monitor.start(); driver=pynvml.nvmlSystemGetDriverVersion(); nvml_version=pynvml.nvmlSystemGetNVMLVersion(); time.sleep(6); result=monitor.finish()
result.update(CUDA_initialized_before=before,CUDA_initialized_after=torch.cuda.is_initialized(),driver=driver,nvml_version=nvml_version)
print(json.dumps(result))
'''
monitor=json.loads(subprocess.run([MONITOR,'-c',monitor_code,str(wall),UUID],env=env,text=True,capture_output=True,check=True).stdout)
assert monitor['backend']=='pynvml' and monitor['samples']>=10 and not monitor['errors']
assert monitor['failed_samples']==monitor['unresolved_device_samples']==0
assert monitor['CUDA_initialized_before'] is False and monitor['CUDA_initialized_after'] is False
assert monitor['max_observed_interval_seconds']<5
result={'scope':'live CPU identity and monitor probe; no scientific GPU work','original_runtime':before,'monitor_runtime':after,'scientific_runtime_equal':True,'previous_preflight':{'path':str(previous),'sha256':sha(previous)},'wall_helper':{'path':str(wall),'sha256':sha(wall)},'direct_nvml_probe':monitor,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
print(json.dumps(result,indent=2,allow_nan=False))
