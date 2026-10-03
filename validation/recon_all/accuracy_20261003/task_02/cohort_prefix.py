"""10例公开T1完整前段AB/BA，保留逐例失败及实际GPU前向。"""
import argparse, fcntl, hashlib, importlib.util, json, os, platform, subprocess, sys, time
from pathlib import Path
import nibabel as nib
import numpy as np
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
RUNTIME=BASE/'accuracy_20261003/baseline_runtime_816e5610'
OUTPUT=BASE/'accuracy_20261003/task_02/cohort_prefix'
COHORT=BASE/'accuracy_20261003/cohort/cohort_verified.json'
MONITOR=BASE/'parallel_20261002/coordinator/run_monitored.py'
UUID='GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def compare(a,b):
 x=nib.load(str(a));y=nib.load(str(b));v=np.asanyarray(x.dataobj);w=np.asanyarray(y.dataobj);d=np.abs(v.astype(np.float64)-w.astype(np.float64))
 return {'shape_a':list(v.shape),'shape_b':list(w.shape),'dtype_a':str(v.dtype),'dtype_b':str(w.dtype),'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),'p99_abs':float(np.quantile(d,.99)),'geometry_max_abs':float(np.max(np.abs(x.affine-y.affine))),'sha_a':sha(a),'sha_b':sha(b)}
def lta_matrix(p):
 s=p.read_text().splitlines();i=s.index('1 4 4')+1;return np.asarray([[float(v) for v in row.split()] for row in s[i:i+4]])
def worker(case,backend):
 sys.path.insert(0,str(RUNTIME/"src"))
 os.environ["PYTHONPATH"]=str(RUNTIME/"src")
 import torch
 torch.manual_seed(0);np.random.seed(0)
 torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 module_path=ROOT/('input_talairach_chain_baseline.py' if backend=='baseline' else 'input_talairach_chain.py')
 spec=importlib.util.spec_from_file_location('fnit.recon_all.input_talairach_chain_'+backend,module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 input_file=Path(case['server_input'])
 resources={name:sha(BASE/name) for name in ('weights/synthstrip.1.pt','weights/synthmorph.affine.2.h5','assets/average/mni305.cor.stripped.mgz')}
 if sha(input_file)!=case['sha256']:raise ValueError('cohort input SHA mismatch')
 target=OUTPUT/case['id']/backend
 torch.cuda.synchronize('cuda:0');start=time.perf_counter()
 result=module.run_input_talairach_chain(t1=input_file,subject_dir=target,weights_dir=BASE/'weights',assets_dir=BASE/'assets',device='cuda:0',threads=4)
 torch.cuda.synchronize('cuda:0')
 result.update(wall_seconds=time.perf_counter()-start,source_file_sha256=sha(module_path),input_sha256=case['sha256'],gpu_uuid=UUID,host=platform.node(),torch_version=torch.__version__,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated('cuda:0'),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved('cuda:0'))
 import fnit.recon_all.conform_gpu as c
 import fnit.recon_all.input_chain as i
 import fnit.recon_all.nifti_import as n
 import fnit.recon_all.talairach_synthmorph as t
 import fnit.synthstrip.pipeline as s
 import fnit.synthmorph.pipeline as m
 result['seed']=0
 result['benchmark_script_sha256']=sha(Path(__file__))
 result['resources_sha256']=resources
 result['cpu_affinity']=sorted(os.sched_getaffinity(0))
 result['thread_environment']={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','PYTORCH_NO_CUDA_MEMORY_CACHING')}
 result['runtime_sha256']={name.__file__:sha(Path(name.__file__)) for name in (c,i,n,t,s,m)}
 (target/'scripts/prefix_run.json').write_text(json.dumps(result,indent=2))
 return result
parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--backend',choices=['baseline','candidate']);args=parser.parse_args()
cohort=json.loads(COHORT.read_text())
if args.case:
 worker(next(c for c in cohort['cases'] if c['id']==args.case),args.backend)
else:
 OUTPUT.mkdir(parents=True,exist_ok=False)
 report={'source_commit':'816e5610417a4c587caf321049438a9554139016','runtime_source':'coordinator SHA-verified baseline_runtime_816e5610/src; worker records actual runtime SHA and paths','cohort_sha256':sha(COHORT),'gpu_uuid':UUID,'threads':4,'overall_equivalence':'not_assessed','cases':[]}
 env=dict(os.environ,PYTHONPATH=str(RUNTIME/'src'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4',PYTORCH_NO_CUDA_MEMORY_CACHING='1')
 for index,case in enumerate(cohort['cases']):
  status={'id':case['id'],'input_sha256':case['sha256'],'runs':[]}
  with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   for backend in (('baseline','candidate') if index%2==0 else ('candidate','baseline')):
    try:
     command=[sys.executable,str(MONITOR),'--gpu-uuid',UUID,'--output',str(OUTPUT/case['id']/(backend+'_monitor')),'--interval','1','--',sys.executable,str(Path(__file__)), '--case',case['id'],'--backend',backend]
     code=subprocess.run(command,env=env).returncode
     status['runs'].append({'backend':backend,'exit_code':code})
    except Exception as error:status['runs'].append({'backend':backend,'error':str(error)})
  if all(r.get('exit_code')==0 for r in status['runs']):
   a=OUTPUT/case['id']/'baseline';b=OUTPUT/case['id']/'candidate'
   status['volumes']={name:compare(a/'mri'/name,b/'mri'/name) for name in ('orig/001.mgz','rawavg.mgz','orig.mgz','synthstrip.mgz')}
   status['xfm_bytes_equal']=(a/'mri/transforms/talairach.xfm').read_bytes()==(b/'mri/transforms/talairach.xfm').read_bytes()
   status['lta_matrix_max_abs']={name:float(np.max(np.abs(lta_matrix(a/'mri/transforms'/name)-lta_matrix(b/'mri/transforms'/name)))) for name in ('synthmorph.mni305/aff.lta','talairach.xfm.lta')}
  report['cases'].append(status); (OUTPUT/'report.json').write_text(json.dumps(report,indent=2))
