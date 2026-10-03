"""Isolate official repeatability and native compiler environment on frozen input."""
import fcntl,hashlib,json,os,platform,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
ORIG=BASE/'reference_sub02_official/subjects/b_official/mri/orig.mgz'
OFFICIAL=Path('/public/software/apps/Freesurfer/8.2.0-1/bin/AntsN4BiasFieldCorrectionFs')
NATIVE=ROOT/'build4_gcc48_clean/fnit_n4_diagnostic'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));return {'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),'p99_abs':float(np.quantile(d,.99))}
report={'host':platform.node(),'input_sha256':sha(ORIG),'official_binary_sha256':sha(OFFICIAL),'native_binary_sha256':sha(NATIVE),'runs':[],'kind':'diagnostic_only_not_production_backend'}
with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',FS_LICENSE='/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt',FREESURFER_HOME='/public/software/apps/Freesurfer/8.2.0-1')
 env['LD_LIBRARY_PATH']='/tmp/fnit_itk4_build_20260929/ITK4-install-gcc48b/lib:/lib64:/usr/lib64'
 env['FNIT_N4_DIAGNOSTIC_PREFIX']=str(ROOT/'gcc48')
 commands=[('gcc48',[str(NATIVE),str(ROOT/'input.raw'),str(ROOT/'gcc48.final.raw'),'256','256','256','1','1','1','4',str(ROOT/'gcc48.profile.json')]),('official',[str(OFFICIAL),'-i',str(ORIG),'-o',str(ROOT/'official_repeat_float.mgz'),'--dtype','float'])]
 for tag,command in commands:
  tick=time.perf_counter()
  with (ROOT/(tag+'.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  report['runs'].append({'tag':tag,'wall_seconds':time.perf_counter()-tick,'command':command})
  (ROOT/'n4_environment.json').write_text(json.dumps(report,indent=2))
old=np.asarray(nib.load(str(BASE/'volume_parity_20260930/sub02/n4_float/reference_float.mgz')).dataobj,dtype=np.float32).ravel(order='F')
repeat=np.asarray(nib.load(str(ROOT/'official_repeat_float.mgz')).dataobj,dtype=np.float32).ravel(order='F')
current=np.fromfile(ROOT/'itk5_0.final.raw',dtype=np.float32)
gcc=np.fromfile(ROOT/'gcc48.final.raw',dtype=np.float32)
report['official_repeat_vs_historical']=metric(repeat,old)
report['gcc48_vs_official']=metric(gcc,repeat)
report['gcc48_vs_itk5_conda']=metric(gcc,current)
report['uchar_gcc48_vs_official']=metric(np.floor(np.clip(gcc,0,255)+.5),np.floor(np.clip(repeat,0,255)+.5))
report['gcc48_vs_itk5_fields']={name:metric(np.fromfile(ROOT/('gcc48.'+name+'.raw'),np.float32),np.fromfile(ROOT/('itk5_0.'+name+'.raw'),np.float32)) for name in ('lattice','logfield','expfield')}
(ROOT/'n4_environment.json').write_text(json.dumps(report,indent=2))
