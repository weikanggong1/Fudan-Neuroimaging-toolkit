"""N4 template/compiler/library cross diagnostics; release lock per variant."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
CURRENT=np.fromfile(ROOT/'itk5_0.final.raw',np.float32)
OLD=np.fromfile(ROOT/'gcc48.final.raw',np.float32)
def metric(a,b):
 d=np.abs(a.astype(np.float64)-b.astype(np.float64));return {'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),'p99_abs':float(np.quantile(d,.99))}
report={'kind':'diagnostic_only_not_production_backend','runs':[]}
for tag,build,libraries in [('floatmask','build4_float_conda','/tmp/fnit_itk4_build_20260929/ITK4-install/lib'),('oldlibraries','build4_oldlibs_conda','/tmp/fnit_itk4_build_20260929/ITK4-install-gcc48b/lib')]:
 binary=ROOT/build/'fnit_n4_diagnostic'
 env=dict(os.environ,LD_LIBRARY_PATH=libraries+':/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/lib',FNIT_N4_DIAGNOSTIC_PREFIX=str(ROOT/tag),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);tick=time.perf_counter()
  subprocess.run([str(binary),str(ROOT/'input.raw'),str(ROOT/(tag+'.final.raw')),'256','256','256','1','1','1','4',str(ROOT/(tag+'.profile.json'))],env=env,check=True)
  elapsed=time.perf_counter()-tick
 a=np.fromfile(ROOT/(tag+'.final.raw'),np.float32)
 report['runs'].append({'tag':tag,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'wall_seconds':elapsed,'vs_current':metric(a,CURRENT),'vs_gcc48_official':metric(a,OLD),'fields_vs_current':{field:metric(np.fromfile(ROOT/(tag+'.'+field+'.raw'),np.float32),np.fromfile(ROOT/('itk5_0.'+field+'.raw'),np.float32)) for field in ('lattice','logfield','expfield')}})
 (ROOT/'n4_cross.json').write_text(json.dumps(report,indent=2))
