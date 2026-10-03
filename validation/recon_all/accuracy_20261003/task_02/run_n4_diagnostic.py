"""Paired N4 diagnostic, frozen real input, exclusive benchmark lock."""
import fcntl, hashlib, json, os, platform, subprocess, time
from pathlib import Path
import nibabel as nib
import numpy as np
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
REF=BASE/'reference_sub02_official/subjects/b_official/mri/orig.mgz'
OLD=BASE/'volume_parity_20260930/sub02/n4_float'
def sha(path):
    with open(path,'rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def metric(a,b):
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'different':int(np.count_nonzero(d)),'max_abs':float(d.max()),'p99_abs':float(np.quantile(d,.99))}
x=np.asarray(nib.load(str(REF)).dataobj,dtype=np.float32).ravel(order='F')
y=np.fromfile(OLD/'input.raw',dtype=np.float32)
if not np.array_equal(x,y): raise ValueError('historical input.raw is not official orig')
(ROOT/'input.raw').write_bytes(x.tobytes())
report={'host':platform.node(),'source_commit':'816e5610417a4c587caf321049438a9554139016','input_sha256':sha(ROOT/'input.raw'),'orig_sha256':sha(REF),'diagnostic_cpp_sha256':sha(ROOT/'n4_diagnostic.cpp'),'lock':'/tmp/fnit-shared-benchmark.lock','fitting_threads':1,'reconstruction_threads':4,'gpu_memory':'not applicable: N4 existing CPU ITK stage','runs':[]}
with open('/tmp/fnit-shared-benchmark.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for version in ('5','4','5'):
        tag=f'itk{version}_{len(report["runs"])}'; binary=ROOT/f'build{version}_conda/fnit_n4_diagnostic'; prefix=ROOT/tag
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='4',FNIT_N4_DIAGNOSTIC_PREFIX=str(prefix))
        env['LD_LIBRARY_PATH']=str(BASE/'fnit_main_env/lib')+':'+env.get('LD_LIBRARY_PATH','') if version=='5' else '/tmp/fnit_itk4_build_20260929/ITK4-install/lib:'+env.get('LD_LIBRARY_PATH','')
        start=time.perf_counter()
        subprocess.run([str(binary),str(ROOT/'input.raw'),str(prefix)+'.final.raw','256','256','256','1','1','1','4',str(prefix)+'.profile.json'],env=env,check=True)
        report['runs'].append({'tag':tag,'binary_sha256':sha(binary),'wall_seconds_including_diagnostic_exports':time.perf_counter()-start,'profile':json.loads(Path(str(prefix)+'.profile.json').read_text())})
        (ROOT/'n4_diagnostic.json').write_text(json.dumps(report,indent=2))
reference=np.asarray(nib.load(str(OLD/'reference_float.mgz')).dataobj,dtype=np.float32).ravel(order='F')
for run in report['runs']:
    vals=np.fromfile(ROOT/(run['tag']+'.final.raw'),dtype=np.float32)
    run['historical_official_float_comparison']=metric(vals,reference)
    run['historical_official_uint8_comparison']=metric(np.floor(np.clip(vals,0,255)+.5).astype(np.uint8),np.floor(np.clip(reference,0,255)+.5).astype(np.uint8))
report['itk4_vs_itk5']={}
report['itk5_repeat']={}
for field in ('lattice','logfield','expfield','final'):
    a=np.fromfile(ROOT/f'itk5_0.{field}.raw',dtype=np.float32);b=np.fromfile(ROOT/f'itk4_1.{field}.raw',dtype=np.float32);c=np.fromfile(ROOT/f'itk5_2.{field}.raw',dtype=np.float32)
    report['itk4_vs_itk5'][field]=metric(a,b)
    report['itk5_repeat'][field]=metric(a,c)
(ROOT/'n4_diagnostic.json').write_text(json.dumps(report,indent=2))
