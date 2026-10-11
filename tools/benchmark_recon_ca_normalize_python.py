"""完整CA normalization同输入Python/独立原版比较，不向producer传参考输出。"""
import argparse,hashlib,inspect,json,os,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np
from fnit.recon_all.ca_normalize_python import run_ca_normalize
p=argparse.ArgumentParser(description=__doc__)
for name in ('nu','mask','atlas','lta','binary','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report={'scope':'same_input_complete_CA_normalization_only','input_sha256':{name:sha(getattr(a,name)) for name in ('nu','mask','atlas','lta')},'reference_binary_sha256':sha(a.binary),'strict_tolerance':0,'overall_equivalence':'not_assessed'}
source=Path(inspect.getfile(run_ca_normalize)).resolve().parent
report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),timing_boundary='Python file API excludes interpreter/imports; native reference includes spawn and IO',source_sha256={name:sha(source/name) for name in ('ca_normalize_python.py','mri_em_register.py')},benchmark_source_sha256=sha(Path(__file__)))
def typed(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {key:typed(item) for key,item in value.items()}
 if isinstance(value,(list,tuple)):return [typed(item) for item in value]
 return value
for mode in ('python','native-reference'):
 output=a.output/mode;output.mkdir()
 tick=time.perf_counter()
 try:
  if mode=='python':
   report['python_api']=run_ca_normalize(a.nu,a.mask,a.atlas,a.lta,output/'norm.mgz',output/'ctrl_pts.mgz')
  else:
   with (output/'stdout.private.log').open('w') as log:
    subprocess.run([str(a.binary),'-c',str(output/'ctrl_pts.mgz'),'-mask',str(a.mask),str(a.nu),str(a.atlas),str(a.lta),str(output/'norm.mgz')],check=True,stdout=log,stderr=subprocess.STDOUT)
 except Exception as error:
  report.update(execution_complete=False,failed_mode=mode,error_type=type(error).__name__)
  (a.output/'metrics.public.json').write_text(json.dumps(typed(report),indent=2,allow_nan=False)+'\n');raise
 report[mode+'_file_API_seconds']=time.perf_counter()-tick
report['volumes']={}
for name in ('norm','ctrl_pts'):
 images=[nib.load(str(a.output/mode/f'{name}.mgz')) for mode in ('python','native-reference')]
 arrays=[np.asarray(image.dataobj) for image in images]
 delta=np.abs(arrays[0].astype(np.float64)-arrays[1].astype(np.float64))
 report['volumes'][name]={'different_elements':int(np.count_nonzero(arrays[0]!=arrays[1])),
  'max_error':float(delta.max()),'p99_error':float(np.quantile(delta,.99)),
  'shape_equal':images[0].shape==images[1].shape,'affine_equal':bool(np.array_equal(images[0].affine,images[1].affine)),
  'dtype_equal':images[0].get_data_dtype()==images[1].get_data_dtype()}
report['strict_passed']=all(row['different_elements']==0 and row['shape_equal'] and row['affine_equal'] and row['dtype_equal'] for row in report['volumes'].values())
report['execution_complete']=True
(a.output/'metrics.public.json').write_text(json.dumps(typed(report),indent=2,allow_nan=False)+'\n')
