"""同输入完整四轮pial AB回归；冻结原碰撞实现仅用于CPU基线。"""
import argparse,hashlib,importlib.util,inspect,json,os,platform,time
from pathlib import Path
import numpy as np,nibabel.freesurfer.io as fs,numba,torch
from fnit.recon_all import place_pial_python as pial
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--subject',type=Path,required=True);p.add_argument('--hemi',choices=['lh','rh'],required=True);p.add_argument('--frozen-collision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);numba.set_num_threads(4)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
files=[a.subject/'surf'/f'{a.hemi}.white',a.subject/'surf'/f'autodet.gw.stats.{a.hemi}.dat',*[a.subject/'label'/f'{a.hemi}.{x}.label' for x in ('cortex','cortex+hipamyg')],*[a.subject/'mri'/f'{x}.mgz' for x in ('brain.finalsurfs','wm','aseg.presurf')]]
report=dict(commit=a.commit,host=platform.node(),scope='frozen same-input complete four-pass pial; not self-generated continuous chain or whole case',input_sha256={str(x):sha(x) for x in files},source_sha256={str(x):sha(x) for x in Path(inspect.getfile(pial)).parent.glob('place*.py')},frozen_collision_sha256=sha(a.frozen_collision),benchmark_sha256=sha(__file__),program_sha256=sha(os.sys.executable),tolerance='exact coordinates, accepted trajectory and cleanup',overall_equivalence='not_assessed',runs={},trace=[])
def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
bootstrap=time.perf_counter()
try:
 bootstrap_tensor=torch.empty(1,device=a.device);torch.cuda.synchronize(a.device);del bootstrap_tensor
 report['cuda_bootstrap_seconds']=time.perf_counter()-bootstrap
 report['gpu_uuid']=str(torch.cuda.get_device_properties(a.device).uuid)
except Exception as error:
 report['cuda_bootstrap_failure']=repr(error);save();raise
report['actual_tf32']=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32)
spec=importlib.util.spec_from_file_location('fnit.recon_all.frozen_collision',a.frozen_collision);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
new_async=pial.asynchronous_first_step;baseline_coordinates={}
for name,backend in [('baseline','cpu'),('candidate','triton')]:
 def frozen_async(*args,**kwargs):
  kwargs.pop('candidate_backend',None)
  return old.asynchronous_first_step(*args,**kwargs)
 pial.asynchronous_first_step=frozen_async if name=='baseline' else new_async
 def trace(step,pass_index,xyz,state):
  item=dict(run=name,step=step,pass_index=pass_index,state=state,coordinates_sha256=hashlib.sha256(xyz.tobytes()).hexdigest())
  if name=='baseline':baseline_coordinates[step]=(xyz,state,pass_index)
  else:
   ref,refstate,refpass=baseline_coordinates[step];delta=np.linalg.norm(xyz.astype(float)-ref,axis=1)
   item.update(different_components=int(np.count_nonzero(xyz!=ref)),max_mm=float(delta.max()),p99_mm=float(np.quantile(delta,.99)),same_state=state==refstate,same_pass=pass_index==refpass)
  report['trace'].append(item);save()
 t=time.perf_counter()
 result=pial.place_pial_t1(subject=a.subject,hemisphere=a.hemi,output=a.output/f'{name}.pial.T1',max_steps=200,sampling_backend=backend,device=a.device if backend!='cpu' else None,trace_callback=trace,candidate_backend='tree')
 report['runs'][name]=dict(seconds=time.perf_counter()-t,stage=result,surface_sha256=sha(a.output/f'{name}.pial.T1'));save()
reference,rf=fs.read_geometry(str(a.output/'baseline.pial.T1'));candidate,cf=fs.read_geometry(str(a.output/'candidate.pial.T1'));np.testing.assert_array_equal(rf,cf)
delta=np.linalg.norm(candidate-reference,axis=1);report['comparison']=dict(ordered_faces_exact=True,different_components=int(np.count_nonzero(reference!=candidate)),max_mm=float(delta.max()),p99_mm=float(np.quantile(delta,.99)),same_pass_ends=report['runs']['baseline']['stage']['pass_ends']==report['runs']['candidate']['stage']['pass_ends'],same_cleanup=report['runs']['baseline']['stage']['cleanup']==report['runs']['candidate']['stage']['cleanup']);save()
np.testing.assert_array_equal(candidate,reference)
assert report['comparison']['same_pass_ends'] and report['comparison']['same_cleanup']
assert all(x['different_components']==0 and x['same_state'] and x['same_pass'] for x in report['trace'] if x['run']=='candidate')
