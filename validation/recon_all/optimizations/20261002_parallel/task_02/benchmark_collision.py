"""两例四半球实际white到pial位移的同输入有序碰撞完整试步回归。"""
import argparse,hashlib,inspect,json,os,platform,time
from pathlib import Path
import nibabel.freesurfer.io as fs,numpy as np,numba,torch
from fnit.recon_all.place_surface_collision import asynchronous_first_step
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);numba.set_num_threads(4);torch.set_num_threads(4)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows=[]
for subject in a.subjects:
 for hemi in ('lh','rh'):
  paths=[subject/'surf'/f'{hemi}.white',subject/'surf'/f'{hemi}.pial.T1',subject/'label'/f'{hemi}.cortex+hipamyg.label']
  xyz,faces=fs.read_geometry(str(paths[0]));pial,pf=fs.read_geometry(str(paths[1]));np.testing.assert_array_equal(faces,pf);xyz=xyz.astype(np.float32);pial=pial.astype(np.float32)
  offsets=np.float32(pial-xyz);length=np.linalg.norm(offsets,axis=1);scale=np.minimum(1,.3/np.maximum(length,1e-30)).astype(np.float32);offsets=np.float32(offsets*scale[:,None]);proposal=np.float32(xyz+offsets)
  ripped=np.ones(len(xyz),bool);ripped[fs.read_label(str(paths[2]))]=False
  outputs={};timings={};accepted={};orders={}
  # Full first trial, including broadphase, Numba cold compilation and copies.
  for backend in ('tree','snapshot'):
   momentum=offsets.copy();tick=time.perf_counter();result,order=asynchronous_first_step(xyz,faces,proposal,ripped,offsets=offsets,accepted_offsets=momentum,candidate_backend=backend)
   timings[backend]=time.perf_counter()-tick;outputs[backend]=result;accepted[backend]=momentum;orders[backend]=order
  np.testing.assert_array_equal(outputs['tree'],outputs['snapshot']);np.testing.assert_array_equal(accepted['tree'],accepted['snapshot']);np.testing.assert_array_equal(orders['tree'],orders['snapshot'])
  # Replay a limited retained-MHT trial: the snapshot selection must retain
  # original source path/state; this is a regression, not its performance claim.
  retries={}
  for backend in ('tree','snapshot'):
   momentum=offsets.copy();result,order=asynchronous_first_step(xyz,faces,proposal,ripped,offsets=offsets,accepted_offsets=momentum,stale_mht_trial=outputs['tree'],limit=1000,candidate_backend=backend);retries[backend]=(result,momentum,order)
  for x,y in zip(retries['tree'],retries['snapshot']):np.testing.assert_array_equal(x,y)
  rows.append(dict(subject=str(subject),hemi=hemi,vertices=len(xyz),input_sha256={str(x):sha(x) for x in paths},tree_seconds=timings['tree'],snapshot_seconds=timings['snapshot'],coordinates_exact=True,accepted_offsets_exact=True,order_exact=True,retained_retry_exact=True,coordinate_sha256=hashlib.sha256(outputs['snapshot'].tobytes()).hexdigest(),timing_scope='full first trial including broadphase and cold JIT; real derived clipped white-to-pial displacement, not full optimizer'))
  (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
report=dict(commit=a.commit,host=platform.node(),threads=4,scope='real frozen mesh complete ordered collision trial; not full pial/white placement',tolerance='exact coordinates, order and accepted momentum',source_sha256={str(x):sha(x) for x in Path(inspect.getfile(asynchronous_first_step)).parent.glob('place*.py')},benchmark_sha256=sha(__file__),program_sha256=sha(os.sys.executable),external_loadavg=os.getloadavg(),rows=rows,overall_equivalence='not_assessed')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
