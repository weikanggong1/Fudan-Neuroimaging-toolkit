"""真实 MRI/网格固定输入算子回归；只写专属结果，不改检查点。"""
import argparse,hashlib,json,os,platform,time,subprocess,inspect
from pathlib import Path
import numpy as np,nibabel as nib,nibabel.freesurfer.io as fs,numba,torch
from fnit.recon_all.place_surface_sampling import PlacementSampling
from fnit.recon_all.place_surface_border import _sample,_voxel,compute_border_values_first_pass
from fnit.recon_all.place_surface_intensity import intensity_gradient
from fnit.recon_all.place_surface_normals import FaceNormalTopology
from fnit.recon_all.place_surface_volume import prepare_placement_volume
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.place_surface_rip import rip_outside_label
from fnit.recon_all.place_surface_collision import subvolume_assignment,_assign_vertices
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',type=Path,nargs=2,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--commit',required=True);p.add_argument('--implementation',choices=['torch','triton'],default='torch');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);numba.set_num_threads(4)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def measure(fn,*args,**kwargs):
 torch.cuda.synchronize(a.device);t=time.perf_counter();r=fn(*args,**kwargs);torch.cuda.synchronize(a.device);return r,time.perf_counter()-t
@numba.njit(cache=True)
def cpu_samples(volume,xyz,affine):
 r=np.empty(len(xyz),np.float64)
 for i in range(len(xyz)):
  x,y,z=_voxel(affine,float(xyz[i,0]),float(xyz[i,1]),float(xyz[i,2]));r[i]=_sample(volume,x,y,z)
 return r
rows=[]
for subject in a.subjects:
 for hemi in ('lh','rh'):
  paths=[subject/'surf'/f'{hemi}.white',subject/'surf'/f'{hemi}.pial.T1',subject/'surf'/f'autodet.gw.stats.{hemi}.dat',subject/'label'/f'{hemi}.cortex+hipamyg.label',*[subject/'mri'/f'{x}.mgz' for x in ('brain.finalsurfs','wm','aseg.presurf')]]
  brain=nib.load(str(paths[4]));white,faces,metadata=fs.read_geometry(str(paths[0]),read_metadata=True);white=white.astype(np.float32);current,_=fs.read_geometry(str(paths[1]));current=current.astype(np.float32)
  ripped=rip_outside_label(len(white),fs.read_label(str(paths[3]))).astype(bool);stats=dict(line.split()[:2] for line in paths[2].read_text().splitlines() if len(line.split())>=2)
  volume,bright=prepare_placement_volume(np.asarray(brain.dataobj),np.asarray(nib.load(str(paths[5])).dataobj),surface='pial',mid_gray=float(stats['MID_GRAY']));placement=volume.copy();placement[bright==130]=0
  affine=surface_ras_to_voxel(brain.header,metadata);normals=FaceNormalTopology(faces,len(white)).evaluate(white)
  thresholds=np.array([float(stats[f'pial_{x}']) for x in ('inside_hi','border_hi','border_low','outside_low','outside_hi')]);border=compute_border_values_first_pass(volume,np.asarray(nib.load(str(paths[6])).dataobj),white,normals,white,ripped,np.full(len(white),-1,np.float32),affine,thresholds,hemisphere=hemi,surface='pial')
  context,load_seconds=measure(PlacementSampling,placement,affine,device=a.device,implementation=a.implementation)
  cpu,cold_cpu=measure(cpu_samples,placement,current,affine);gpu,cold_gpu=measure(context.sample,current)
  sample_difference=np.abs(cpu-gpu);np.testing.assert_allclose(cpu,gpu,rtol=0,atol=1e-10)
  cpu_times=[];gpu_times=[]
  args=(white,normals,ripped,border[0],border[5],brain.header.get_zooms()[:3])
  reference,gradient_cpu_cold=measure(intensity_gradient,placement,*args[:5],affine,args[5]);candidate,gradient_gpu_cold=measure(context.gradient,*args)
  difference=np.abs(reference-candidate);np.testing.assert_allclose(reference,candidate,rtol=0,atol=1e-6)
  for iteration in range(4):
   functions=[('cpu',lambda:intensity_gradient(placement,*args[:5],affine,args[5])),('gpu',lambda:context.gradient(*args))]
   if iteration%2:functions.reverse()
   for name,fn in functions:
    result,seconds=measure(fn);(cpu_times if name=='cpu' else gpu_times).append(seconds)
  geometry,face_svi,new=subvolume_assignment(white,faces,white,ripped)
  flat=faces.ravel();incident=np.argsort(flat,kind='stable')//3;offsets=np.zeros(len(white)+1,np.int64);offsets[1:]=np.cumsum(np.bincount(flat,minlength=len(white)))
  def old_assignment():
   result=np.full(len(white),-1,np.int32)
   for vertex in range(len(white)):
    if ripped[vertex] or offsets[vertex]==offsets[vertex+1]:continue
    adjacent=face_svi[incident[offsets[vertex]:offsets[vertex+1]]];result[vertex]=int(adjacent[0]) if np.all(adjacent==adjacent[0]) else 64
   return result
  old,old_seconds=measure(old_assignment);new,new_seconds=measure(_assign_vertices,face_svi,incident,offsets,ripped);np.testing.assert_array_equal(old,new)
  rows.append(dict(subject=str(subject),hemi=hemi,vertices=len(white),input_sha256={str(x):sha(x) for x in paths},sampling_max_error=float(sample_difference.max()),gradient_max_error=float(difference.max()),gradient_different_components=int(np.count_nonzero(reference!=candidate)),context_load_seconds=load_seconds,sample_cold_cpu_seconds=cold_cpu,sample_cold_gpu_seconds=cold_gpu,gradient_cold_cpu_seconds=gradient_cpu_cold,gradient_cold_gpu_seconds=gradient_gpu_cold,gradient_warm_cpu_seconds=cpu_times,gradient_warm_gpu_seconds=gpu_times,subvolume_old_seconds=old_seconds,subvolume_new_seconds=new_seconds,subvolume_exact=True))
  (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
report=dict(commit=a.commit,host=platform.node(),device=a.device,gpu_uuid=str(torch.cuda.get_device_properties(a.device).uuid) if hasattr(torch.cuda.get_device_properties(a.device),'uuid') else None,threads=4,precision='native FP32 affine and displacement; native FP64 interpolation; no autocast; global TF32 untouched',tolerances=dict(sample_atol=1e-10,gradient_atol=1e-6,rtol=0,subvolume='exact int32'),implementation=a.implementation,scope='frozen real-data operators; not full placement or whole case',source_sha256={str(x):sha(x) for x in Path(inspect.getfile(PlacementSampling)).parent.glob('place*.py')},benchmark_sha256=sha(__file__),rows=rows,torch_memory_stats_status='unavailable_allocator_disabled' if 'PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ else 'available',torch_peak_allocated=None if 'PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ else torch.cuda.max_memory_allocated(a.device),torch_peak_reserved=None if 'PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ else torch.cuda.max_memory_reserved(a.device),external_load=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True),overall_equivalence='not_assessed')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
