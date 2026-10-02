"""自产冻结 nu/brainmask/SynthSeg/affine 输入，完整执行 GCA→filled；非原始T1整例。"""
import time
START=time.perf_counter()
import argparse,hashlib,json,os,platform,shutil,subprocess,threading
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from fnit.recon_all.assets import ASSET_FILES
from fnit.weights import WEIGHT_FILES
from fnit.recon_all.native_free import (_run_native_em_register,_segment_callosum,
                                      _run_native_wm_segment,_run_native_wm_edit)
from fnit.recon_all.ca_normalize_python import run_ca_normalize
from fnit.recon_all.normalization.aseg_pipeline import normalize_t1_aseg
from fnit.recon_all.ants_denoise_python import denoise_volume
from fnit.recon_all.fill_cutting_plane_python import fill_mgz
from fnit.recon_all.pretess_python import pretess_mgh
from fnit.recon_all.sclimbic import mri_entowm_seg,ENTOWM_MODEL,ENTOWM_CTAB
from fnit.recon_all.wm_edits_gpu import fix_ento_wm_gpu
from fnit.recon_all.wm_edits_python import fix_ento_wm as fix_ento_wm_cpu

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def geometry(image):return {'shape':[int(value) for value in image.shape],'affine':image.affine.tolist(),'dtype':str(image.get_data_dtype())}
def compare_volume(actual,reference,labels=False):
 x,y=nib.load(str(actual)),nib.load(str(reference));a,b=np.asarray(x.dataobj),np.asarray(y.dataobj)
 same=a.shape==b.shape and np.array_equal(x.affine,y.affine)
 result={'candidate_geometry':geometry(x),'reference_geometry':geometry(y),'same_geometry':same,'same_dtype':x.get_data_dtype()==y.get_data_dtype(),'candidate_sha256':sha(actual),'reference_sha256':sha(reference)}
 if not same:return result
 delta=np.abs(a.astype(np.float64)-b.astype(np.float64));result.update(different_elements=int(np.count_nonzero(a!=b)),max_error=float(delta.max()),p99_error=float(np.quantile(delta,.99)))
 if labels:
  values=np.union1d(np.unique(a),np.unique(b));result['per_label_dice']={str(int(v)):float(2*np.count_nonzero((a==v)&(b==v))/(np.count_nonzero(a==v)+np.count_nonzero(b==v))) for v in values}
 return result

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=('whole_sub01_candidate_retry1','whole_sub02_candidate_retry2'),required=True);p.add_argument('--commit',required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
mri=args.output/'mri';mri.mkdir();(mri/'transforms').mkdir()
for name in ('scripts','stats'):(args.output/name).mkdir()
reference=args.root/'serial_20261001'/args.case/'mri';assets=args.root/'assets';weights=args.root/'weights';binaries=args.root/'serial_20261001/native_bundle/bin'
snapshot=Path(__file__).resolve().parents[5]/'source_commit.txt'
actual_commit=snapshot.read_text().strip() if snapshot.exists() else args.commit
report={'commit':actual_commit,'dispatch_commit':args.commit,'case':args.case,'host':platform.node(),'pid':os.getpid(),'scope':'FNIT_frozen_prefix_continuous_GCA_to_filled_not_raw_T1_whole','overall_equivalence':'not_assessed','tolerance_declared':0,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS')},'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_version':torch.__version__,'input_sha256':{},'resource_sha256':{},'binary_sha256':{},'stages':[],'gpu_samples':[],'comparison':{},'execution_complete':False}
def json_scalar(value):
 if isinstance(value,np.generic):return value.item()
 raise TypeError('unsupported report value: '+type(value).__name__)
def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2,default=json_scalar)+'\n')
# Frozen side inputs are all previously self-produced; no intermediate result is copied.
for name in ('nu.mgz','brainmask.mgz','synthseg.rca.mgz','transforms/talairach.xfm.lta'):
 source=reference/name;report['input_sha256'][name]=sha(source);shutil.copyfile(source,mri/name)
for name in ('average/RB_all_2020-01-02.gca','SubCorticalMassLUT.txt'):
 source=assets/name;size,digest,*_=ASSET_FILES[name]
 if source.stat().st_size!=size or sha(source)!=digest:raise ValueError('asset verification failed: '+name)
 report['resource_sha256'][name]=digest
for name in (ENTOWM_MODEL,ENTOWM_CTAB):
 source=weights/name;url,size,digest=WEIGHT_FILES[name]
 if source.stat().st_size!=size or sha(source)!=digest:raise ValueError('weight verification failed: '+name)
 report['resource_sha256'][name]=digest
for name in ('mri_em_register','mri_segment','mri_edit_wm_with_aseg'):report['binary_sha256'][name]=sha(binaries/name)
# Baseline native default is kept; the slower optional cache is not enabled.
os.environ.pop('FNIT_GCA_SCORER',None)
os.environ.pop('FNIT_GCA_QUERY_CAPABILITIES',None)
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
report['precision']={'default_matmul_tf32':True,'default_cudnn_tf32':True,'autocast':False,'EntoWM_actual_forwards':[]}
stop=threading.Event()
def monitor():
 while not stop.is_set():
  tick=time.time()
  try:
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
   # Capture actual ancestry at the same sample, avoiding addition of unrelated peaks.
   processes=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
   parents={int(s.split()[0]):int(s.split()[1]) for s in processes.splitlines() if len(s.split())==2}
   family={os.getpid()}
   while True:
    added={child for child,parent in parents.items() if parent in family}-family
    if not added:break
    family.update(added)
   report['gpu_samples'].append({'time':tick,'process_tree_pids':sorted(family),'compute_apps':apps,'loadavg':os.getloadavg()})
  except Exception as error:report['gpu_samples'].append({'time':tick,'error':str(error)})
  stop.wait(.5)
watcher=threading.Thread(target=monitor);watcher.start()
def stage(name,function,*values,**parameters):
 torch.cuda.synchronize('cuda:0');tick=time.perf_counter();result=function(*values,**parameters);torch.cuda.synchronize('cuda:0')
 report['stages'].append({'name':name,'seconds_including_io':time.perf_counter()-tick,'torch_allocated_bytes':torch.cuda.memory_allocated(),'torch_reserved_bytes':torch.cuda.memory_reserved()});save();return result
try:
 gca=assets/'average/RB_all_2020-01-02.gca';lta=mri/'transforms/talairach.lta'
 stage('mri_em_register',_run_native_em_register,binaries/'mri_em_register',mri,gca,assets)
 stage('mri_ca_normalize',run_ca_normalize,mri/'nu.mgz',mri/'brainmask.mgz',gca,lta,mri/'norm.mgz',mri/'ctrl_pts.mgz')
 stage('mri_cc',_segment_callosum,mri)
 stage('brain_second_normalize',normalize_t1_aseg,mri/'norm.mgz',mri/'aseg.presurf.mgz',mri/'brainmask.mgz',mri/'brain.mgz',device='cuda:0')
 with torch.backends.cudnn.flags(allow_tf32=False):
  stage('entowm',mri_entowm_seg,mri/'nu.mgz',mri/'entowm.mgz',weights,device='cuda:0',stats_path=args.output/'stats/entowm.stats',talairach_lta=mri/'transforms/talairach.xfm.lta',precision_report=report['precision']['EntoWM_actual_forwards'])
 stage('ants_denoise',denoise_volume,mri/'brain.mgz',mri/'antsdn.brain.mgz')
 stage('mri_segment',_run_native_wm_segment,binaries/'mri_segment',mri,assets)
 stage('mri_edit_wm_with_aseg',_run_native_wm_edit,binaries/'mri_edit_wm_with_aseg',mri,assets)
 stage('wm_pretess',pretess_mgh,mri/'wm.asegedit.mgz','wm',mri/'norm.mgz',mri/'wm.mgz')
 # Independently recompute only the two affected edits on this chain's own
 # pretess output; diagnostics are never read back by production calculations.
 diagnostic=args.output/'current_cpu_wm_point_control.mgz'
 def cpu_point_control():
  fix_ento_wm_cpu(mri/'wm.mgz',mri/'entowm.mgz',diagnostic,level=3,left_value=255,right_value=255)
  fix_ento_wm_cpu(diagnostic,mri/'aseg.presurf.mgz',diagnostic,level=3,left_value=255,right_value=255,acj=True)
 stage('CPU_WM_point_validation_only',cpu_point_control)
 stage('wm_fix_ento',fix_ento_wm_gpu,mri/'wm.mgz',mri/'entowm.mgz',mri/'wm.mgz',level=3,left_value=255,right_value=255,device='cuda:0')
 stage('wm_fix_acj',fix_ento_wm_gpu,mri/'wm.mgz',mri/'aseg.presurf.mgz',mri/'wm.mgz',level=3,left_value=255,right_value=255,device='cuda:0',acj=True)
 report['comparison']['GPU_WM_vs_current_CPU_point_control']=compare_volume(mri/'wm.mgz',diagnostic)
 assert report['comparison']['GPU_WM_vs_current_CPU_point_control']['different_elements']==0,'current GPU WM point edits differ from current mature CPU on self-produced input'
 stage('mri_fill',fill_mgz,mri/'wm.mgz',mri/'aseg.presurf.mgz',lta,assets/'SubCorticalMassLUT.txt',mri/'filled.mgz',args.output/'scripts/ponscc.cut.log')
 shutil.copyfile(mri/'filled.mgz',mri/'filled.auto.mgz')
 for name in ('norm','ctrl_pts','aseg.presurf','brain','entowm','antsdn.brain','wm.seg','wm.asegedit','wm','filled'):
  report['comparison'][name]=compare_volume(mri/(name+'.mgz'),reference/(name+'.mgz'),labels=name in ('aseg.presurf','entowm','filled'))
 from fnit.recon_all.ca_normalize_python import read_voxel_lta
 before=read_voxel_lta(reference/'transforms/talairach.lta');after=read_voxel_lta(lta)
 report['comparison']['LTA_matrix']={'different_elements':int(np.count_nonzero(before!=after)),'max_error':float(np.max(np.abs(before-after))),'p99_error':float(np.quantile(np.abs(before-after),.99))}
 def volume_info(path):
  text=Path(path).read_text();text=text[text.index('src volume info'):];return '\n'.join(line for line in text.splitlines() if not line.startswith('filename'))
 report['comparison']['LTA_volume_info_equal']=volume_info(lta)==volume_info(reference/'transforms/talairach.lta')
 report['execution_complete']=True
except Exception as error:
 report['error']=str(error);raise
finally:
 stop.set();watcher.join();report['wall_seconds_including_imports_validation_outputs']=time.perf_counter()-START
 samples=report['gpu_samples'];gaps=[b['time']-a['time'] for a,b in zip(samples,samples[1:])];report['gpu_sample_interval_requested_seconds']=.5;report['gpu_max_sample_gap_seconds']=max(gaps) if gaps else None;report['gpu_failed_samples']=sum('error' in row for row in samples)
 own=[]
 for row in samples:
  if 'compute_apps' not in row:continue
  entries=[line.split(',') for line in row['compute_apps'].splitlines()];total=sum(int(v[2].strip())*1048576 for v in entries if len(v)==3 and int(v[0].strip()) in row['process_tree_pids'] and v[1].strip()==report['gpu_uuid']);own.append(total)
 report['gpu_process_tree_peak_sampled_bytes']=max(own) if own else None
 source=Path(__file__).resolve().parents[5]/'src/fnit/recon_all';report['source_sha256']={str(path.relative_to(source)):sha(path) for path in source.rglob('*.py')};report['script_sha256']=sha(__file__);save()
