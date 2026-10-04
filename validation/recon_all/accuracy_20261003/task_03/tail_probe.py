"""官方冻结相同输入 brain normalize/denoise/fill 算子回放；benchmark 专用。"""
import argparse,fcntl,json,os,platform,subprocess,threading,time
from pathlib import Path
import torch
from volume_probe import sha,compare
from fnit.recon_all.normalization import normalize_t1_aseg
from fnit.recon_all.ants_denoise_python import denoise_volume
from fnit.recon_all.fill_cutting_plane_python import fill_mgz
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());out=Path(c['output'])/'tail';out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
report={'commit':c['commit'],'host':platform.node(),'pid':os.getpid(),'scope':'official_frozen_same_input_component_only','overall_equivalence':'not_assessed','tf32':{'matmul':True,'cudnn':True},'device':'cuda:0','cases':[],'gpu_samples':[]}
stop=threading.Event()
def monitor():
 while not stop.is_set():
  try:
   report['gpu_samples'].append({'time':time.time(),'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True),'gpu':subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)})
  except Exception as e:report['gpu_samples'].append({'time':time.time(),'error':repr(e)})
  stop.wait(.5)
def save():(out/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
for case in c['cases']:
 mri=Path(case['official'])/'mri';folder=out/case['id'];folder.mkdir();r={'id':case['id'],'input_sha256':{n:sha(mri/n) for n in ('norm.mgz','brainmask.mgz','aseg.presurf.mgz','brain.mgz','wm.mgz','transforms/talairach.lta')},'stages':[]};report['cases'].append(r);save()
 for name,function,inputs,reference in [('brain',normalize_t1_aseg,[mri/'norm.mgz',mri/'aseg.presurf.mgz',mri/'brainmask.mgz'],mri/'brain.mgz'),('antsdn.brain',denoise_volume,[mri/'brain.mgz'],mri/'antsdn.brain.mgz'),('filled',fill_mgz,[mri/'wm.mgz',mri/'aseg.presurf.mgz',mri/'transforms/talairach.lta',Path(c['assets'])/'SubCorticalMassLUT.txt'],mri/'filled.mgz')]:
  lock=open('/tmp/fnit-shared-benchmark.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);stop.clear();watch=threading.Thread(target=monitor);watch.start();torch.cuda.synchronize('cuda:0');tick=time.perf_counter()
  try:
   target=folder/(name+'.mgz');details=function(*inputs,target,**({'device':'cuda:0'} if name=='brain' else {}));torch.cuda.synchronize('cuda:0');r['stages'].append({'name':name,'seconds_including_io':time.perf_counter()-tick,'details':details,'comparison':compare(target,reference,name=='filled'),'allocated_bytes':torch.cuda.max_memory_allocated('cuda:0'),'reserved_bytes':torch.cuda.max_memory_reserved('cuda:0')});save()
  finally:stop.set();watch.join();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
report['complete']=True;save()
