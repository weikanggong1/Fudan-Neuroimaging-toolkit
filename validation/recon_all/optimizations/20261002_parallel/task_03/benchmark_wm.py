"""真实自产 WM 和 aseg/ento：两个成熟点编辑的完整读写 CPU/GPU 比较。"""
import argparse,hashlib,json,os,platform,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.wm_edits_python import fix_ento_wm
from fnit.recon_all.wm_edits_gpu import fix_ento_wm_gpu
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'commit':a.commit,'scope':'frozen_same_input_point_edits_not_full_WM_core_or_continuous_chain','host':platform.node(),'tolerance_declared':0,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'rows':[],'overall_equivalence':'not_assessed'}
torch.set_num_threads(4)
for case in ('whole_sub01_candidate_retry1','whole_sub02_candidate_retry2'):
 mri=a.root/'serial_20261001'/case/'mri'
 for name,acj,label in [('ento',False,'entowm.mgz'),('acj',True,'aseg.presurf.mgz')]:
  row={'case':case,'stage':name,'input_sha256':sha(mri/'wm.mgz'),'label_sha256':sha(mri/label),'timings':[]}
  for mode in ('cpu','gpu','gpu','cpu'):
   out=a.output/(case+'_'+name+'_'+mode+'.mgz')
   tick=time.perf_counter()
   if mode=='cpu':count=fix_ento_wm(mri/'wm.mgz',mri/label,out,level=3,left_value=255,right_value=255,acj=acj)
   else:count=fix_ento_wm_gpu(mri/'wm.mgz',mri/label,out,level=3,left_value=255,right_value=255,acj=acj,device='cuda:0');torch.cuda.synchronize()
   row['timings'].append({'backend':mode,'seconds_including_io':time.perf_counter()-tick,'edited_voxels':count})
  cpu=nib.load(str(a.output/(case+'_'+name+'_cpu.mgz')));gpu=nib.load(str(a.output/(case+'_'+name+'_gpu.mgz')))
  c=np.asarray(cpu.dataobj);g=np.asarray(gpu.dataobj)
  row.update(different_voxels=int(np.count_nonzero(c!=g)),geometry_equal=bool(np.array_equal(cpu.affine,gpu.affine)),dtype_equal=cpu.get_data_dtype()==gpu.get_data_dtype(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),per_value_dice={str(v):float(2*np.count_nonzero((c==v)&(g==v))/(np.count_nonzero(c==v)+np.count_nonzero(g==v))) for v in np.union1d(np.unique(c),np.unique(g))})
  row['process_gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
  report['rows'].append(row)
source_root=Path(__file__).resolve().parents[5]/'src'
report['source_sha256']={str(p.relative_to(source_root)):sha(p) for p in (source_root/'fnit/recon_all').glob('wm_edits*.py')};report['script_sha256']=sha(__file__)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
# Targeted unit case exercises overlapping bilateral seeds and edge exclusion.
from fnit.recon_all.wm_edits_python import amygdala_cortex_junction
from fnit.recon_all.wm_edits_gpu import amygdala_cortex_junction_gpu
labels=np.full((7,7,7),3,np.int32);labels[2,2,2]=18;labels[4,4,4]=54;labels[0,3,3]=18
expected=amygdala_cortex_junction(labels)
actual=amygdala_cortex_junction_gpu(torch.tensor(labels,device='cuda:0')).cpu().numpy()
assert np.array_equal(expected,actual),'ACJ overlap/edge regression'
report['targeted_overlap_edge_test']='passed'
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
