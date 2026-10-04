"""同输入诊断FP32坐标累计；猴补丁仅存在本进程，生产源码不改。"""
import argparse,fcntl,json,time
from pathlib import Path
import numpy as np
from volume_probe import sha,compare
import fnit.recon_all.ca_normalize_python as ca
from fnit.recon_all.mri_em_register import _vnl_affine_inverse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());root=Path(cfg['output'])/'coordinates_fp32';root.mkdir(parents=True,exist_ok=False)
original=ca.atlas_samples

def atlas_samples_fp32(gca,voxel_lta):
 samples=original(gca,voxel_lta)
 mapping=_vnl_affine_inverse(voxel_lta)
 mapping[:,:3]=np.float32(mapping[:,:3]*np.float32(gca.prior_spacing))
 coordinates=samples.prior_coordinates.astype(np.float32)
 xyz=np.zeros((len(coordinates),3),np.float32)
 for axis in range(3):
  for component in range(3):xyz[:,axis]=np.float32(xyz[:,axis]+np.float32(mapping[axis,component]*coordinates[:,component]))
  xyz[:,axis]=np.float32(xyz[:,axis]+mapping[axis,3])
 # Native float coordinates are promoted before nint; do not round the +0.5 in FP32.
 v=xyz.astype(np.float64);source=np.where(v>=0,np.floor(v+.5),np.ceil(v-.5)).astype(np.int32)
 return ca.AtlasSamples(samples.prior_coordinates,source,samples.labels,samples.priors)
ca.atlas_samples=atlas_samples_fp32
report={'commit':cfg['commit'],'scope':'same_input_coordinate_operator_diagnostic_only','equivalence':'not_assessed','module_sha256':sha(ca.__file__),'script_sha256':sha(__file__),'cases':[]}
def save():(root/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
for case in cfg['cases']:
 mri=Path(case['official'])/'mri';out=root/case['id'];out.mkdir();r={'id':case['id'],'input_sha256':{n:sha(mri/n) for n in ('nu.mgz','brainmask.mgz','transforms/talairach.lta')}};report['cases'].append(r);save()
 lock=open('/tmp/fnit-shared-benchmark.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  tick=time.perf_counter();r['normalization']=ca.run_ca_normalize(mri/'nu.mgz',mri/'brainmask.mgz',cfg['atlas'],mri/'transforms/talairach.lta',out/'norm.mgz',out/'ctrl_pts.mgz');r['seconds']=time.perf_counter()-tick;r['norm']=compare(out/'norm.mgz',mri/'norm.mgz');r['controls']=compare(out/'ctrl_pts.mgz',mri/'ctrl_pts.mgz');save()
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
report['complete']=True;save()
