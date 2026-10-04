"""冻结同输入体积诊断；官方输入仅用于benchmark，不用于生产。"""
import argparse, hashlib, json, os, platform, shutil, time, fcntl
from pathlib import Path
import nibabel as nib
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
 return h.hexdigest()

def compare(a,b,labels=False):
 x,y=nib.load(str(a)),nib.load(str(b));v,w=np.asarray(x.dataobj),np.asarray(y.dataobj)
 r={'paths':[str(a),str(b)],'sha256':[sha(a),sha(b)],'shape':[list(x.shape),list(y.shape)],'dtype':[str(x.get_data_dtype()),str(y.get_data_dtype())],'affine_max_abs':float(np.max(np.abs(x.affine-y.affine))),'same_grid':x.shape==y.shape and np.array_equal(x.affine,y.affine)}
 if not r['same_grid']:return r
 d=np.abs(v.astype(np.float64)-w.astype(np.float64)); nz=d[d!=0]
 r.update(different_voxels=int(nz.size),max_abs=float(d.max()),p99_abs=float(np.quantile(d,.99)),p99_changed_abs=float(np.quantile(nz,.99)) if nz.size else 0)
 if labels:
  r['dice']={str(int(k)):float(2*np.count_nonzero((v==k)&(w==k))/(np.count_nonzero(v==k)+np.count_nonzero(w==k))) for k in np.union1d(np.unique(v),np.unique(w))}
 return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--mode',choices=['audit','normalize','cross'],required=True);a=p.parse_args();c=json.loads(a.config.read_text());out=Path(c['output'])/a.mode;out.mkdir(parents=True,exist_ok=False)
 import fnit.recon_all.ca_normalize_python as ca
 from fnit.recon_all.mri_em_register_cached_conda import run_cached_em_register
 report={'mode':a.mode,'hostname':platform.node(),'commit':c['commit'],'module':str(Path(ca.__file__).resolve()),'module_sha256':sha(ca.__file__),'scope':'frozen_real_input_diagnostic','equivalence':'not_assessed','cases':[]}
 def save():(out/'report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
 for case in c['cases']:
  official=Path(case['official'])/'mri';fnit=Path(case['fnit'])/'mri';item={'id':case['id'],'runs':[]};report['cases'].append(item);save()
  if a.mode=='audit':
   for n in ('orig','nu','T1','brainmask','norm','ctrl_pts','aseg.presurf','brain','antsdn.brain','wm.seg','wm.asegedit','wm','filled'):
    try:item['runs'].append({'stage':n,'comparison':compare(fnit/(n+'.mgz'),official/(n+'.mgz'),n in ('aseg.presurf','filled'))})
    except Exception as e:item['runs'].append({'stage':n,'error':repr(e)})
   lm=ca.read_voxel_lta(fnit/'transforms/talairach.lta')-ca.read_voxel_lta(official/'transforms/talairach.lta');item['lta_max_abs']=float(np.abs(lm).max());save();continue
  combinations=[('official','official'),('fnit','official'),('official','fnit'),('fnit','fnit')] if a.mode=='cross' else [('official','official')]
  for nu_name,mask_name in combinations:
   run=out/case['id']/(nu_name+'nu_'+mask_name+'mask');run.mkdir(parents=True);mri=run/'mri';mri.mkdir();(mri/'transforms').mkdir();sources={'official':official,'fnit':fnit}
   for name,owner in [('nu.mgz',nu_name),('brainmask.mgz',mask_name)]:shutil.copyfile(sources[owner]/name,mri/name)
   rr={'combination':[nu_name,mask_name],'input_sha256':{n:sha(mri/n) for n in ('nu.mgz','brainmask.mgz')}};item['runs'].append(rr);save()
   lock=open('/tmp/fnit-shared-benchmark.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
   if a.mode=='cross':
    t=time.perf_counter();run_cached_em_register(binary=c['binary'],mri=mri,atlas=c['atlas'],assets=c['assets'],binary_sha256=c['binary_sha256'],threads=4);rr['em_seconds_including_io']=time.perf_counter()-t
   else:shutil.copyfile(official/'transforms/talairach.lta',mri/'transforms/talairach.lta')
   t=time.perf_counter();rr['normalization']=ca.run_ca_normalize(mri/'nu.mgz',mri/'brainmask.mgz',c['atlas'],mri/'transforms/talairach.lta',mri/'norm.mgz',mri/'ctrl_pts.mgz');rr['ca_seconds_including_io']=time.perf_counter()-t
   rr['lta_max_abs_vs_official']=float(np.abs(ca.read_voxel_lta(mri/'transforms/talairach.lta')-ca.read_voxel_lta(official/'transforms/talairach.lta')).max());rr['norm_vs_official']=compare(mri/'norm.mgz',official/'norm.mgz');rr['controls_vs_official']=compare(mri/'ctrl_pts.mgz',official/'ctrl_pts.mgz');save();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 report['complete']=True;save()
if __name__=='__main__':main()
