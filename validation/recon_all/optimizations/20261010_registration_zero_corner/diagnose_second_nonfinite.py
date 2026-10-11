"""只读回放自产球面配准，捕获第一个非有限内核输出，不修补生产网格。

输入subject_dir中同序sphere/smoothwm/sulc和已声明atlas，输出新诊断目录。
坐标surface RAS/mm；原版程序和参考结果不是输入。观察包装器增加检查成本，
本工具的耗时不是性能benchmark；失败保存输入哈希、模块版本及首差NPZ。
"""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
import fnit.recon_all.mris_register_nonlinear as nonlinear
import fnit.recon_all.mris_register_sulc_run as sulc_run

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(t):
 a=t.detach().cpu().numpy();finite=np.isfinite(a)
 r={'shape':list(a.shape),'dtype':str(a.dtype),'nonfinite_elements':int((~finite).sum())}
 if a.ndim==2 and a.shape[1]==3:
  norms=np.linalg.norm(a.astype(np.float64),axis=1);good=norms[np.isfinite(norms)]
  r.update(nonfinite_vertices=int((~finite.all(axis=1)).sum()),zero_vectors=int((norms==0).sum()),min_norm=float(good.min()) if len(good) else None,max_norm=float(good.max()) if len(good) else None)
 return r

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('subject-dir','atlas','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--hemi',choices=('lh','rh'),default='lh');p.add_argument('--averaging-device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(a.threads)
 report={'scope':'first-nonfinite replay; diagnostic overhead; not pipeline continuation or performance timing','subject_dir':str(a.subject_dir),'hemi':a.hemi,'atlas':str(a.atlas),'averaging_device':a.averaging_device,'threads':a.threads,'torch_version':torch.__version__,'modules':{m.__name__:sha(m.__file__) for m in (nonlinear,sulc_run)},'diagnostic_sha256':sha(__file__),'input_sha256':{n:sha(a.subject_dir/'surf'/f'{a.hemi}.{n}') for n in ('sphere','smoothwm','sulc')},'atlas_sha256':sha(a.atlas),'observations':[]}
 def wrapped(name,original):
  def invoke(*args,**kwargs):
   result=original(*args,**kwargs);outputs=result if isinstance(result,tuple) else (result,)
   row={'function':name,'call':sum(x['function']==name for x in report['observations'])+1,'inputs':[summary(x) for x in args if isinstance(x,torch.Tensor) and x.dtype.is_floating_point],'outputs':[summary(x) for x in outputs if isinstance(x,torch.Tensor)]}
   report['observations'].append(row)
   if any(x['nonfinite_elements'] for x in row['outputs']):
    arrays={f'arg{i}':x.detach().cpu().numpy() for i,x in enumerate(args) if isinstance(x,torch.Tensor)}
    arrays.update({f'output{i}':x.detach().cpu().numpy() for i,x in enumerate(outputs) if isinstance(x,torch.Tensor)})
    np.savez(a.output/'first_nonfinite.npz',**arrays)
    report['first_bad_function']=name;report['checkpoint_sha256']=sha(a.output/'first_nonfinite.npz')
    if name in ('sphere_vertex_normals','face_area_normals'):
     xyz=args[0].float();faces=args[1];f=xyz[faces];e0=f-f[:,[2,0,1]];e1=f[:,[1,2,0]]-f
     cross=torch.cross(e0,e1,dim=2);cross64=torch.cross(e0.double(),e1.double(),dim=2)
     report['normal_degeneracy']={'zero_edges_float32':int((torch.sum(e0*e0,dim=2)==0).sum()),'zero_corner_cross_float32':int((torch.sum(cross*cross,dim=2)==0).sum()),'zero_corner_cross_double':int((torch.sum(cross64*cross64,dim=2)==0).sum())}
    raise FloatingPointError('First nonfinite output: '+name)
   return result
  return invoke
 for name in ('sphere_vertex_normals','sphere_arc_distances','face_area_normals','first_distance_gradient','first_area_gradient','tangent_basis','apply_spherical_gradient','correlation_gradient_add','sample_correlation_atlas'):
  fn=wrapped(name,getattr(nonlinear,name));setattr(nonlinear,name,fn)
  if hasattr(sulc_run,name):setattr(sulc_run,name,fn)
 started=time.perf_counter()
 try:
  report['stage']=sulc_run.run_register_sulc(a.subject_dir/'surf'/f'{a.hemi}.sphere',a.subject_dir/'surf'/f'{a.hemi}.smoothwm',a.subject_dir/'surf'/f'{a.hemi}.sulc',a.atlas,a.output/'sulc-seed',averaging_device=a.averaging_device)
  report['status']='no_nonfinite_in_complete_sulc_stage'
 except Exception as e:report.update(status='diagnostic_captured_failure',error=repr(e),traceback=traceback.format_exc())
 report['diagnostic_seconds_including_observation']=time.perf_counter()-started
 (a.output/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'first_bad_function':report.get('first_bad_function'),'normal_degeneracy':report.get('normal_degeneracy'),'seconds':report['diagnostic_seconds_including_observation']}))
if __name__=='__main__':main()
