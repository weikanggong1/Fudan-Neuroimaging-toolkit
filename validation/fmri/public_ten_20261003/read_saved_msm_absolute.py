#!/usr/bin/env python3
"""从已完成后验绑定读取两链最终MSM绝对向外取向；不运行MRI/注册。"""
import argparse,json,hashlib,time,sys
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('posthoc-case','source','output-root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.posthoc_case=a.posthoc_case.resolve();a.source=a.source.resolve();o=a.output_root.resolve()
 for x in (a.posthoc_case,a.source,Path(__file__).resolve().parent):
  if o==x or o.is_relative_to(x) or x.is_relative_to(o):raise ValueError('isolated output overlaps source/input')
 if o.exists():raise FileExistsError('new output only')
 pp=a.posthoc_case/'report.public.json';bp=a.posthoc_case/'files.private.json';post=json.loads(pp.read_text());b=json.loads(bp.read_text())
 if not post['inputs_unchanged'] or not post['code_unchanged'] or post['candidate_source_revision']!='1128bc52c7a0233266e5b8a8d7dc0b382994e676':raise ValueError('formal posthoc guard failed')
 cp=Path(b['candidate_report']);rp=Path(b['reference_report']);cf=cp.parent/'files.private.json';cm=Path(json.loads(cf.read_text())['metadata']);meta=json.loads(cm.read_text());ref=json.loads(rp.read_text());roots=[q for q in cm.parents if (q/'dataset_description.json').is_file()];root=roots[0];paths={};inputs={'posthoc':pp,'posthoc_binding':bp,'candidate_report':cp,'reference_report':rp,'candidate_files':cf,'candidate_metadata':cm}
 for hemi in ('L','R'):
  paths['candidate',hemi]=root/meta['FNIT']['RegisteredSpheres'][hemi]['File'].removeprefix('bids::')
  rows=[x for x in ref['QC']['outputs'] if x['kind']=='new_MSMSulc_sphere' and f'hemi-{hemi}_' in x['relative_path']]
  if len(rows)!=1:raise ValueError('reference MSM binding ambiguous')
  paths['reference',hemi]=rp.parent/rows[0]['relative_path']
  for role,expected in [('candidate',meta['FNIT']['RegisteredSpheres'][hemi]['SHA256']),('reference',rows[0]['sha256'])]:
   if sha(paths[role,hemi])!=expected:raise ValueError('saved sphere bound SHA mismatch')
   inputs[f'{role}/{hemi}/saved_sphere']=paths[role,hemi];inputs[f'{role}/{hemi}/native_sphere']=Path(b[f'{role}_subject'])/'surf'/f'{"lh" if hemi=="L" else "rh"}.sphere'
 raw_path=Path(b['raw_t1w']).resolve();raw_roots=[q for q in raw_path.parents if (q/'dataset_description.json').is_file()]
 protected_roots=[cp.parent.parent.resolve(),rp.parent.resolve(),raw_roots[0] if raw_roots else raw_path.parent]
 for protected in protected_roots:
  if o==protected or o.is_relative_to(protected) or protected.is_relative_to(o):raise ValueError('output overlaps protected candidate/reference/raw root')
 before={k:sha(v) for k,v in inputs.items()};sys.path.insert(0,str(a.source/'src'));tick=time.perf_counter()
 import numpy as np,nibabel as nib,nibabel.freesurfer.io as fs,torch
 from fnit.recon_all.mris_register_nonlinear import face_area_normals
 import fnit.recon_all.mris_register_nonlinear as mod
 torch.set_num_threads(1);module=Path(mod.__file__).resolve()
 if not module.is_relative_to(a.source/'src'):raise ValueError('wrong mature source')
 module_before=sha(module);rows={}
 for (role,hemi),p in paths.items():
  im=nib.load(str(p));v=im.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data;f=im.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data;n,nf=fs.read_geometry(str(inputs[f'{role}/{hemi}/native_sphere']))
  if v.shape!=n.shape or not np.array_equal(f,nf) or not np.isfinite(v).all():raise ValueError('saved/native topology or finiteness failed')
  t=v.astype(np.float64)[f];s=(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])*t[:,0]).sum(1);area,_=face_area_normals(torch.as_tensor(v.astype(np.float32)),torch.as_tensor(f.astype(np.int64)),signed_sphere=True);area=area.numpy();rows.setdefault(role,{})[hemi]={'vertices':int(len(v)),'faces':int(len(f)),'saved_dtype':str(v.dtype),'saved_file_sha256':before[f'{role}/{hemi}/saved_sphere'],'all_coordinates_finite':bool(np.isfinite(v).all()),'native_face_order_exact':True,'absolute_negative_faces_float64':int((s<0).sum()),'zero_triple_products_float64':int((s==0).sum()),'nonfinite_triple_products_float64':int((~np.isfinite(s)).sum()),'negative_faces_mature_float32':int((area<0).sum()),'zero_area_faces_mature_float32':int((area==0).sum()),'minimum_mature_signed_area_mm2':float(area.min()),'negative_face_sets_f32_f64_equal':bool(np.array_equal(np.flatnonzero(s<0),np.flatnonzero(area<0))),'negative_face_ids_float64':np.flatnonzero(s<0).tolist()}
 after={k:sha(v) for k,v in inputs.items()};d={'status':'measured','case_id':post['case_id'],'created_utc':datetime.now(timezone.utc).isoformat(),'helper_sha256':sha(Path(__file__)),'mature_orientation_helper_sha256':module_before,'mature_helper_unchanged':module_before==sha(module),'input_sha256':before,'input_sha256_after':after,'inputs_unchanged':before==after,'chains':rows,'threads':1,'cuda_initialized':torch.cuda.is_initialized(),'diagnostic_wall_seconds':time.perf_counter()-tick,'scope':'Absolute outward orientation of each exact saved final MSM sphere, both chains and hemispheres. Independent of relative normalized-rotated/native-baseline ratio and unavailable original solver coordinates. No geometry change or registration rerun.'}
 if not d['inputs_unchanged'] or not d['mature_helper_unchanged']:raise ValueError('protected guard changed')
 o.mkdir(parents=True);(o/'files.private.json').write_text(json.dumps({k:str(v) for k,v in inputs.items()},indent=2)+'\n');(o/'report.public.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':d['status'],'sha256':sha(o/'report.public.json'),'absolute_negative':{c:{h:x['absolute_negative_faces_float64'] for h,x in z.items()} for c,z in rows.items()}}))
if __name__=='__main__':main()
