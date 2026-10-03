#!/usr/bin/env python3
"""只读保存球面的有向面积；重建并校验临时注册基线，不重跑 MRI 或 MSM。"""
import argparse,hashlib,json,os,sys,time,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def signed(x,faces):
 t=np.asarray(x,dtype=np.float64)[faces]
 return (np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])*t[:,0]).sum(1)
def scalar(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:scalar(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [scalar(v) for v in x]
 return x

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cohort-root','source','posthoc-case','output-root'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--case-id',required=True)
 a=p.parse_args();a.source=a.source.resolve();a.cohort_root=a.cohort_root.resolve();a.posthoc_case=a.posthoc_case.resolve();out=a.output_root.resolve()
 subject=a.cohort_root/'candidate_v4'/a.case_id/'reconstruction/subject';cp=a.cohort_root/'candidate_v4'/a.case_id
 for protected in (a.source,a.cohort_root,a.posthoc_case):
  if out==protected or out.is_relative_to(protected) or protected.is_relative_to(out):raise ValueError('isolated output must not overlap protected input/source/cohort')
 if out.exists():raise FileExistsError('output exists')
 config_path=a.cohort_root/'configs_candidate_v4'/f'{a.case_id}.json';config=json.loads(config_path.read_text());assets=Path(config['hcp_assets_dir']).resolve();wb=Path(shutil.which(config['wb_command'])).resolve()
 binding_path=cp/'report/files.private.json';binding=json.loads(binding_path.read_text());meta_path=Path(binding['metadata']);qc_path=Path(binding['qc_report']);meta=json.loads(meta_path.read_text());qc=json.loads(qc_path.read_text());posthoc=json.loads((a.posthoc_case/'report.public.json').read_text())
 if posthoc['candidate_source_revision']!='1128bc52c7a0233266e5b8a8d7dc0b382994e676' or not posthoc['inputs_unchanged'] or not posthoc['code_unchanged']:raise ValueError('formal frozen posthoc contract failed')
 if json.loads((cp/'report/report.public.json').read_text())['status']!='complete':raise ValueError('candidate not complete')
 registered=cp/'derivatives'/meta['FNIT']['RegisteredSpheres']['L']['File'].removeprefix('bids::')
 if sha(registered)!=meta['FNIT']['RegisteredSpheres']['L']['SHA256']:raise ValueError('saved sphere SHA mismatch')
 inputs={'candidate_report':cp/'report/report.public.json','candidate_binding':binding_path,'candidate_metadata':meta_path,'candidate_qc':qc_path,'posthoc_report':a.posthoc_case/'report.public.json','config':config_path,'workbench':wb,'saved_L':registered}
 mesh=assets/'global/templates/standard_mesh_atlases'
 for hemi in ('L','R'):
  fshemi='lh' if hemi=='L' else 'rh'
  for name in ('sphere','sphere.reg','sulc'):inputs[f'{hemi}/{name}']=subject/'surf'/f'{fshemi}.{name}'
  for tag,name in [('average',f'fs_{fshemi}/fsaverage.{hemi}.sphere.164k_fs_{fshemi}.surf.gii'),('transform',f'fs_{fshemi}/fs_{fshemi}-to-fs_LR_fsaverage.{hemi}_LR.spherical_std.164k_fs_{fshemi}.surf.gii'),('reference_sphere',f'fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii'),('reference_sulc',f'{hemi}.refsulc.164k_fs_LR.shape.gii')]:inputs[f'{hemi}/{tag}']=mesh/name
 before={k:sha(v) for k,v in inputs.items()};source_files={str(f.relative_to(a.source)):f for f in (a.source/'src').rglob('*') if f.is_file() and f.suffix in ('.py','.so','.cpp','.h')};source_before={k:sha(v) for k,v in source_files.items()}
 sys.path.insert(0,str(a.source/'src'))
 global np
 import numpy as np,nibabel as nib,nibabel.freesurfer.io as fs,torch
 import fnit.msm.prepare as prep,fnit.fmri.surface_prepare as fprep,fnit.msm.msmsulc as msm
 from fnit.recon_all.mris_register_nonlinear import face_area_normals
 torch.set_num_threads(1)
 modules={m.__name__:Path(m.__file__).resolve() for m in (prep,fprep,msm)}
 for v in modules.values():
  if not v.is_relative_to(a.source/'src'):raise ValueError('wrong imported source')
 out.mkdir(parents=True);tick=time.perf_counter();commands=[];generated=out/'baseline';generated.mkdir();initial=[]
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 for hemi in ('L','R'):
  fshemi='lh' if hemi=='L' else 'rh';v,f=fs.read_geometry(str(subject/'surf'/f'{fshemi}.sphere.reg'));native=generated/f'{hemi}.sphere.FS.native.surf.gii';fprep._write_gifti(native,v,f,fshemi);dest=generated/f'{hemi}.sphere.FS_to_fsLR.native.surf.gii'
  args=[str(wb),'-surface-sphere-project-unproject',str(native),str(inputs[f'{hemi}/average']),str(inputs[f'{hemi}/transform']),str(dest)]
  subprocess.run(args,check=True,capture_output=True,text=True,env=env);commands.append(['wb_command',args[1],'/data/native_sphere','/assets/average','/assets/transform','/new/initial_sphere']);initial.append(dest)
 entries=prep.prepare_msmsulc_inputs(subject,tuple(initial),assets,generated/'msmsulc_inputs',wb_command=wb,parallel=False,cpu_threads=1)
 checks={};
 for hemi,entry in entries.items():
  checks[hemi]={name:{'actual_sha256':sha(Path(value)),'expected_sha256':qc['MSM']['InputsSHA256'][hemi][name],'equal':sha(Path(value))==qc['MSM']['InputsSHA256'][hemi][name]} for name,value in entry.__dict__.items()}
 if not all(x['equal'] for d in checks.values() for x in d.values()):raise ValueError('regenerated MSM inputs differ from actual production SHA; cannot recover baseline')
 native,faces=fs.read_geometry(str(subject/'surf/lh.sphere'));reg,regfaces=fs.read_geometry(str(subject/'surf/lh.sphere.reg'));img=nib.load(str(registered));saved=img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data;sf=img.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
 if not np.array_equal(faces,sf) or not np.array_equal(faces,regfaces):raise ValueError('face order changed')
 rawrot,rotfaces=msm._surface(entries['L'].rotated_sphere);rotated=msm._normalize_sphere(rawrot)
 if not np.array_equal(faces,rotfaces):raise ValueError('rotated face order changed')
 arrays={'native':native,'native_registered':reg,'saved_MSM':saved,'rotated_raw':rawrot,'production_baseline_normalized_rotated':rotated};signs={k:signed(x,faces) for k,x in arrays.items()};base=signs['production_baseline_normalized_rotated'];nb=signs['native'];s=signs['saved_MSM'];pr=s/base;nr=s/nb
 rows={}
 for k,x in arrays.items():
  tt=torch.as_tensor(np.asarray(x).astype(np.float32));fa=torch.as_tensor(faces.astype(np.int64));area,_=face_area_normals(tt,fa,signed_sphere=True);af=area.numpy();v=np.asarray(x);n=signs[k]
  rows[k]={'loaded_array_dtype':str(v.dtype),'source_coordinate_storage': 'float64 in memory normalized from float32 GIFTI' if k=='production_baseline_normalized_rotated' else 'float32 persisted coordinates; FreeSurfer reader promotes coordinates to float64' if k.startswith('native') else 'float32 GIFTI', 'negative_faces_float64_exact_saved_coordinates':int((n<0).sum()),'zero_faces_float64':int((n==0).sum()),'negative_faces_mature_float32_predicate':int((af<0).sum()),'minimum_mature_signed_area_mm2':float(af.min()),'minimum_float64_triple_product_mm3':float(n.min()),'negative_face_ids_float64':np.flatnonzero(n<0).tolist(),'negative_face_ids_mature_float32':np.flatnonzero(af<0).tolist()}
 selected=np.union1d(np.flatnonzero(pr<=0),np.flatnonzero(nr<=0))
 details=[]
 for i in selected:
  details.append({'face_id':int(i),'vertex_ids':faces[i].tolist(),'native_triple_product_mm3':float(nb[i]),'sphere_reg_triple_product_mm3':float(signs['native_registered'][i]),'saved_MSM_triple_product_mm3':float(s[i]),'rotated_raw_triple_product_mm3':float(signs['rotated_raw'][i]),'production_normalized_rotated_denominator_mm3':float(base[i]),'ratio_to_own_native':float(nr[i]),'ratio_to_production_normalized_rotated':float(pr[i]),'saved_GIFTI_coordinates':saved[faces[i]].tolist()})
 after={k:sha(v) for k,v in inputs.items()};source_after={k:sha(v) for k,v in source_files.items()}
 report={'case_id':a.case_id,'status':'measured','created_utc':datetime.now(timezone.utc).isoformat(),'helper_sha256':sha(Path(__file__)),'software_versions':{'nibabel':nib.__version__,'numpy':np.__version__,'torch':torch.__version__},'candidate_source_revision':posthoc['candidate_source_revision'],'input_sha256':before,'input_sha256_after':after,'inputs_unchanged':before==after,'source_inventory_sha256':hashlib.sha256(json.dumps(source_before,sort_keys=True).encode()).hexdigest(),'source_files_checked':len(source_before),'source_unchanged':source_before==source_after,'imported_modules_sha256':{k:sha(v) for k,v in modules.items()},'regenerated_input_checks':checks,'generated_baseline_scope':'Only surface-preparation Workbench operators, no MRI reconstruction or MSM solver rerun; actual temporary-input bytes all exactly match the original saved production hashes.','mature_orientation_scope':'mature signed area uses float32 arithmetic. The float64 triple product checks exactly the same already saved float32 coordinates, without geometry correction. Ratio denominators can be negative; ratio fold count is relative orientation, not absolute outward count.','sphere_summaries':rows,'saved_MSM_ratio_QC':{'own_native':msm._native_output_qc(saved,faces,native),'production_normalized_rotated':msm._native_output_qc(saved,faces,rotated)},'production_original_registration_QC':meta['FNIT']['RegistrationDetails']['Hemispheres']['L'],'unsaved_solver_coordinates_available':False,'selected_ratio_anomaly_faces':details,'threads':1,'cuda_initialized':torch.cuda.is_initialized(),'diagnostic_wall_seconds':time.perf_counter()-tick,'timing_scope':'new independent CPU saved-data and baseline-regeneration diagnostic; excluded from ten-case whole benchmark','commands':commands}
 if not report['inputs_unchanged'] or not report['source_unchanged']:raise ValueError('guard changed')
 (out/'files.private.json').write_text(json.dumps({k:str(v) for k,v in inputs.items()},indent=2)+'\n')
 (out/'report.public.json').write_text(json.dumps(scalar(report),indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'report_sha256':sha(out/'report.public.json'),'saved_QC':report['saved_MSM_ratio_QC'],'negative_counts':{k: {'f64':v['negative_faces_float64_exact_saved_coordinates'],'mature_f32':v['negative_faces_mature_float32_predicate']} for k,v in rows.items()},'diagnostic_wall_seconds':report['diagnostic_wall_seconds']}))
if __name__=='__main__':main()
