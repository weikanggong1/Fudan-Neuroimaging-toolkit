#!/usr/bin/env python3
"""同一公开 T1 已完成三条重建的原生 aseg24计数/PV；只读，不运行MRI。"""
import argparse,json,hashlib,time,os
from pathlib import Path
from datetime import datetime,timezone

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('cohort-root','fs82-cold-binding','fs82-report','output-root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.cohort_root.resolve();o=a.output_root.resolve();cold=json.loads(a.fs82_cold_binding.read_text());r82=json.loads(a.fs82_report.read_text());cand=b/'candidate_v4/CON01/report';ref=b/'reference_fmriprep25_2_4_v1/cases/sub-CON01/attempt-02';rc=json.loads((cand/'report.public.json').read_text());rr=json.loads((ref/'report.corrected.public.json').read_text());raw=Path(cold['source_t1w']).resolve();rawsha=sha(raw)
 if any(x!=rawsha for x in [cold['metadata']['request']['source_sha256'],r82['raw_t1w_sha256_before'],r82['raw_t1w_sha256_after'],rc['input_sha256']['t1w'],rr['input_sha256']['t1w']]):raise ValueError('same raw T1 identity mismatch')
 if any(x['status']!='complete' for x in [cold['metadata'],rc,rr]) or cold['reused'] or not all(r82[k] for k in ['raw_input_guard_equal','frozen_source_guard_equal','validation_helper_guard_equal','runner_guard_equal']):raise ValueError('complete source/input guards failed')
 cf=json.loads((cand/'files.private.json').read_text());rf=json.loads((ref/'files.corrected.private.json').read_text());subjects={'FNIT_frozen1128':Path(cf['recon_all']),'fMRIPrep_FS7_3_2':Path(rf['recon_all']),'standalone_FS8_2':Path(cold['subject_dir'])}
 for x in [b,a.fs82_cold_binding.parent.resolve(),raw.parent.resolve(),*subjects.values()]:
  x=x.resolve()
  if o==x or o.is_relative_to(x) or x.is_relative_to(o):raise ValueError('output overlaps protected input')
 if o.exists():raise FileExistsError('new isolated output required')
 inp={'raw_t1':raw,'candidate_report':cand/'report.public.json','candidate_files':cand/'files.private.json','reference_report':ref/'report.corrected.public.json','reference_files':ref/'files.corrected.private.json','fs82_cold_binding':a.fs82_cold_binding,'fs82_report':a.fs82_report}
 for k,s in subjects.items():
  for name in ['mri/aseg.mgz','stats/aseg.stats']:inp[f'{k}/{name}']=s/name
 before={k:sha(v) for k,v in inp.items()};tick=time.perf_counter()
 import numpy as np,nibabel as nib
 rows={}
 for key,subject in subjects.items():
  im=nib.load(str(subject/'mri/aseg.mgz'));array=np.asarray(im.dataobj);matches=[]
  for line in (subject/'stats/aseg.stats').read_text().splitlines():
   fields=line.split()
   if line.startswith('#') or len(fields)<5:continue
   if fields[1]=='24':matches.append({'NVoxels':int(fields[2]),'Volume_mm3':float(fields[3]),'StructName':fields[4]})
  if len(matches)!=1:raise ValueError('aseg24 stats unique row absent')
  stats=matches[0];count=int((array==24).sum());rows[key]={'label_id':24,'label_name_in_original_stats':stats['StructName'],'native_hard_voxel_count':count,'native_shape':list(im.shape),'native_voxel_volume_mm3':float(abs(np.linalg.det(im.affine[:3,:3]))),'stats_NVoxels':stats['NVoxels'],'stats_partial_volume_mm3':stats['Volume_mm3'],'stats_Nvoxels_equal_native_count':stats['NVoxels']==count,'source_aseg_sha256':before[f'{key}/mri/aseg.mgz'],'source_stats_sha256':before[f'{key}/stats/aseg.stats']}
 after={k:sha(v) for k,v in inp.items()};report={'status':'measured','case_id':'CON01','created_utc':datetime.now(timezone.utc).isoformat(),'helper_sha256':sha(Path(__file__)),'same_raw_T1_sha256':rawsha,'input_sha256':before,'input_sha256_after':after,'inputs_unchanged':before==after,'rows':rows,'reference_versions':{'fMRIPrep':'25.2.4','fMRIPrep_FreeSurfer':'7.3.2','standalone_FreeSurfer':'8.2.0-1'},'formal_candidate_source_revision':rc['source_revision'],'FS82_standalone_request_device':r82['requested_device'],'FS82_standalone_actual_device':r82['actual_reconstruction_device'],'diagnostic_wall_seconds':time.perf_counter()-tick,'timing_scope':'Independent CPU1 read of three already saved native labels and their original statistics; no MRI/PV writer rerun and no ten-case workflow timing aggregation','measurement_scope':'Same raw T1 identity; native aseg24 hard count and original stats PV are distinct definitions. Separate standalone FS8.2 backend adapter result, not a ten-case fMRI whole reference or cross-version equivalence test. No label relabelling or world-grid resampling is performed.','conclusion':'Version alone does not establish segmentation-domain equivalence or a bug. These three native counts retain each method original label24 coverage.','software_versions':{'nibabel':nib.__version__,'numpy':np.__version__},'threads':1,'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES')}
 if not report['inputs_unchanged']:raise ValueError('protected input changed')
 o.mkdir(parents=True);(o/'files.private.json').write_text(json.dumps({k:str(v) for k,v in inp.items()},indent=2)+'\n');(o/'report.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'rows':rows,'report_sha256':sha(o/'report.public.json'),'seconds':report['diagnostic_wall_seconds']}))
if __name__=='__main__':main()
