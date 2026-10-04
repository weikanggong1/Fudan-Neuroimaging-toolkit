"""收集已固定10例前段产物；未完成或失败逐例保留。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929')
ROOT=BASE/'accuracy_20261003/task_02/cohort_prefix'
COHORT=BASE/'accuracy_20261003/cohort/cohort_verified.json'
OUTPUT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
original=json.loads((ROOT/'report.json').read_text());fixed=json.loads(COHORT.read_text())
completed={c['id']:c for c in original['cases']}
summary={'kind':'complete_prefix_stage_only_not_recon_all','baseline_commit':'816e5610417a4c587caf321049438a9554139016','candidate_commit':'c24b3d6d','cohort_manifest_sha256':hashlib.sha256(COHORT.read_bytes()).hexdigest(),'expected_cases':len(fixed['cases']),'completed_pairs':len(completed),'overall_equivalence':'not_assessed','formal_speed_tolerance':None,'cases':[]}
rows=[];peaks=[];gaps=[];failed=0
for case in fixed['cases']:
 result=completed.get(case['id']);item={'id':case['id'],'input_sha256':case['sha256'],'status':'pending' if result is None else 'failed'}
 if result is not None:
  item['comparison']=result
  success=all(r.get('exit_code')==0 for r in result['runs'])
  item['status']='execution_complete' if success else 'failed'
  item['runs']={}
  for backend in ('baseline','candidate'):
   runpath=ROOT/case['id']/backend/'scripts/prefix_run.json';monpath=ROOT/case['id']/(backend+'_monitor')/'monitor.json'
   if not runpath.exists() or not monpath.exists():continue
   run=json.loads(runpath.read_text());monitor=json.loads(monpath.read_text())
   item['runs'][backend]={'run':run,'monitor':monitor,'run_path':str(runpath),'monitor_path':str(monpath)}
   if monitor['peak_sampled_process_bytes'] is not None:peaks.append(monitor['peak_sampled_process_bytes'])
   if monitor['maximum_sampling_gap_seconds'] is not None:gaps.append(monitor['maximum_sampling_gap_seconds'])
   failed+=monitor['failed_app_queries']
  if success:
   volumes=result['volumes'];item['numeric_preservation']=all(v['different']==0 and v['geometry_max_abs']==0 for v in volumes.values()) and result['xfm_bytes_equal'] and all(v==0 for v in result['lta_matrix_max_abs'].values())
   a=item['runs']['baseline']['run'];b=item['runs']['candidate']['run']
   rows.append({'case':case['id'],'baseline_seconds':a['wall_seconds'],'candidate_seconds':b['wall_seconds'],'candidate_minus_baseline_seconds':b['wall_seconds']-a['wall_seconds'],'baseline_synthstrip_seconds':a['synthstrip_seconds'],'candidate_synthstrip_seconds':b['synthstrip_seconds'],'baseline_talairach_seconds':a['talairach_seconds'],'candidate_talairach_seconds':b['talairach_seconds'],'synthstrip_different_voxels':volumes['synthstrip.mgz']['different'],'baseline_dtype':volumes['synthstrip.mgz']['dtype_a'],'candidate_dtype':volumes['synthstrip.mgz']['dtype_b'],'xfm_bytes_equal':result['xfm_bytes_equal']})
 summary['cases'].append(item)
summary['all_10_completed']=len(completed)==10 and all(c['status']=='execution_complete' for c in summary['cases'])
summary['all_completed_numeric_preserved']=all(c.get('numeric_preservation',False) for c in summary['cases'] if c['status']=='execution_complete')
summary['memory']={'peak_sampled_parent_children_bytes':max(peaks,default=None),'budget_bytes':20000000000,'maximum_sampling_gap_seconds':max(gaps,default=None),'failed_app_queries':failed,'continuous_peak_verified':False}
summary['performance']={'paired_case_count':len(rows),'median_baseline_seconds':float(np.median([r['baseline_seconds'] for r in rows])) if rows else None,'median_candidate_seconds':float(np.median([r['candidate_seconds'] for r in rows])) if rows else None,'median_paired_delta_seconds':float(np.median([r['candidate_minus_baseline_seconds'] for r in rows])) if rows else None,'repeatability_assessment':'see prefix_repeat_collected.json; no formal tolerance assigned'}
(OUTPUT/'cohort_prefix_collected.json').write_text(json.dumps(summary,indent=2))
with (OUTPUT/'cohort_prefix_times.csv').open('w',newline='') as f:
 if rows:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
