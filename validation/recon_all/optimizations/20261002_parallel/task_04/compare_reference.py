"""官方同输入对照；仅有面序对应时比较同索引，否则精确双向点到三角面。"""
import argparse,importlib.util,json,pathlib
import nibabel.freesurfer.io as fsio
import numpy as np
from fnit.recon_all.compare_subject import _topology


def main():
 p=argparse.ArgumentParser();p.add_argument('--pairs',type=pathlib.Path,required=True);p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--chains',type=pathlib.Path);p.add_argument('--distance-helper',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('point_triangle_reference_helpers',a.distance_helper);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 rows=[]
 for source in sorted(a.reference.glob('*/*/*/report.json')):
  subject,hemi,stage=source.parent.relative_to(a.reference).parts;c=a.pairs/subject/hemi/stage/'candidate';baseline=a.pairs/subject/hemi/stage/'baseline'
  candidate_kind='same-input staged candidate'
  if not (c/'report.json').exists() and a.chains is not None:
   c=a.chains/subject/hemi/'output';candidate_kind='actual continuous-chain candidate; formal paired timing separate'
  if not (c/'report.json').exists():continue
  ref=json.loads(source.read_text());cand=json.loads((c/'report.json').read_text())
  if ref['status']!='complete' or cand['status']!='complete':continue
  monitor_path=c/'monitor/monitor.json'
  inner_command_seconds=cand['command_wall_seconds']
  if monitor_path.exists():cand['command_wall_seconds']=json.loads(monitor_path.read_text())['command_wall_seconds']
  rv,rf=fsio.read_geometry(str(source.parent/(hemi+'.'+stage)));cv,cf=fsio.read_geometry(str(c/(hemi+'.'+stage)))
  same=rv.shape==cv.shape and np.array_equal(rf,cf)
  row={'comparison_candidate_kind':candidate_kind,'subject':subject,'hemisphere':hemi,'stage':stage,'official_program_sha256':ref['program_sha256'],'candidate_commit':cand['args']['commit'],'same_input_sha256':ref['input_sha256']==cand['input_sha256'],'ordered_faces_equal':bool(same),'official_command_seconds':ref['command_wall_seconds'],'candidate_api_seconds':cand['stage']['total_seconds_including_io'],'candidate_command_seconds':cand['command_wall_seconds'],'candidate_inner_report_seconds':inner_command_seconds,'reference_topology':_topology(rv,rf),'candidate_topology':_topology(cv,cf),'unit':'surface RAS mm','official_equivalence':'not_assessed','tolerance_policy':'raw max/P99 metrics; no post-hoc pass threshold; existing 138 diagnostics handled by whole acceptance'}
  if same:
   d=np.linalg.norm(rv-cv,axis=1);row['indexed_displacement_mm']=helper._summary(d);row['different_vertices']=int(np.count_nonzero(d));row['strict_official_coordinates']=bool(np.array_equal(rv,cv));row['strict_official_coordinate_bits']=rv.astype('>f4').tobytes()==cv.astype('>f4').tobytes()
  else:
   row['candidate_to_official_triangle_mm']=helper._summary(helper._point_to_mesh(cv,rv,rf));row['official_to_candidate_triangle_mm']=helper._summary(helper._point_to_mesh(rv,cv,cf));row['strict_official_coordinates']=False
  if (baseline/'report.json').exists():
   bv,bf=fsio.read_geometry(str(baseline/(hemi+'.'+stage)));row['candidate_equals_frozen_baseline']=bool(np.array_equal(bv,cv) and np.array_equal(bf,cf));row['new_official_degradation']='none in measured coordinates' if row['candidate_equals_frozen_baseline'] else 'needs baseline differential assessment'
  rows.append(row)
 a.output.write_text(json.dumps({'rows':rows,'whole_equivalence':'not_assessed','timing_boundary':'Official CLI wall includes command IO; FNIT API includes complete stage IO but excludes Python launcher imports; FNIT command wall additionally includes imports/hash checks. Official private context-copy preparation excluded; no whole-pipeline speed claim.'},indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k not in ('reference_topology','candidate_topology')} for r in rows],indent=2))

if __name__=='__main__':main()
