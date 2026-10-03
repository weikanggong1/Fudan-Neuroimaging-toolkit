"""Read-only independent final actual ten-case/raw/route audit; never modifies original contracts/index."""
import json,hashlib,time,os
from pathlib import Path
from datetime import datetime,timezone
import nibabel as nib
import numpy as np
R=Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002/task_01')
O=R/'CPU_final10_independent_audit_v1';O.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()
manifest=R/'final_manifest.json';m=load(manifest);expected=['sub-CON01','sub-CON03']+[f'sub-CON{x:02d}' for x in range(4,12)];cases=m['cases']
assert sha(manifest)=='d707f7a990372e50fb27de0023b2cddfc4eb74c7e5cdd9d909e41b90c2fa1b88'
assert len(cases)==10 and len({c['case_id'] for c in cases})==10 and {c['case_id'] for c in cases}==set(expected)
rows={};raw_hashes=[]
for c in cases:
 row={'raw_AP':c['dwi'],'raw_PA':c['reverse_dwi']}
 for name,path in [('AP',c['dwi']),('PA',c['reverse_dwi'])]:
  relative=str(Path(path).relative_to(Path(c['bids_root'])));digest=sha(path);assert digest==c['input_sha256'][relative]
  im=nib.load(path);a=np.asanyarray(im.dataobj);finite=np.isfinite(a)
  row[name]={'SHA256':digest,'shape':list(im.shape),'nonfinite_voxels':int(np.count_nonzero(~finite)),'negative_finite_voxels':int(np.count_nonzero(a[finite]<0)),'finite_min':float(a[finite].min()),'finite_max':float(a[finite].max()),'raw_values_changed':False}
  if name=='AP':raw_hashes.append(digest)
 rows[c['case_id']]=row
assert len(set(raw_hashes))==10
save(O/'raw_index_audit.json',{'observed_UTC':utc(),'manifest_SHA256':sha(manifest),'ten_distinct_expected_case_ids':True,'ten_distinct_actual_raw_AP_SHA256':True,'cases':rows,'raw_negative_and_nonfinite_retained_without_clipping':True,'tool_SHA256':sha(__file__)})
D=R/'CPU_reference_explicit_ten_delivery_v1'
while True:
 status=load(D/'status.json')
 if status['state']=='actual_ten_verified_and_compared_CPU_delivery_collected':break
 save(O/'status.json',{'state':'raw_index_verified_waiting_actual_final_delivery','observed_UTC':utc(),'collector_state':status['state']});time.sleep(30)
s=load(D/'summary.json');route=load(D/'explicit_actual_CPU_case_routes_v1.json');assert len(s['cases'])==10 and set('sub-'+x for x in s['cases'])==set(expected) and len(route['cases'])==10 and set(route['cases'])==set(expected) and route['all_ten_actual_verified_ready']
assert s['all_ten_CPU_references_completed'] and s['all_ten_actual_comparisons_completed']
checks={}
for subject,row in s['cases'].items():
 cid='sub-'+subject;rr=route['cases'][cid];d=Path(rr['official_case_directory']);r=load(d/'report.json');v=load(d/'completed_contract_verified.json')
 assert sha(d/'report.json')==row['report_SHA256']==rr['report_SHA256']==v['report_sha256']
 assert sha(d/'completed_contract_verified.json')==row['verified_contract_SHA256']==rr['verified_contract_SHA256']
 for name,digest in r['output_sha256'].items():assert sha(d/name)==digest
 for name,digest in row['evidence_files_SHA256'].items():assert sha(Path(row['evidence_directory'])/name)==digest
 ap=Path(rr['data']);im=nib.load(ap);a=np.asanyarray(im.dataobj);bv=np.loadtxt(rr['rotated_bvecs']);bl=np.loadtxt(rr['bvals']);mask=nib.load(rr['mask'])
 assert im.shape==tuple(rows[cid]['AP']['shape']) and bv.shape==(3,102) and bl.size==102 and np.isfinite(a).all() and np.isfinite(bv).all() and np.isfinite(bl).all() and np.allclose(im.affine,mask.affine,rtol=0,atol=1e-5)
 if subject=='CON11':
  assert not r.get('source_CPU_stage_lineage') and row['execution_kind']=='fresh_official_CPU_rawprep' and 'fresh_contiguous_CPU_rawprep_wall_seconds' in row and 'CPU_reference_activation_wall_seconds' not in row
  assert '/official_CON11_CPU_fresh_origin_v1/sub-CON11' in str(d) and '/formal_selected_monitor_recovery_v3/baseline/sub-CON11' in rr['actual_FNIT_case_directory']
 else:assert r['source_CPU_stage_lineage'] and row['execution_kind']=='restored_own_CPU_stages_plus_new_CPU_EDDY' and 'fresh_contiguous_CPU_rawprep_wall_seconds' not in row
 checks[cid]={'report_SHA256':sha(d/'report.json'),'verified_contract_SHA256':sha(d/'completed_contract_verified.json'),'actual_raw_AP_SHA256':rows[cid]['AP']['SHA256'],'official_case_directory':str(d),'actual_FNIT_case_directory':rr['actual_FNIT_case_directory'],'execution_kind':row['execution_kind'],'full_DWI_shape':list(im.shape),'rotated_bvecs_shape':list(bv.shape),'finite_output':True,'negative_output_voxels_retained':int(np.count_nonzero(a<0))}
index=load(D/'SHA256_manifest.json');mismatches={name:{'recorded_SHA256':digest,'actual_SHA256':sha(D/name)} for name,digest in index.items() if sha(D/name)!=digest}
assert not set(mismatches)-{'status.json'}
actualindex={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()};save(O/'final_delivery_actual_SHA256_index.json',actualindex)
result={'state':'actual_ten_CPU_independently_verified_and_compared','observed_UTC':utc(),'cases':checks,'ten_distinct_raw_AP':True,'final_summary_SHA256':sha(D/'summary.json'),'actual_route_SHA256':sha(D/'explicit_actual_CPU_case_routes_v1.json'),'original_collector_index_mismatches_preserved':mismatches,'actual_current_delivery_index_SHA256':sha(O/'final_delivery_actual_SHA256_index.json'),'original_delivery_or_contract_rewritten':False,'science_changed':False,'continuous_cold_wall_fabricated':False,'tool_SHA256':sha(__file__)}
save(O/'audit.json',result);save(O/'status.json',{'state':result['state'],'observed_UTC':utc(),'audit_SHA256':sha(O/'audit.json')});print(json.dumps({'state':result['state'],'index_mismatches':mismatches}),flush=True)
