"""Read existing receipts only; writes JSON to stdout, never launches algorithms."""
import pathlib,json,hashlib,datetime
R=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003')
N=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/FNIT/runs/recon_accuracy_20261003')
receipts={}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):
 p=pathlib.Path(p)
 if not p.is_file():return {}
 b=p.read_bytes();receipts[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return json.loads(b)
def ref(p):return {'path':str(p),'sha256':receipts.get(str(p),{}).get('sha256')}
cohort=read(R/'cohort/cohort_verified.json'); index=read('/cwStorage/home/gongwk/Notebook_code/FNIT/INDEX.json')
def run(cp,expected,kind):
 c=read(cp)
 if not c:return None
 diag=pathlib.Path(c['diagnostic_root']); launch=read(diag/'launch.json');done=read(diag/'completion.json');out=pathlib.Path(c['output']);pipe=read(out/'fnit-native-free-run.json')
 val=done.get('output_validation',{});checks={'input_config_matches_cohort':c.get('input_sha256')==expected,'launch_input_matches_cohort':launch.get('input_sha256')==expected,'launch_config_sha_matches':launch.get('config_sha256')==receipts[str(cp)]['sha256'],'completion_exit_zero':done.get('exit_code')==0 and done.get('child_exit_code')==0,'completion_complete':done.get('execution_status')=='complete'}
 if kind!='official':checks.update(pipeline_complete=pipe.get('status')=='complete',completion_source_matches=done.get('code_commit')==c.get('code_commit'),outputs_138=val.get('expected')==138 and val.get('present')==138 and val.get('missing')==[])
 else:checks['official_done_sha_matches']= (out/'scripts/recon-all.done').is_file() and sha(out/'scripts/recon-all.done')==done.get('done_sha256')
 return {'kind':kind,'actual_config':ref(cp),'actual_config_fields':{k:c.get(k) for k in ['code_root','code_commit','source_archive_sha256','code_version','input','input_sha256','output','diagnostic_root','invocation','device','threads','pipeline_kwargs']},'launch':ref(diag/'launch.json'),'completion':ref(diag/'completion.json'),'completion_fields':done,'pipeline':ref(out/'fnit-native-free-run.json'),'pipeline_status':pipe.get('status'),'checks':checks,'complete_bound_receipts':all(checks.values()),'output_138':val or None}
def comparison(p,case):
 d=read(p/'checkpoint.json'); b=read(p/'execution_binding.json'); role=d.get('evaluated_role');prefix='baseline_vs_official' if role=='baseline' else 'startup_only_candidate_vs_official'; s=read(p/('strict_'+prefix+'.json'))
 checks={'case_matches':d.get('case')==case,'checkpoint_complete':d.get('status')=='complete','all_18_phases_complete':len(d.get('phases',{}))==18 and all(v.get('status')=='complete' for v in d.get('phases',{}).values()),'strict_checked_138':s.get('checked')==138}
 bound={}
 for name in [role,'official']:
  x=b.get(name,{})
  if x.get('subject'):
   out=pathlib.Path(x['subject']);bound[name]={'subject':str(out),'completion_sha256_recorded':x.get('completion_sha256'),'config_sha256_recorded':x.get('config_sha256'),'code_commit':x.get('completion',{}).get('code_commit')}
 return {'checkpoint':ref(p/'checkpoint.json'),'execution_binding':ref(p/'execution_binding.json'),'status':d.get('status'),'evaluated_role':role,'phase_count':len(d.get('phases',{})),'checks':checks,'numerical_report_complete':all(checks.values()),'strict_138':{k:s.get(k) for k in ['checked','passed','all_pass','reference','candidate']},'recorded_input_sha256':b.get('input_sha256'),'input_metadata_matches_cohort':b.get('input_sha256')==case_sha[case],'execution_subjects':bound,'overall_metric_equivalence':d.get('overall_metric_equivalence'),'files':[{ 'path':str(x),'sha256':sha(x)} for x in sorted(p.glob('*.json'))]}
case_sha={c['id']:c['sha256'] for c in cohort['cases']};rows=[]
output_names=list(read(R/'task_01/pair_baseline_official_v1/ds000030_sub-10159/strict_baseline_vs_official.json').get('files',{}))
for c in cohort['cases']:
 case=c['id'];actual=sha(c['server_input']);orig=run(R/'configs_v1'/('baseline_'+case+'.json'),actual,'baseline_original');offs=run(R/'configs_v1'/('official_'+case+'.json'),actual,'official');replays=[]
 for cp in sorted((N/'baseline_resource_replays_v1'/case).glob('attempt_*/retry_config.json')):replays.append(run(cp,actual,'baseline_resource_replay'))
 candidates=[orig]+replays;usable=[x for x in candidates if x and x['complete_bound_receipts']];comps=[]
 for b in [R/'task_01',N/'evaluation']:
  for p in b.rglob('checkpoint.json'):
   if read(p).get('case')==case:comps.append(comparison(p.parent,case))
 row={'case':case,'input_sha256':actual,'cohort_sha_matches':actual==c['sha256'],'baseline_original':orig,'baseline_replays':replays,'official':offs,'baseline_complete_for_future_same_input_comparison':bool(usable),'selected_baseline_actual_config':usable[-1]['actual_config'] if usable else None,'existing_baseline_official_comparisons':comps,'existing_baseline_official_metrics_available':any(x['numerical_report_complete'] for x in comps),'startup_8f_separate':None}
 if case=='ds000114_sub-06':
  st=run(N/'startup_whole_sub06_v2/attempt_01/retry_config.json',actual,'startup_only_candidate');comp=comparison(N/'startup_whole_sub06_v2/evaluation_startup_vs_official_v1',case);corr=read(N/'startup_pair_binding_correction_v1/correction.json');row['startup_8f_separate']={'run':st,'comparison':comp,'correction':ref(N/'startup_pair_binding_correction_v1/correction.json'),'correction_status':corr.get('status'),'corrected_input_sha256':corr.get('corrected_input_sha256'),'correction_original_binding_sha_matches':corr.get('frozen_receipts',{}).get('original_binding',{}).get('sha256')==comp['execution_binding']['sha256']}
 for item in [orig,offs]+replays+([row['startup_8f_separate']['run']] if row['startup_8f_separate'] else []):
  missing=[name for name in output_names if not (pathlib.Path(item['actual_config_fields']['output'])/name).is_file()]
  item['present_138_files_now']={'expected':len(output_names),'present':len(output_names)-len(missing),'missing':missing,'profile_source':'completed strict report 10159 files keys; presence only, no numerical assessment'}
 for comp in comps:
  comp['side_receipt_hashes_match']={}
  for role,item in [('baseline',orig),('official',offs)]:
   x=comp['execution_subjects'].get(role,{})
   comp['side_receipt_hashes_match'][role]=all([x.get('subject')==item['actual_config_fields']['output'],x.get('completion_sha256_recorded')==item['completion']['sha256'],x.get('config_sha256_recorded')==item['actual_config']['sha256']])
 rows.append(row)
print(json.dumps({'schema':'fnit-baseline-comparison-inventory-v1','collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'read-only existing metadata and input SHA; no algorithm, comparison, or GPU launch','overall_metric_equivalence':'not_assessed','cases':rows,'receipt_inventory':receipts,'discovery_roots':[str(R/'task_01'),str(N/'evaluation'),str(N/'baseline_resource_replays_v1'),str(N/'startup_whole_sub06_v2')],'limitations':['Does not rerun verify_binding/resource manifests or numerical comparisons.','138 output presence receipt differs from strict numerical 138 passes.','Uncorrected original top-level input SHA metadata is retained and flagged.','Future precision candidate must finish its own bound same-input execution/comparison.']},indent=2))
