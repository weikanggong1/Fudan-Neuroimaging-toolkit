import pathlib,json,hashlib,importlib.util,sys,os,time,datetime,subprocess,traceback
sys.dont_write_bytecode=True
root=pathlib.Path('/cwStorage/home/gongwk/Notebook_code/fnit_connectome_tenraw_20261002');p=root/'task_04/raw_matrix_envelope_recovery_readonly_v2_baseline_CON05';p.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
producer=root/'formal_selected_monitor_recovery_tools_v3/benchmark_connectome_selected_raw_recovery.py'
# Exact deployed producer identity is frozen in configuration before functions are imported.
c=json.loads(pathlib.Path(__file__).with_name('recovery_config.json').read_text());assert sha(producer)==c['producer_sha256']
selection_path=producer.parent/'selection.json';assert sha(selection_path)==c['selection_sha256'];selection=json.loads(selection_path.read_text())
spec=importlib.util.spec_from_file_location('actual_selected_producer',producer);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
rows=s.validate_selection(selection);status={'scope':'read-only actual recovery proofs and early raw envelope; final logical10 origin freeze belongs to root','producer':{'path':str(producer),'sha256':sha(producer)},'selection':{'path':str(selection_path),'sha256':sha(selection_path)},'runs':[],'GPU_science_started':False}
def require(x,msg):
 if not x:raise ValueError(msg)
for arm,caseid in [('baseline','sub-CON05')]:
 t=time.monotonic();dst=p/f'{arm}_{caseid}';dst.mkdir();rec={'arm':arm,'case_id':caseid,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  row=next(r for r in rows if r['arm']==arm and r['case_id']==caseid);cohort,rerun,staged=s.load_helpers(row)
  original,oldstate,case=s.origin(row,cohort,rerun,staged)
  job=root/'formal_selected_monitor_recovery_v3'/arm/caseid
  values={n:json.loads((job/n).read_text()) for n in ('recovery_binding.json','recovery_eligibility.json','gpu_report.json','recovery_config.json','raw_bids_wall.json')}
  for n in values:(dst/n).write_bytes((job/n).read_bytes())
  b=values['recovery_binding.json'];e=values['recovery_eligibility.json'];g=values['gpu_report.json'];new=values['recovery_config.json'];w=values['raw_bids_wall.json']
  require(b['mode']==s.MODE and b['arm']==arm and b['case_id']==caseid and b['case']==case and b['original_config']==row['origin_config'] and b['original_driver']==row['origin_driver'],'canonical recovery selection/original proof mismatch')
  require(b['FS_recomputed'] is False and b['old_DWI_outputs_used'] is False and b['raw_DWI_execution_policy']=='complete_from_raw','recovery scientific scope mismatch')
  require(e['status']=='execution_complete_memory_observed_below_budget' and e['validation_error'] is None and e['full_ten_complete'] is False and e['memory_budget']==g['memory_budget'] and s.eligible(g['memory_budget']),'recovery eligibility/memory mismatch')
  require(g['status']=='completed' and g['exit_code']==0 and g['source_before']==g['source_after']==original['frozen_sources'][arm],'recovery execution/source mismatch')
  expected=s.fresh_config(original,str(root/'formal_selected_monitor_recovery_v3'),new['gpu_python'],new['stop_dispatch_path']);observed={k:v for k,v in new.items() if k!='resources_manifest'}
  require(observed==expected,'recovery altered non-permitted config');rerun.verify_resources(new)
  resources=s.read_bound(new['resources_manifest']);prior=s.read_bound(original['resources_manifest'])
  ledger=lambda d:{(v['role'],v['path']):v for v in d['files'] if v['role']!='gpu_python'}
  require(resources==b['resources'] and ledger(resources)==ledger(prior),'non-monitor scientific resource change')
  runtime=b['runtime'];require(runtime['scientific_runtime_equal'] is True and runtime['CUDA_initialized'] is False and runtime['original']['modules']==runtime['isolated']['modules'],'recorded scientific runtime differs')
  for rr in (runtime['original'],runtime['isolated']):
   require(sha(rr['python_binary'])==rr['python_binary_sha256'],'runtime python bytes changed')
   for v in rr['modules'].values():require(sha(v['path'])==v['sha256'],'runtime science module bytes changed')
  oldjob=pathlib.Path(original['run_root'])/arm/caseid;require(s.validate_reason(row,oldjob,cohort)==b['reason'],'original reason mismatch')
  for name,record in b['old_case_reports'].items():
   if 'binding' in record:require(s.read_bound(record['binding'])==record['value'],'old case report bytes changed')
   else:require(not (oldjob/name).exists(),'originally absent old result appeared')
  recon,subject,proof=s.anatomy(row,original,case,cohort,rerun,staged)
  require(recon['anatomy']==b['anatomy'] and subject==b['anatomy_subject_dir'],'same-round FS current data differs')
  require(g['wall_report']==str(job/'raw_bids_wall.json') and w['status']=='completed' and w['exit_code']==0 and w['outputs']['status']=='complete','actual CLI wall incomplete')
  require(w['initial_output_state']['output_directory_existed'] is False and w['initial_output_state']['preexisting_run_state'] is False and w['initial_output_state']['preexisting_state_files']==[],'reused raw output namespace')
  require(w['preprocessing']==[{'topup':'completed','eddy':'completed','recon_all':'supplied'}],'fresh raw preprocessing contract differs')
  for f,meta in g['outputs']['files'].items():
   q=pathlib.Path(f);require(q.is_file() and sha(q)==meta['sha256'] and q.stat().st_size==meta['size_bytes'],'GPU output binding changed')
  for f,meta in w['outputs']['files'].items():require(sha(job/'connectome'/f)==meta['sha256'],'actual CLI output changed')
  rec.update(recovery_current_readonly_validation='passed',memory_budget=g['memory_budget'],proof_files={n:{'path':str(job/n),'sha256':sha(job/n)} for n in values},helper_files=row['helpers'],root_final_immutable_origins_freeze=False)
  official=root/'task_04/official_raw10_cpu_view_v3'/caseid/'reference_manifest.json'
  if not official.exists():rec['matrix_scope']='waiting_completed_official_same_case'
  else:
   argv=[sys.executable,str(pathlib.Path(__file__).parent/'tools/benchmark_connectome_raw_cohort_envelope.py'),'--official-root',str(official.parent),'--fnit',str(job/'connectome'),'--fnit-gpu-reports',str(job/'gpu_report.json'),'--raw-manifest',c['raw_manifest']['path'],'--raw-manifest-sha256',c['raw_manifest']['sha256'],'--case-id',caseid,'--fnit-seeds','0','--output',str(dst/'envelope.json')]
   rec['argv']=argv;tt=time.monotonic();r=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);rec.update(returncode=r.returncode,envelope_readonly_wall_seconds=time.monotonic()-tt);(dst/'stdout.log').write_text(r.stdout);(dst/'stderr.log').write_text(r.stderr)
   if r.returncode:rec['guard_failure']=r.stderr[-4000:]
   else:
    d=json.loads((dst/'envelope.json').read_text());rec.update(accepted=sum(v['accepted_count'] for x in d['profiles'].values() for v in x['ranges'].values()),total=sum(len(v['comparison_accepted']) for x in d['profiles'].values() for v in x['ranges'].values()),matrix_envelope_status=d['matrix_envelope_status'],fnit_reproducibility_status=d['fnit_reproducibility_status'],envelope_sha256=sha(dst/'envelope.json'))
 except Exception as error:rec['validation_error']={'type':type(error).__name__,'message':str(error)};(dst/'validation_failure.log').write_text(traceback.format_exc())
 rec.update(readonly_wall_seconds=time.monotonic()-t,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(dst/'execution.json').write_text(json.dumps(rec,indent=2)+'\n');status['runs'].append(rec);(p/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps({k:rec.get(k) for k in ('arm','case_id','recovery_current_readonly_validation','accepted','total','matrix_envelope_status','matrix_scope','validation_error','guard_failure')}),flush=True)
status['finished']=True;(p/'status.json').write_text(json.dumps(status,indent=2)+'\n')
