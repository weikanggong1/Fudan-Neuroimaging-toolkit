import argparse,hashlib,json,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--version',required=True);a=p.parse_args();r=a.root/'runs'/('topology_candidate_state_cache_20261011_'+a.version);w=a.root/'workspaces'/('topology_candidate_state_cache_20261011_'+a.version)
source=r/'abba/report.private.json';s=json.loads(source.read_text());m=json.loads((w/'SOURCE_MANIFEST.json').read_text());out=r/'public';out.mkdir(mode=0o700,exist_ok=False)
records=[]
for row in s['records']:
 safe={k:row[k]for k in ('status','backend','index','fresh_cli_seconds','complete_api_seconds','child_seconds_including_imports','first_candidate_source_gate','final_repeat_gate','input_sha256','target_program_sha256','frozen_patch_manifest_sha256','actual_fnit_loaded_source_sha256','thread_environment','cpu_affinity','target_cuda','process_gpu_memory_bytes','complete_ga','final_mesh_quality','explicit_state_context_used','context_count','driver_sha256','delegate_driver_sha256','baseline_source_sha256')}
 safe['calls']=[q for q in row['instrumented_function_calls']if q['function']in('complete_state_assembly','context_preparation')]
 safe['complete_records']=row['records'];records.append(safe)
checks=[]
for row in s['comparisons']:
 groups={k.replace('.private.npz','.npz').replace('.private.json','.json'):v for k,v in row['groups'].items()};checks.append(dict(index=row['index'],backend=row['backend'],groups=groups))
medians={}
for key in ['fresh_cli_seconds','complete_api_seconds']:
 old=statistics.median(q[key]for q in records if q['backend']=='control');new=statistics.median(q[key]for q in records if q['backend']=='candidate');medians[key]={'control':old,'candidate':new,'reduction_percent':100*(old-new)/old}
report={'schema_version':1,'version':a.version,'status':s['status'],'declared_case':'public ds000114 sub07 ses-test LH; FNIT79 upstream frozen','vertices':110534,'faces':216860,'inside_vertices':38,'selected_vertices':99,'records':records,'comparisons':checks,'medians':medians,'original_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((w/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),'source_payload':m['payload'],'bundle_git_head':m['actual_bundle_git_head'],'baseline_git_commit':m['baseline_git_commit'],'driver_total_seconds':s['complete_driver_seconds'],'reference_after_used_to_generate':False,'complete_ga':False,'raw_t1_whole':False,'overall_metric_equivalence':'not_assessed','final_mesh_quality':'not_assessed'}
# Private launcher is retained server-side. Export only its hash/size, not body.
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(out/'exit.json').write_text(json.dumps({'driver_exit':int((r/'driver.exitcode').read_text()),'regression_exit':json.loads((r/'abba/EXIT.json').read_text())['exit_code']})+'\n')
print(json.dumps({p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in out.iterdir()},sort_keys=True))
