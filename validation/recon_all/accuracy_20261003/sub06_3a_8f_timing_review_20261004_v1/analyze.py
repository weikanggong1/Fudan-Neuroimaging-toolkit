"""CPU-only arithmetic on saved actual execution receipts; no algorithm launch."""
import json,pathlib,hashlib
P=pathlib.Path(__file__).parent
raw=json.loads((P/'receipts.json').read_text()); old=raw['old']['fnit-native-free-run.json']['data'];new=raw['new']['fnit-native-free-run.json']['data']
a={x['name']:x for x in old['stages']};b={x['name']:x for x in new['stages']};assert list(a)==list(b)
rows=[]
for name,x in a.items():
 y=b[name];rows.append({'name':name,'parent':'pipeline_seconds','additive_at_pipeline_level':True,'old_seconds':x['seconds'],'new_seconds':y['seconds'],'delta_seconds':y['seconds']-x['seconds'],'old_parent_cpu_seconds':x.get('parent_cpu_seconds'),'new_parent_cpu_seconds':y.get('parent_cpu_seconds'),'old_child_cpu_seconds':x.get('child_cpu_seconds'),'new_child_cpu_seconds':y.get('child_cpu_seconds'),'old_pre_post_cuda_sync_seconds':x.get('cuda_pre_sync_seconds',0)+x.get('cuda_post_sync_seconds',0),'new_pre_post_cuda_sync_seconds':y.get('cuda_pre_sync_seconds',0)+y.get('cuda_post_sync_seconds',0)})
groups={}
for role,q in [('old',old),('new',new)]:
 gs=[]
 for g in q['hemisphere_scheduling']['groups']:
  item={k:g.get(k) for k in ['operation','group_wall_seconds','private_copy_seconds','startup_wait_seconds_actual','operation_worker_span_seconds','operation_worker_sum_seconds','operation_overlap_seconds','publish_seconds','cleanup_seconds','parent_idle_cuda_cache']};item['parent']=''+g['operation']+'_hemisphere_group';item['additive_at_pipeline_level']=False
  item['attempts']={h:[{'attempt':z.get('attempt'),'pid':z.get('pid'),'exit_code':z.get('exit_code'),'operation_entered':z.get('operation_entered'),'failure_stage':z.get('worker_report',{}).get('stage'),'bootstrap_stage':z.get('worker_report',{}).get('cuda_bootstrap_stage'),'status':z.get('worker_report',{}).get('status'),'total_seconds':z.get('worker_report',{}).get('total_seconds')} for z in v] for h,v in g['startup_attempts'].items()}
  item['worker_substages']={h:g['values'][h].get('stages') for h in ['lh','rh']};gs.append(item)
 groups[role]=gs
summary={}
for role,q in [('old',old),('new',new)]:
 t=q['timing'];s=sum(x['seconds'] for x in q['stages']);summary[role]={'entry_seconds':raw[role]['completion.json']['data']['command_seconds'],'api_seconds':q['total_seconds'],'pipeline_seconds':t['pipeline_seconds'],'additive_stages_sum_seconds':s,'unprofiled_pipeline_residual_seconds':t['pipeline_seconds']-s,'validation_seconds':t['validation_seconds'],'public_wrapper_residual_seconds':t['thread_setup_and_restore_seconds'],'entry_minus_api_seconds':raw[role]['completion.json']['data']['command_seconds']-q['total_seconds'],'launch_loadavg':raw[role]['launch.json']['data'].get('loadavg_at_launch'),'monitor':{k:v for k,v in raw[role]['monitor.json']['data'].items() if k!='command'},'cache_counters':q['surface_stats_cache'],'cuda_allocator':q['cuda_allocator']}
summary['delta']={k:summary['new'][k]-summary['old'][k] for k in ['entry_seconds','api_seconds','pipeline_seconds','additive_stages_sum_seconds','unprofiled_pipeline_residual_seconds','validation_seconds','public_wrapper_residual_seconds','entry_minus_api_seconds']}
out={'schema':'fnit-actual-sub06-timing-delta-v1','input_equal':old['input']==new['input'],'source_commits':{'old':raw['old']['completion.json']['data']['code_commit'],'new':raw['new']['completion.json']['data']['code_commit']},'summary':summary,'additive_pipeline_stages':rows,'descending_positive_deltas':sorted(rows,key=lambda x:x['delta_seconds'],reverse=True),'nonadditive_group_diagnostics':groups,'actual_mni_nested':{role:next(x for x in q['stages'] if x['name']=='mni_nonlinear') for role,q in [('old',old),('new',new)]},'source_evidence':{'native_free_3a':'group stage call lines1000-1002 returns before metrics lines1024-1030; worker defer_metrics at684','profiling':'StageProfiler.run seconds includes pre/body/post; CPU times not additive to wall','policy':'mris_register_average_numba.py new cached parallel source-order reduction; surface_stats_cache.py new float64 ROI area only; GPU allocator disabled both runs'},'receipts_sha256':hashlib.sha256((P/'receipts.json').read_bytes()).hexdigest(),'assessment':'single-run observations; no causal attribution to five precision changes; no equivalence claim'}
(P/'timing_delta.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary['delta'],indent=2));print([(x['name'],round(x['delta_seconds'],3)) for x in out['descending_positive_deltas'][:12]])
