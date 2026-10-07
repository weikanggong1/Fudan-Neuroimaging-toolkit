"""Rebuild source-bound scalar acceptance from saved receipts; no MRI/model imports."""
import hashlib
import json
from pathlib import Path
import statistics

LEAF = Path(__file__).resolve().parent

def identity(path):
    data=path.read_bytes()
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}

def read(name):
    return json.loads((LEAF/name).read_text())

def build():
    plan=read('PLAN.json');plan_sha=identity(LEAF/'PLAN.json')['sha256']
    assert plan_sha=='6daa84b1b18e6c343c493ef9bbff7e6fc83836c8b8149845caaef1f87655e8ed'
    local_source=LEAF.parents[2]/'src/fnit/synthseg_parc'
    for name,sha in plan['production_sources']['candidate'].items():
        assert identity(local_source/name)['sha256']==sha,(name,'candidate source changed')
    phase=read('phase1_results/phase1/QUEUE.json')
    interface=read('phase1_results/phase1/INTERFACE_LOAD.json')
    short=read('phase1_results/phase1/CONTRACTS.json')
    assert phase['status']=='metadata_and_bounded_contracts_exit0_no_whole' and len(phase['arms'])==2 and all(row['returncode']==0 for row in phase['arms'])
    assert interface['valid_interface'] and interface['compile_calls']==1 and interface['copy_calls']==interface['SGEMM_calls']==0
    assert short['valid_bounded_contracts'] and short['completed'] and len(short['numeric_rows'])==6 and len(short['copy_rows'])==13 and len(short['guard_rows'])==23
    assert all(row['different_bits']==0 for row in short['numeric_rows']+short['copy_rows'])
    assert short['candidate_copy_calls']==short['candidate_SGEMM_calls']==12
    for item in (interface,short):
        assert item['sources_unchanged'] and item['flags_unchanged'] and item['RSS_maximum_bytes']<=32_000_000_000
    rows={}; queues={}
    for mode,names in [('cpu',('A1_baseline','B1_cold','B2_warm','A2_baseline')),('gpu',('A_baseline','B_candidate'))]:
        queue=read('whole_results/whole_'+mode+'/QUEUE.json');queues[mode]=queue
        assert queue['status']=='complete' and len(queue['jobs'])==len(names) and queue['PLAN_sha256']==plan_sha
        assert read('whole_results/'+mode.upper()+'_EXIT.json')['returncode']==0
        for job,name in zip(queue['jobs'],names):
            assert job['name']==name and job['returncode']==0 and job['immediate_full_file_SHA_gate_passed']
            sub='whole_results/whole_'+mode+'/'+name+'/';whole=read(sub+'WHOLE.json')
            assert identity(LEAF/(sub+'WHOLE.json'))['sha256']==job['WHOLE_sha256']
            assert whole['valid_complete_arm'] and whole['PLAN_sha256']==plan_sha
            assert whole['source_before']==whole['source_after']==plan['production_sources'][whole['arm']]
            assert whole['common_support_before']==whole['common_support_after']==plan['common_support_sha256']
            assert whole['resources_before']==whole['resources_after']=={k:{n:r[n] for n in ('bytes','sha256')} for k,r in plan['resources'].items()}
            assert whole['source_unchanged'] and whole['resources_unchanged'] and whole['common_support_unchanged'] and whole['flags_restored']
            assert len(whole['forward_rows'])==2 and not whole['observations_are_Module_hooks']
            assert whole['saved_outputs']==job['saved_outputs']
            assert identity(LEAF/(sub+'volumes.csv'))==job['saved_outputs']['volumes.csv']
            assert all(f['input_dtype']==f['output_dtype']=='torch.float32' and not f['flags']['CPU_autocast'] and not f['flags']['CUDA_autocast'] for f in whole['forward_rows'])
            if mode=='cpu':
                assert whole['RSS_maximum_bytes']<=32_000_000_000 and whole['threads']==8
                assert len(whole['columns_rows'])==(2 if whole['arm']=='candidate' else 0)
                assert whole['copy_calls']==whole['SGEMM_calls']==(28 if whole['arm']=='candidate' else 0)
                assert all(not f['flags']['oneDNN'] for f in whole['forward_rows'])
            else:
                assert whole['optional_CPU_modules_imported']==[] and whole['compile_calls']==whole['copy_calls']==whole['SGEMM_calls']==whole['build_input_calls']==whole['compiler_probe_calls']==0
                assert all(f['flags']['matmul_TF32'] and f['flags']['cuDNN_TF32'] for f in whole['forward_rows'])
                assert whole['gpu']['allocated_peak_bytes']<=20_000_000_000 and whole['gpu']['reserved_peak_bytes']<=20_000_000_000
            rows[name]={**{k:whole[k] for k in ('API_seconds','preflight_seconds','construct_seconds','save_seconds','worker_seconds','RSS_maximum_bytes','compile_calls','copy_calls','SGEMM_calls','compiler_probe_calls','build_input_calls','cache_directory_exists_before')},
                'cold_process_seconds':job['cold_process_seconds'],'actual_comparison_executed':job['comparison_executed'],'saved_outputs':job['saved_outputs'],'WHOLE':identity(LEAF/(sub+'WHOLE.json'))}
            if mode=='gpu': rows[name].update({'Torch_memory':whole['gpu'],'driver_process_tree_sampling':job['gpu_sampling'],'optional_CPU_modules_imported':whole['optional_CPU_modules_imported']})
    assert rows['B1_cold']['compile_calls']==1 and rows['B2_warm']['compile_calls']==0
    assert rows['B1_cold']['cache_directory_exists_before'] is False and rows['B2_warm']['cache_directory_exists_before'] is True
    assert rows['A1_baseline']['saved_outputs']==rows['B1_cold']['saved_outputs']==rows['B2_warm']['saved_outputs']==rows['A2_baseline']['saved_outputs']
    assert rows['A_baseline']['saved_outputs']==rows['B_candidate']['saved_outputs']
    assert rows['A_baseline']['Torch_memory']==rows['B_candidate']['Torch_memory']
    score=read('whole_results/posthoc_v2/POSTHOC.json');exit=read('whole_results/posthoc_v2/EXIT.json')
    assert exit['returncode']==0 and exit['all_original_six_arms_sources_exits_unchanged']
    assert score['status']=='saved_output_score_passed_no_inference' and score['saved_files_unchanged']
    assert read('whole_results/posthoc_v2/PRESERVATION_BEFORE.json')==read('whole_results/posthoc_v2/PRESERVATION_AFTER.json')
    assert identity(LEAF/'case02_current_cpu_labels.png')=={k:score['figure'][k] for k in ('bytes','sha256')}
    pair=score['pairs']['baseline_vs_candidate'];native=score['pairs']['official_vs_candidate']
    assert pair['map']['different_voxels']==0 and pair['soft_CSV']['max_absolute_difference_mm3']==0 and pair['header']['all_13_exact'] and pair['header']['all_struct_fields_exact']
    assert all(score['official_error_changes'][k]==0 for k in ('introduced','removed','retained_with_changed_label'))
    baseline_median={k:statistics.median(rows[a][k] for a in ('A1_baseline','A2_baseline')) for k in ('cold_process_seconds','API_seconds')}
    candidate_median={k:statistics.median(rows[a][k] for a in ('B1_cold','B2_warm')) for k in ('cold_process_seconds','API_seconds')}
    timing={k:{'baseline_two_arm_median':baseline_median[k],'candidate_cold_warm_two_arm_median':candidate_median[k],
        'relative_seconds_reduction':1-candidate_median[k]/baseline_median[k]} for k in baseline_median}
    summary={'schema':'fnit_columns_integration_actual_acceptance/v1','status':'real_contract_CPU4_GPU2_saved_output_gates_passed_pending_root_integration',
       'production_sources':plan['production_sources'],'PLAN':identity(LEAF/'PLAN.json'),'prepare_commit':'41ede608','source_commit':'70ad6537',
       'phase1':{'numeric_cases':6,'copy_oracles':13,'fallback_cases':23,'all_numeric_copy_bits_different':0,'candidate_copy_SGEMM_calls':12,'actual_new_compile_calls':1,
         'interface':identity(LEAF/'phase1_results/phase1/INTERFACE_LOAD.json'),'contracts':identity(LEAF/'phase1_results/phase1/CONTRACTS.json'),'actual_binary':interface['artifact'],
         'phase1_controller_outer_returncode_separately_sampled':False,'limitation':'Both child RC0 and complete controller state are saved; no separate original phase1 outer OS EXIT receipt.'},
       'complete_CPU4_GPU2':rows,'CPU_two_arm_observations':timing,
       'CPU_clock_boundary':'Fresh worker includes resource/source preflight, imports, construct, API and segmentation map/CSV save; API includes each eligible engine guard/input/lib/header/provider SHA, compiler probe and cold compile.',
       'official_reference':plan['existing_official_reference'],'official_native_normal_node7_CPU8_cold_seconds':55.04640325624496,
       'official_clock_boundary':'Existing independent module/CLI worker: import/inference/main map/CSV; reused same case/node/8 physical cores, no new native run; earlier373s outlier excluded.',
       'official_full_CPU_speed_goal_passed':False,'parc_fast_new_whole_executed':False,
       'old_new_complete_output_exact':True,'old_new_NIfTI_all_fields_exact':True,'old_new_soft_CSV_exact':True,
       'official_candidate':{'different_voxels':native['map']['different_voxels'],'foreground_minimum_dice':native['map']['foreground_minimum_dice'],
         'foreground_median_dice':native['map']['foreground_median_dice'],'per_label_nonexact':[r for r in native['map']['per_label'] if r['dice']!=1],
         'soft_CSV_max_absolute_difference_mm3':native['soft_CSV']['max_absolute_difference_mm3'],'soft_CSV_nonzero':[r for r in native['soft_CSV']['columns'] if r['signed_difference_mm3']!=0],
         'all_NIfTI_struct_and13_controls_exact':native['header']['all_13_exact'] and native['header']['all_struct_fields_exact'],'gzip_file_SHA_equal':native['header']['gzip_file_SHA_equal'],
         'official_error_changes':score['official_error_changes']},
       'GPU_observation_only_no_speed_ratio':True,'GPU_old_new_map_CSV_SHA_exact':True,'GPU_allocated_reserved_exact':True,
       'GPU_driver_sampling_is_absolute_peak':False,'saved_score':identity(LEAF/'whole_results/posthoc_v2/POSTHOC.json'),'figure':score['figure'],
       'v1_posthoc_failure_retained':'Matplotlib missing after scalar comparisons; no atomic scalar result; independently authorized v2 reads saved files only.',
       'fresh_independent_Conda_environment_install_executed':False,'existing_Conda_GCC_compile_load_real_contracts_executed':True,
       'new_whole_native_runs':0,'new_low_precision_or_GPU_backend_changes':False}
    (LEAF/'RESULTS.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':summary['status'],'CPU_medians':timing,'official_different_voxels':native['map']['different_voxels'],'new_compiles':interface['compile_calls']+rows['B1_cold']['compile_calls'],'GPU_memory_exact':True}))

if __name__=='__main__':build()
