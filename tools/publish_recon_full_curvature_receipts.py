"""整理完整曲率同输入机器报告；公开副本去除主机、私有路径及逐PID记录。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-report',type=Path,required=True)
    p.add_argument('--options-report',type=Path,required=True)
    p.add_argument('--memory-report',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True)
    p.add_argument('--diagnostic',action='append',default=[],help='名称=私有诊断JSON；失败证据单列')
    a=p.parse_args()
    helper=Path(__file__).resolve().parents[1]/'validation/recon_all/python_gpu_port/publish_pial_norm_receipts.py'
    spec=importlib.util.spec_from_file_location('curvature_receipt_sanitize',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OMIT |= {'target_gpu_uuid','benchmark_pid','worker_pids','native_commands'}
    clean=module.sanitize
    if a.output_directory.exists():raise FileExistsError(a.output_directory)
    a.output_directory.mkdir(parents=True)
    raw=json.loads(a.full_report.read_text());options=json.loads(a.options_report.read_text());memory=json.loads(a.memory_report.read_text())
    if raw['status']!='complete' or options['status']!='complete':raise ValueError('final receipts must be complete')
    if raw['source_sha256']!=raw['source_sha256_after']:raise ValueError('candidate source changed')
    rows=[]
    for case in raw['cases']:
        if case['input_sha256']!=case['input_sha256_after']:raise ValueError('stage input changed')
        h=case['hemisphere']
        strict_maps=all(m['different_elements']==0 for t in case['trials'] for m in t['maps'].values())
        strict_stats=all(t['stats']['same_file_bytes'] for t in case['trials'])
        file_bytes=all(all(t['output_sha256'][name]==case['native_output_sha256'][0][f'{h}.smoothwm.{name}.crv']
                 for name in ('K','H','K1','K2','S','C','BE','FI')) for t in case['trials'])
        native_repeat=case['native_repeat_stats']['same_file_bytes'] and all(m['different_elements']==0 for m in case['native_repeat_maps'].values())
        rows.append({'public_case':f"ds000114 sub-{6+case['case_index']:02d}", 'hemisphere':h,
         'vertices':case['input_mesh']['vertices'],'faces':case['input_mesh']['faces'],
         'reference_reproducibility':'passed' if native_repeat else 'failed',
         'strict_map_values':'passed' if strict_maps else 'failed','strict_map_file_bytes':'passed' if file_bytes else 'failed',
         'strict_complete_stats':'passed' if strict_stats else 'failed',
         'existing_eight_map_gate':'passed' if all(m['within_existing_eight_map_exploration_gate'] for t in case['trials'] for m in t['maps'].values()) else 'failed',
         'native_complete_cli_seconds':case['native_seconds'],
         'own_cpu_complete_api_seconds':[t['result']['seconds']['total_including_io'] for t in case['trials'] if t['device']=='cpu'],
         'own_gpu_complete_api_seconds':[t['result']['seconds']['total_including_io'] for t in case['trials'] if t['device'].startswith('cuda')],
         'precision_scope':'same-input frozen own smoothwm; not original T1 end to end',
         'whole_pipeline_improvement':'not_measured'})
    report={'schema_version':1,'scope':'complete_discrete_curvature_stats_same_input_four_hemispheres',
       'candidate_base_commit':raw['code_base_commit'],'common_dependency_commit':'79a41cdd',
       'reference_source_commit':'d932c45b7941662ea380a05efef580568b98d41a',
       'reference_kind':'same-host independent Conda source build; no installed-distribution equivalence inferred',
       'publication_script_sha256':sha(__file__),'sanitizer_sha256':sha(helper),
       'private_full_report_sha256':sha(a.full_report),'private_options_report_sha256':sha(a.options_report),
       'hardware':{'cpu':raw.get('cpu_model'),'gpu':memory['gpu_name'],'threads':raw['threads'],'torch':raw['torch'],'numpy':raw['numpy'],'system_libc':raw['system_libc']},
       'execution':'complete','output_integrity':'passed' if len(rows)==4 and all(len(t['result']['outputs'])==9 for r in raw['cases'] for t in r['trials']) else 'failed',
       'strict_stage_reproduction':'passed' if all(r['strict_map_values']=='passed' and r['strict_complete_stats']=='passed' for r in rows) else 'failed',
       'mesh_quality':'input coordinates/topology unchanged; independent whole-mesh quality not recomputed by this scalar stage',
       'optimized_stage_degradation':'none observed in tested complete standard stage',
       'overall_metric_equivalence':'not_assessed','raw_T1_recon_all':'not_run_for_this_module',
       'cases':rows,'full_report':clean(raw),'optional_profiles':clean(options),
       'memory':{'private_report_sha256':sha(a.memory_report),'report':clean(memory),'number_of_samples':len(memory.get('samples',[])),
           'budget_bytes':20000000000,'max_pytorch_allocated_bytes':max(t.get('peak_allocated_bytes',0) for r in raw['cases'] for t in r['trials']),
           'max_pytorch_reserved_bytes':max(t.get('peak_reserved_bytes',0) for r in raw['cases'] for t in r['trials']),
           'process_tree_budget':'unverified when attribution unresolved; shared whole-card occupancy includes other tasks'},
       'diagnostics':{}}
    for item in a.diagnostic:
        name,path=item.split('=',1);path=Path(path)
        report['diagnostics'][name]={'private_report_sha256':sha(path),'report':clean(json.loads(path.read_text()))}
    (a.output_directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('execution','output_integrity','strict_stage_reproduction','cases')},indent=2))


if __name__=='__main__':main()
