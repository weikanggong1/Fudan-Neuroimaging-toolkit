"""离线schema和隐私导出合同；没有影像、候选算法运行或速度结论。"""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[3]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


CONTRACT=load(REPO/'tests/recon_all/test_topology_edge_predicate_population_prefix.py','fnit_prefix_fixture_contract')
API=CONTRACT.api


def fixture(tmp_path):
    """完整假报告只验证schema，不冒充实际source import或真实T1结果。"""
    binding=CONTRACT.fake_pair(tmp_path);root=tmp_path/'frozen';root.mkdir()
    manifest=json.loads(binding['source_manifest'].read_text())
    export_relative='validation/recon_all/optimizations/20261011_topology_edge_predicate_population_prefix/export_metrics.py'
    manifest['files'][export_relative]=dict(size=0,sha256='0'*64)
    real={API.TOOL_RELATIVE:REPO/API.TOOL_RELATIVE,API.identity.TOOL_RELATIVE:REPO/API.identity.TOOL_RELATIVE,
          export_relative:BASE/'export_metrics.py'}
    for relative in manifest['files']:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if relative in real:shutil.copyfile(real[relative],path)
        else:path.write_text('# offline metadata fixture only\n')
        manifest['files'][relative]=dict(size=path.stat().st_size,sha256=API.sha(path))
    for name in manifest['expected_fnit_module_sha256']:
        relative=API.identity.module_relative(name,json.loads(binding['source_manifest'].read_text()))
        manifest['expected_fnit_module_sha256'][name]=manifest['files'][relative]['sha256']
    API.write(binding['source_manifest'],manifest)
    protocol=json.loads(binding['protocol'].read_text());protocol['source_manifest_sha256']=API.sha(binding['source_manifest'])
    protocol['input_sha256']={k:'b'*64 for k in ('nofix','qsphere','brain')};API.write(binding['protocol'],protocol)
    for arm in ('baseline','memo'):
        path=tmp_path/arm
        def report_mutation(r):
            r['source_manifest_sha256']=API.sha(binding['source_manifest']);r['protocol_sha256']=API.sha(binding['protocol'])
            r['actual_tool_sha256']=manifest['files'][API.TOOL_RELATIVE]['sha256'];r['input_sha256']=protocol['input_sha256']
            for name in r['actual_loaded_fnit_sha256']:
                relative=r['actual_loaded_fnit_sources'][name]['relative_path'];record=manifest['files'][relative]
                r['actual_loaded_fnit_sha256'][name]=record['sha256']
                r['actual_loaded_fnit_sources'][name]=dict(relative_path=relative,origin_relative_path=relative,**record)
            for name in r['actual_loaded_benchmark_tool_sources']:
                relative='tools/'+name+'.py';r['actual_loaded_benchmark_tool_sources'][name]=dict(
                    relative_path=relative,origin_relative_path=relative,**manifest['files'][relative])
            r['peak_process_rss_bytes']=20000000
            r['stages_seconds']={k:1.0 for k in ('load_seconds','own_base_seconds','own_priors_edges_local_seconds')}
            r['private_extra']='NEVER_EXPORT_HOST_PATH_OR_RAW_ARRAY'
            for record in r['records']:
                record['base_context_state_sha256']='1'*64
                record.update({k:1.0 for k in ('preparation_seconds','controller_with_output_seconds','summary_output_write_seconds',
                    'complete_prefix_api_and_output_seconds','source_controller_seconds','memo_release_seconds')})
                for s in record['slots']:
                    s.update({k:.1 for k in ('complete_cycle_seconds','ordering_seconds','complete_scan_api_seconds',
                        'scanner_preparation_seconds','output_npz_control_write_seconds','retessellation_seconds')})
                    s['cycle_subseconds']={k:.1 for k in ('candidate_complete','statistics_update','selective_full_restore','complete_cycle')}
                    s['candidate_subseconds']={k:.1 for k in ('geometry_complete_api','displacement_complete_api','sff_complete_api',
                        'complete_prepared_five_api','complete_candidate_api_including_validation')}
                    for cache in (s['cache_before'],s['cache_after']):
                        if cache is not None:cache.update(vertex_count=6,hit_rate=cache['cache_hits']/cache['logical_predicate_calls'] if cache['logical_predicate_calls'] else 0.,preparation_seconds=0.)
        CONTRACT.refresh(path,report_mutation)
    controller=tmp_path/'controller';controller.mkdir()
    API.compare_completed(baseline=tmp_path/'baseline',candidate=tmp_path/'memo',output=controller/'pair.private.json',**binding)
    API.write(controller/'cli.private.json',dict(records=[dict(arm=arm,exit_code=0,complete_cold_cli_wall_seconds=10.) for arm in ('baseline','memo')],
        includes_imports_preflight_raw_preparation_nine_candidates_io_exit=True,no_other_arm_outputs_read_before_both_complete=True))
    exporter=load(root/export_relative,'fnit_prefix_export_frozen_contract')
    return exporter,dict(baseline=tmp_path/'baseline',memo=tmp_path/'memo',controller=controller,source_root=root,
        **binding,output=tmp_path/'safe.json')


def test_complete_closed_export_only_discloses_typed_metrics(tmp_path):
    exporter,kwargs=fixture(tmp_path);r=exporter.export(**kwargs)
    assert r['complete_comparison_groups']==114 and r['artifact_count_per_arm']==45
    assert r['arms']['memo']['runs']['warm']['slots']['2']['cache_delta']['cache_hits']==2
    assert r['gpu_exercised'] is False and r['allocated_bytes'] is None and r['full_ga'] is False
    assert r['hemisphere']=='lh'
    text=kwargs['output'].read_text();assert 'NEVER_EXPORT' not in text and str(tmp_path) not in text


@pytest.mark.parametrize('configured',['rh','other',None])
def test_actual_configured_hemisphere_rejected_before_input_read(tmp_path,configured):
    exporter,kwargs=fixture(tmp_path)
    manifest=json.loads(kwargs['source_manifest'].read_text());declared=json.loads(kwargs['protocol'].read_text())
    reader=exporter.load_reader(kwargs['source_root'],manifest)
    config=dict(source_root=str(kwargs['source_root']),source_manifest=str(kwargs['source_manifest']),
        source_manifest_sha256=API.sha(kwargs['source_manifest']),protocol=str(kwargs['protocol']),
        protocol_sha256=API.sha(kwargs['protocol']),threads=2,cpu_affinity=[1,2],maximum_entries=250000,
        input_sha256=declared['input_sha256'],hemisphere=configured)
    # This source-complete fixture contains no MRI input paths: rejection is pre-IO.
    with pytest.raises(ValueError,match='configured hemisphere'):
        reader.validate_source(config)
    config['hemisphere']='lh';reader.validate_source(config)


@pytest.mark.parametrize('wrong',['missing_receipt','nan_clock','child_clock','code_payload','original_gate','failed_cli'])
def test_incomplete_or_corrupt_evidence_cannot_export(tmp_path,wrong):
    exporter,kwargs=fixture(tmp_path)
    if wrong=='missing_receipt':(kwargs['memo']/'COMPLETE.json').unlink()
    elif wrong=='nan_clock':
        path=kwargs['baseline'];p=path/'report.private.json';r=json.loads(p.read_text())
        r['records'][0]['preparation_seconds']=float('nan');p.write_text(json.dumps(r)+'\n')
        API.write(path/'COMPLETE.json',dict(status=r['status'],report_sha256=API.sha(p)))
    elif wrong=='child_clock':CONTRACT.refresh(kwargs['baseline'],lambda r:r['records'][0]['slots'][0]['cycle_subseconds'].pop('complete_cycle'))
    elif wrong=='code_payload':(kwargs['source_root']/API.TOOL_RELATIVE).write_text('# corrupted\n')
    elif wrong=='original_gate':
        p=kwargs['controller']/'pair.private.json';r=json.loads(p.read_text());r['checks'].pop();API.write(p,r)
    elif wrong=='failed_cli':
        p=kwargs['controller']/'cli.private.json';r=json.loads(p.read_text());r['records'][1]['exit_code']=1;API.write(p,r)
    with pytest.raises(ValueError):exporter.export(**kwargs)
    assert not kwargs['output'].exists()


@pytest.mark.parametrize('value',[True,-1,float('nan'),float('inf'),'host/path'])
def test_numeric_schema_rejects_boolean_nonfinite_and_path_values(value):
    exporter=load(BASE/'export_metrics.py','fnit_prefix_typed_unit')
    with pytest.raises(ValueError):exporter.number(value)
