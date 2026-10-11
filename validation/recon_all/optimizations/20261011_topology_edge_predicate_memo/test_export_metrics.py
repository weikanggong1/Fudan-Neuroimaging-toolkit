"""报告schema/安全导出合同夹具；没有执行MRI或科学算子。"""
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


EXPORT = load(BASE / 'export_metrics.py', 'fnit_memo_typed_export_contract')
DRIVER_PATH = REPO / 'tools/benchmark_recon_topology_edge_predicate_cache.py'
DRIVER = load(DRIVER_PATH, 'fnit_memo_schema_contract')


def fixture(tmp_path):
    """显式伪造的固定schema报告，仅用于导出器合同，不是阶段benchmark。"""
    root = tmp_path / 'frozen'; root.mkdir(); (root / 'tools').mkdir()
    shutil.copyfile(DRIVER_PATH, root / DRIVER.TOOL_RELATIVE)
    files = {}
    modules = {}
    for name in DRIVER.REQUIRED_FNIT:
        relative = 'src/' + name.replace('.', '/') + ('/__init__.py' if name in ('fnit','fnit.recon_all') else '.py')
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# metadata-only test fixture\n')
        files[relative] = dict(size=path.stat().st_size, sha256=EXPORT.sha(path)); modules[name] = files[relative]['sha256']
    for name in DRIVER.REQUIRED_HELPERS:
        relative = 'tools/' + name + '.py'; path = root / relative; path.write_text('# metadata-only test fixture\n')
        files[relative] = dict(size=path.stat().st_size, sha256=EXPORT.sha(path))
    entry = root / DRIVER.TOOL_RELATIVE
    files[DRIVER.TOOL_RELATIVE] = dict(size=entry.stat().st_size, sha256=EXPORT.sha(entry))
    manifest = root / 'manifest.json'; DRIVER.write(manifest, dict(git_base='a'*40,files=files,expected_fnit_module_sha256=modules))
    protocol = root / 'protocol.json'; DRIVER.write(protocol, dict(git_base='a'*40,source_manifest_sha256=EXPORT.sha(manifest),
        resources=dict(requested_threads=2,candidate_cpu_affinity=[1,2]), input_sha256={k:'b'*64 for k in ('nofix','qsphere','brain')},
        cache=dict(maximum_entries=10),strict_gates=dict(candidate_npz_fields=sorted(DRIVER.CANDIDATE_FIELDS),candidate_npz_field_count=36,
        state_npz_fields=sorted(DRIVER.STATE_FIELDS),state_npz_field_count=40,full_control_fields=sorted(DRIVER.CONTROL_FIELDS))))
    actual_sources = {}
    for name in modules:
        relative = DRIVER.module_relative(name,json.loads(manifest.read_text())); record = files[relative]
        actual_sources[name]=dict(relative_path=relative,origin_relative_path=relative,size=record['size'],sha256=record['sha256'])
    helpers = {name:dict(relative_path='tools/'+name+'.py',origin_relative_path='tools/'+name+'.py',
        size=files['tools/'+name+'.py']['size'],sha256=files['tools/'+name+'.py']['sha256']) for name in DRIVER.REQUIRED_HELPERS}
    environment = {k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
        'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','NUMBA_NUM_THREADS')}
    clock_fields = ('preparation_seconds','ordering_seconds','complete_cycle_seconds','preparation_and_cycle_seconds',
        'full_candidate_api_seconds','candidate_api_and_output_seconds','complete_scan_api_seconds','scanner_preparation_seconds',
        'output_npz_control_write_seconds','retessellation_seconds')
    paths=[]
    for arm in ('baseline','memo'):
        path=tmp_path/arm;path.mkdir();paths.append(path);artifacts={};records=[]
        for run in ('cold','warm','audit'):
            for kind,fields in (('candidate',DRIVER.CANDIDATE_FIELDS),('state',DRIVER.STATE_FIELDS)):
                artifact=path/(run+'.'+kind+'.private.npz');np.savez_compressed(artifact,**{k:np.asarray([1,-0.0],np.float32) for k in fields})
                artifacts[artifact.name]=EXPORT.sha(artifact)
            control=path/(run+'.control.private.json');DRIVER.write(control,{k:1 for k in sorted(DRIVER.CONTROL_FIELDS)});artifacts[control.name]=EXPORT.sha(control)
            cache=None
            if arm=='memo':
                cache=dict(vertex_count=4,maximum_entries=10,entries=2,logical_predicate_calls=3,cache_hits=1,
                    actual_predicate_evaluations=2,inserted_unique_keys=2,capacity_bypass_evaluations=0,complete_unique_key_count=2,
                    hit_rate=1/3,preparation_seconds=0.1,canonical_sha256='c'*64)
            records.append(dict(run=run,logical_predicate_calls=3,sequence_sha256='d'*64 if run=='audit' else None,
                cache=cache,**{k:1.0 for k in clock_fields}))
        report=dict(status='complete_fixed_first_identity_candidate_diagnostic',arm=arm,records=records,artifact_sha256=artifacts,
            input_sha256=json.loads(protocol.read_text())['input_sha256'],source_manifest_sha256=EXPORT.sha(manifest),protocol_sha256=EXPORT.sha(protocol),
            actual_tool_sha256=files[DRIVER.TOOL_RELATIVE]['sha256'],actual_loaded_fnit_sha256=modules,
            actual_loaded_fnit_sources=actual_sources,actual_loaded_benchmark_tool_sources=helpers,all_fnit_modules_inside_frozen_root=True,
            threads=2,cpu_affinity=[1,2],actual_rng_draws=0,peak_process_rss_bytes=16000000,
            actual_thread_policy=dict(requested_threads=2,restoration_complete=True,torch=dict(before=2,effective=2,restored=2),numba=dict(before=2,effective=2,restored=2)),
            fresh_thread_environment=environment,global_function_replacement=True,production_wiring=False,cuda_hidden=True,
            cache_shared_between_repeats=False,private_ignored_marker='NEVER_EXPORT_PRIVATE_LOCATION')
        DRIVER.write(path/'report.private.json',report);DRIVER.write(path/'COMPLETE.json',dict(status=report['status'],report_sha256=EXPORT.sha(path/'report.private.json')))
    controller=tmp_path/'controller';controller.mkdir()
    DRIVER.compare_completed(baseline=paths[0],candidate=paths[1],output=controller/'pair.private.json',source_manifest=manifest,protocol=protocol)
    DRIVER.write(controller/'cli.private.json',dict(records=[dict(arm=arm,exit_code=0,complete_cold_cli_wall_seconds=10.0) for arm in ('baseline','memo')],
        includes_imports_preflight_raw_preparation_three_candidates_io_exit=True,no_other_arm_outputs_read_before_both_complete=True))
    return dict(baseline=paths[0],memo=paths[1],controller=controller,source_root=root,source_manifest=manifest,protocol=protocol,output=tmp_path/'safe.json')


def test_complete_export_keeps_only_typed_metrics_not_private_markers(tmp_path):
    kwargs=fixture(tmp_path);result=EXPORT.export(**kwargs)
    assert result['complete_gate_count']==24 and result['candidate_npz_field_count']==36 and result['state_npz_field_count']==40
    assert result['arms']['memo']['runs']['warm']['cache']['cache_hits']==1
    assert result['allocated_bytes'] is None and result['whole_recon_all']=='not_assessed'
    text=kwargs['output'].read_text();assert 'NEVER_EXPORT_PRIVATE_LOCATION' not in text and str(tmp_path) not in text


def test_export_requires_both_receipts_and_declines_nan_metric(tmp_path):
    kwargs=fixture(tmp_path);receipt=kwargs['memo']/'COMPLETE.json';raw=receipt.read_bytes();receipt.unlink()
    with pytest.raises(ValueError,match='both complete'):EXPORT.export(**kwargs)
    assert not kwargs['output'].exists();receipt.write_bytes(raw)
    path=kwargs['baseline']/'report.private.json';report=json.loads(path.read_text());report['records'][0]['full_candidate_api_seconds']=float('nan')
    path.write_text(json.dumps(report)+'\n')
    DRIVER.write(kwargs['baseline']/'COMPLETE.json',dict(status=report['status'],report_sha256=EXPORT.sha(path)))
    refreshed=kwargs['controller']/'refreshed.private.json'
    DRIVER.compare_completed(baseline=kwargs['baseline'],candidate=kwargs['memo'],output=refreshed,
        source_manifest=kwargs['source_manifest'],protocol=kwargs['protocol'])
    refreshed.replace(kwargs['controller']/'pair.private.json')
    with pytest.raises(ValueError,match='numeric metric'):EXPORT.export(**kwargs)
    assert not kwargs['output'].exists()


def test_typed_metric_and_commit_reject_nonfinite_private_strings_and_bool():
    for value in (float('nan'),float('inf'),-1,True,'NEVER_EXPORT_PRIVATE_LOCATION'):
        with pytest.raises(ValueError):EXPORT.number(value)
    with pytest.raises(ValueError):EXPORT.commit('/private/path')
