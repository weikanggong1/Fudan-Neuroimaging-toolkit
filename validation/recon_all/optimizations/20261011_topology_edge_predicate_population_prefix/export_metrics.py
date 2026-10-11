"""双方完整收据后独立重核114项门，只导出具名数值、布尔值和SHA。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

RELATIVE = 'validation/recon_all/optimizations/20261011_topology_edge_predicate_population_prefix/export_metrics.py'
ENTRY = 'tools/benchmark_recon_topology_edge_predicate_population_prefix.py'
IDENTITY = 'tools/benchmark_recon_topology_edge_predicate_cache.py'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, *, integer=False, nullable=False):
    if nullable and value is None: return None
    if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
            or value < 0 or integer and not isinstance(value,int)):
        raise ValueError('finite nonnegative typed metric required')
    return value


def digest(value, *, length=64):
    if not isinstance(value,str) or len(value)!=length or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('hexadecimal source digest required')
    return value


def load_reader(root, manifest):
    """离线比较也绑定新入口及复用v5入口，拒绝sys.modules中外部同名工具。"""
    for relative in (ENTRY,IDENTITY,RELATIVE):
        path=root/relative; record=manifest['files'][relative]
        if sha(path)!=record['sha256'] or path.stat().st_size!=record['size']:
            raise ValueError('actual frozen reader/exporter payload mismatch')
    if Path(__file__).resolve() != (root/RELATIVE).resolve():
        raise ValueError('actual exporter must load from declared frozen root')
    name='benchmark_recon_topology_edge_predicate_cache'; previous=sys.modules.get(name)
    try:
        spec=importlib.util.spec_from_file_location(name,root/IDENTITY)
        identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity);sys.modules[name]=identity
        spec=importlib.util.spec_from_file_location('fnit_population_prefix_frozen_reader',root/ENTRY)
        reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
        return reader
    finally:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous


def cache_public(cache):
    if cache is None:return None
    counts=('vertex_count','maximum_entries','entries','logical_predicate_calls','cache_hits',
            'actual_predicate_evaluations','inserted_unique_keys','capacity_bypass_evaluations')
    result={k:number(cache[k],integer=True) for k in counts}
    result.update(hit_rate=number(cache['hit_rate']),preparation_seconds=number(cache['preparation_seconds']),
        complete_unique_key_count=number(cache['complete_unique_key_count'],integer=True,nullable=True),
        canonical_sha256=digest(cache['canonical_sha256']))
    if result['hit_rate']>1:raise ValueError('bounded cache hit rate required')
    return result


def export(*, baseline, memo, controller, source_root, source_manifest, protocol, output):
    """重复核闭合schema/来源/反馈/RNG门；候选计算结束后才读取两臂结果。"""
    baseline,memo,controller,output=map(Path,(baseline,memo,controller,output))
    if output.exists():raise FileExistsError('fresh public metrics required')
    if not all((p/'COMPLETE.json').is_file() for p in (baseline,memo)):
        raise ValueError('both complete receipts required before export')
    declared=json.loads(Path(protocol).read_text());manifest=json.loads(Path(source_manifest).read_text())
    root=Path(source_root).resolve();reader=load_reader(root,manifest)
    reader.validate_source(dict(source_root=str(root),source_manifest=str(source_manifest),
        source_manifest_sha256=sha(source_manifest),protocol=str(protocol),protocol_sha256=sha(protocol),
        threads=declared['resources']['requested_threads'],cpu_affinity=declared['resources']['candidate_cpu_affinity'],
        input_sha256=declared['input_sha256'],maximum_entries=declared['cache']['maximum_entries'],hemisphere=declared['hemisphere']))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fnit-prefix-export-',dir=output.parent) as temporary:
        checked=reader.compare_completed(baseline=baseline,candidate=memo,
            output=Path(temporary)/'check.private.json',source_manifest=source_manifest,protocol=protocol)
    original=json.loads((controller/'pair.private.json').read_text())
    if (checked['status']!='passed_fixed_initial3_prefix_pair' or len(checked['checks'])!=114
            or any(r['equal'] is not True for r in checked['checks'])
            or original['status']!=checked['status'] or original['checks']!=checked['checks']
            or original['reports_sha256']!=checked['reports_sha256']):
        raise ValueError('independent full114 gate disagrees with original receipt')
    cli=json.loads((controller/'cli.private.json').read_text())
    if ([r['arm'] for r in cli['records']]!=['baseline','memo'] or any(r['exit_code']!=0 for r in cli['records'])
            or cli['includes_imports_preflight_raw_preparation_nine_candidates_io_exit'] is not True
            or cli['no_other_arm_outputs_read_before_both_complete'] is not True):
        raise ValueError('complete sequential fresh-process CLI receipt required')
    arms={}
    prefix_clocks=('preparation_seconds','controller_with_output_seconds','summary_output_write_seconds',
                   'complete_prefix_api_and_output_seconds','source_controller_seconds','memo_release_seconds')
    slot_clocks=('complete_cycle_seconds','ordering_seconds','complete_scan_api_seconds','scanner_preparation_seconds',
                 'output_npz_control_write_seconds')
    for index,(arm,path) in enumerate((('baseline',baseline),('memo',memo))):
        r=json.loads((path/'report.private.json').read_text());runs={}
        for record in r['records']:
            slots={}
            for s in record['slots']:
                seconds={k:number(s[k]) for k in slot_clocks}
                seconds['retessellation_seconds']=number(s['retessellation_seconds'],nullable=True)
                for field,allowed in (('cycle_subseconds',{'candidate_complete','statistics_update',
                        'selective_full_restore','complete_cycle'}),('candidate_subseconds',{
                        'geometry_complete_api','displacement_complete_api','sff_complete_api',
                        'complete_prepared_five_api','complete_candidate_api_including_validation'})):
                    if set(s[field])!=allowed:raise ValueError('complete mature named child clocks required')
                    seconds[field]={k:number(s[field][k]) for k in sorted(allowed)}
                before,after=cache_public(s['cache_before']),cache_public(s['cache_after'])
                delta=None
                if after is not None:
                    keys=('logical_predicate_calls','cache_hits','actual_predicate_evaluations',
                          'inserted_unique_keys','capacity_bypass_evaluations')
                    delta={k:number(after[k]-before[k],integer=True) for k in keys}
                slots[str(s['slot'])]=dict(seconds=seconds,cache_before=before,cache_after=after,cache_delta=delta,
                    logical_predicate_calls=number(s['logical_predicate_calls'],integer=True),
                    ordered_predicate_bool_sha256=digest(s['sequence_sha256']) if record['run']=='audit' else None)
            runs[record['run']]=dict(seconds={k:number(record[k]) for k in prefix_clocks},slots=slots,
                actual_rng_draws=number(record['actual_rng_draws'],integer=True),
                full_rng_ordinal_bounds_value_sha256=digest(record['full_rng_stream_sha256']),
                canonical_sha256=digest(record['canonical_sha256']),base_context_state_sha256=digest(record['base_context_state_sha256']),
                actual_hooks_restored=True,memo_shared_across_slots=arm=='memo')
        arms[arm]=dict(runs=runs,complete_cold_cli_wall_seconds=number(cli['records'][index]['complete_cold_cli_wall_seconds']),
            stages_seconds={k:number(r['stages_seconds'][k]) for k in ('load_seconds','own_base_seconds','own_priors_edges_local_seconds')},
            peak_process_rss_bytes=number(r['peak_process_rss_bytes'],integer=True),
            input_sha256={k:digest(r['input_sha256'][k]) for k in ('nofix','qsphere','brain')},
            report_sha256=digest(checked['reports_sha256'][index]),actual_tool_sha256=digest(r['actual_tool_sha256']),
            actual_loaded_fnit_module_count=len(r['actual_loaded_fnit_sha256']),
            actual_loaded_benchmark_helper_count=len(r['actual_loaded_benchmark_tool_sources']),actual_torch_numba_restored=True)
    result=dict(status='passed_initial3_cross_candidate_diagnostic_not_full_ga',
        git_base=digest(declared['git_base'],length=40),hemisphere=declared['hemisphere'],source_manifest_sha256=digest(sha(source_manifest)),
        protocol_sha256=digest(sha(protocol)),producer_sha256=digest(sha(root/ENTRY)),exporter_sha256=digest(sha(__file__)),
        complete_comparison_groups=114,candidate_fields=36,state_fields=40,control_fields=6,ordering_fields=14,
        prefix_fields=14,artifact_count_per_arm=45,arms=arms,seed=1234,threads=2,
        strict_candidate_state_ordering_mutation_feedback_full_rng_and_audit_reproduction=True,
        cap_entries=250000,cache_shared_across_three_slots=True,cache_shared_between_repeats=False,
        comparison_recheck_seconds=number(checked['comparison_hash_npz_control_read_seconds']),
        gpu_exercised=False,allocated_bytes=None,reserved_bytes=None,production_wiring=False,default_changed=False,
        full_ga=False,full_495_search=False,official_new_comparison=False,whole_recon_all='not_assessed',
        final_mesh_quality='not_assessed',overall_equivalence='not_assessed',os_isolation_verified=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','memo','controller','source-root','source-manifest','protocol','output'):
        p.add_argument('--'+name,type=Path,required=True)
    export(**vars(p.parse_args()))


if __name__=='__main__':main()
