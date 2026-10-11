"""保留原raw比较失败，只离线重核prefix JSON对象键序及原111项科学门。"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import time

FROZEN_PRODUCER_SHA256 = '029a8bc613aa0b74f3f205ca113090953d5e7a8053788719bbac8a86be68ba1f'
ENTRY = 'tools/benchmark_recon_topology_edge_predicate_population_prefix.py'
EXPORTER = 'validation/recon_all/optimizations/20261011_topology_edge_predicate_population_prefix/export_metrics.py'
FIELDS = frozenset(('evaluation_trace','generation_trace','termination','counts',
    'final_tessellation_complete','complete_ga','overall_metric_equivalence'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON object key forbidden')
        result[key]=value
    return result


def _reject_constant(value):raise ValueError('nonfinite JSON constant forbidden')


def read_control(path):
    value=json.loads(Path(path).read_text(),object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)
    if type(value) is not dict or set(value)!=FIELDS:raise ValueError('complete seven-field prefix object required')
    return value


def strict_tree_equal(left,right):
    """对象键序无语义；完整类型/键/数组顺序保持，浮点按F64位比较含负零。"""
    if type(left) is not type(right):return False
    if type(left) is dict:
        return set(left)==set(right) and all(strict_tree_equal(left[key],right[key]) for key in left)
    if type(left) is list:
        return len(left)==len(right) and all(strict_tree_equal(a,b) for a,b in zip(left,right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and struct.pack('>d',left)==struct.pack('>d',right)
    if left is None or type(left) in (str,bool,int):return left==right
    raise ValueError('declared JSON leaf type required')


def canonical_prefix_equal(left,right):
    a,b=read_control(left),read_control(right)
    if not strict_tree_equal(a,b):return False
    # 额外保留完整规范化字节门；不转换标签、浮点或列表排序。
    return json.dumps(a,sort_keys=True,indent=2,allow_nan=False)==json.dumps(b,sort_keys=True,indent=2,allow_nan=False)


def clone_comparator(*, reader, producer):
    """仅克隆完整旧比较器的一项prefix JSON表达式；原模块/源码不变。"""
    producer=Path(producer)
    if sha(producer)!=FROZEN_PRODUCER_SHA256:raise ValueError('actual frozen original producer required')
    tree=ast.parse(producer.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compare_completed']
    if len(nodes)!=1:raise ValueError('one original full comparator required')
    function=copy.deepcopy(nodes[0]);replaced=0
    for node in ast.walk(function):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='dict':continue
        keywords={v.arg:v for v in node.keywords}
        kind=keywords.get('kind');equal=keywords.get('equal')
        if kind is None or not isinstance(kind.value,ast.Constant) or kind.value.value!='prefix_control':continue
        if equal is None or not isinstance(equal.value,ast.Compare) or len(equal.value.ops)!=1 or not isinstance(equal.value.ops[0],ast.Eq):
            raise ValueError('original prefix raw-byte equality expression required')
        values=[equal.value.left,*equal.value.comparators]
        if len(values)!=2 or any(not isinstance(v,ast.Call) or v.args or v.keywords
                or not isinstance(v.func,ast.Attribute) or v.func.attr!='read_bytes' for v in values):
            raise ValueError('original prefix read_bytes pair required')
        equal.value=ast.Call(func=ast.Name(id='_canonical_prefix_equal',ctx=ast.Load()),
            args=[v.func.value for v in values],keywords=[])
        replaced+=1
    if replaced!=1:raise ValueError('exactly one prefix-only format expression replacement required')
    namespace=dict(vars(reader));namespace['_canonical_prefix_equal']=canonical_prefix_equal
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module,str(Path(__file__).resolve()),'exec'),namespace)
    return namespace['compare_completed']


def original_failure_gate(original):
    if original.get('status')!='failed_fixed_initial3_prefix_pair' or len(original.get('checks',[]))!=114:
        raise ValueError('original complete failed114 receipt required')
    failed=[row for row in original['checks'] if row.get('equal') is not True]
    expected=[dict(scope='cross_arm',run=run,kind='prefix_control',equal=False) for run in ('cold','warm','audit')]
    if failed!=expected:raise ValueError('only three proven prefix-format failures eligible for offline recheck')


def export(*, baseline, memo, controller, source_root, source_manifest, protocol, output):
    """同一冻结产物重核，不进行候选评分、不覆盖原始失败收据。"""
    tick=time.perf_counter();baseline,memo,controller,root,output=map(Path,(baseline,memo,controller,source_root,output))
    if output.exists():raise FileExistsError('fresh offline recheck output required')
    manifest=json.loads(Path(source_manifest).read_text());declared=json.loads(Path(protocol).read_text())
    actual=root/EXPORTER
    if sha(actual)!=manifest['files'][EXPORTER]['sha256']:raise ValueError('actual frozen field exporter binding required')
    spec=importlib.util.spec_from_file_location('frozen_prefix_field_exporter',actual)
    fields=importlib.util.module_from_spec(spec);spec.loader.exec_module(fields)
    reader=fields.load_reader(root.resolve(),manifest)
    original=json.loads((controller/'pair.private.json').read_text());original_failure_gate(original)
    original_function=reader.compare_completed
    rechecker=clone_comparator(reader=reader,producer=root/ENTRY)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prefix-object-recheck-',dir=output.parent) as temporary:
        checked=rechecker(baseline=baseline,candidate=memo,output=Path(temporary)/'recheck.private.json',
            source_manifest=source_manifest,protocol=protocol)
    if (reader.compare_completed is not original_function or checked['reports_sha256']!=original['reports_sha256']
            or checked['status']!='passed_fixed_initial3_prefix_pair' or len(checked['checks'])!=114):
        raise ValueError('complete source-bound independent recheck required')
    for left,right in zip(original['checks'],checked['checks']):
        expected=dict(left)
        if left['equal'] is False:expected['equal']=True
        if right!=expected:raise ValueError('any other original scientific check changed')
    prefixes=[]
    for run in ('cold','warm','audit'):
        paths=[path/(run+'.prefix_control.private.json') for path in (baseline,memo)]
        data=[read_control(path) for path in paths]
        canonical=[(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode() for value in data]
        prefixes.append(dict(run=run,full_field_count=7,all_recursive_JSON_types_and_F64_bits_equal=strict_tree_equal(*data),
            raw_bytes_equal=paths[0].read_bytes()==paths[1].read_bytes(),raw_sha256=[sha(path) for path in paths],
            original_key_order=[list(value) for value in data],canonical_bytes_equal=canonical[0]==canonical[1],
            canonical_sha256=[hashlib.sha256(value).hexdigest() for value in canonical]))
    cli=json.loads((controller/'cli.private.json').read_text())
    if ([row['arm'] for row in cli['records']]!=['baseline','memo'] or any(row['exit_code']!=0 for row in cli['records'])
            or cli['no_other_arm_outputs_read_before_both_complete'] is not True
            or cli['includes_imports_preflight_raw_preparation_nine_candidates_io_exit'] is not True):
        raise ValueError('both fresh scientific arm exit receipts required')
    arms={}
    prefix_clocks=('preparation_seconds','controller_with_output_seconds','summary_output_write_seconds',
        'complete_prefix_api_and_output_seconds','source_controller_seconds','memo_release_seconds')
    slot_clocks=('complete_cycle_seconds','ordering_seconds','complete_scan_api_seconds','scanner_preparation_seconds','output_npz_control_write_seconds')
    for index,(arm,path) in enumerate((('baseline',baseline),('memo',memo))):
        report=json.loads((path/'report.private.json').read_text());runs={}
        for record in report['records']:
            slots={}
            for slot in record['slots']:
                before,after=fields.cache_public(slot['cache_before']),fields.cache_public(slot['cache_after'])
                delta=None
                if after is not None:
                    delta={key:fields.number(after[key]-before[key],integer=True) for key in ('logical_predicate_calls',
                        'cache_hits','actual_predicate_evaluations','inserted_unique_keys','capacity_bypass_evaluations')}
                seconds={key:fields.number(slot[key]) for key in slot_clocks}
                seconds['retessellation_seconds']=fields.number(slot['retessellation_seconds'],nullable=True)
                for key,expected in (('cycle_subseconds',{'candidate_complete','statistics_update','selective_full_restore','complete_cycle'}),
                    ('candidate_subseconds',{'geometry_complete_api','displacement_complete_api','sff_complete_api','complete_prepared_five_api','complete_candidate_api_including_validation'})):
                    if set(slot[key])!=expected:raise ValueError('complete named mature clock fields required')
                    seconds[key]={name:fields.number(slot[key][name]) for name in sorted(expected)}
                slots[str(slot['slot'])]=dict(seconds=seconds,cache_before=before,cache_after=after,cache_delta=delta,
                    logical_predicate_calls=fields.number(slot['logical_predicate_calls'],integer=True),
                    ordered_predicate_bool_sha256=fields.digest(slot['sequence_sha256']) if record['run']=='audit' else None)
            runs[record['run']]=dict(seconds={key:fields.number(record[key]) for key in prefix_clocks},slots=slots,
                actual_rng_draws=fields.number(record['actual_rng_draws'],integer=True),
                full_rng_ordinal_bounds_value_sha256=fields.digest(record['full_rng_stream_sha256']),
                canonical_sha256=fields.digest(record['canonical_sha256']),base_context_state_sha256=fields.digest(record['base_context_state_sha256']),
                actual_hooks_restored=True,memo_shared_across_slots=arm=='memo')
        arms[arm]=dict(runs=runs,complete_cold_cli_wall_seconds=fields.number(cli['records'][index]['complete_cold_cli_wall_seconds']),
            stages_seconds={key:fields.number(report['stages_seconds'][key]) for key in ('load_seconds','own_base_seconds','own_priors_edges_local_seconds')},
            peak_process_rss_bytes=fields.number(report['peak_process_rss_bytes'],integer=True),
            input_sha256={key:fields.digest(report['input_sha256'][key]) for key in ('nofix','qsphere','brain')},
            report_sha256=fields.digest(checked['reports_sha256'][index]),actual_tool_sha256=fields.digest(report['actual_tool_sha256']),
            actual_loaded_fnit_module_count=len(report['actual_loaded_fnit_sha256']),
            actual_loaded_benchmark_helper_count=len(report['actual_loaded_benchmark_tool_sources']),actual_torch_numba_restored=True)
    answer=dict(status='offline_prefix_object_recheck_passed_original_raw114_failed',
        git_base=declared['git_base'],hemisphere='lh',source_manifest_sha256=sha(source_manifest),protocol_sha256=sha(protocol),
        original_producer_sha256=sha(root/ENTRY),original_field_exporter_sha256=sha(actual),rechecker_sha256=sha(__file__),
        original_comparison_sha256=sha(controller/'pair.private.json'),original_cli_receipt_sha256=sha(controller/'cli.private.json'),
        original_comparison_group_count=114,original_raw_failed_groups=3,original_unchanged_passed_groups=111,
        offline_canonical_prefix_object_groups=3,all_original_111_scientific_gates_preserved=True,
        strict_raw_file_reproduction=False,optimization_introduced_state_control_full_RNG_or_audit_degradation=False,
        prefixes=prefixes,arms=arms,seed=1234,threads=2,cap_entries=250000,artifact_count_per_arm=45,
        candidate_fields=36,state_fields=40,control_fields=6,ordering_fields=14,prefix_fields=14,
        cache_shared_across_three_slots=True,cache_shared_between_repeats=False,
        comparison_recheck_seconds=fields.number(checked['comparison_hash_npz_control_read_seconds']),
        complete_offline_export_seconds=time.perf_counter()-tick,scientific_calculation_rerun=False,
        frozen_source_modified=False,original_failure_overwritten=False,global_comparator_function_replaced=False,
        gpu_exercised=False,allocated_bytes=None,reserved_bytes=None,production_wiring=False,default_changed=False,
        full_ga=False,full_495_search=False,official_new_comparison=False,whole_recon_all='not_assessed',
        final_mesh_quality='not_assessed',overall_equivalence='not_assessed',os_isolation_verified=False)
    output.write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n');return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','memo','controller','source-root','source-manifest','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    export(**vars(parser.parse_args()))


if __name__=='__main__':main()
