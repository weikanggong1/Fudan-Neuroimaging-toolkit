"""完整默认首GA显式编译阻挡扫描；沿用既有完整face-cache/优化验收。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import benchmark_recon_topology_genetic_face_cache as complete_face_cache
from benchmark_recon_topology_genetic_optimization import sha,write


def run(*,blocker_scan_backend,**options):
    """原始三输入→全部默认首缺陷→保存→完整旧评分/状态控制比较。

    options完全复用benchmark_recon_topology_genetic_optimization.run的
    必填具名参数及冻结policy。blocker_scan_backend仅python或numba，
    透明prepare观测在既有prepare返回后向本次evaluator自身retess参数
    写入显式backend。循环和finalizer共用本次dict，不改全局算法默认。
    成熟face_detection_context仍每缺陷一次，不混state_context。
    original prepare/retess aliases总在finally恢复；每次只forward一次。
    逐次观测实际backend、完整首阻挡控制trace SHA与调用数；没有
    序列审计缓冲，不改变随机draw或扫描范围。参考/旧赢家只在继承
    完整工具保存全部结果后读取。成功须完整495+final、497重连、
    原全部80fields/3NPZ/29trace/RNG门，不是九缺陷或recon-all。
    """
    if blocker_scan_backend not in ('python','numba'):
        raise ValueError('explicit python or numba blocker_scan_backend required')
    from fnit.recon_all import topology_genetic_search as search
    from fnit.recon_all import topology_retessellation_dynamic as dynamic
    original_prepare=search.prepare_genetic_defect_evaluator
    original_retess=dynamic.retessellate_ordered_edges
    prepared=[];calls=[];started=time.perf_counter()
    def prepare(**kwargs):
        value=original_prepare(**kwargs)
        value['evaluator'].retessellation_inputs['blocker_scan_backend']=blocker_scan_backend
        prepared.append(value['evaluator'].retessellation_inputs)
        return value
    def observe_retess(**kwargs):
        value=original_retess(**kwargs)
        trace=json.dumps(value['retessellation_trace'],sort_keys=True,separators=(',',':')).encode()
        calls.append(dict(backend=value['blocker_scan_backend'],predicate_calls=value['blocker_predicate_calls'],
            scanner_calls=value['blocker_scan_calls'],scanner_preparation_seconds=value['blocker_scan_preparation_seconds'],
            retessellation_trace_sha256=hashlib.sha256(trace).hexdigest(),rounds=value['rounds']))
        with (Path(options['output'])/'blocker_calls.private.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(index=len(calls)-1,**calls[-1]),allow_nan=False)+'\n')
        return value
    search.prepare_genetic_defect_evaluator=prepare
    dynamic.retessellate_ordered_edges=observe_retess
    try:
        result=complete_face_cache.run(face_detection_cache=True,**options)
    finally:
        search.prepare_genetic_defect_evaluator=original_prepare
        dynamic.retessellate_ordered_edges=original_retess
    output=Path(options['output'])
    search_control=json.loads((output/'search_control.json').read_text())
    trace=search_control.get('generation_trace',[])
    passed=bool(result['strict_complete_pair_passed'] and len(prepared)==1 and len(calls)==497
        and all(row['backend']==blocker_scan_backend for row in calls)
        and options['population_size']==10 and options['niters']==-1
        and result['search_evaluations']==495 and len(trace)==29)
    report=dict(status='complete_default_first_GA_blocker_scan'if passed else'failed_default_first_GA_blocker_scan',
        strict_pair_passed=passed,blocker_scan_backend=blocker_scan_backend,prepared_evaluators=len(prepared),
        retessellation_calls=len(calls),retessellation_records=calls,
        inherited_face_cache_report_sha256=sha(output/'face_cache.json'),
        inherited_optimization_report_sha256=sha(output/'optimization.json'),
        actual_tool_sha256=sha(__file__),dynamic_sha256=sha(dynamic.__file__),
        helper_sha256=sha(Path(dynamic.__file__).with_name('topology_edge_blocker_scan.py')),
        complete_API_before_report_seconds=time.perf_counter()-started,
        search_evaluations=result['search_evaluations'],final_scores=1,
        generation_trace_count=len(trace),last_zero_based_generation_index=result['last_zero_based_generation_index'],
        random_draws=result['random_draws'],whole_GA=False,whole_recon_all='not_assessed',
        production_default_changed=False,state_context_enabled=False,sequence_audit_enabled=False)
    write(output/'blocker_scan.json',report)
    if not passed:raise RuntimeError('complete blocker scan GA gate failed; all output retained')
    write(output/'BLOCKER_SCAN_COMPLETE.json',dict(status=report['status'],
        report_sha256=sha(output/'blocker_scan.json'),inherited_complete_sha256=sha(output/'FACE_CACHE_COMPLETE.json')))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface','input-qsphere','brain','output','reference-statistics','reference-volume',
            'reference-fitness','baseline-directory','source-manifest','protocol'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hemisphere',choices=('lh','rh'),required=True)
    for name in ('threads','population-size','max-unchanged','niters','seed'):
        parser.add_argument('--'+name,type=int,required=True)
    parser.add_argument('--initial-selection',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--vertex-eliminate',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--blocker-scan-backend',choices=('python','numba'),required=True)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({key:result[key]for key in ('status','strict_pair_passed','retessellation_calls')}))


if __name__=='__main__':main()
