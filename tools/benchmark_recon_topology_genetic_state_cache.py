"""完整默认首缺陷GA的显式状态上下文回归；不修改生产调度。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import benchmark_recon_topology_genetic_optimization as complete_optimization


@contextmanager
def explicit_state_cache(*, enabled):
    """每次成熟prepare后创建一个上下文；全部评分和末态显式复用。

    enabled为必填bool。仅在这个串行benchmark作用域透明包装成熟
    prepare/cycle/final别名，退出或异常恢复全部函数身份。完整当前
    一环基底是工厂唯一输入，不缓存coords/K/E/H/curvbak/RP/rip。
    每个候选仍消费完整公共装配校验；跨域状态由成熟核心拒绝。
    返回仅计时、计数、基底SHA的audit；无文件读取、GPU或随机调用。
    """
    if not isinstance(enabled, bool):
        raise ValueError('explicit bool enabled required')
    audit = dict(preparations=[], materializations=[], final_score_calls=0,
                 final_write_calls=0)
    if not enabled:
        yield audit
        return
    from fnit.recon_all import topology_genetic_search as search
    from fnit.recon_all import topology_candidate_cycle as cycle
    from fnit.recon_all import topology_ga_finalize as final
    from fnit.recon_all.topology_candidate_state_cache import prepare_candidate_state_context
    original_prepare = search.prepare_genetic_defect_evaluator
    original_materialize = cycle.materialize_candidate_state
    original_final_score = final.evaluate_candidate_cycle
    original_final_write = final.materialize_candidate_state
    active = None

    def prepare(**options):
        nonlocal active
        value = original_prepare(**options)
        evaluator = value['evaluator']
        tick = time.perf_counter()
        active = prepare_candidate_state_context(surface_state=evaluator.surface_state)
        seconds = time.perf_counter()-tick
        evaluator.state_context = active
        audit['preparations'].append(dict(seconds=seconds,
            vertex_count=len(active.topology), logical_base_faces=len(active.faces),
            base_faces_sha256=hashlib.sha256(active.faces.tobytes()).hexdigest()))
        return dict(value, prepare_seconds=value['prepare_seconds']+seconds)

    def materialize(*, initial, result, state_context=None):
        if active is None or state_context is not active:
            raise ValueError('explicit current prepared context required in every materialization')
        tick = time.perf_counter()
        value = original_materialize(initial=initial, result=result, state_context=state_context)
        audit['materializations'].append(dict(seconds=time.perf_counter()-tick,
            actual_current_context=True, vertices=len(active.topology),
            initial_logical_faces=int(initial['nfaces']), candidate_logical_faces=int(value['nfaces'])))
        return value

    def final_score(**options):
        if active is None or 'state_context' in options:
            raise ValueError('one explicit prepared final score context required')
        audit['final_score_calls'] += 1
        return original_final_score(**options, state_context=active)

    def final_write(**options):
        if active is None or 'state_context' in options:
            raise ValueError('one explicit prepared final write context required')
        audit['final_write_calls'] += 1
        return materialize(**options, state_context=active)

    search.prepare_genetic_defect_evaluator = prepare
    cycle.materialize_candidate_state = materialize
    final.evaluate_candidate_cycle = final_score
    final.materialize_candidate_state = final_write
    try:
        yield audit
    finally:
        search.prepare_genetic_defect_evaluator = original_prepare
        cycle.materialize_candidate_state = original_materialize
        final.evaluate_candidate_cycle = original_final_score
        final.materialize_candidate_state = original_final_write


def run(*, state_cache, **options):
    """完整自产默认495评分+最终评分/写回→旧完整结果事后比较。

    state_cache为必填bool，options完整传入已验证optimization.run。
    新产物全部落盘后才读取baseline；默认10/10/-1、seed1234由事先
    protocol校验。构造/JIT/加载/完整校验/读写均计入API和外部CLI。
    成功额外写STATE_CACHE_COMPLETE绑定旧完整门；失败保留新结果。
    不改变RNG、剔除/代际/终止、全局精度、其他缓存或生产默认。
    """
    if not isinstance(state_cache, bool):
        raise ValueError('explicit bool state_cache required')
    output = Path(options['output'])
    started = time.perf_counter()
    with explicit_state_cache(enabled=state_cache) as audit:
        result = complete_optimization.run(**options)
    control = json.loads((output/'search_control.json').read_text())
    trace = control['generation_trace']
    generations = [row['generation'] for row in trace]
    context_passed = (len(audit['preparations']) == 1
        and len(audit['materializations']) == result['search_evaluations']+2
        and audit['final_score_calls'] == 1 and audit['final_write_calls'] == 1
        and all(row['actual_current_context'] for row in audit['materializations'])) if state_cache else (
        not audit['preparations'] and not audit['materializations']
        and audit['final_score_calls'] == 0 and audit['final_write_calls'] == 0)
    passed = bool(result['strict_optimization_pair_passed'] and context_passed
        and generations == list(range(len(trace)))
        and trace[-1]['generation'] == result['search_generations'])
    report = dict(status='complete_default_first_GA_state_cache' if passed else 'failed_first_GA_state_cache',
        strict_complete_pair_passed=passed, state_cache=state_cache, **audit,
        context_preparations=len(audit['preparations']), materialization_count=len(audit['materializations']),
        generation_trace_count=len(trace), last_zero_based_generation_index=trace[-1]['generation'],
        search_evaluations=result['search_evaluations'], random_draws=result['random_draws'],
        complete_wrapper_seconds=time.perf_counter()-started,
        actual_tool_sha256=complete_optimization.sha(__file__),
        delegate_tool_sha256=complete_optimization.sha(complete_optimization.__file__),
        optimization_report_sha256=complete_optimization.sha(output/'optimization.json'),
        inherited_complete_sha256=complete_optimization.sha(output/'OPTIMIZATION_COMPLETE.json'),
        production_default_changed=False, whole_GA=False, whole_recon_all='not_assessed',
        overall_metric_equivalence='not_assessed')
    complete_optimization.write(output/'state_cache.json', report)
    if not passed:
        raise RuntimeError('complete first GA state context gate failed; all outputs retained')
    complete_optimization.write(output/'STATE_CACHE_COMPLETE.json', dict(status=report['status'],
        state_cache_report_sha256=complete_optimization.sha(output/'state_cache.json'),
        inherited_complete_sha256=report['inherited_complete_sha256']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface', 'input-qsphere', 'brain', 'output', 'reference-statistics',
                 'reference-volume', 'reference-fitness', 'baseline-directory', 'source-manifest', 'protocol'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--state-cache', action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument('--hemisphere', choices=('lh','rh'), required=True)
    for name in ('threads','population-size','max-unchanged','niters','seed'):
        parser.add_argument('--'+name,type=int,required=True)
    parser.add_argument('--initial-selection',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--vertex-eliminate',action=argparse.BooleanOptionalAction,default=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key:result[key] for key in ('status','strict_complete_pair_passed',
        'search_evaluations','generation_trace_count','context_preparations','materialization_count')},indent=2))


if __name__=='__main__':
    main()
