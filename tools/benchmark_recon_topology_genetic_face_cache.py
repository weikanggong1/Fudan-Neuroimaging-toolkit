"""完整默认首缺陷GA的显式保留面缓存回归；只复用成熟完整搜索。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import benchmark_recon_topology_genetic_optimization as complete_optimization


def run(*, face_detection_cache, **options):
    """完整同输入首缺陷→缓存一次准备→全部评分/写出→旧完整结果比较。

    face_detection_cache必须显式bool；其余具名参数完整传给既有
    optimization.run。缓存只在实际自产evaluator构造完成后，依据它的
    全base_faces和顶点数创建一次，追加到retessellation_inputs；不改
    geometry/cycle、随机流、MRI先验、接受/拒绝、恢复或停止。关闭时
    不准备上下文。工厂和检测器观察仅各forward一次，局部函数在异常
    和成功后均恢复。计时包括工厂；每次动态面判定的有序报告SHA单列。

    先完成所有自产候选和写出再复用旧完整评分、控制和80字段门。
    另写FACE_CACHE_COMPLETE.json绑定继承完成门及本次缓存调用报告；
    它不是全缺陷/整例或原版完整赢家验收。所有输出要求新目录，
    不更改生产默认、父环境、CUDA策略或算法输入；失败保留诊断。
    """
    if not isinstance(face_detection_cache, bool):
        raise ValueError('explicit bool face_detection_cache required')
    from fnit.recon_all import topology_genetic_search as search
    from fnit.recon_all import topology_face_detection_cache as detector
    original_prepare = search.prepare_genetic_defect_evaluator
    original_detect = detector.detect_patch_faces_cached
    preparations, calls = [], []
    contexts = []

    def prepare(**kwargs):
        value = original_prepare(**kwargs)
        if face_detection_cache:
            r = value['evaluator'].retessellation_inputs
            tick = time.perf_counter()
            context = detector.prepare_face_detection_context(
                base_faces=r['base_faces'], vertex_count=len(r['canonical']))
            seconds = time.perf_counter() - tick
            r['face_detection_context'] = context
            contexts.append(context)
            preparations.append(dict(seconds=seconds,
                base_faces=len(context.faces), vertex_count=context.vertex_count,
                base_faces_sha256=context.faces_sha256))
            # 原prepare返回的时间不含新增工厂；显式计入本次prepare总耗时。
            value = dict(value, prepare_seconds=value['prepare_seconds'] + seconds)
        return value

    def detect(**kwargs):
        tick = time.perf_counter()
        result = original_detect(**kwargs)
        seconds = time.perf_counter() - tick
        context = kwargs['context']
        decision_bytes = json.dumps(result[1], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()
        calls.append(dict(index=len(calls), seconds=seconds,
            actual_prepared_context=any(context is item for item in contexts),
            base_faces_sha256=context.faces_sha256,
            modified_vertices=len(kwargs['vertices']), accepted_faces=len(result[0]),
            complete_ordered_decisions_sha256=hashlib.sha256(decision_bytes).hexdigest()))
        return result

    search.prepare_genetic_defect_evaluator = prepare
    detector.detect_patch_faces_cached = detect
    output = Path(options['output'])
    started = time.perf_counter()
    try:
        result = complete_optimization.run(**options)
    finally:
        search.prepare_genetic_defect_evaluator = original_prepare
        detector.detect_patch_faces_cached = original_detect
    control = json.loads((output / 'search_control.json').read_text())
    trace = control['generation_trace']
    generations = [row['generation'] for row in trace]
    context_passed = (len(preparations) == 1 and len(calls) >= result['search_evaluations'] + 1
        and all(row['actual_prepared_context'] for row in calls)) if face_detection_cache else (
        len(preparations) == 0 and len(calls) == 0)
    passed = bool(result['strict_optimization_pair_passed'] and context_passed
        and generations == list(range(len(trace)))
        and trace[-1]['generation'] == result['search_generations'])
    report = dict(status='complete_default_first_GA_face_cache' if passed else 'failed_first_GA_face_cache',
        strict_complete_pair_passed=passed, face_detection_cache=face_detection_cache,
        preparations=preparations, detector_calls=calls, context_preparations=len(preparations),
        detector_call_count=len(calls), generation_trace_count=len(trace),
        last_zero_based_generation_index=trace[-1]['generation'],
        search_evaluations=result['search_evaluations'], random_draws=result['random_draws'],
        complete_wrapper_seconds=time.perf_counter()-started,
        actual_tool_sha256=complete_optimization.sha(__file__),
        delegate_tool_sha256=complete_optimization.sha(complete_optimization.__file__),
        helper_sha256=complete_optimization.sha(detector.__file__),
        optimization_report_sha256=complete_optimization.sha(output / 'optimization.json'),
        inherited_complete_sha256=complete_optimization.sha(output / 'OPTIMIZATION_COMPLETE.json'),
        production_default_changed=False, whole_GA=False, whole_recon_all='not_assessed',
        overall_metric_equivalence='not_assessed')
    complete_optimization.write(output / 'face_cache.json', report)
    if not passed:
        raise RuntimeError('complete first GA face cache gate failed; all outputs retained')
    complete_optimization.write(output / 'FACE_CACHE_COMPLETE.json', dict(status=report['status'],
        face_cache_report_sha256=complete_optimization.sha(output / 'face_cache.json'),
        inherited_complete_sha256=report['inherited_complete_sha256']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface', 'input-qsphere', 'brain', 'output', 'reference-statistics',
                 'reference-volume', 'reference-fitness', 'baseline-directory', 'source-manifest', 'protocol'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--face-detection-cache', action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument('--hemisphere', choices=('lh', 'rh'), required=True)
    for name in ('threads', 'population-size', 'max-unchanged', 'niters', 'seed'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--initial-selection', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vertex-eliminate', action=argparse.BooleanOptionalAction, default=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ('status', 'strict_complete_pair_passed',
        'generation_trace_count', 'search_evaluations', 'context_preparations', 'detector_call_count')}, indent=2))


if __name__ == '__main__':
    main()
