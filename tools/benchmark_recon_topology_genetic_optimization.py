"""完整首缺陷默认GA合并优化回归；旧赢家只在全部新结果落盘后读取。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

import benchmark_recon_topology_genetic_search as original_benchmark
from benchmark_recon_topology_candidate_state import _compare_value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def compare_npz(*, candidate, reference):
    """事后比较完整无pickle数组的字段、shape、dtype和全部原始字节。"""
    with np.load(candidate, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        same_keys = set(a.files) == set(b.files)
        fields = {key: _compare_value(a[key], b[key]) for key in sorted(set(a.files) & set(b.files))}
    return dict(same_keys=same_keys, fields=fields,
                all_fields_byte_equal=same_keys and all(v['byte_equal'] for v in fields.values()),
                complete_file_sha_equal=sha(candidate) == sha(reference))


def run(*, baseline_directory, source_manifest, protocol, **kwargs):
    """同输入完整默认GA→落盘→旧完整495候选/控制/末态严格比较。

    kwargs完全复用既有genetic_search.run，不更改随机流/停止/淘汰。
    baseline_directory仅在既有接口完整成功、所有新结果落盘后读取；
    protocol是事先声明输入SHA及默认政策，source_manifest绑定本次全部
    冻结源码与预期实际加载模块，不含用于算法的旧属性/赢家/参考场。
    成功产生OPTIMIZATION_COMPLETE.json；失败保留完整自产结果与失败
    回执。既有COMPLETE只表示旧首候选源门，不冒充本次优化配对完成。
    CPU四线程，无CUDA；这个完整首缺陷阶段仍不是全部缺陷或recon-all。
    """
    start = time.perf_counter()
    output = Path(kwargs['output'])
    declared = json.loads(Path(protocol).read_text())
    frozen = json.loads(Path(source_manifest).read_text())
    source_root = Path(source_manifest).parent
    if output.exists():
        raise FileExistsError('fresh optimized first-defect output required')
    requested = {name: kwargs[name] for name in ('threads', 'population_size',
        'max_unchanged', 'niters', 'seed', 'initial_selection', 'vertex_eliminate')}
    if requested != declared['policy']:
        raise ValueError('same predeclared complete default GA policy required')
    for name, key in (('nofix', 'input_surface'), ('qsphere', 'input_qsphere'), ('brain', 'brain')):
        if sha(kwargs[key]) != declared['input_sha256'][name]:
            raise ValueError('same original scientific input SHA required: ' + name)
    for name, record in frozen['payload'].items():
        p = source_root / name
        if p.stat().st_size != record['size'] or sha(p) != record['sha256']:
            raise ValueError('immutable source payload changed: ' + name)
    # Import/hash the complete original actual module set before algorithm input
    # loading. This verifies cloning/overlay resolution, not environment isolation.
    loaded_before = {}
    for name, expected in frozen['expected_module_sha256'].items():
        module = importlib.import_module(name)
        p = Path(module.__file__)
        actual = sha(p)
        if actual != expected:
            raise ValueError('unexpected actual module SHA before run: ' + name)
        loaded_before[name] = dict(sha256=actual, path=str(p))
    validated = time.perf_counter() - start
    marker_calls = 0
    marker = importlib.import_module('fnit.recon_all.place_surface_intersection_marking')
    original_marker = marker.mark_source_intersections
    def observe_marker(*args, **options):
        nonlocal marker_calls
        marker_calls += 1
        return original_marker(*args, **options)
    marker.mark_source_intersections = observe_marker
    try:
        tick = time.perf_counter()
        result = original_benchmark.run(**kwargs)
        complete_original_benchmark = time.perf_counter() - tick
    finally:
        marker.mark_source_intersections = original_marker
    write(output / 'optimization_source.private.json', dict(
        actual_before_module_sha256=loaded_before, source_manifest_sha256=sha(source_manifest),
        protocol_sha256=sha(protocol), candidate_intersection_marker_calls=marker_calls,
        validation_seconds=validated, original_complete_benchmark_seconds=complete_original_benchmark))
    # First access to the previous complete GA result. All new outputs exist now.
    tick = time.perf_counter()
    try:
        baseline_directory = Path(baseline_directory)
        receipt = json.loads((baseline_directory / 'COMPLETE.json').read_text())
        baseline_report_path = baseline_directory / 'report.json'
        if (receipt['report_sha256'] != sha(baseline_report_path)
                or sha(baseline_report_path) != declared['baseline_report_sha256']
                or receipt['status'] != 'complete_self_generated_first_defect_search'):
            raise ValueError('exact old complete GA receipt required for post-run comparison')
        baseline = json.loads(baseline_report_path.read_text())
        if baseline['input_sha256'] != result['input_sha256']:
            raise ValueError('baseline input identity differs')
        for name, digest in baseline['checkpoint_sha256'].items():
            if sha(baseline_directory / name) != digest:
                raise ValueError('old completed output SHA differs: ' + name)
        arrays = {name: compare_npz(candidate=output / name, reference=baseline_directory / name)
                  for name in ('first_candidate.private.npz', 'final_state.private.npz', 'search.private.npz')}
        control = json.loads((output / 'search_control.json').read_text())
        old_control = json.loads((baseline_directory / 'search_control.json').read_text())
        same_control = control == old_control
        scores = np.asarray([row['fitness'] for row in result['candidate_timings']], np.float64)
        old_scores = np.asarray([row['fitness'] for row in baseline['candidate_timings']], np.float64)
        score_comparison = _compare_value(scores, old_scores)
        summary_keys = ('seed', 'random_draws', 'actual_policy', 'search_counts', 'search_termination',
            'search_generations', 'best_fitness', 'actual_final_fitness', 'fitness_matches_best',
            'final_faces', 'final_additional_discarded', 'base_vertices', 'all_defects',
            'first_defect_edges', 'first_defect_vertices')
        summaries = {key: result[key] == baseline[key] for key in summary_keys}
        loaded_after = {name: sha(module.__file__) for name, module in sorted(sys.modules.items())
            if name.startswith('fnit.') and getattr(module, '__file__', None)
            and str(module.__file__).endswith('.py')}
        unchanged_loaded_sources = all(loaded_after.get(name) == row['sha256']
                                      for name, row in loaded_before.items())
        passed = bool(all(v['all_fields_byte_equal'] and v['complete_file_sha_equal'] for v in arrays.values())
            and score_comparison['byte_equal'] and same_control and all(summaries.values())
            and unchanged_loaded_sources and result['strict_first_source_candidate_passed'])
        report = dict(status='complete_optimized_first_default_GA' if passed else 'failed_optimized_first_default_GA',
            all_new_outputs_written_before_previous_winner_read=True, previous_outputs_used_in_algorithm=False,
            input_sha256=result['input_sha256'], source_manifest_sha256=sha(source_manifest),
            protocol_sha256=sha(protocol), loaded_module_sha256=loaded_after,
            unchanged_loaded_sources=unchanged_loaded_sources, complete_npz_comparisons=arrays,
            all_candidate_fitness_scores=score_comparison, exact_search_control=same_control,
            exact_summary=summaries, strict_optimization_pair_passed=passed,
            source_first_candidate_passed=result['strict_first_source_candidate_passed'],
            search_generations=result['search_generations'], search_evaluations=result['search_counts']['evaluated'],
            random_draws=result['random_draws'], candidate_marker_calls=marker_calls,
            source_marker_performance_effect='not_consumed_in_this_GA' if marker_calls == 0 else 'included',
            stage_seconds=result['stages'], nested_search_seconds=result['search_api_seconds'],
            original_benchmark_seconds=complete_original_benchmark, validation_seconds=validated,
            post_run_pair_comparison_seconds=time.perf_counter() - tick,
            full_optimization_API_before_report_seconds=time.perf_counter() - start,
            baseline_stage_seconds=baseline['stages'], baseline_nested_search_seconds=baseline['search_api_seconds'],
            current_cpu_affinity=result['cpu_affinity'], baseline_cpu_affinity=baseline['cpu_affinity'],
            threads=result['threads'], gpu_memory_bytes=None, allocated_bytes=None, reserved_bytes=None,
            whole_GA=False, whole_recon_all='not_assessed', overall_metric_equivalence='not_assessed',
            production_default_changed=False, environment_isolation='not_assessed')
        write(output / 'optimization.json', report)
        if not passed:
            raise RuntimeError('complete optimized default first GA differs; all outputs preserved')
        write(output / 'OPTIMIZATION_COMPLETE.json', dict(status=report['status'],
            optimization_report_sha256=sha(output / 'optimization.json'),
            original_report_sha256=sha(output / 'report.json'),
            complete_output_sha256={name: sha(output / name) for name in (*arrays, 'search_control.json')}))
        return report
    except Exception as error:
        write(output / 'optimization_failure.json', dict(type=type(error).__name__, message=str(error),
            all_new_outputs_preserved=True, whole_GA=False, whole_recon_all='not_assessed'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-surface', 'input-qsphere', 'brain', 'output', 'reference-statistics',
                 'reference-volume', 'reference-fitness', 'baseline-directory', 'source-manifest', 'protocol'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hemisphere', choices=('lh', 'rh'), required=True)
    for name in ('threads', 'population-size', 'max-unchanged', 'niters', 'seed'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--initial-selection', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vertex-eliminate', action=argparse.BooleanOptionalAction, default=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ('status', 'strict_optimization_pair_passed',
                     'search_generations', 'search_evaluations', 'candidate_marker_calls')}, indent=2))


if __name__ == '__main__':
    main()
