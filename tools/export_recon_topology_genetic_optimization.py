"""完整首缺陷GA优化的指标白名单导出；原始私有收据和影像留在服务器。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from export_recon_gca_receipts import numeric_leaves


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_metrics(*, input_run, baseline_run, frozen_source, output_directory):
    """新完整回归/旧完整回归/冻结源→公开指标JSON，四参均必填。

    input_run、baseline_run是包含candidate和process.private.json的两个
    实际完整运行目录；frozen_source含本轮OVERLAY_SHA，只导出
    明确安全字段和原SHA。output_directory必须
    不存在。不读取MRI/NPZ，不发布地址、用户名、主机、命令、路径或
    源码清单。每个被选指标在JSON往返后必须保留数值及Python类型。
    完整495搜索+final配对门失败时不输出成功白名单；旧报告不改写。
    """
    input_run, baseline_run, frozen_source = map(Path, (input_run, baseline_run, frozen_source))
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError('fresh public metrics directory required')
    rpath = input_run / 'candidate/optimization.json'
    report = json.loads(rpath.read_text())
    complete_path = input_run / 'candidate/OPTIMIZATION_COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    process_path = input_run / 'process.private.json'
    process = json.loads(process_path.read_text())
    if (process['exit_code'] != 0 or not report['strict_optimization_pair_passed']
            or report['status'] != 'complete_optimized_first_default_GA'
            or complete['optimization_report_sha256'] != sha(rpath)
            or complete['status'] != report['status']):
        raise ValueError('completed full first default GA optimization gate required')
    for name, digest in complete['complete_output_sha256'].items():
        if sha(input_run / 'candidate' / name) != digest:
            raise ValueError('new completed output SHA changed')
    stage_path = input_run / 'candidate/report.json'
    stage = json.loads(stage_path.read_text())
    old_stage_path = baseline_run / 'candidate/report.json'
    old_stage = json.loads(old_stage_path.read_text())
    old_process_path = baseline_run / 'process.private.json'
    old_process = json.loads(old_process_path.read_text())
    protocol_path = frozen_source / 'protocol.private.json'
    protocol = json.loads(protocol_path.read_text())
    baseline_complete = json.loads((baseline_run / 'candidate/COMPLETE.json').read_text())
    if (sha(protocol_path) != report['protocol_sha256']
            or sha(frozen_source / 'SOURCE_MANIFEST.private.json') != report['source_manifest_sha256']
            or process['source_manifest_sha256'] != report['source_manifest_sha256']):
        raise ValueError('same measured immutable protocol and source manifest required')
    if (sha(old_stage_path) != protocol['baseline_report_sha256']
            or baseline_complete['report_sha256'] != sha(old_stage_path)
            or baseline_complete['status'] != old_stage['status']
            or old_process['exit_code'] != 0):
        raise ValueError('same completed baseline binding required; no alternate timing directory')
    if complete['original_report_sha256'] != sha(stage_path):
        raise ValueError('completed stage report changed')
    safe_keys = ('status', 'all_new_outputs_written_before_previous_winner_read',
        'previous_outputs_used_in_algorithm', 'input_sha256', 'source_manifest_sha256',
        'protocol_sha256', 'unchanged_loaded_sources', 'all_candidate_fitness_scores',
        'exact_search_control', 'exact_summary', 'strict_optimization_pair_passed',
        'source_first_candidate_passed', 'search_generations', 'search_evaluations',
        'random_draws', 'candidate_marker_calls', 'source_marker_performance_effect',
        'stage_seconds', 'nested_search_seconds', 'original_benchmark_seconds',
        'validation_seconds', 'post_run_pair_comparison_seconds',
        'full_optimization_API_before_report_seconds', 'baseline_stage_seconds',
        'baseline_nested_search_seconds', 'current_cpu_affinity', 'baseline_cpu_affinity',
        'threads', 'gpu_memory_bytes', 'allocated_bytes', 'reserved_bytes', 'whole_GA',
        'whole_recon_all', 'overall_metric_equivalence', 'production_default_changed',
        'environment_isolation')
    metrics = {key: report[key] for key in safe_keys}
    control = json.loads((input_run / 'candidate/search_control.json').read_text())
    generation_trace = control['generation_trace']
    if (not generation_trace
            or [row['generation'] for row in generation_trace] != list(range(len(generation_trace)))
            or generation_trace[-1]['generation'] != report['search_generations']):
        raise ValueError('complete zero-based generation trace required')
    # 继承字段generations是末次零基索引；保留原数值并明确完整记录数。
    metrics['generation_trace_count'] = len(generation_trace)
    metrics['last_zero_based_generation_index'] = generation_trace[-1]['generation']
    metrics['legacy_search_generations_semantics'] = 'last zero-based generation index; not executed generation count'
    metrics['complete_state_comparisons'] = {
        name.removesuffix('.private.npz'): fields
        for name, fields in report['complete_npz_comparisons'].items()}
    metrics['execution'] = {key: process[key] for key in ('exit_code',
        'complete_fresh_cli_seconds', 'driver_seconds_before_receipt', 'preflight',
        'threads', 'cpu_affinity', 'target_cuda', 'gpu_memory_bytes', 'allocated_bytes',
        'reserved_bytes', 'child_peak_rss_kib', 'sampler_interval_seconds')}
    metrics['baseline_execution'] = {key: old_process[key] for key in ('exit_code',
        'complete_fresh_cli_seconds', 'preflight', 'threads', 'cpu_affinity',
        'target_cuda', 'gpu_memory_bytes', 'child_peak_rss_kib')}
    metrics['first_source_checks'] = stage['first_source_checks']
    metrics['search_counts'] = stage['search_counts']
    metrics['search_termination'] = stage['search_termination']
    metrics['best_fitness'] = stage['best_fitness']
    metrics['actual_final_fitness'] = stage['actual_final_fitness']
    metrics['outputs_sha256'] = {k.removesuffix('.private.npz'): v for k, v in stage['checkpoint_sha256'].items()}
    metrics['baseline_outputs_sha256'] = {k.removesuffix('.private.npz'): v for k, v in old_stage['checkpoint_sha256'].items()}
    overlay = json.loads((frozen_source / 'OVERLAY_SHA.json').read_text())
    metrics['six_mature_optimization_module_sha256'] = {
        Path(name).stem: row['sha256'] for name, row in overlay['payload'].items()
        if name.startswith('src/')}
    metrics['binding'] = dict(source_report_sha256=sha(rpath),
        source_process_sha256=sha(process_path), source_stage_report_sha256=sha(stage_path),
        baseline_process_sha256=sha(old_process_path), baseline_report_sha256=sha(old_stage_path),
        new_tool_sha256=overlay['payload']['tools/benchmark_recon_topology_genetic_optimization.py']['sha256'],
        driver_sha256=process['driver_sha256'], exporter_sha256=sha(__file__),
        load_curve_sha256=process['load_curve_sha256'])
    affinity_note = ('same CPU affinity' if metrics['current_cpu_affinity'] == metrics['baseline_cpu_affinity']
                     else 'different CPU affinity')
    metrics['timing_interpretation'] = ('same four-thread budget; '+affinity_note+
        ' and shared cross-time load; no controlled ABBA throughput claim')
    scores = dict(candidate=stage['candidate_timings'], baseline=old_stage['candidate_timings'],
                 scores_all_byte_equal=report['all_candidate_fitness_scores']['byte_equal'])
    rows = [json.loads(line) for line in (input_run / 'load_curve.private.jsonl').read_text().splitlines()]
    curve = []
    for row in rows:
        parsed = row['loadavg'].split() if row['loadavg'] else []
        curve.append(dict(seconds=row['seconds'], load_1m=float(parsed[0]) if parsed else None,
            load_5m=float(parsed[1]) if parsed else None, load_15m=float(parsed[2]) if parsed else None,
            runnable=int(parsed[3].split('/')[0]) if parsed else None,
            cgroup_available={key: value is not None for key, value in row['cgroup'].items()}))
    selected = {'metrics.json': metrics, 'candidate_timings.json': scores,
                'shared_load.json': dict(interval_seconds=process['sampler_interval_seconds'], samples=curve,
                    exact_parent_peak_memory_bytes=None, past_baseline_load_trace='not_available')}
    encoded = {}
    preserved = 0
    for name, value in selected.items():
        data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        before, after = numeric_leaves(value), numeric_leaves(json.loads(data))
        if [(type(v), v) for v in before] != [(type(v), v) for v in after]:
            raise RuntimeError('numeric value/type changed in public JSON')
        preserved += len(before)
        encoded[name] = data
    output_directory.mkdir(parents=True)
    for name, data in encoded.items():
        (output_directory / name).write_bytes(data)
    receipt = dict(scope='metrics/algorithm SHA whitelist; no raw MRI mesh NPZ private logs host paths or source lists',
        original_report_sha256=sha(rpath), exporter_sha256=sha(__file__),
        numeric_boolean_null_leaves_retained=preserved, numeric_values_and_types_preserved=True,
        payload={name: dict(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) for name,data in encoded.items()})
    (output_directory / 'EXPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-run', 'baseline-run', 'frozen-source', 'output-directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(export_metrics(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
