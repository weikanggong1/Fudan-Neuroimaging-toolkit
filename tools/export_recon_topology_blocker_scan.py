"""有序阻挡扫描真实配对的数值白名单；完整私有记录留在服务器。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from export_recon_gca_receipts import numeric_leaves


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialize(value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    before, after = numeric_leaves(value), numeric_leaves(json.loads(data))
    if [(type(x), x) for x in before] != [(type(x), x) for x in after]:
        raise RuntimeError('selected numeric value or type changed')
    return data, len(before)


def verify_source(source, expected):
    path = source/'SOURCE_MANIFEST.private.json'
    if sha(path) != expected:
        raise ValueError('same actual frozen source manifest required')
    manifest = json.loads(path.read_text())
    for name, item in manifest['payload'].items():
        path = source/name
        if sha(path) != item['sha256'] or path.stat().st_size != item['size']:
            raise ValueError('actual frozen source payload changed')
    return manifest


def export_fourth(*, input_run, frozen_source, output_directory):
    """具名运行/源码/新输出目录→完成门、产物SHA后的安全JSON。

    input_run是实际两个fresh子进程完成的父目录，frozen_source为本次
    不可变源码，output_directory必须不存在。先验证EXIT0、父及两个
    child的COMPLETE/report SHA、源码全payload、实际加载模块、冷/热/
    审计72字段及有序控制门、完整谓词序列与输入SHA。白名单只保留
    原数值、状态与SHA；不输出路径、命令、日志、MRI或NPZ。
    所有选中数值/布尔/null的值和类型逐项保留；不重新执行修复。
    """
    run, source, out = map(Path, (input_run, frozen_source, output_directory))
    path = run/'report.private.json'
    report = json.loads(path.read_text())
    query_half = report['status'] == 'complete_fixed_fourth_candidate_query_half_pair'
    variants = ('baseline', 'query_half') if query_half else ('python', 'numba')
    complete = json.loads((run/'COMPLETE.json').read_text())
    exit_record = json.loads((run/'EXIT.json').read_text())
    if (out.exists() or exit_record['exit_code'] != 0 or not exit_record['strict_pair_passed']
            or report['status'] not in ('complete_fixed_fourth_candidate_numba_blocker_pair',
                                       'complete_fixed_fourth_candidate_query_half_pair')
            or complete['status'] != report['status'] or complete['report_sha256'] != sha(path)
            or not report['all_controls_and_fields_exact'] or report['whole_search']
            or report['source_rng_draws'] != 0 or report['candidate_reference_consumed_before_outputs']
            or report['protocol_sha256'] != sha(source/'protocol.private.json')):
        raise ValueError('same complete actual fourth-stage pair required')
    manifest = verify_source(source, report['source_manifest_sha256'])
    same_process_variants = ([r['backend'] for r in report['process']] == ['numba', 'numba']
        and [r.get('variant') for r in report['process']] == list(variants)) if query_half else (
        [r['backend'] for r in report['process']] == list(variants))
    if not same_process_variants:
        raise ValueError('same declared fresh source variants required')
    if ([r['run'] for r in report['fields']] != ['cold', 'warm', 'audit']
            or any(not r['all_bytes_equal'] or r['field_count'] != 72 for r in report['fields'])
            or any(not r['all_bytes_equal'] for r in report['controls'])):
        raise ValueError('complete source ordering and 72-field gates required')
    children = []
    source_names = ('topology_edge_blocker_scan', 'topology_retessellation_dynamic')
    if query_half:
        prefix = 'baseline/src/'
        before = {key.removeprefix(prefix): value for key, value in manifest['payload'].items()
            if key.startswith(prefix)}
        prefix = 'query_half/src/'
        after = {key.removeprefix(prefix): value for key, value in manifest['payload'].items()
            if key.startswith(prefix)}
        helper = 'fnit/recon_all/topology_edge_blocker_scan.py'
        if set(before) != set(after) or any(before[key] != after[key] for key in before if key != helper):
            raise ValueError('only declared query-half scanner source may differ')
    actual_source = ({variant: {name: sha(source/variant/'src/fnit/recon_all'/str(name+'.py'))
        for name in source_names} for variant in variants} if query_half else
        {name: sha(source/'src/fnit/recon_all'/str(name+'.py')) for name in source_names})
    for index, (variant, records) in enumerate(zip(variants, report['records'])):
        backend = 'numba' if query_half else variant
        folder = run/variant/'candidate'
        child_path = folder/'report.private.json'
        child = json.loads(child_path.read_text())
        receipt = json.loads((folder/'COMPLETE.json').read_text())
        process = json.loads((run/variant/'process.private.json').read_text())
        if (receipt['status'] != child['status'] or receipt['report_sha256'] != sha(child_path)
                or process['exit_code'] != 0 or process != report['process'][index]
                or child['blocker_scan_backend'] != backend or child['records'] != records
                or child['frozen_source_manifest_sha256'] != report['source_manifest_sha256']
                or child['input_sha256'] != report['input_sha256']
                or child['actual_rng_draws'] != 0 or not child['all_repeated_fields_byte_equal']
                or not child['all_repeated_controls_equal']):
            raise ValueError('same actual completed child and input identity required')
        for name, digest in child['actual_module_sha256'].items():
            relative = (variant+'/' if query_half else '')+'src/'+name.replace('.', '/')
            item = manifest['payload'].get(relative+'.py') or manifest['payload'].get(relative+'/__init__.py')
            if not item or item['sha256'] != digest:
                raise ValueError('actual loaded source differs from freeze')
        for name, digest in (actual_source[variant] if query_half else actual_source).items():
            # The Python reference does not import the unused Numba scanner.
            if backend == 'python' and name == 'topology_edge_blocker_scan':
                continue
            if child['actual_module_sha256'].get('fnit.recon_all.'+name) != digest:
                raise ValueError('actual scanner or dynamic source differs')
        for record in records:
            name = record['run']
            for suffix, key in (('.private.npz', 'candidate_sha256'),
                                ('.state.private.npz', 'state_sha256'),
                                ('.control.private.json', 'complete_control_sha256')):
                if sha(folder/(name+suffix)) != record[key]:
                    raise ValueError('complete child artifact changed')
        audit = records[-1]
        if (audit['blocker_predicate_sequence_sha256'] != report['predicate_sequence_sha256']
                or audit['blocker_predicate_calls'] != report['predicate_calls']):
            raise ValueError('same complete predicate sequence required')
        if query_half:
            protocol = json.loads((source/'protocol.private.json').read_text())
            if (report['predicate_calls'] != protocol['expected_predicate_calls']
                    or report['predicate_sequence_sha256'] != protocol['expected_predicate_sequence_sha256']):
                raise ValueError('complete query-half audit must match its fixed protocol')
        request = json.loads((run/variant/'request.private.json').read_text())['command']
        for flag, key in (('--input-surface', 'nofix'), ('--input-qsphere', 'qsphere'),
                          ('--brain', 'brain'), ('--state-checkpoint', 'checkpoint'),
                          ('--producer-progress', 'producer_progress_snapshot')):
            if sha(request[request.index(flag)+1]) != child['input_sha256'][key]:
                raise ValueError('actual input changed after execution')
        selected_keys = ('status', 'defect_number', 'population_index', 'identity_ordering', 'actual_rng_draws',
            'checkpoint_complete_fields_restored', 'checkpoint_fields_count',
            'initial_state_from_own_previous_complete_checkpoint', 'initial_attributes_from_reference',
            'shared_priors_and_local_mri_self_generated', 'ordered_edge_count', 'inside_count', 'hull_count',
            'local_volume_shape', 'input_sha256', 'actual_tool_sha256', 'records', 'stages_seconds',
            'loading_seconds', 'complete_tool_seconds_before_report', 'threads', 'cpu_affinity',
            'cuda_hidden', 'peak_rss_kib', 'full_ga', 'final_mesh_quality', 'overall_equivalence',
            'whole_recon_all', 'blocker_scan_backend', 'sequence_audit_enabled_only_last')
        selected_child = {key: child[key] for key in selected_keys}
        if query_half: selected_child['source_variant'] = variant
        children.append(selected_child)
    selected = {key: report[key] for key in ('status', 'source_manifest_sha256', 'protocol_sha256',
        'process', 'fields', 'controls', 'predicate_calls', 'predicate_sequence_sha256',
        'all_controls_and_fields_exact', 'source_rng_draws', 'input_sha256',
        'candidate_reference_consumed_before_outputs', 'complete_pair_cli_seconds',
        'whole_search', 'overall_equivalence')}
    selected['children'] = children
    selected['source_sha256'] = actual_source
    selected['stage_tool_sha256'] = sha(source/'tools/benchmark_recon_topology_fourth_candidate_profile.py')
    selected['allocated_bytes'] = None
    selected['reserved_bytes'] = None
    data, count = serialize(selected)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out/'metrics.json').write_bytes(data)
    receipt = dict(status=('complete_safe_query_half_pair_metric_whitelist' if query_half else
                           'complete_safe_blocker_pair_metric_whitelist'),
        source_manifest_sha256=report['source_manifest_sha256'],
        actual_private_report_sha256=sha(path), actual_complete_sha256=sha(run/'COMPLETE.json'),
        actual_exit_sha256=sha(run/'EXIT.json'), exporter_sha256=sha(__file__),
        numeric_boolean_null_leaves_preserved=count, numeric_values_and_types_preserved=True,
        payload=dict(metrics=dict(sha256=sha(out/'metrics.json'), size=len(data))))
    (out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def export_first(*, input_run, baseline_run, frozen_source, output_directory,
        expected_baseline_process_sha256, baseline_metrics=None,
        expected_baseline_metrics_sha256=None):
    """完整默认首缺陷运行/实际旧运行/冻结源/新目录→数值白名单。

    只在EXIT0、BLOCKER/FACE/OPTIMIZATION/原COMPLETE全部同链、497重连、
    495搜索+final、29trace、80字段/三NPZ和RNG门通过后导出。baseline_run
    必须是科学protocol绑定的原报告，不可替换耗时来源。实际新加载的
    scanner/dynamic SHA来自manifest与优化报告，不借旧overlay的SHA回标。
    expected_baseline_process_sha256须来自已有公开原运行绑定，与该
    baseline_report共同核对，不能仅由本次现场重新哈希冒充旧执行身份。
    旧运行若是同一full_cli_seconds格式，还需baseline_metrics及其原公开
    SHA，逐字段验证旧已发布执行数值/类型与source/protocol，保留原键名。
    原MRI、NPZ、完整日志、私有路径和源清单不导出。
    """
    run, baseline, source, out = map(Path, (input_run, baseline_run, frozen_source, output_directory))
    folder = run/'candidate'
    block_path = folder/'blocker_scan.json'
    block = json.loads(block_path.read_text())
    optimization_path = folder/'optimization.json'
    optimization = json.loads(optimization_path.read_text())
    face_path = folder/'face_cache.json'
    face = json.loads(face_path.read_text())
    stage_path = folder/'report.json'
    stage = json.loads(stage_path.read_text())
    process = json.loads((run/'process.private.json').read_text())
    exit_record = json.loads((run/'EXIT.json').read_text())
    block_complete = json.loads((folder/'BLOCKER_SCAN_COMPLETE.json').read_text())
    face_complete = json.loads((folder/'FACE_CACHE_COMPLETE.json').read_text())
    optimization_complete = json.loads((folder/'OPTIMIZATION_COMPLETE.json').read_text())
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if (out.exists() or exit_record['exit_code'] != 0 or process['exit_code'] != 0
            or not exit_record['blocker_complete_present'] or not exit_record['optimization_complete_present']
            or block['status'] != 'complete_default_first_GA_blocker_scan' or not block['strict_pair_passed']
            or block_complete['status'] != block['status'] or block_complete['report_sha256'] != sha(block_path)
            or block_complete['inherited_complete_sha256'] != sha(folder/'FACE_CACHE_COMPLETE.json')
            or block['inherited_face_cache_report_sha256'] != sha(face_path)
            or block['inherited_optimization_report_sha256'] != sha(optimization_path)
            or not face['strict_complete_pair_passed'] or face_complete['status'] != face['status']
            or face_complete['face_cache_report_sha256'] != sha(face_path)
            or face_complete['inherited_complete_sha256'] != sha(folder/'OPTIMIZATION_COMPLETE.json')
            or optimization_complete['status'] != optimization['status']
            or optimization_complete['optimization_report_sha256'] != sha(optimization_path)
            or optimization_complete['original_report_sha256'] != sha(stage_path)
            or complete['report_sha256'] != sha(stage_path) or complete['status'] != stage['status']
            or not optimization['strict_optimization_pair_passed']):
        raise ValueError('same complete actual full first GA receipt chain required')
    manifest = verify_source(source, optimization['source_manifest_sha256'])
    if process['source_manifest_sha256'] != optimization['source_manifest_sha256']:
        raise ValueError('same actual full CLI and source identity required')
    protocol = json.loads((source/'protocol.private.json').read_text())
    baseline_path = baseline/'candidate/report.json'
    baseline_stage = json.loads(baseline_path.read_text())
    baseline_complete = json.loads((baseline/'candidate/COMPLETE.json').read_text())
    baseline_process = json.loads((baseline/'process.private.json').read_text())
    baseline_optimization = json.loads((baseline/'candidate/optimization.json').read_text())
    if (sha(source/'protocol.private.json') != optimization['protocol_sha256']
            or process['protocol_sha256'] != optimization['protocol_sha256']
            or protocol['baseline_report_sha256'] != sha(baseline_path)
            or baseline_complete['report_sha256'] != sha(baseline_path)
            or baseline_complete['status'] != baseline_stage['status'] or baseline_process['exit_code'] != 0
            or sha(baseline/'process.private.json') != expected_baseline_process_sha256
            or baseline_process['source_manifest_sha256'] != baseline_optimization['source_manifest_sha256']
            or ('protocol_sha256' in baseline_process and
                baseline_process['protocol_sha256'] != baseline_optimization['protocol_sha256'])):
        raise ValueError('same completed baseline and immutable protocol required')
    for name, digest in optimization_complete['complete_output_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('completed new artifact changed')
    for name, digest in baseline_stage['checkpoint_sha256'].items():
        if sha(baseline/'candidate'/name) != digest:
            raise ValueError('completed baseline artifact changed')
    control = json.loads((folder/'search_control.json').read_text())
    trace = control['generation_trace']
    comparisons = optimization['complete_npz_comparisons']
    if (block['prepared_evaluators'] != 1 or block['retessellation_calls'] != 497
            or block['search_evaluations'] != 495 or block['final_scores'] != 1
            or block['generation_trace_count'] != 29 or block['last_zero_based_generation_index'] != 28
            or [row['generation'] for row in trace] != list(range(29))
            or not optimization['exact_search_control'] or not all(optimization['exact_summary'].values())
            or not optimization['all_candidate_fitness_scores']['byte_equal']
            or not all(row['all_fields_byte_equal'] and row['complete_file_sha_equal'] for row in comparisons.values())
            or len(block['retessellation_records']) != 497
            or any(row['backend'] != block['blocker_scan_backend'] for row in block['retessellation_records'])):
        raise ValueError('complete default count, ordered control and full numerical gates required')
    source_hashes = {}
    for name in ('topology_edge_blocker_scan', 'topology_retessellation_dynamic'):
        digest = sha(source/'src/fnit/recon_all'/str(name+'.py'))
        if optimization['loaded_module_sha256'].get('fnit.recon_all.'+name) != digest:
            raise ValueError('actual full-run scanner or dynamic source differs')
        source_hashes[name] = digest
    if block['helper_sha256'] != source_hashes['topology_edge_blocker_scan'] or block['dynamic_sha256'] != source_hashes['topology_retessellation_dynamic']:
        raise ValueError('actual full wrapper and module binding differs')
    block_keys = ('status', 'strict_pair_passed', 'blocker_scan_backend', 'prepared_evaluators',
        'retessellation_calls', 'retessellation_records', 'actual_tool_sha256', 'dynamic_sha256',
        'helper_sha256', 'complete_API_before_report_seconds', 'search_evaluations', 'final_scores',
        'generation_trace_count', 'last_zero_based_generation_index', 'random_draws', 'whole_GA',
        'whole_recon_all', 'production_default_changed', 'state_context_enabled', 'sequence_audit_enabled')
    selected = {key: block[key] for key in block_keys}
    selected['source_manifest_sha256'] = optimization['source_manifest_sha256']
    selected['protocol_sha256'] = optimization['protocol_sha256']
    selected['source_sha256'] = source_hashes
    selected['comparisons'] = {name.removesuffix('.private.npz'): value
        for name, value in comparisons.items()}
    selected['optimization'] = {key: optimization[key] for key in ('all_new_outputs_written_before_previous_winner_read',
        'previous_outputs_used_in_algorithm', 'input_sha256', 'unchanged_loaded_sources',
        'all_candidate_fitness_scores', 'exact_search_control', 'exact_summary', 'source_first_candidate_passed',
        'search_generations', 'candidate_marker_calls', 'source_marker_performance_effect',
        'stage_seconds', 'nested_search_seconds', 'baseline_stage_seconds', 'baseline_nested_search_seconds',
        'full_optimization_API_before_report_seconds', 'current_cpu_affinity', 'baseline_cpu_affinity',
        'threads', 'allocated_bytes', 'reserved_bytes', 'overall_metric_equivalence')}
    selected['execution'] = {key: process[key] for key in ('exit_code', 'full_cli_seconds', 'threads',
        'cpu_affinity', 'peak_child_rss_kib', 'allocated_bytes', 'reserved_bytes')}
    if 'full_cli_seconds' in baseline_process:
        if baseline_metrics is None or expected_baseline_metrics_sha256 is None:
            raise ValueError('previous published full-run timing binding required')
        baseline_metrics = Path(baseline_metrics)
        if sha(baseline_metrics) != expected_baseline_metrics_sha256:
            raise ValueError('previous published metrics identity changed')
        published = json.loads(baseline_metrics.read_text())
        old_execution = {key: baseline_process[key] for key in published['execution']}
        if (published['status'] != 'complete_default_first_GA_blocker_scan'
                or not published['strict_pair_passed']
                or published['source_manifest_sha256'] != baseline_process['source_manifest_sha256']
                or published['protocol_sha256'] != baseline_process['protocol_sha256']
                or old_execution != published['execution']
                or [(type(x), x) for x in numeric_leaves(old_execution)] !=
                   [(type(x), x) for x in numeric_leaves(published['execution'])]):
            raise ValueError('actual baseline execution differs from its published numeric record')
        selected['baseline_execution'] = old_execution
        selected['baseline_published_metrics_sha256'] = expected_baseline_metrics_sha256
    else:
        if baseline_metrics is not None or expected_baseline_metrics_sha256 is not None:
            raise ValueError('published full-cli metrics binding applies only to full-cli baseline schema')
        selected['baseline_execution'] = {key: baseline_process[key] for key in ('exit_code',
            'complete_fresh_cli_seconds', 'threads', 'cpu_affinity', 'child_peak_rss_kib')}
    selected['query_half_per_scan'] = bool(protocol.get('query_half_per_scan', False))
    selected['search_counts'] = stage['search_counts']
    selected['best_fitness'] = stage['best_fitness']
    selected['actual_final_fitness'] = stage['actual_final_fitness']
    selected['search_termination'] = stage['search_termination']
    selected['candidate_timings'] = stage['candidate_timings']
    selected['baseline_candidate_timings'] = baseline_stage['candidate_timings']
    selected['outputs_sha256'] = {name.removesuffix('.private.npz'): value
        for name, value in stage['checkpoint_sha256'].items()}
    selected['baseline_outputs_sha256'] = {name.removesuffix('.private.npz'): value
        for name, value in baseline_stage['checkpoint_sha256'].items()}
    selected['timing_interpretation'] = 'same four-thread budget; separate shared-load windows, no controlled ABBA throughput'
    selected['baseline_report_sha256'] = sha(baseline_path)
    selected['baseline_process_sha256'] = expected_baseline_process_sha256
    data, count = serialize(selected)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out/'metrics.json').write_bytes(data)
    receipt = dict(status='complete_safe_full_first_blocker_metric_whitelist',
        source_manifest_sha256=optimization['source_manifest_sha256'], actual_private_report_sha256=sha(block_path),
        actual_complete_sha256=sha(folder/'BLOCKER_SCAN_COMPLETE.json'), actual_exit_sha256=sha(run/'EXIT.json'),
        baseline_report_sha256=sha(baseline_path), baseline_process_sha256=expected_baseline_process_sha256,
        exporter_sha256=sha(__file__),
        numeric_boolean_null_leaves_preserved=count, numeric_values_and_types_preserved=True,
        payload=dict(metrics=dict(sha256=sha(out/'metrics.json'), size=len(data))))
    (out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt



def export_wiring(*, input_run, frozen_source, output_directory):
    """直接库六评分配对的完成门、五实际模块与产物校验→白名单。

    仅search/prepare/finalize实际MRI阶段，repair/files为入口契约范围。
    原报告、COMPLETE、EXIT及五模块/完整产物SHA全部核验，保存所有选中
    数字/type；不发布源码清单、文件路径、日志、NPZ或原MRI。
    """
    run, source, out = map(Path, (input_run, frozen_source, output_directory))
    folder = run/'pair'; path = folder/'pair.private.json'
    report = json.loads(path.read_text()); complete = json.loads((folder/'COMPLETE.json').read_text())
    exit_path = run/'EXIT_CODE'
    if exit_path.exists():
        exit_ok = int(exit_path.read_text()) == 0
    else:
        exit_path = run/'EXIT.json'
        exit_record = json.loads(exit_path.read_text())
        exit_ok = (exit_record['exit_code'] == 0 and exit_record['pair_complete_present']
            and exit_record['source_manifest_sha256'] == report['source_manifest_sha256']
            and exit_record['protocol_sha256'] == sha(source/'protocol.private.json'))
    if (out.exists() or not exit_ok
            or report['status'] != 'passed_six_score_explicit_blocker_scan_pair'
            or complete['status'] != report['status'] or complete['report_sha256'] != sha(path)
            or report['complete_default_GA'] or report['algorithm_reference_inputs']
            or not all(report['controls'].values())
            or not all(v['all_fields_byte_equal'] and v['complete_file_sha_equal'] for v in report['comparisons'].values())):
        raise ValueError('completed exact actual direct-library pair required')
    manifest = verify_source(source, report['source_manifest_sha256'])
    protocol = json.loads((source/'protocol.private.json').read_text())
    source_names = ('topology_edge_blocker_scan', 'topology_retessellation_dynamic',
        'topology_genetic_search', 'topology_genetic_repair', 'topology_genetic_files')
    sources = {n:sha(source/'src/fnit/recon_all'/str(n+'.py')) for n in source_names}
    records = []
    for backend, record in zip(('python','numba'), report['records']):
        child_dir=folder/backend; child_path=child_dir/'report.private.json'
        child=json.loads(child_path.read_text()); receipt=json.loads((child_dir/'COMPLETE.json').read_text())
        process=json.loads((folder/(backend+'.EXIT.json')).read_text())
        if (record['label'] != backend or record['report'] != child or process['exit_code'] != 0
                or process['fresh_cli_seconds'] != record['fresh_cli_seconds']
                or receipt['status'] != child['status'] or receipt['report_sha256'] != sha(child_path)
                or child['status'] != 'complete_six_score_production_API_stage'
                or child['blocker_scan_backend'] != backend or not child['face_detection_cache']
                or child['source_manifest_sha256'] != report['source_manifest_sha256']
                or child['input_sha256'] != protocol['input_sha256']
                or child['scores'] != 6 or child['search_evaluations'] != 5
                or not child['explicit_blocker_backend_parameter'] or child['global_function_replacement']):
            raise ValueError('same actual completed child/backend/input required')
        for name,digest in child['actual_loaded_module_sha256'].items():
            relative='src/'+name.replace('.','/')
            item=manifest['payload'].get(relative+'.py') or manifest['payload'].get(relative+'/__init__.py')
            if not item or item['sha256'] != digest:raise ValueError('actual loaded module differs from scientific freeze')
        if any(child['actual_loaded_module_sha256'].get('fnit.recon_all.'+n) != digest for n,digest in sources.items()):
            raise ValueError('five actual source modules required')
        for name,digest in child['checkpoint_sha256'].items():
            if sha(child_dir/name) != digest:raise ValueError('completed child artifact changed')
        keys=('status','blocker_scan_backend','face_detection_cache','input_sha256','policy','scores',
            'search_evaluations','stages','nested_search_seconds','random_draws','random_sequence_sha256',
            'cpu_affinity','target_cuda','allocated_bytes','reserved_bytes','self_peak_rss_kib',
            'complete_worker_seconds_before_report','explicit_production_search_parameter',
            'explicit_blocker_backend_parameter','global_function_replacement','complete_default_GA',
            'whole_recon_all','final_mesh_quality','overall_metric_equivalence')
        selected={k:child[k] for k in keys}
        selected['output_sha256']={n.replace('.private',''):v for n,v in child['checkpoint_sha256'].items()}
        records.append(dict(label=backend,fresh_cli_seconds=record['fresh_cli_seconds'],report=selected))
    if (records[0]['report']['random_draws'] != records[1]['report']['random_draws']
            or records[0]['report']['random_sequence_sha256'] != records[1]['report']['random_sequence_sha256']):
        raise ValueError('actual ordered RNG bounds/value sequence differs')
    selected={k:report[k] for k in ('status','source_manifest_sha256','controls','compared_numeric_fields',
        'complete_pair_wall_seconds','production_default_changed','complete_default_GA',
        'whole_recon_all','final_mesh_quality','overall_metric_equivalence','algorithm_reference_inputs')}
    selected['records']=records;selected['source_sha256']=sources
    selected['protocol_sha256']=sha(source/'protocol.private.json')
    selected['actual_tool_sha256']=sha(source/'tools/benchmark_recon_topology_genetic_blocker_wiring.py')
    selected['comparisons']={n.replace('.private',''):v for n,v in report['comparisons'].items()}
    selected['scope']='raw arrays to direct prepare/search/finalize; repair/files dispatcher contracts only'
    data,count=serialize(selected);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    (out/'metrics.json').write_bytes(data)
    receipt=dict(status='complete_safe_direct_blocker_wiring',source_manifest_sha256=report['source_manifest_sha256'],
        actual_private_report_sha256=sha(path),actual_complete_sha256=sha(folder/'COMPLETE.json'),
        actual_exit_sha256=sha(exit_path),exporter_sha256=sha(__file__),
        numeric_boolean_null_leaves_preserved=count,numeric_values_and_types_preserved=True,
        payload=dict(metrics=dict(sha256=sha(out/'metrics.json'),size=len(data))))
    (out/'EXPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fourth', 'first', 'wiring'), default='fourth')
    parser.add_argument('--baseline-run', type=Path)
    parser.add_argument('--expected-baseline-process-sha256')
    parser.add_argument('--baseline-metrics', type=Path)
    parser.add_argument('--expected-baseline-metrics-sha256')
    for name in ('input-run', 'frozen-source', 'output-directory'):
        parser.add_argument('--'+name, type=Path, required=True)
    options = vars(parser.parse_args()); stage = options.pop('stage'); baseline = options.pop('baseline_run')
    process_hash = options.pop('expected_baseline_process_sha256')
    baseline_metrics = options.pop('baseline_metrics')
    baseline_metrics_hash = options.pop('expected_baseline_metrics_sha256')
    if stage == 'wiring':
        if any(x is not None for x in (baseline, process_hash, baseline_metrics, baseline_metrics_hash)):
            parser.error('baseline controls apply only to first stage')
        value = export_wiring(**options)
    elif stage == 'first':
        if baseline is None or process_hash is None: parser.error('--baseline-run and --expected-baseline-process-sha256 required for full first GA export')
        value = export_first(baseline_run=baseline, expected_baseline_process_sha256=process_hash,
            baseline_metrics=baseline_metrics, expected_baseline_metrics_sha256=baseline_metrics_hash, **options)
    else:
        if any(x is not None for x in (baseline, process_hash, baseline_metrics, baseline_metrics_hash)):
            parser.error('baseline bindings belong only to full first GA export')
        value = export_fourth(**options)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
