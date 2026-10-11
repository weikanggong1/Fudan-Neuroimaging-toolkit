"""显式距离 route 的原 child 与独立后验比较安全导出；不重算评分。"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import zipfile
from export_recon_topology_blocker_scan import sha, serialize, verify_source

NPZ_FILES = ('final_candidate.private.npz', 'final_state.private.npz',
             'search.private.npz', 'all_scores.private.npz')
CONTROL_FILES = ('candidate_control.private.json', 'search_control.private.json')
STAGES = ('import_and_source_validation', 'load', 'raw_base',
          'global_prior_and_input_context', 'ordered_edges', 'local_prior',
          'search_and_final_API', 'write')
POLICY_KEYS = ('population_size', 'max_unchanged', 'niters', 'initial_selection',
               'vertex_eliminate', 'seed', 'threads')
CHILD_STATUS = 'complete_six_score_production_API_stage'
ROUTE_MODULES = tuple('fnit.recon_all.' + name for name in
    ('topology_candidate_pipeline', 'topology_genetic_search',
     'topology_genetic_repair', 'topology_genetic_files'))


def same_json(a, b):
    """JSON 结构、数值类型和数值均相同；bool 不等同 int。"""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return a == b


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError('complete declared key set required')


def _digest(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('valid SHA-256 required')
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('strict integer required')
    return value


def _number(value, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError('finite numeric value required')
    return value


def _thread_budget(value):
    _keys(value, ('requested_threads', 'scope', 'torch', 'numba',
                  'environment_modified', 'restoration_complete'))
    if (type(value['requested_threads']) is not int or value['requested_threads'] != 2
            or value['scope'] != 'Torch process intraop and current caller Numba thread mask'
            or value['environment_modified'] is not False
            or value['restoration_complete'] is not True):
        raise ValueError('actual caller budget and restoration required')
    _keys(value['torch'], ('before', 'effective', 'after_numba_initialization',
                          'interop_unchanged', 'restored'))
    for item in value['torch'].values():
        _integer(item, 1)
    _keys(value['numba'], ('before', 'effective', 'initial_capacity', 'mask_only',
                          'threading_layer', 'restored'))
    for key in ('before', 'effective', 'initial_capacity', 'restored'):
        _integer(value['numba'][key], 1)
    if (value['numba']['mask_only'] is not True
            or value['numba']['threading_layer'] not in ('tbb', 'omp', 'workqueue')
            or value['torch']['effective'] != 2 or value['numba']['effective'] != 2
            or value['torch']['restored'] != value['torch']['before']
            or value['numba']['restored'] != value['numba']['before']):
        raise ValueError('unchanged global capacity and restored mask required')
    return value


def _comparison(value, archive):
    _keys(value, ('same_keys', 'fields', 'all_fields_byte_equal', 'complete_file_sha_equal'))
    if any(value[k] is not True for k in ('same_keys', 'all_fields_byte_equal', 'complete_file_sha_equal')):
        raise ValueError('strict byte equality booleans required')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    if len(names) != len(set(names)) or any(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*\.npy', n) is None for n in names):
        raise ValueError('declared numeric archive field names required')
    _keys(value['fields'], (n[:-4] for n in names))
    if not names:
        raise ValueError('nonempty numeric archive required')
    for row in value['fields'].values():
        _keys(row, ('shape_equal', 'dtype_equal', 'byte_equal', 'differing_values',
                    'differing_bytes', 'max_abs', 'p99_abs'))
        if any(row[k] is not True for k in ('shape_equal', 'dtype_equal', 'byte_equal')):
            raise ValueError('strict per-field equality required')
        for key in ('differing_values', 'differing_bytes'):
            if _integer(row[key]) != 0:
                raise ValueError('zero differing values and bytes required')
        for key in ('max_abs', 'p99_abs'):
            if _number(row[key]) != 0:
                raise ValueError('zero numerical difference required')
    return value


def export_completed(*, input_run, child_run, frozen_source, output_directory,
        expected_source_manifest_sha256, expected_protocol_sha256,
        expected_report_sha256, expected_complete_sha256, expected_exit_sha256):
    """校验完整四 NPZ、两控制、全部模块、原协议和实际 RNG，再导出白名单。

    input_run 为新后验比较目录，child_run 为原两 fresh 计算目录，
    frozen_source 为原 child 实际源码。五 expected SHA 绑定源码、协议和
    后验三收据。output_directory 必须不存在。仅读取已完成产物，不重算
    MRI/评分；原父 exit1 保留，后验耗时不是原父成功 CLI 或整例耗时。
    """
    D, C, S, out = map(Path, (input_run, child_run, frozen_source, output_directory))
    if out.exists():
        raise FileExistsError('fresh publication directory required')
    def load(path):
        return json.loads(Path(path).read_text())
    for value in (expected_source_manifest_sha256, expected_protocol_sha256,
                  expected_report_sha256, expected_complete_sha256, expected_exit_sha256):
        _digest(value)
    for rel, digest in (('report.private.json', expected_report_sha256),
                        ('COMPLETE.json', expected_complete_sha256), ('EXIT.json', expected_exit_sha256)):
        if sha(D / rel) != digest:
            raise ValueError('actual posthoc record SHA differs')
    report, complete, exit_info = (load(D / rel) for rel in ('report.private.json', 'COMPLETE.json', 'EXIT.json'))
    if (report['status'] != 'passed_posthoc_explicit_distance_short_pair'
            or complete['status'] != report['status'] or complete['report_sha256'] != expected_report_sha256
            or type(exit_info['exit_code']) is not int or exit_info['exit_code'] != 0
            or exit_info['complete_present'] is not True
            or report['source_manifest_sha256'] != expected_source_manifest_sha256
            or report['algorithm_rerun'] is not False
            or report['successful_original_pair_cli_seconds'] is not None
            or type(report['original_parent_exit_code']) is not int or report['original_parent_exit_code'] != 1
            or report['original_parent_exit_sha256'] != sha(C / 'EXIT_CODE')
            or (C / 'EXIT_CODE').read_text() != '1'):
        raise ValueError('completed posthoc comparison and failed original parent binding required')
    mf = verify_source(S, expected_source_manifest_sha256)
    if sha(S / 'protocol.private.json') != expected_protocol_sha256:
        raise ValueError('frozen original protocol differs')
    protocol = load(S / 'protocol.private.json')
    if (protocol['source_manifest_sha256'] != expected_source_manifest_sha256
            or not same_json(protocol['cpu_affinity'], [84, 85])
            or not same_json(protocol['distance_backends'], ['serial', 'parallel'])
            or protocol['fresh_processes'] is not True or protocol['global_function_replacement'] is not False):
        raise ValueError('same original fresh-process protocol required')
    _keys(protocol['input_sha256'], ('nofix', 'qsphere', 'brain'))
    for digest in protocol['input_sha256'].values():
        _digest(digest)
    comparator = _digest(mf['payload']['tools/benchmark_recon_topology_genetic_optimization.py']['sha256'])
    if report['comparator_sha256'] != comparator:
        raise ValueError('actual comparator source differs')
    expected_modules = mf['expected_module_sha256']
    if not expected_modules or any(name not in expected_modules for name in ROUTE_MODULES):
        raise ValueError('complete explicit-route module manifest required')
    for name in expected_modules:
        if re.fullmatch(r'fnit\.recon_all(?:\.[A-Za-z][A-Za-z0-9_]*)*', name) is None:
            raise ValueError('declared FNIT module names required')
    if type(report['records']) is not list or len(report['records']) != 2:
        raise ValueError('two complete original children required')
    _keys(report['comparisons'], NPZ_FILES)
    _keys(report['controls'], CONTROL_FILES)
    rows, reports, actual_controls = [], [], []
    for label, observed in zip(('serial', 'parallel'), report['records']):
        P = C / 'pair' / label
        v, comp, process = load(P / 'report.private.json'), load(P / 'COMPLETE.json'), load(C / 'pair' / (label + '.EXIT.json'))
        if (observed['label'] != label or not same_json(v, observed['report'])
                or not same_json(process, observed['process'])
                or observed['actual_report_sha256'] != sha(P / 'report.private.json')
                or observed['actual_complete_sha256'] != sha(P / 'COMPLETE.json')
                or observed['actual_process_sha256'] != sha(C / 'pair' / (label + '.EXIT.json'))
                or v['status'] != CHILD_STATUS or comp['status'] != CHILD_STATUS
                or comp['report_sha256'] != sha(P / 'report.private.json')
                or type(process['exit_code']) is not int or process['exit_code'] != 0
                or v['source_manifest_sha256'] != expected_source_manifest_sha256
                or v['distance_backend'] != label or type(v['scores']) is not int or v['scores'] != 6
                or type(v['search_evaluations']) is not int or v['search_evaluations'] != 5
                or not same_json(v['cpu_affinity'], protocol['cpu_affinity'])
                or v['global_function_replacement'] is not False):
            raise ValueError('actual complete same-input child/source/budget differs')
        _keys(v['policy'], POLICY_KEYS)
        if not same_json(v['policy'], {k: protocol[k] for k in POLICY_KEYS}):
            raise ValueError('actual child policy differs from frozen protocol')
        for key in ('population_size', 'max_unchanged', 'niters', 'seed', 'threads'):
            _integer(v['policy'][key])
        if v['policy']['initial_selection'] is not True or v['policy']['vertex_eliminate'] is not True:
            raise ValueError('strict policy booleans required')
        _keys(v['input_sha256'], ('nofix', 'qsphere', 'brain'))
        if not same_json(v['input_sha256'], protocol['input_sha256']):
            raise ValueError('three original input SHA bindings required')
        policy = v['distance_policy']
        _keys(policy, ('requested_backend', 'executed', 'actual_backend', 'skipped_unmri',
                      'caller_numba_thread_mask', 'slab_count', 'effective_unmri_weight', 'worker_count_observed'))
        if (policy['requested_backend'] != label or policy['actual_backend'] != label
                or policy['executed'] is not True or policy['skipped_unmri'] is not False
                or not same_json(policy['caller_numba_thread_mask'], 2 if label == 'parallel' else None)
                or not same_json(policy['slab_count'], 2 if label == 'parallel' else 1)
                or policy['worker_count_observed'] is not False):
            raise ValueError('actual executed route differs')
        _number(policy['effective_unmri_weight'])
        budget = _thread_budget(v['thread_budget'])
        _keys(v['stages'], STAGES)
        for item in v['stages'].values():
            _number(item)
        if not same_json(v['actual_loaded_module_sha256'], expected_modules):
            raise ValueError('all actual loaded module bindings required')
        for name, digest in expected_modules.items():
            rel = 'src/' + name.replace('.', '/')
            item = mf['payload'].get(rel + '.py') or mf['payload'].get(rel + '/__init__.py')
            if not item or item['sha256'] != _digest(digest):
                raise ValueError('actual module source differs')
        _keys(v['checkpoint_sha256'], NPZ_FILES + CONTROL_FILES)
        for name, digest in v['checkpoint_sha256'].items():
            if sha(P / name) != _digest(digest):
                raise ValueError('actual child artifact differs')
        controls = {n: load(P / n) for n in CONTROL_FILES}
        rng = controls['search_control.private.json']
        if (_integer(v['random_draws'], 1) != _integer(rng['random_draws'], 1)
                or v['random_draws'] != _integer(rng['observed_random_draws'], 1)
                or _digest(v['random_sequence_sha256']) != _digest(rng['random_sequence_sha256'])):
            raise ValueError('actual RNG control and report binding required')
        if (v['target_cuda'] != 'hidden' or v['allocated_bytes'] is not None or v['reserved_bytes'] is not None):
            raise ValueError('CPU-only observed memory policy required')
        _integer(v['self_peak_rss_kib'], 1)
        _number(v['complete_worker_seconds_before_report'])
        selected = {k: v[k] for k in ('distance_backend', 'policy', 'scores', 'search_evaluations',
                    'stages', 'random_draws', 'random_sequence_sha256', 'input_sha256', 'cpu_affinity',
                    'target_cuda', 'allocated_bytes', 'reserved_bytes', 'self_peak_rss_kib',
                    'complete_worker_seconds_before_report', 'actual_loaded_module_sha256')}
        selected.update(distance_policy=policy, thread_budget=budget,
            output_sha256={k.replace('.private', ''): h for k, h in v['checkpoint_sha256'].items()})
        rows.append(dict(label=label, actual_report_sha256=observed['actual_report_sha256'],
            actual_complete_sha256=observed['actual_complete_sha256'], actual_process_sha256=observed['actual_process_sha256'],
            fresh_cli_seconds=_number(process['fresh_cli_seconds']), report=selected))
        reports.append(v)
        actual_controls.append(controls)
    for name in NPZ_FILES:
        if reports[0]['checkpoint_sha256'][name] != reports[1]['checkpoint_sha256'][name]:
            raise ValueError('actual complete numeric archive SHA differs')
        _comparison(report['comparisons'][name], C / 'pair' / 'serial' / name)
    for name in CONTROL_FILES:
        if report['controls'][name] is not True or not same_json(actual_controls[0][name], actual_controls[1][name]):
            raise ValueError('actual type-sensitive control equality required')
    if (reports[0]['random_draws'] != reports[1]['random_draws']
            or reports[0]['random_sequence_sha256'] != reports[1]['random_sequence_sha256']):
        raise ValueError('actual RNG sequence and draw count differ')
    count_fields = sum(len(v['fields']) for v in report['comparisons'].values())
    if type(report['compared_numeric_fields']) is not int or report['compared_numeric_fields'] != count_fields or count_fields != 81:
        raise ValueError('all 81 original numeric fields required')
    selected = dict(status=report['status'], source_manifest_sha256=expected_source_manifest_sha256,
        protocol_sha256=expected_protocol_sha256, records=rows,
        comparisons={k.replace('.private', ''): v for k, v in report['comparisons'].items()},
        controls={k.replace('.private', ''): v for k, v in report['controls'].items()},
        compared_numeric_fields=count_fields, actual_random_bounds_and_values_sha256=reports[0]['random_sequence_sha256'],
        posthoc_comparison_seconds=_number(report['posthoc_comparison_seconds']), successful_original_pair_cli_seconds=None,
        original_parent_exit_code=1, original_parent_failure_diagnosis='checker_import_setup_retained_separately',
        algorithm_rerun=False, complete_default_GA=False, whole_recon_all='not_assessed',
        overall_metric_equivalence='not_assessed', production_default_distance_backend='serial', final_mesh_quality='not_assessed')
    data, count = serialize(selected)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / 'metrics.json').write_bytes(data)
    receipt = dict(status='complete_safe_explicit_distance_route', source_manifest_sha256=expected_source_manifest_sha256,
        protocol_sha256=expected_protocol_sha256, actual_report_sha256=expected_report_sha256,
        actual_complete_sha256=expected_complete_sha256, actual_exit_sha256=expected_exit_sha256,
        original_parent_exit_sha256=report['original_parent_exit_sha256'], comparator_sha256=comparator,
        exporter_sha256=sha(__file__), numeric_values_and_types_preserved=True,
        numeric_boolean_null_leaves_preserved=count,
        payload={'metrics': dict(sha256=sha(out / 'metrics.json'), size=len(data))})
    (out / 'EXPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-run', 'child-run', 'frozen-source', 'output-directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source-manifest-sha256', 'expected-protocol-sha256', 'expected-report-sha256',
                 'expected-complete-sha256', 'expected-exit-sha256'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(export_completed(**vars(parser.parse_args()))))

if __name__ == '__main__':
    main()
