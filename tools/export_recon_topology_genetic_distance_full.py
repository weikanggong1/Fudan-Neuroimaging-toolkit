"""完整首缺陷距离配对的安全导出；读取已完成产物，不重新评分。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import zipfile

from export_recon_topology_genetic_distance_wiring import (
    CONTROL_FILES, NPZ_FILES, POLICY_KEYS, ROUTE_MODULES, STAGES,
    _comparison, _digest, _integer, _keys, _number, _thread_budget, same_json,
)
from export_recon_topology_blocker_scan import sha, serialize, verify_source

CHILD_STATUS = 'complete_default_first_GA_explicit_distance_API'
PAIR_STATUS = 'passed_default_first_GA_explicit_distance_pair'
SOURCE_GIT = 'eb164824de5dec324f38a42626037f56bd040eb7'
NESTED_STAGES = ('prepare', 'search', 'finalization', 'complete_api')
SEARCH_CONTROL_FIELDS = ('actual_policy', 'additional_discarded', 'best_fitness', 'counts',
    'deletion_count', 'elimination_threshold', 'evaluation_trace', 'final_fitness', 'fitness_matches_best',
    'generation_trace', 'generations', 'observed_random_draws', 'random_draws', 'random_sequence_sha256', 'termination')
CANDIDATE_CONTROL_FIELDS = ('face_check', 'face_decisions', 'ndiscarded', 'ninside', 'retessellation_trace', 'rounds')
COUNT_FIELDS = ('evaluated', 'removed_vertices', 'best_removed_vertices', 'best_patch_number',
    'mutation_attempts', 'mutation_improvements', 'crossover_attempts', 'crossover_improvements')
# 来自本次EB164 candidate_payload、DVS state serializer及search写出域。
# 字段名也属于公开白名单，不能仅凭“81项”允许任意标识符。
ARCHIVE_FIELDS = {
    'final_candidate.private.npz': (
        'cached_original_face_area_mm2', 'core_fields', 'displacement_mm', 'displacement_vertices',
        'distance_mm', 'edge_used', 'face_area', 'face_normals', 'faces', 'fitness', 'flattened_vertices',
        'incident_corners', 'incident_faces', 'incident_offsets', 'k1', 'k2', 'marks', 'neighbor_offsets',
        'ordered_neighbors', 'original_vertices', 'patch_faces', 'removed_vertices', 'selected_edges',
        'selected_faces', 'selected_vertex_normals', 'selected_vertices', 'smoothed_original_vertices',
        'tangent_e1', 'tangent_e2', 'temporary_faces', 'vertex_curvbak', 'vertex_normals'),
    'final_state.private.npz': (
        'H', 'K', 'canonical_vertices', 'current_vertices', 'curv', 'curvbak', 'face_area', 'face_normals',
        'face_rip', 'faces', 'flattened_vertices', 'imag_val', 'incident_corners', 'incident_faces',
        'incident_offsets', 'k1', 'k2', 'mean', 'mean_imag', 'neighbor_offsets', 'neighbor_storage', 'nfaces',
        'old_undefval', 'original_vertices', 'stat', 'std_error', 'tangent_e1', 'tangent_e2', 'temporary_vertices',
        'undefval', 'val', 'val2', 'val2bak', 'valbak', 'vertex_area', 'vertex_border', 'vertex_marks',
        'vertex_normals', 'vertex_rip', 'vertex_topology'),
    'search.private.npz': ('best_discarded', 'best_ordering', 'population_fitness', 'population_orderings',
        'population_ranks', 'use_counts', 'vertex_fitness', 'vertex_rip'),
    'all_scores.private.npz': ('fitness',),
}
OLD_FILES = {
    'EXIT.json': 'exit', 'process.private.json': 'process',
    'candidate/BLOCKER_SCAN_COMPLETE.json': 'blocker_completion',
    'candidate/OPTIMIZATION_COMPLETE.json': 'optimization_completion',
    'candidate/optimization.json': 'optimization_report',
    'candidate/search_control.json': 'search_control',
    'candidate/report.json': 'scientific_report', 'candidate/COMPLETE.json': 'scientific_completion',
    'candidate/final_state.private.npz': 'final_state', 'candidate/search.private.npz': 'search',
}


def _signed_number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite signed fitness required')
    return value


def _load(path):
    return json.loads(Path(path).read_text())


def _source_modules(source, manifest, observed):
    expected = manifest['expected_module_sha256']
    if len(expected) != 71 or any(n not in expected for n in ROUTE_MODULES):
        raise ValueError('all 71 explicit-route scientific modules required')
    if not same_json(observed, expected):
        raise ValueError('actual scientific module bindings differ')
    for name, digest in expected.items():
        if re.fullmatch(r'fnit\.recon_all(?:\.[A-Za-z][A-Za-z0-9_]*)*', name) is None:
            raise ValueError('declared FNIT module name required')
        rel = 'src/' + name.replace('.', '/')
        item = manifest['payload'].get(rel + '.py') or manifest['payload'].get(rel + '/__init__.py')
        if not item or item['sha256'] != _digest(digest):
            raise ValueError('actual module payload differs')


def _loaded_helpers(manifest, observed):
    if type(observed) is not dict or not observed:
        raise ValueError('nonempty actual loaded source bindings required')
    for name, digest in observed.items():
        if name == 'fnit' or name.startswith('fnit.'):
            rel = 'src/' + name.replace('.', '/')
            item = manifest['payload'].get(rel + '.py') or manifest['payload'].get(rel + '/__init__.py')
        elif re.fullmatch(r'(benchmark_recon_|export_recon_)[A-Za-z0-9_]+', name):
            item = manifest['payload'].get('tools/' + name + '.py')
        else:
            raise ValueError('unexpected actual helper name')
        if not item or item['sha256'] != _digest(digest):
            raise ValueError('actual helper payload differs')


def _archive_bytes(a, b, fields):
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        names = za.namelist()
        if (not names or len(names) != len(set(names)) or set(names) != set(zb.namelist())
                or set(names) != {name + '.npy' for name in fields}
                or any(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*\.npy', n) is None for n in names)):
            raise ValueError('complete numeric archive fields required')
        if not all(za.read(name) == zb.read(name) for name in names):
            raise ValueError('actual NPY bytes differ')
    return len(names)


def _full_controls(control, report, archive, candidate_archive, candidate_control):
    import numpy as np
    _keys(control, SEARCH_CONTROL_FIELDS)
    _keys(control['counts'], COUNT_FIELDS)
    for count in control['counts'].values():
        _integer(count)
    actual_policy = dict(population_size=10, max_unchanged=10, niters=-1,
        initial_selection=True, vertex_eliminate=True, overlap='disabled')
    if not same_json(control['actual_policy'], actual_policy):
        raise ValueError('actual source search policy required')
    _integer(control['additional_discarded'])
    _integer(control['deletion_count'], 1)
    _number(control['elimination_threshold'])
    _keys(candidate_control, CANDIDATE_CONTROL_FIELDS)
    for name in ('rounds', 'ninside', 'ndiscarded'):
        _integer(candidate_control[name])
    if (type(candidate_control['retessellation_trace']) is not list
            or not candidate_control['retessellation_trace']
            or type(candidate_control['face_decisions']) is not dict
            or not candidate_control['face_decisions']):
        raise ValueError('complete ordered retessellation and face controls required')
    _keys(candidate_control['face_check'], ('source_result', 'first_invalid', 'checked_directed_edges'))
    if (not same_json(candidate_control['face_check']['source_result'], 1)
            or candidate_control['face_check']['first_invalid'] is not None):
        raise ValueError('actual source face check required')
    _integer(candidate_control['face_check']['checked_directed_edges'])
    for key in ('random_draws', 'observed_random_draws'):
        if _integer(control[key], 1) != report['random_draws']:
            raise ValueError('actual RNG counts differ')
    if _digest(control['random_sequence_sha256']) != report['random_sequence_sha256']:
        raise ValueError('actual RNG bounds and values SHA differs')
    if (type(control['evaluation_trace']) is not list or len(control['evaluation_trace']) != 495
            or type(control['generation_trace']) is not list or len(control['generation_trace']) != 29
            or _integer(control['generations']) != 28
            or _integer(control['counts']['evaluated']) != 495
            or control['fitness_matches_best'] is not True
            or not same_json(control['final_fitness'], control['best_fitness'])):
        raise ValueError('complete default search and final fitness required')
    # Source trace uses a generation index field; reject bool-as-int or gaps.
    indices = [row['generation'] for row in control['generation_trace']]
    if not same_json(indices, list(range(29))):
        raise ValueError('all 29 ordered generation indices required')
    values = [_signed_number(row['fitness']) for row in control['evaluation_trace']]
    values.append(_signed_number(control['final_fitness']))
    with np.load(candidate_archive, allow_pickle=False) as data:
        fitness = data['fitness']
        if (fitness.dtype != np.dtype('float64') or fitness.shape != () or not np.isfinite(fitness)
                or fitness.tobytes() != np.asarray(control['final_fitness'], np.float64).tobytes()):
            raise ValueError('actual final candidate fitness must match final control bits')
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {'fitness'}:
            raise ValueError('complete saved fitness array required')
        actual = data['fitness']
        wanted = np.asarray(values, np.float64)
        if (actual.dtype != np.dtype('float64') or actual.shape != (496,)
                or not np.isfinite(actual).all() or actual.tobytes() != wanted.tobytes()):
            raise ValueError('all 496 saved scores must match actual trace and final')


def _old_reference(*, root, protocol, candidate):
    """仅在新配对门后读取旧参照；不把旧first candidate对应到新final。"""
    ref = protocol['prior_scientific_reference']
    run_name = ref['run']
    if type(run_name) is not str or re.fullmatch(r'[A-Za-z0-9_]+', run_name) is None:
        raise ValueError('declared relative old run name required')
    old, source = root / 'runs' / run_name, root / 'workspaces' / run_name
    verify_source(source, _digest(ref['source_manifest_sha256']))
    if sha(source / 'protocol.private.json') != _digest(ref['protocol_sha256']):
        raise ValueError('actual old protocol differs')
    _keys(ref['files'], OLD_FILES)
    for name, digest in ref['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(old / name) != _digest(digest):
            raise ValueError('actual pinned old completion or output differs')
    old_exit, old_process = _load(old / 'EXIT.json'), _load(old / 'process.private.json')
    old_report = _load(old / 'candidate/report.json')
    old_complete = _load(old / 'candidate/COMPLETE.json')
    if (type(old_exit['exit_code']) is not int or old_exit['exit_code'] != 0
            or type(old_process['exit_code']) is not int or old_process['exit_code'] != 0
            or old_process['source_manifest_sha256'] != ref['source_manifest_sha256']
            or old_process['protocol_sha256'] != ref['protocol_sha256']
            or old_complete['report_sha256'] != sha(old / 'candidate/report.json')
            or old_complete['status'] != old_report['status']
            or old_report['input_sha256'] != protocol['input_sha256']
            or _integer(old_report['random_draws']) != 1362381):
        raise ValueError('actual complete same-input old producer required')
    if not same_json(ref['comparable_npz'], ['final_state.private.npz', 'search.private.npz']):
        raise ValueError('only declared old final-state and search subset allowed')
    arrays = {}
    for name in ref['comparable_npz']:
        count = _archive_bytes(old / 'candidate' / name, candidate / name, ARCHIVE_FIELDS[name])
        arrays[name.replace('.private', '')] = dict(fields=count, actual_npy_bytes_equal=True,
            complete_file_sha_equal=sha(old / 'candidate' / name) == sha(candidate / name),
            old_sha256=sha(old / 'candidate' / name), new_sha256=sha(candidate / name))
    old_control = _load(old / 'candidate/search_control.json')
    new_control = _load(candidate / 'search_control.private.json')
    expected = ('counts', 'evaluation_trace', 'generation_trace', 'termination', 'generations',
                'best_fitness', 'deletion_count', 'elimination_threshold')
    _keys(old_control, expected)
    if not same_json(old_control, {k: new_control[k] for k in expected}):
        raise ValueError('actual old comparable typed controls differ')
    if not same_json(old_report['actual_final_fitness'], new_control['final_fitness']):
        raise ValueError('actual old final fitness differs')
    return dict(status='passed_read_only_old_final_state_search_control_subset',
        source_manifest_sha256=ref['source_manifest_sha256'], protocol_sha256=ref['protocol_sha256'],
        input_sha256=old_report['input_sha256'], compared_numeric_fields=sum(v['fields'] for v in arrays.values()),
        comparisons=arrays, typed_control_values_equal=True,
        control_scope='typed_json_value_equality_not_signed_zero_bits_or_text_bytes',
        actual_final_fitness=old_report['actual_final_fitness'], random_draws=old_report['random_draws'],
        old_rng_bounds_and_values_sha256=None, old_rng_sequence_comparison='not_assessed',
        old_first_candidate_compared_to_new_final=False, old_time_used_as_paired_baseline=False,
        actual_reference_file_sha256={OLD_FILES[k]: v for k, v in ref['files'].items()})


def export_completed(*, input_run, frozen_source, output_directory,
        expected_source_manifest_sha256, expected_protocol_sha256,
        expected_report_sha256, expected_complete_sha256, expected_exit_sha256):
    """完整成功父链、两个实际child、81数组/496评分/RNG→新安全目录。

    所有Path和expected SHA均具名。源码目录是原计算不可变冻结，input_run
    是其完整父输出，output_directory必须不存在。校验全部科学/helper源码、
    request/driver/process绑定后，再读取旧query的已声明最终state/search子集。
    不读取MRI、不重算GA、不导出原控制文本、私有路径或数组。
    """
    D, S, out = map(Path, (input_run, frozen_source, output_directory))
    if out.exists():
        raise FileExistsError('fresh publication directory required')
    for rel, digest in (('candidate/pair.private.json', expected_report_sha256),
            ('candidate/COMPLETE.json', expected_complete_sha256), ('EXIT.json', expected_exit_sha256)):
        if sha(D / rel) != _digest(digest):
            raise ValueError('actual complete parent SHA differs')
    mf = verify_source(S, _digest(expected_source_manifest_sha256))
    if sha(S / 'protocol.private.json') != _digest(expected_protocol_sha256):
        raise ValueError('actual frozen protocol differs')
    protocol = _load(S / 'protocol.private.json')
    if (protocol['git_base'] != SOURCE_GIT or protocol['scientific_source_overlays'] != []
            or protocol['source_manifest_sha256'] != expected_source_manifest_sha256
            or not same_json(protocol['cpu_affinity'], [92, 93])
            or not same_json(protocol['distance_backends'], ['serial', 'parallel'])
            or protocol['fresh_processes'] is not True
            or protocol['serial_child_finishes_before_parallel'] is not True
            or protocol['isolated_numba_cache_per_child'] is not True):
        raise ValueError('actual immutable complete fresh-pair protocol required')
    policy = {k: protocol[k] for k in POLICY_KEYS}
    if not same_json(policy, dict(population_size=10, max_unchanged=10, niters=-1,
            initial_selection=True, vertex_eliminate=True, seed=1234, threads=2)):
        raise ValueError('unchanged complete default policy required')
    _keys(protocol['input_sha256'], ('nofix', 'qsphere', 'brain'))
    for value in protocol['input_sha256'].values():
        _digest(value)
    report, complete, process = (_load(D / n) for n in
        ('candidate/pair.private.json', 'candidate/COMPLETE.json', 'EXIT.json'))
    request = _load(D / 'request.private.json')
    driver_sha = sha(S / 'control.private.py')
    if (type(process['exit_code']) is not int or process['exit_code'] != 0
            or process['complete_present'] is not True
            or process['source_manifest_sha256'] != expected_source_manifest_sha256
            or process['protocol_sha256'] != expected_protocol_sha256
            or process['request_sha256'] != sha(D / 'request.private.json')
            or process['driver_sha256'] != driver_sha
            or process['report_sha256'] != expected_report_sha256
            or process['complete_sha256'] != expected_complete_sha256
            or sha(D / 'process.private.json') != sha(D / 'EXIT.json')
            or request['source_manifest_sha256'] != expected_source_manifest_sha256
            or request['protocol_sha256'] != expected_protocol_sha256 or request['driver_sha256'] != driver_sha
            or not same_json(process['cpu_affinity'], [92, 93]) or not same_json(process['threads'], 2)
            or report['status'] != PAIR_STATUS or complete['status'] != PAIR_STATUS
            or complete['report_sha256'] != expected_report_sha256
            or report['source_manifest_sha256'] != expected_source_manifest_sha256
            or report['complete_default_GA'] is not True
            or report['global_function_replacement'] is not False
            or report['production_default_changed'] is not False or report['algorithm_reference_inputs'] is not False):
        raise ValueError('actual complete successful parent/request/driver binding required')
    if type(report['records']) is not list or len(report['records']) != 2:
        raise ValueError('two complete actual fresh children required')
    rows, originals, controls = [], [], []
    _keys(report['comparisons'], NPZ_FILES)
    _keys(report['controls'], CONTROL_FILES)
    for label, embedded in zip(('serial', 'parallel'), report['records']):
        C = D / 'candidate' / label
        v, c, p = _load(C / 'report.private.json'), _load(C / 'COMPLETE.json'), _load(D / 'candidate' / (label + '.EXIT.json'))
        if (embedded['label'] != label or not same_json(embedded['report'], v)
                or not same_json(embedded['fresh_cli_seconds'], p['fresh_cli_seconds'])
                or type(p['exit_code']) is not int or p['exit_code'] != 0
                or v['status'] != CHILD_STATUS or c['status'] != CHILD_STATUS
                or c['report_sha256'] != sha(C / 'report.private.json')
                or v['complete_default_GA'] is not True or not same_json(v['policy'], policy)
                or not same_json(v['cpu_affinity'], [92, 93]) or v['distance_backend'] != label
                or v['source_manifest_sha256'] != expected_source_manifest_sha256
                or v['diagnostic_entry_sha256'] != protocol['diagnostic_entry_sha256']
                or v['diagnostic_entry_sha256'] != mf['payload']['tools/benchmark_recon_topology_genetic_distance_wiring.py']['sha256']
                or not same_json(v['input_sha256'], protocol['input_sha256'])
                or not same_json(v['scores'], 496) or not same_json(v['search_evaluations'], 495)
                or not same_json(v['generation_trace_count'], 29) or not same_json(v['last_zero_based_generation_index'], 28)
                or not same_json(v['random_draws'], 1362381)
                or v['global_function_replacement'] is not False
                or v['explicit_production_search_parameter'] is not True
                or v['explicit_distance_backend_parameter'] is not True):
            raise ValueError('actual complete same-input default child binding required')
        _source_modules(S, mf, v['actual_loaded_module_sha256'])
        _loaded_helpers(mf, v['actual_loaded_source_sha256'])
        _thread_budget(v['thread_budget'])
        distance = v['distance_policy']
        _keys(distance, ('requested_backend', 'executed', 'actual_backend', 'skipped_unmri',
            'caller_numba_thread_mask', 'slab_count', 'effective_unmri_weight', 'worker_count_observed'))
        if (distance['requested_backend'] != label or distance['actual_backend'] != label
                or distance['executed'] is not True or distance['skipped_unmri'] is not False
                or not same_json(distance['caller_numba_thread_mask'], 2 if label == 'parallel' else None)
                or not same_json(distance['slab_count'], 2 if label == 'parallel' else 1)
                or distance['worker_count_observed'] is not False):
            raise ValueError('actual executed distance route differs')
        _number(distance['effective_unmri_weight'])
        for timings, keys in ((v['stages'], STAGES), (v['nested_search_seconds'], NESTED_STAGES)):
            _keys(timings, keys)
            for value in timings.values():
                _number(value)
        _keys(v['checkpoint_sha256'], NPZ_FILES + CONTROL_FILES)
        for name, digest in v['checkpoint_sha256'].items():
            if sha(C / name) != _digest(digest):
                raise ValueError('actual complete child artifact differs')
        current_controls = {n: _load(C / n) for n in CONTROL_FILES}
        _full_controls(current_controls['search_control.private.json'], v, C / 'all_scores.private.npz',
            C / 'final_candidate.private.npz', current_controls['candidate_control.private.json'])
        if v['target_cuda'] != 'hidden' or v['allocated_bytes'] is not None or v['reserved_bytes'] is not None:
            raise ValueError('actual CPU-only memory policy required')
        _integer(v['self_peak_rss_kib'], 1)
        _number(v['complete_worker_seconds_before_report'])
        for key in ('whole_recon_all', 'final_mesh_quality', 'overall_metric_equivalence'):
            if v[key] != 'not_assessed':
                raise ValueError('first-defect scope required')
        selected = {k: v[k] for k in ('distance_backend', 'distance_policy', 'thread_budget', 'policy',
            'scores', 'search_evaluations', 'generation_trace_count', 'last_zero_based_generation_index',
            'stages', 'nested_search_seconds', 'random_draws', 'random_sequence_sha256', 'input_sha256',
            'cpu_affinity', 'self_peak_rss_kib', 'allocated_bytes', 'reserved_bytes',
            'actual_loaded_module_sha256', 'complete_worker_seconds_before_report')}
        selected.update(actual_final_fitness=current_controls['search_control.private.json']['final_fitness'],
            actual_search_policy=current_controls['search_control.private.json']['actual_policy'],
            additional_discarded=current_controls['search_control.private.json']['additional_discarded'])
        rows.append(dict(label=label, fresh_child_cli_seconds=_number(p['fresh_cli_seconds']), report=selected,
            actual_report_sha256=sha(C / 'report.private.json'), actual_complete_sha256=sha(C / 'COMPLETE.json'),
            actual_child_exit_sha256=sha(D / 'candidate' / (label + '.EXIT.json')),
            output_sha256={k.replace('.private', ''): h for k, h in v['checkpoint_sha256'].items()}))
        originals.append(v)
        controls.append(current_controls)
    fields = 0
    for name in NPZ_FILES:
        a, b = D / 'candidate/serial' / name, D / 'candidate/parallel' / name
        _comparison(report['comparisons'][name], a)
        fields += _archive_bytes(a, b, ARCHIVE_FIELDS[name])
        if sha(a) != sha(b):
            raise ValueError('complete numeric archive SHA differs')
    if fields != 81 or not same_json(report['compared_numeric_fields'], 81):
        raise ValueError('all 81 numeric fields required')
    for name in CONTROL_FILES:
        if report['controls'][name] is not True or not same_json(controls[0][name], controls[1][name]):
            raise ValueError('actual type-sensitive controls differ')
    rng = _digest(originals[0]['random_sequence_sha256'])
    if rng != originals[1]['random_sequence_sha256'] or rng != report['actual_random_bounds_and_values_sha256']:
        raise ValueError('actual complete RNG bounds/value sequence differs')
    old = _old_reference(root=D.parent.parent, protocol=protocol, candidate=D / 'candidate/serial')
    selected = dict(status=PAIR_STATUS, source_git=SOURCE_GIT,
        source_manifest_sha256=expected_source_manifest_sha256, protocol_sha256=expected_protocol_sha256,
        driver_sha256=driver_sha, request_sha256=sha(D / 'request.private.json'),
        records=rows, comparisons={k.replace('.private', ''): v for k, v in report['comparisons'].items()},
        controls={k.replace('.private', ''): v for k, v in report['controls'].items()},
        control_scope='typed_json_value_equality_not_signed_zero_bits_or_text_bytes',
        compared_numeric_fields=fields, actual_random_bounds_and_values_sha256=rng,
        parent_driver_wall_seconds=_number(process['complete_pair_wall_seconds']),
        pair_internal_wall_seconds=_number(report['complete_pair_wall_seconds']),
        external_fresh_parent_cli_seconds=None, complete_default_GA=True, old_reference=old,
        production_default_distance_backend='serial', whole_recon_all='not_assessed',
        final_mesh_quality='not_assessed', overall_metric_equivalence='not_assessed',
        shared_load_fresh_pair=True, bounded_worker_pool_observation=False)
    data, count = serialize(selected)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / 'metrics.json').write_bytes(data)
    receipt = dict(status='complete_safe_default_first_distance_pair', exporter_sha256=sha(__file__),
        source_manifest_sha256=expected_source_manifest_sha256, protocol_sha256=expected_protocol_sha256,
        actual_report_sha256=expected_report_sha256, actual_complete_sha256=expected_complete_sha256,
        actual_exit_sha256=expected_exit_sha256, driver_sha256=driver_sha,
        numeric_values_and_types_preserved=True, numeric_boolean_null_leaves_preserved=count,
        payload={'metrics': dict(sha256=sha(out / 'metrics.json'), size=len(data))})
    (out / 'EXPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'COMPLETE.json').write_text(json.dumps(dict(status=receipt['status'],
        metrics_sha256=sha(out / 'metrics.json'), receipt_sha256=sha(out / 'EXPORT_RECEIPT.json')), indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-run', 'frozen-source', 'output-directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source-manifest-sha256', 'expected-protocol-sha256',
            'expected-report-sha256', 'expected-complete-sha256', 'expected-exit-sha256'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(export_completed(**vars(parser.parse_args()))))


if __name__ == '__main__':
    main()
