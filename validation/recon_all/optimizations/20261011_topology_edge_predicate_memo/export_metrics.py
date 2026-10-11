"""仅在双方收据完整后复核固定全字段，导出typed计时/hash；不输出私有影像或路径。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, *, integer=False, nullable=False):
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid numeric metric')
    if integer and not isinstance(value, int):
        raise ValueError('integer metric required')
    return value


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('SHA-256 metric required')
    return value


def commit(value):
    if not isinstance(value, str) or len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Git commit hash required')
    return value


def export(*, baseline, memo, controller, source_root, source_manifest, protocol, output):
    """重核完整24比较项/全schema和来源收据，只发布预定义数值及hash。"""
    baseline, memo, controller, output = map(Path, (baseline, memo, controller, output))
    if output.exists():
        raise FileExistsError('fresh typed output required')
    # No report inspection before both stage receipts have arrived.
    if not all((p / 'COMPLETE.json').is_file() for p in (baseline, memo)):
        raise ValueError('both complete receipts required before export')
    declared = json.loads(Path(protocol).read_text())
    git_base = commit(declared['git_base'])
    threads = number(declared['resources']['requested_threads'], integer=True)
    if threads not in (2,4):
        raise ValueError('declared fixed thread budget required')
    frozen_root = Path(source_root).resolve()
    tool = frozen_root / 'tools/benchmark_recon_topology_edge_predicate_cache.py'
    manifest = json.loads(Path(source_manifest).read_text())
    if sha(tool) != manifest['files']['tools/benchmark_recon_topology_edge_predicate_cache.py']['sha256']:
        raise ValueError('actual comparison driver source mismatch')
    spec = importlib.util.spec_from_file_location('fnit_edge_memo_frozen_export_reader', tool)
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    reader.validate_source(dict(source_root=str(frozen_root), source_manifest=str(source_manifest),
        source_manifest_sha256=sha(source_manifest), protocol=str(protocol), protocol_sha256=sha(protocol),
        threads=declared['resources']['requested_threads'], cpu_affinity=declared['resources']['candidate_cpu_affinity'],
        input_sha256=declared['input_sha256'], maximum_entries=declared['cache']['maximum_entries']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fnit-memo-export-check-', dir=output.parent) as temporary:
        checked = reader.compare_completed(baseline=baseline, candidate=memo,
            output=Path(temporary) / 'checked.private.json', source_manifest=source_manifest, protocol=protocol)
    original = json.loads((controller / 'pair.private.json').read_text())
    if (checked['status'] != 'passed_fixed_candidate_pair' or len(checked['checks']) != 24
            or original['status'] != checked['status'] or original['checks'] != checked['checks']
            or original['reports_sha256'] != checked['reports_sha256']):
        raise ValueError('original and independently rechecked complete gates differ')
    for row in checked['checks']:
        expected = {'candidate': 36, 'state': 40}.get(row['kind'])
        if row['all_shape_dtype_bytes_equal'] is not True or expected is not None and row['field_count'] != expected:
            raise ValueError('complete field/control/sequence gate required')
    cli = json.loads((controller / 'cli.private.json').read_text())
    if ([r['arm'] for r in cli['records']] != ['baseline', 'memo']
            or any(r['exit_code'] != 0 for r in cli['records'])
            or cli['includes_imports_preflight_raw_preparation_three_candidates_io_exit'] is not True
            or cli['no_other_arm_outputs_read_before_both_complete'] is not True):
        raise ValueError('complete ordered CLI receipts required')
    rows = {}
    clock_fields = ('preparation_seconds','ordering_seconds','complete_cycle_seconds',
        'preparation_and_cycle_seconds','full_candidate_api_seconds','candidate_api_and_output_seconds',
        'complete_scan_api_seconds','scanner_preparation_seconds','output_npz_control_write_seconds')
    cache_counts = ('vertex_count','maximum_entries','entries','logical_predicate_calls','cache_hits',
        'actual_predicate_evaluations','inserted_unique_keys','capacity_bypass_evaluations')
    for index, (arm, path) in enumerate((('baseline', baseline), ('memo', memo))):
        report = json.loads((path / 'report.private.json').read_text())
        policy = report['actual_thread_policy']; requested = threads
        expected_environment = {'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','NUMBA_NUM_THREADS'}
        if (policy['restoration_complete'] is not True or policy['requested_threads'] != requested
                or policy['torch']['effective'] != requested or policy['numba']['effective'] != requested
                or policy['torch']['before'] != policy['torch']['restored']
                or policy['numba']['before'] != policy['numba']['restored']
                or set(report['fresh_thread_environment']) != expected_environment
                or any(v != str(requested) for v in report['fresh_thread_environment'].values())
                or report['global_function_replacement'] is not True or report['production_wiring'] is not False
                or report['cuda_hidden'] is not True or report['cache_shared_between_repeats'] is not False):
            raise ValueError('actual thread/cache/diagnostic strategy gate required')
        runs = {}
        for record in report['records']:
            clocks = {name: number(record[name]) for name in clock_fields}
            clocks['retessellation_seconds'] = number(record['retessellation_seconds'], nullable=True)
            cache = None
            if arm == 'baseline' and record['cache'] is not None:
                raise ValueError('baseline must be uncached')
            if arm == 'memo':
                c = record['cache']; cache = {name: number(c[name], integer=True) for name in cache_counts}
                cache.update(hit_rate=number(c['hit_rate']), preparation_seconds=number(c['preparation_seconds']),
                    complete_unique_key_count=number(c['complete_unique_key_count'], integer=True, nullable=True),
                    canonical_sha256=digest(c['canonical_sha256']))
                if (cache['cache_hits'] + cache['actual_predicate_evaluations'] != cache['logical_predicate_calls']
                        or cache['inserted_unique_keys'] + cache['capacity_bypass_evaluations'] != cache['actual_predicate_evaluations']
                        or cache['entries'] != cache['inserted_unique_keys'] or cache['hit_rate'] > 1
                        or cache['maximum_entries'] != declared['cache']['maximum_entries']
                        or cache['entries'] > cache['maximum_entries']
                        or cache['capacity_bypass_evaluations'] > 0 and cache['complete_unique_key_count'] is not None):
                    raise ValueError('invalid cache count/capacity metric')
            runs[record['run']] = dict(seconds=clocks, cache=cache,
                logical_predicate_calls=number(record['logical_predicate_calls'], integer=True),
                sequence_sha256=digest(record['sequence_sha256']) if record['run'] == 'audit' else None)
        rows[arm] = dict(runs=runs, complete_cold_cli_wall_seconds=number(cli['records'][index]['complete_cold_cli_wall_seconds']),
            peak_process_rss_bytes=number(report['peak_process_rss_bytes'], integer=True),
            input_sha256={name:digest(report['input_sha256'][name]) for name in ('nofix','qsphere','brain')},
            report_sha256=digest(checked['reports_sha256'][index]),
            actual_loaded_fnit_module_count=len(report['actual_loaded_fnit_sha256']),
            actual_loaded_benchmark_helper_count=len(report['actual_loaded_benchmark_tool_sources']),
            actual_tool_sha256=digest(report['actual_tool_sha256']),
            actual_torch_numba_restored=True, actual_rng_draws=0)
    result = dict(status='passed_fixed_identity_candidate_pair_not_full_ga',
        git_base=git_base, source_manifest_sha256=digest(sha(source_manifest)),
        protocol_sha256=digest(sha(protocol)), producer_sha256=digest(sha(tool)), exporter_sha256=digest(sha(__file__)),
        complete_gate_count=24, candidate_npz_field_count=36, state_npz_field_count=40, control_field_count=6,
        strict_candidate_state_control_and_sequence_reproduction=True, arms=rows,
        threads=threads, gpu_exercised=False, allocated_bytes=None, reserved_bytes=None,
        comparison_recheck_seconds=number(checked['comparison_hash_npz_control_read_seconds']),
        production_wiring=False, default_changed=False, official_new_comparison=False, full_ga=False,
        real_cross_candidate_search=False, final_mesh_quality='not_assessed', whole_recon_all='not_assessed',
        overall_equivalence='not_assessed', os_isolation_verified=False,
        cold_cli_includes_three_candidates_and_io=True, same_arm_independent_memo_each_repeat=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','memo','controller','source-root','source-manifest','protocol','output'):
        p.add_argument('--' + name, type=Path, required=True)
    export(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
