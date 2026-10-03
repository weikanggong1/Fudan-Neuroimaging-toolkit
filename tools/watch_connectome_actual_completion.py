#!/usr/bin/env python3
"""Metadata gate for immutable selected recovery and frozen CPU comparisons.

No MRI/GPU work is dispatched. Final CPU comparison requires --launch and all
actual selected origins finalized. The reader implementation is pinned by SHA.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    check(path.is_absolute() and path.is_file() and not path.is_symlink(), 'missing/linked metadata: ' + str(path))
    return json.loads(path.read_bytes())


def bound(identity):
    check(set(identity) == {'path', 'sha256'}, 'explicit metadata identity required')
    value = read(identity['path'])
    check(sha(identity['path']) == identity['sha256'], 'immutable metadata changed: ' + identity['path'])
    return value


def atomic(path, value):
    path = Path(path); temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def load_readers(config):
    for identity in config['helper_files']:
        check(not Path(identity['path']).is_symlink() and sha(identity['path']) == identity['sha256'], 'frozen v8 helper changed')
    sys.path.insert(0, config['helper_directory'])
    compare = importlib.import_module('benchmark_connectome_cohort_compare')
    summary = importlib.import_module('summarize_connectome_actual_cohort')
    check(Path(compare.__file__).resolve() == Path(config['comparison_tool']).resolve(), 'wrong comparison reader imported')
    check(Path(summary.__file__).resolve() == Path(config['summary_tool']).resolve(), 'wrong summary reader imported')
    return compare, summary


def metadata_gate(config, compare):
    # Original failure, two completed v2 pairs, and original terminal bytes remain
    # bound independently of v8's v1-only supported cached anatomy namespace.
    for identity in config['static_JSON_bindings']:
        bound(identity)
    prior = bound(config['prior_v2'])
    check(prior['status'] == 'failed_actual_comparison' and prior.get('end_utc'), 'actual v2 terminal failure required')
    completed = {key for key, value in prior['cases'].items() if value['status'] == 'completed_comparison'}
    check(completed == {'sub-CON01', 'sub-CON03'}, 'prior v2 completed pair set changed')
    for case_id, reports in config['prior_v2_pairs'].items():
        check(case_id in completed, 'unknown prior pair')
        for name, identity in reports.items():
            value = bound(identity)
            check(prior['cases'][case_id][name]['path'] == identity['path'] and
                  prior['cases'][case_id][name]['sha256'] == identity['sha256'], 'prior pair binding differs')
            check(value['status'] == 'completed', 'prior pair report incomplete')
    path = Path(config['selected_driver'])
    if not path.exists():
        return {'status': 'pending_selected_driver', 'final_CPU_comparison_started': False}
    state = read(path); expected = config['selected_pairs']
    check(state.get('selected_pairs') == expected and state.get('full_ten_complete') is False, 'actual selected scope changed')
    if state.get('status') != 'completed_selected_subset':
        check(state.get('status') in {'preflighting', 'running', 'waiting'}, 'selected recovery ended without eligible completion: ' + str(state.get('status')))
        return {'status': 'pending_selected_completion', 'selected_completed_observed': sum(x.get('status') == 'completed' for x in state.get('cases', {}).values()),
                'selected_expected': 10, 'final_CPU_comparison_started': False}
    check(state.get('selected_attempted') == state.get('selected_completed') == 10 and set(state['cases']) == set(expected), 'selected ten-pair completion coverage changed')
    origins_path = Path(config['GPU_origin_bindings'])
    if not origins_path.exists():
        return {'status': 'pending_finalized_origin_bindings', 'final_CPU_comparison_started': False}
    origins = read(origins_path)
    rows = origins.get('bindings', [])
    keys = [row['arm'] + '/' + row['case_id'] for row in rows]
    if len(rows) < 10:
        return {'status': 'pending_finalized_origin_bindings', 'bindings_observed': len(rows), 'final_CPU_comparison_started': False}
    check(len(rows) == 10 and len(set(keys)) == 10 and set(keys) == set(expected), 'origin bindings contain duplicates/unknown/missing pairs')
    current_driver = {'path': str(path), 'sha256': sha(path)}
    if any(row['replacement']['driver_status'] != current_driver for row in rows):
        # Producer writes terminal driver first, then atomically finalizes origins.
        return {'status': 'pending_finalized_origin_bindings', 'final_CPU_comparison_started': False}
    options = SimpleNamespace(**{key: Path(value) for key, value in config['comparison_inputs'].items()})
    cases = compare.manifest_cases(read(options.manifest))
    bindings, origin_identity = compare.gpu_origins.load_bindings(origins_path, cases, options)
    check(set(bindings) == {(row['arm'], row['case_id']) for row in rows}, 'frozen reader dropped an actual origin')
    atlases = read(Path(config['comparison_inputs']['baseline_root']) / 'cohort_config.json')['atlases']
    reports = []; logical_sources = {}
    for case in cases:
        for arm in ('baseline', 'candidate'):
            root, driver, binding = compare.gpu_origins.selected_origin(options, arm, case['case_id'], bindings)
            wall = read(root / arm / case['case_id'] / 'raw_bids_wall.json')
            subject = (compare.gpu_origins.selected_proof(binding, case)[0]['anatomy_subject_dir'] if binding else wall['selected_inputs']['freesurfer_subject_dir'])
            touched = {}
            _, evidence = compare.validate_gpu_run(root, driver, arm, case, subject, atlases, touched, GPU_origin_binding=binding)
            reports.append({'arm': arm, 'case_id': case['case_id'], 'GPU_report': evidence['gpu_report'], 'wall_report': evidence['wall_report']})
            logical_sources[arm + '/' + case['case_id']] = str(root / arm / case['case_id'])
    check(len(reports) == 20 and len(logical_sources) == 20, 'twenty logical source reports required')
    return {'status': 'ready_actual_finalized_twenty_reports', 'selected_driver': current_driver, 'GPU_origin_bindings': origin_identity,
            'logical_sources': logical_sources, 'immutable_completed_reports': reports, 'prior_v2': config['prior_v2'],
            'prior_v2_pairs': config['prior_v2_pairs'], 'final_CPU_comparison_started': False, 'MRI_or_GPU_started': False}


def launch(config, readiness, compare, summary, report_dir):
    check(readiness['status'] == 'ready_actual_finalized_twenty_reports', 'cannot launch before actual readiness')
    for name in ('comparison_report_dir', 'summary_report_dir'):
        check(not Path(config[name]).exists(), 'never overwrite final comparison/summary namespace')
    arguments = [config['CPU_python'], config['comparison_tool']]
    for key, value in config['comparison_inputs'].items():
        arguments += ['--' + key.replace('_', '-'), value]
    arguments += ['--report-dir', config['comparison_report_dir'], '--prior-comparison-dir', config['supported_v1_prior_directory'],
                  '--gpu-origin-bindings', config['GPU_origin_bindings'], '--once']
    summary_arguments = [config['CPU_python'], config['summary_tool'], '--comparison-root', config['comparison_report_dir'],
                         '--report-dir', config['summary_report_dir'], '--candidate-source-label', '641f16b frozen source; eligible selected raw-DWI origins']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    # Only the pinned read-only CPU reference CLI is allowed. No science worker.
    stages = []
    for name, argv in (('comparison', arguments), ('summary', summary_arguments)):
        with (Path(report_dir) / (name + '.log')).open('xb') as log:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment)
        stages.append({'stage': name, 'argv': argv, 'exit_code': result.returncode})
        check(result.returncode == 0, 'frozen ' + name + ' reader rejected actual reports; preserve logs and report minimal failure')
        if name == 'comparison':
            state = read(Path(config['comparison_report_dir']) / 'status.json')
            check(state.get('status') == 'completed_actual_ten_case_comparison' and state.get('completed_cases') == state.get('anatomy_compared_cases') == 10 and state.get('failed_cases') == 0,
                  'comparison did not finish ten actual pairs')
    table = read(Path(config['summary_report_dir']) / 'summary.json')
    check(table['status'] == 'complete_actual_ten_case_tables' and table['ready_for_ten_case_render'] is True and table['completed_pairs'] == 10,
          'summary incomplete; no final completion claim')
    matrix_keys = {(row['case_id'], row['atlas'], row['kind']) for row in table['matrix_rows']}
    check(len(table['matrix_rows']) == len(matrix_keys) == 320 and len(table['anatomy_rows']) == 130 and
          len({row['case_id'] for row in table['anatomy_rows']}) == 10 and
          len({(row['case_id'], row['file']) for row in table['anatomy_rows']}) == 130, 'final 320 matrix / 10 x 13 FS coverage or uniqueness failed')
    for identity in config['static_JSON_bindings']:
        bound(identity)
    return {'status': 'completed_actual_CPU_comparison_and_tables', 'stages': stages, 'matrix_checks': 320, 'FS_case_comparisons': 10,
            'FS_array_checks': 130, 'summary': {'path': str(Path(config['summary_report_dir']) / 'summary.json'), 'sha256': sha(Path(config['summary_report_dir']) / 'summary.json')},
            'prior_v2': config['prior_v2'], 'prior_v2_pairs': config['prior_v2_pairs'], 'prior_v2_pairs_counted_again_in_aggregate': False,
            'scope': 'ten unique actual pairs; v8 revalidates prior two pairs, prior failures preserved; no GPU/MRI and no continuous cold-wall or MRtrix acceptance claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--launch', action='store_true', help='only after ten finalized eligible origins; CPU read-only comparison')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--timeout-hours', type=float, default=72)
    options = parser.parse_args()
    check(1 <= options.poll_seconds <= 60 and 0 < options.timeout_hours <= 168, 'invalid bounded watch interval')
    config = read(options.configuration)
    check(options.report_dir.is_absolute() and not options.report_dir.exists(), 'fresh absolute waiter report directory required')
    compare, summary = load_readers(config)
    protected = [Path(x['path']).parent for x in config['static_JSON_bindings']]
    protected += [config['helper_directory'], Path(config['selected_driver']).parent, config['comparison_report_dir'], config['summary_report_dir']]
    compare.check_report_namespace(options.report_dir, protected)
    options.report_dir.mkdir(parents=True, exist_ok=False)
    (options.report_dir / 'configuration.original_bytes.json').write_bytes(options.configuration.read_bytes())
    tool_sha = sha(__file__); config_sha = sha(options.configuration); started = time.monotonic()
    state = {'status': 'pending', 'waiter_sha256': tool_sha, 'configuration_sha256': config_sha, 'MRI_or_GPU_started': False}
    try:
        while True:
            check(sha(__file__) == tool_sha and sha(options.configuration) == config_sha, 'frozen waiter/config changed')
            load_readers(config)
            observation = metadata_gate(config, compare)
            state.update(observation); atomic(options.report_dir / 'status.json', state)
            if observation['status'] == 'ready_actual_finalized_twenty_reports':
                if options.launch:
                    state.update(status='launching_frozen_CPU_comparison', final_CPU_comparison_started=True)
                    atomic(options.report_dir / 'status.json', state)
                    state.update(launch(config, observation, compare, summary, options.report_dir))
                    load_readers(config)
                break
            if not options.watch:
                break
            if (options.report_dir / 'STOP_OBSERVATION').exists():
                state['status'] = 'stopped_pending'; break
            if time.monotonic() - started >= options.timeout_hours * 3600:
                state['status'] = 'timed_out_pending'; break
            time.sleep(options.poll_seconds)
    except Exception as error:
        state.update(status='failed_metadata_or_frozen_CPU_reader', error={'type': type(error).__name__, 'message': str(error)})
    state['waiter_observation_seconds'] = time.monotonic() - started
    atomic(options.report_dir / 'status.json', state)
    print(json.dumps({key: state.get(key) for key in ('status', 'error', 'matrix_checks', 'FS_case_comparisons')}))
    return int(state['status'].startswith('failed'))


if __name__ == '__main__':
    raise SystemExit(main())
