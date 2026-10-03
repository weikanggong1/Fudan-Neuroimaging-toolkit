#!/usr/bin/env python3
"""Run only the frozen mixed metadata waiter and final CPU export, when ready."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import watch_connectome_actual_completion as base


def require_completed_tables(waiter, summary):
    base.check(waiter.get('status') == 'completed_actual_CPU_comparison_and_tables' and
               waiter.get('matrix_checks') == 320 and waiter.get('FS_case_comparisons') == 10,
               'final export cannot precede actual ten-pair CPU comparison')
    base.check(summary.get('status') == 'complete_actual_ten_case_tables' and
               summary.get('ready_for_ten_case_render') is True and summary.get('completed_pairs') == 10,
               'final export requires actual complete immutable summary')
    matrices = summary.get('matrix_rows', []); anatomy = summary.get('anatomy_rows', [])
    base.check(len(matrices) == len({(r['case_id'], r['atlas'], r['kind']) for r in matrices}) == 320 and
               len(anatomy) == len({(r['case_id'], r['file']) for r in anatomy}) == 130 and
               len({r['case_id'] for r in anatomy}) == 10, 'unique 320 matrix and 10 x 13 FS summary coverage required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--mixed-tool', type=Path, required=True)
    parser.add_argument('--mixed-report-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--export-report-dir', type=Path, required=True)
    parser.add_argument('--representative-cases', nargs='+', default=['sub-CON01', 'sub-CON09'])
    parser.add_argument('--figure-atlas', default='fs-aparc')
    options = parser.parse_args()
    config = base.read(options.configuration); compare, _ = base.load_readers(config)
    base.check(options.mixed_tool.name == 'watch_connectome_actual_mixed_completion.py' and
               {'path': str(options.mixed_tool), 'sha256': base.sha(options.mixed_tool)} in config['helper_files'], 'explicit pinned mixed metadata entrypoint required')
    for p in (options.report_dir, options.mixed_report_dir, options.export_report_dir):
        base.check(p.is_absolute() and not p.exists(), 'fresh absolute CPU chain/waiter/export namespaces required')
    protected = [config['helper_directory'], options.configuration.parent, *[Path(x['path']).parent for x in config['static_JSON_bindings']], config['comparison_report_dir'], config['summary_report_dir']]
    compare.check_report_namespace(options.report_dir, protected + [options.mixed_report_dir, options.export_report_dir])
    compare.check_report_namespace(options.export_report_dir, protected + [options.report_dir, options.mixed_report_dir])
    options.report_dir.mkdir(parents=True, exist_ok=False)
    (options.report_dir / 'configuration.original_bytes.json').write_bytes(options.configuration.read_bytes())
    config_sha = base.sha(options.configuration); tool_sha = base.sha(__file__); started = time.monotonic()
    receipt_identity = config['selected_attempts'][0]['completed_case_subset_receipt']
    receipt = base.bound(receipt_identity)
    state = {'status': 'waiting_actual_mixed_completion', 'CPU_chain_sha256': tool_sha, 'configuration_sha256': config_sha,
             'failed_v3_driver': receipt['driver_status'], 'original_v3_driver_status': receipt['original_driver_status'],
             'completed_case_subset_receipt': receipt_identity, 'failed_v3_case_keys': receipt['failed_case_keys'],
             'not_dispatched_v3_case_keys': receipt['not_dispatched_case_keys'], 'GPU_or_MRI_started': False, 'stages': []}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    base.atomic(options.report_dir / 'status.json', state)
    try:
        argv = [config['CPU_python'], str(options.mixed_tool), '--configuration', str(options.configuration), '--report-dir', str(options.mixed_report_dir), '--watch', '--launch', '--poll-seconds', '60', '--timeout-hours', '72']
        with (options.report_dir / 'mixed_waiter.log').open('xb') as log:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment)
        state['stages'].append({'stage': 'mixed_metadata_wait_then_original_CPU_comparison_summary', 'argv': argv, 'exit_code': result.returncode})
        waiter = base.read(options.mixed_report_dir / 'status.json')
        state['mixed_waiter_status'] = {'path': str(options.mixed_report_dir / 'status.json'), 'sha256': base.sha(options.mixed_report_dir / 'status.json')}
        base.check(result.returncode == 0, 'mixed metadata/comparison failed; preserve actual failure, no export')
        if waiter.get('status') != 'completed_actual_CPU_comparison_and_tables':
            state.update(status='pending_or_stopped_actual_mixed_outputs', actual_mixed_status=waiter.get('status'), export_started=False)
        else:
            summary = base.read(Path(config['summary_report_dir']) / 'summary.json')
            require_completed_tables(waiter, summary)
            base.check(base.sha(options.configuration) == config_sha and base.sha(__file__) == tool_sha, 'frozen CPU metadata chain/config changed')
            base.load_readers(config)
            export_tool = Path(config['helper_directory']) / 'export_connectome_actual_cohort.py'
            argv = [config['CPU_python'], str(export_tool), '--comparison-root', config['comparison_report_dir'], '--summary-root', config['summary_report_dir'],
                    '--report-dir', str(options.export_report_dir), '--representative-cases', *options.representative_cases, '--figure-atlas', options.figure_atlas,
                    '--candidate-source-label', '641f16b frozen source; six verified v3 receipt cases +four normal v4 cases']
            state.update(status='running_original_CPU_export', export_started=True)
            base.atomic(options.report_dir / 'status.json', state)
            with (options.report_dir / 'export.log').open('xb') as log:
                result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment)
            state['stages'].append({'stage': 'original_CPU_export', 'argv': argv, 'exit_code': result.returncode})
            export = base.read(options.export_report_dir / 'status.json')
            base.check(result.returncode == 0 and export.get('status') == 'completed_actual_ten_case_export' and export.get('completed_pairs') == 10, 'original CPU export did not finish ten actual pairs')
            base.load_readers(config)
            for item in config['static_JSON_bindings']: base.bound(item)
            state.update(status='completed_actual_ten_unique_comparison_summary_export', matrix_checks=320, FS_case_comparisons=10, FS_array_checks=130,
                         export_status={'path': str(options.export_report_dir / 'status.json'), 'sha256': base.sha(options.export_report_dir / 'status.json')},
                         scope='read-only CPU comparison/export of actual ten unique pairs; original v3 failure retained; no MRI/GPU rerun or continuous cold-wall/MRtrix acceptance claim')
    except Exception as error:
        state.update(status='failed_actual_mixed_CPU_chain', error={'type': type(error).__name__, 'message': str(error)})
    state['CPU_chain_observation_seconds'] = time.monotonic() - started
    base.atomic(options.report_dir / 'status.json', state)
    print(json.dumps({'status': state['status'], 'error': state.get('error'), 'GPU_or_MRI_started': False}))
    return int(state['status'].startswith('failed'))


if __name__ == '__main__':
    raise SystemExit(main())
