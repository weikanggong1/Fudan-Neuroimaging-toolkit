#!/usr/bin/env python3
"""Conditional CPU final comparison for six explicit v3 successes + four v4.

The failed v3 controller remains failed. Receipts and actual origin declarations
are metadata bindings, never a synthesized completed controller.
"""
from __future__ import annotations
import json
from pathlib import Path
import watch_connectome_actual_completion as base


def metadata_gate(config, compare):
    for item in config['static_JSON_bindings']:
        base.bound(item)
    prior = base.bound(config['prior_v2'])
    base.check(prior['status'] == 'failed_actual_comparison' and prior.get('end_utc'), 'actual preserved v2 failure required')
    base.check({key for key, row in prior['cases'].items() if row['status'] == 'completed_comparison'} == {'sub-CON01', 'sub-CON03'}, 'prior pair set changed')
    for key, pairs in config['prior_v2_pairs'].items():
        for name, item in pairs.items():
            value = base.bound(item)
            base.check(value['status'] == 'completed' and all(prior['cases'][key][name][field] == item[field] for field in ('path', 'sha256')), 'prior pair chain differs')
    attempts = config['selected_attempts']
    base.check(len(attempts) == 2 and 'completed_case_subset_receipt' in attempts[0] and 'completed_case_subset_receipt' not in attempts[1], 'explicit failed-v3 receipt plus normal-v4 declarations required')
    subset = compare.gpu_origins.completed_subset()
    receipt = subset.bound_json(attempts[0]['completed_case_subset_receipt'])
    rebuilt = subset.build_receipt(receipt['driver_status'], receipt['recovery_origins'], receipt['completed_case_keys'], receipt['producer'])
    base.check(receipt == rebuilt and receipt['driver_status']['path'] == attempts[0]['driver_status'] and
               receipt['completed_case_keys'] == attempts[0]['completed_case_keys'], 'actual failed v3 receipt changed')
    first_origins = subset.bound_json(receipt['recovery_origins'])
    completed = [{**row, 'completed_case_subset_receipt': attempts[0]['completed_case_subset_receipt']} for row in first_origins['bindings']
                 if row['arm'] + '/' + row['case_id'] in receipt['completed_case_keys']]
    second = attempts[1]; driver_path = Path(second['driver_status'])
    if not driver_path.exists():
        return {'status': 'pending_actual_v4_driver', 'v3_completed_subset_preserved': 6, 'original_v3_driver_status': receipt['original_driver_status'], 'final_CPU_comparison_started': False}
    driver = base.read(driver_path)
    base.check(driver.get('mode') == compare.gpu_origins.SELECTED_MODE and driver.get('selected_pairs') == second['completed_case_keys'] and driver.get('full_ten_complete') is False, 'actual v4 declared scope differs')
    if driver.get('status') != 'completed_selected_subset':
        base.check(driver.get('status') in {'preflighting', 'running', 'waiting'}, 'v4 ended without complete eligibility: ' + str(driver.get('status')))
        return {'status': 'pending_actual_v4_completed_subset', 'v3_completed_subset_preserved': 6,
                'v4_completed_observed': sum(x.get('status') == 'completed' for x in driver.get('cases', {}).values()),
                'v4_expected': 4, 'original_v3_driver_status': receipt['original_driver_status'], 'final_CPU_comparison_started': False}
    base.check(driver.get('selected_attempted') == driver.get('selected_completed') == 4 and set(driver.get('cases', {})) == set(second['completed_case_keys']), 'v4 four-pair actual completion required')
    origins_path = Path(second['recovery_origins'])
    if not origins_path.exists():
        return {'status': 'pending_finalized_v4_origins', 'final_CPU_comparison_started': False}
    origins = base.read(origins_path); rows = origins.get('bindings', [])
    keys = [row['arm'] + '/' + row['case_id'] for row in rows]
    if len(rows) < 4:
        return {'status': 'pending_finalized_v4_origins', 'final_CPU_comparison_started': False}
    base.check(len(keys) == len(set(keys)) == 4 and set(keys) == set(second['completed_case_keys']), 'v4 origin duplicates/unknown cases')
    driver_identity = {'path': str(driver_path), 'sha256': base.sha(driver_path)}
    if any(row['replacement']['driver_status'] != driver_identity for row in rows):
        return {'status': 'pending_finalized_v4_origins', 'final_CPU_comparison_started': False}
    base.check(all('completed_case_subset_receipt' not in row for row in rows), 'normal v4 must not borrow failed v3 receipt')
    final_rows = completed + rows
    final_keys = [row['arm'] + '/' + row['case_id'] for row in final_rows]
    base.check(len(final_keys) == len(set(final_keys)) == 10 and final_keys == config['selected_pairs'], 'final explicit six+four unique origin map differs')
    # Only merge immutable real origin declarations. Driver files/status/cases
    # are untouched, and failed v3 CON08 cannot replace successful v4 CON08.
    value = {'schema_version': 1, 'scope': 'explicit_actual_GPU_monitor_recovery', 'bindings': final_rows}
    path = Path(config['GPU_origin_bindings'])
    if path.exists():
        base.check(base.read(path) == value, 'existing final origin map differs; never overwrite')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    # Preserve the original full report/outputs/source/runtime/CLI/memory gates.
    from types import SimpleNamespace
    options = SimpleNamespace(**{k: Path(v) for k, v in config['comparison_inputs'].items()})
    cases = compare.manifest_cases(base.read(options.manifest))
    bindings, origin_identity = compare.gpu_origins.load_bindings(path, cases, options)
    base.check(set(bindings) == {(row['arm'], row['case_id']) for row in final_rows}, 'frozen reader omitted origin')
    atlases = base.read(options.baseline_root / 'cohort_config.json')['atlases']
    reports = []; logical_sources = {}
    for case in cases:
        for arm in ('baseline', 'candidate'):
            root, actual_driver, binding = compare.gpu_origins.selected_origin(options, arm, case['case_id'], bindings)
            wall = base.read(root / arm / case['case_id'] / 'raw_bids_wall.json')
            subject = compare.gpu_origins.selected_proof(binding, case)[0]['anatomy_subject_dir'] if binding else wall['selected_inputs']['freesurfer_subject_dir']
            _, evidence = compare.validate_gpu_run(root, actual_driver, arm, case, subject, atlases, {}, GPU_origin_binding=binding)
            reports.append({'arm': arm, 'case_id': case['case_id'], 'GPU_report': evidence['gpu_report'], 'wall_report': evidence['wall_report']})
            logical_sources[arm + '/' + case['case_id']] = str(root / arm / case['case_id'])
    base.check(len(reports) == len(logical_sources) == 20, 'twenty unique logical report sources required')
    return {'status': 'ready_actual_finalized_twenty_reports', 'GPU_origin_bindings': origin_identity,
            'failed_v3_driver': receipt['driver_status'], 'completed_case_subset_receipt': attempts[0]['completed_case_subset_receipt'],
            'v4_driver': driver_identity, 'v4_origins': {'path': str(origins_path), 'sha256': base.sha(origins_path)},
            'original_v3_driver_status': receipt['original_driver_status'], 'failed_v3_case_keys': receipt['failed_case_keys'],
            'not_dispatched_v3_case_keys': receipt['not_dispatched_case_keys'], 'logical_sources': logical_sources,
            'immutable_completed_reports': reports, 'prior_v2': config['prior_v2'], 'prior_v2_pairs': config['prior_v2_pairs'],
            'final_CPU_comparison_started': False, 'MRI_or_GPU_started': False}


if __name__ == '__main__':
    base.metadata_gate = metadata_gate
    raise SystemExit(base.main())
