"""Explicit immutable six-case receipt for one finished selected science failure.

Only metadata is read/written. A receipt never relabels a failed controller or
admits its failed case. Original scientific readers still validate each result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

SCOPE = 'explicit_immutable_completed_case_subset_from_failed_selected_driver'
MODE = 'selected_same_round_anatomy_fresh_raw_dwi_monitor_recovery'
REPORTS = ('gpu_report.json', 'raw_bids_wall.json', 'recovery_eligibility.json', 'recovery_config.json', 'recovery_binding.json')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    check(path.is_absolute() and path.is_file() and not path.is_symlink(), 'actual unlinked receipt source required')
    return {'path': str(path), 'sha256': sha(path)}


def bound_json(item):
    check(isinstance(item, dict) and set(item) == {'path', 'sha256'} and identity(item['path']) == item,
          'immutable completed-subset source changed')
    return json.loads(Path(item['path']).read_bytes())


def record_sha(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def terminal_layout(driver, completed_keys):
    selected = driver.get('selected_pairs', [])
    check(driver.get('mode') == MODE and driver.get('status') == 'selected_recovery_failed_or_ineligible' and
          driver.get('full_ten_complete') is False and len(selected) == len(set(selected)) == 10,
          'only explicit known finished selected failure qualifies; running/unknown failure rejected')
    check(isinstance(completed_keys, list) and len(completed_keys) == len(set(completed_keys)) == 6 and
          completed_keys == selected[:6], 'receipt requires exact six explicit completed keys')
    records = driver.get('cases', {})
    check(driver.get('selected_attempted') == 7 and driver.get('selected_completed') == 6 and
          set(records) == set(selected[:7]), 'failed driver attempted/completed/pending coverage differs')
    for key in completed_keys:
        record = records[key]; eligibility = record.get('response', {}).get('eligibility', {})
        check(record.get('status') == 'completed' and not record.get('error') and not record.get('validation_error') and
              eligibility.get('status') == 'execution_complete_memory_observed_below_budget' and
              eligibility.get('execution_status') == 'completed' and eligibility.get('validation_error') is None and
              eligibility.get('full_ten_complete') is False, 'receipt includes a failed/ineligible/unknown case')
        budget = eligibility.get('memory_budget', {})
        check(budget.get('status') == 'observed_below_budget' and not budget.get('monitor_issues'), 'receipt success has incomplete memory eligibility')
    failed_key = selected[6]; failed = records[failed_key]
    response = failed.get('response', {}); GPU = response.get('gpu_result', {}); eligibility = response.get('eligibility', {})
    check(failed.get('status') == 'failed_or_ineligible' and eligibility.get('status') == 'not_eligible' and
          eligibility.get('execution_status') == 'failed' and GPU.get('status') == 'failed' and
          isinstance(GPU.get('exit_code'), int) and not isinstance(GPU['exit_code'], bool) and GPU['exit_code'] != 0,
          'unknown terminal failure or monitor-only ineligibility cannot create this receipt')
    return selected, records, failed_key


def build_receipt(driver_identity, origins_identity, completed_keys, producer):
    driver = bound_json(driver_identity); origins = bound_json(origins_identity)
    check(identity(producer['path']) == producer, 'receipt producer bytes changed')
    selected, records, failed_key = terminal_layout(driver, completed_keys)
    rows = origins.get('bindings', [])
    check(set(origins) == {'schema_version', 'scope', 'bindings'} and origins['schema_version'] == 1 and
          origins['scope'] == 'explicit_actual_GPU_monitor_recovery' and len(rows) == 7 and
          [row['arm'] + '/' + row['case_id'] for row in rows] == selected[:7], 'actual terminal seven-row origins required')
    ledger = []
    for index, key in enumerate(selected):
        if key not in records:
            ledger.append({'case_key': key, 'driver_record_absent': True, 'status': 'not_dispatched_after_terminal_failure'})
            continue
        record = records[key]; response = record['response']; row = rows[index]
        arm, case_id = key.split('/', 1)
        check(record.get('version') == arm and record.get('case_id') == case_id and
              response.get('gpu_result', {}).get('version') == arm and response.get('gpu_result', {}).get('case_id') == case_id,
              'actual attempted record/GPU case identity differs')
        check(row['replacement']['driver_status'] == driver_identity, 'actual terminal origins driver identity differs')
        job = Path(row['replacement']['root'])
        check(str(job) == response.get('job_root') and job.name == key.split('/')[1] and job.parent.name == key.split('/')[0], 'actual report job identity differs')
        report_bindings = {name: identity(job / name) for name in REPORTS}
        check(response.get('GPU_report') == report_bindings['gpu_report.json'] and
              response.get('wall_report') == report_bindings['raw_bids_wall.json'] and
              response.get('configuration') == report_bindings['recovery_config.json'] and
              bound_json(report_bindings['gpu_report.json']) == response.get('gpu_result') and
              bound_json(report_bindings['recovery_eligibility.json']) == response.get('eligibility') and
              bound_json(report_bindings['recovery_config.json']) == response.get('new_config') and
              bound_json(report_bindings['recovery_binding.json']) == response.get('binding'),
              'actual attempted record/report/config/proof differs')
        ledger.append({'case_key': key, 'status': record['status'], 'driver_record_sha256': record_sha(record), 'report_bindings': report_bindings})
    return {'schema_version': 1, 'scope': SCOPE, 'producer': producer, 'driver_status': driver_identity,
            'recovery_origins': origins_identity, 'selected_pairs': selected, 'completed_case_keys': completed_keys,
            'failed_case_keys': [failed_key], 'not_dispatched_case_keys': selected[7:], 'case_ledger': ledger,
            'failed_records': {failed_key: records[failed_key]}, 'original_driver_status': driver['status'],
            'full_ten_complete': False, 'original_driver_rewritten': False,
            'scope_note': 'six actual completed cases only; whole terminal failed driver and all failed/pending records retained; each accepted case still requires original raw/FS/source/output/runtime/CLI/UUID/memory gates'}


def verify_receipt(receipt_identity, declaration, driver):
    receipt = bound_json(receipt_identity)
    check(receipt.get('schema_version') == 1 and receipt.get('scope') == SCOPE and
          receipt.get('driver_status') == declaration['replacement']['driver_status'] and
          driver == bound_json(receipt['driver_status']), 'receipt is outside the declared actual failed driver')
    rebuilt = build_receipt(receipt['driver_status'], receipt['recovery_origins'], receipt['completed_case_keys'], receipt['producer'])
    check(receipt == rebuilt, 'receipt ledger altered or incomplete; no synthesized completed controller accepted')
    key = declaration['arm'] + '/' + declaration['case_id']
    check(key in receipt['completed_case_keys'], 'failed/pending case cannot use completed-subset receipt')
    origins = bound_json(receipt['recovery_origins'])
    original_row = next(row for row in origins['bindings'] if row['arm'] + '/' + row['case_id'] == key)
    actual = {k: v for k, v in declaration.items() if k != 'completed_case_subset_receipt'}
    check(actual == original_row, 'receipt declaration differs from immutable actual original completed origin')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-status', type=Path, required=True)
    parser.add_argument('--origins', type=Path, required=True)
    parser.add_argument('--case-key', action='append', required=True, help='exact six completed keys, explicitly ordered as actual selected list')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    check(options.output.is_absolute() and not options.output.exists(), 'new absolute receipt output required')
    value = build_receipt(identity(options.driver_status), identity(options.origins), options.case_key, identity(Path(__file__).resolve()))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': 'immutable_six_completed_receipt_created', 'receipt': identity(options.output), 'GPU_or_MRI_started': False}))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
