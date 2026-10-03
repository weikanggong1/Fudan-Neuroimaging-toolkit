"""Attempt accounting contracts; small report fixtures are not benchmark evidence."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'validation/dmri_pipeline/public10_20261002/compare_public10.py'
spec = importlib.util.spec_from_file_location('public10_intermediate_comparison', SOURCE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def history(tmp_path):
    digest = 'a' * 64
    names = ('AP.nii.gz', 'AP.bval', 'AP.bvec', 'AP.json', 'PA.nii.gz', 'PA.bval', 'PA.json', 'T1w.nii.gz')
    reference = {'case_id': 'case01', 'registration_backend': 'mmorf',
        'input_files': {name: {'sha256': digest} for name in names},
        'templates': {name: {'sha256': digest} for name in ('FA_reference', 'T1_reference', 'tensor_reference')},
        'SynthStrip': {'weights': {'sha256': digest}}, 'parameters': {'gp_seed': 12345}}
    candidate = {'case_id': 'case01', 'registration_backend': 'mmorf', 'status': 'complete',
        'input_and_resource_provenance': {
            **{'raw_' + name: {'sha256': digest} for name in names if name != 'T1w.nii.gz'},
            **{name: {'sha256': digest} for name in ('fa_template', 'synthstrip_weights', 't1', 't1_template', 'tensor_template')}},
        'parameters': {'eddy_gp_seed': 12345}, 'timing': {'api_wall_seconds': 20}}
    write(tmp_path / 'candidate/report.json', candidate)
    write(tmp_path / 'selected/report.json', {**reference, 'status': 'complete', 'total_processing_seconds': 200})
    write(tmp_path / 'initial/report.json', {**reference, 'status': 'failed', 'failure_type': 'RuntimeError', 'total_processing_seconds': 1000})
    write(tmp_path / 'middle/report.json', {**reference, 'status': 'failed', 'failure_type': 'RuntimeError',
        'total_processing_seconds': 392.96265, 'failure_message': '/private/secret_command',
        'subprocess_steps': [{'arguments': ['secret_command', '/private/raw.nii.gz']}]})
    for name, wall, code in (('candidate', '0:50.00', 0), ('selected', '8:20.00', 0),
                              ('initial', '16:45.00', 1), ('middle', '6:35.24', 1)):
        write(tmp_path / name / 'process_metrics.json', {'wall_seconds': 9999, 'exit_code': code,
            'sampled_own_gpu_memory_peak_mib': 1186, 'sampled_other_gpu_memory_peak_mib': 34000,
            'gpu_samples': [{'secret_path': '/private/ignored'}]})
        (tmp_path / name / 'time.txt').write_text('Elapsed (wall clock) time (h:mm:ss or m:ss): ' + wall + '\n'
            'Maximum resident set size (kbytes): 12345\nExit status: ' + str(code) + '\n')
    write(tmp_path / 'comparison.json', {'case_id': 'case01', 'registration_backend': 'mmorf', 'status': 'complete'})
    row = {'case_id': 'case01', 'backend': 'mmorf', 'candidate_report': 'candidate/report.json',
           'reference_report': 'selected/report.json', 'comparison_report': 'comparison.json',
           'candidate_process_metrics': 'candidate/process_metrics.json',
           'reference_process_metrics': 'selected/process_metrics.json',
           'initial_failed_reference_report': 'initial/report.json',
           'initial_failed_reference_process_metrics': 'initial/process_metrics.json',
           'intermediate_reference_attempts': [{'report': 'middle/report.json',
                'process_metrics': 'middle/process_metrics.json',
                'reason': 'native constructor failure; private log /private/middle/log'}]}
    return tmp_path, row


def result(history):
    root, row = history
    aggregate = comparison.aggregate_plan({'planned_cases': [row]}, manifest_root=root)
    selected = next(item for item in aggregate['cases'] if item['case_id'] == 'case01' and item['backend'] == 'mmorf')
    return aggregate, selected


def test_three_attempts_only_selected_primary_clock_enters_ratio_and_rendering(history):
    root, _ = history
    aggregate, selected = result(history)
    attempts = aggregate['reference_attempts']
    assert attempts['initial']['observed_attempts'] == attempts['initial']['failed_attempts'] == 1
    assert attempts['intermediate']['observed_attempts'] == attempts['intermediate']['failed_attempts'] == 1
    assert attempts['rerun']['successful_attempts'] == attempts['selected_primary']['successful_attempts'] == 1
    assert attempts['all']['observed_attempts'] == 3 and attempts['all']['failed_attempts'] == 2
    assert attempts['all']['success_fraction'] == pytest.approx(1 / 3)
    assert selected['paired_ratios'] == {'processing_reference_over_candidate': 10, 'full_process_reference_over_candidate': 10}
    middle = selected['retained_intermediate_reference_attempts'][0]
    assert middle['attempt_index'] == 1
    assert middle['timing_seconds'] == {'processing': 392.96265, 'full_process': 395.24}
    assert middle['report']['sha256'] == comparison.file_hash(root / 'middle/report.json')
    assert middle['external_process_metrics']['wall_time_source'] == 'gnu_time_v'
    assert middle['external_process_metrics']['exit_status'] == 1
    assert not middle['included_in_paired_speed_ratios']
    serialized = json.dumps(aggregate)
    assert '/private' not in serialized and 'secret_command' not in serialized and str(root) not in serialized
    assert '[private filesystem path]' in serialized

    renderer_spec = importlib.util.spec_from_file_location('intermediate_renderer', SOURCE.with_name('render_report.py'))
    renderer = importlib.util.module_from_spec(renderer_spec)
    renderer_spec.loader.exec_module(renderer)
    write(root / 'aggregate.json', aggregate)
    binding = renderer.render(root / 'aggregate.json', root / 'rendered')
    assert binding['reference_full_run_attempts']['all'] == {'observed': 3, 'successful': 1, 'failed': 2, 'pending': 0}
    assert binding['reference_full_run_attempts']['intermediate']['failed'] == 1
    with (root / 'rendered/cases.csv').open(newline='', encoding='utf-8') as stream:
        row = next(row for row in csv.DictReader(stream) if row['case_id'] == 'case01' and row['backend'] == 'mmorf')
    records = json.loads(row['retained_intermediate_reference_attempts_json'])
    assert row['retained_intermediate_reference_attempt_count'] == '1'
    assert records[0]['full_process_seconds'] == 395.24 and records[0]['report_sha256'] == middle['report']['sha256']
    assert float(row['full_process_reference_over_candidate']) == 10
    assert records[0]['included_in_paired_speed_ratios'] is False
    assert '中间保留的参考恢复运行' in (root / 'rendered/RESULTS.md').read_text()


@pytest.mark.parametrize('duplicate', ['same_path', 'same_sha', 'initial', 'selected'])
def test_duplicate_identity_is_binding_error_never_adds_attempt(history, duplicate):
    root, row = history
    if duplicate in ('same_path', 'same_sha'):
        extra = dict(row['intermediate_reference_attempts'][0])
        if duplicate == 'same_sha':
            write(root / 'alias/report.json', json.loads((root / 'middle/report.json').read_text()))
            extra = {'report': 'alias/report.json'}
        row['intermediate_reference_attempts'].append(extra)
        expected_all = 3
    else:
        row['intermediate_reference_attempts'] = [{'report': duplicate + '/report.json'}]
        expected_all = 2
    aggregate, selected = result(history)
    assert any('duplicate_report_identity' in value for value in selected['binding_errors'])
    assert aggregate['reference_attempts']['all']['observed_attempts'] == expected_all
    assert not selected['paired_runs_complete']
    assert all(value is None for value in selected['paired_ratios'].values())


@pytest.mark.parametrize('mismatch', ['case', 'backend', 'input', 'seed', 'metrics_directory', 'successful'])
def test_wrong_binding_or_nonfailed_terminal_is_not_counted(history, mismatch):
    root, row = history
    path = root / 'middle/report.json'
    report = json.loads(path.read_text())
    if mismatch == 'case':
        report['case_id'] = 'case02'
    elif mismatch == 'backend':
        report['registration_backend'] = 'tbss'
    elif mismatch == 'input':
        report['input_files']['AP.nii.gz']['sha256'] = 'b' * 64
    elif mismatch == 'seed':
        report['parameters']['gp_seed'] = 99
    elif mismatch == 'metrics_directory':
        row['intermediate_reference_attempts'][0]['process_metrics'] = 'selected/process_metrics.json'
    elif mismatch == 'successful':
        report['status'] = 'complete'
        report.pop('failure_type')
        write(root / 'middle/process_metrics.json', {'exit_code': 0, 'wall_seconds': 395.24})
        (root / 'middle/time.txt').write_text('Elapsed (wall clock) time (h:mm:ss or m:ss): 6:35.24\nExit status: 0\n')
    write(path, report)
    aggregate, selected = result(history)
    assert selected['binding_errors'] and selected['retained_intermediate_reference_attempts'] == []
    assert aggregate['reference_attempts']['all']['observed_attempts'] == 2
    assert aggregate['reference_attempts']['selected_primary']['successful_attempts'] == 1
    assert not selected['paired_runs_complete'] and all(value is None for value in selected['paired_ratios'].values())


def test_initial_duplicate_selected_is_not_an_extra_attempt(history):
    _, row = history
    row['initial_failed_reference_report'] = row['reference_report']
    row['initial_failed_reference_process_metrics'] = row['reference_process_metrics']
    row.pop('intermediate_reference_attempts')
    aggregate, selected = result(history)
    assert 'initial_reference_duplicate_selected_report' in selected['binding_errors']
    assert selected['retained_initial_reference_attempt'] is None
    assert aggregate['reference_attempts']['all']['observed_attempts'] == 1
