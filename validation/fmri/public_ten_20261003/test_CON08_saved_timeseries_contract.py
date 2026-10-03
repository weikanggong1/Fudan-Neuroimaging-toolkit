"""CON08 只读后验比较合同；不执行或模拟 MRI benchmark。"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('CON08_saved_contract',
    Path(__file__).with_name('compare_CON08_backend_saved_timeseries.py'))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def reports():
    common = {'status': 'complete', 'subject': 'CON08', 'source_revision': comparison.REVISION}
    formal = dict(common, raw_inputs_unchanged=True, configuration_unchanged=True,
        source_unchanged_during_run=True, driver_unchanged=True, provenance_guards_passed=True,
        input_sha256={'t1w': comparison.RAW_T1, 'bold': comparison.RAW_BOLD},
        frames=180, repetition_time=2.4, backend='fnit')
    cold = dict(common, readonly_input_guards_equal=True, frozen_source_guards_equal=True,
        binding_guard_equal=True, volume_reused=True, raw_t1w_sha256=comparison.RAW_T1,
        raw_bold_sha256=comparison.RAW_BOLD, frame_count=180, tr_seconds=2.4,
        backend='freesurfer', surface_device='cpu', cuda_visible_devices='',
        torch_cuda_available=False, volume_executed=False, reconstruction_reused=False)
    return formal, cold


@pytest.mark.parametrize('role,key,value', [
    (0, 'status', 'running'), (1, 'status', 'failed'),
    (0, 'source_unchanged_during_run', None), (1, 'binding_guard_equal', False),
    (1, 'raw_bold_sha256', 'other'), (1, 'tr_seconds', 2.1),
    (1, 'volume_executed', True), (1, 'reconstruction_reused', True),
])
def test_whole_binding_scope_failures_are_rejected(role, key, value):
    values = list(reports())
    comparison.verify_reports(*values)
    values[role][key] = value
    with pytest.raises(ValueError):
        comparison.verify_reports(*values)


def test_output_refuses_original_parents_and_symlinks(tmp_path):
    root = tmp_path / 'FNIT'
    runs = root / 'runs'
    runs.mkdir(parents=True)
    protected = runs / 'original' / 'resource'
    protected.mkdir(parents=True)
    with pytest.raises(ValueError):
        comparison.protect_output(root, protected / 'new', [protected])
    with pytest.raises(FileExistsError):
        comparison.protect_output(root, protected.parent, [protected])
    broken = runs / 'broken'
    broken.symlink_to(runs / 'missing')
    with pytest.raises(FileExistsError):
        comparison.protect_output(root, broken, [protected])
    comparison.protect_output(root, runs / 'new', [protected])
    assert not (runs / 'new').exists()


def late_bindings(tmp_path):
    import json
    original = {'status': 'failed', 'error_type': 'TypeError', 'runner_sha256': comparison.COLD_HELPER_SHA,
        'readonly_input_guards_equal': True, 'frozen_source_guards_equal': True, 'binding_guard_equal': True,
        'configuration_sha256': 'fixed', 'readonly_inputs_before': {'raw': 'same'},
        'readonly_inputs_after': {'raw': 'same'}, 'raw_t1w_sha256': comparison.RAW_T1,
        'raw_bold_sha256': comparison.RAW_BOLD, 'driver_through_saved_output_validation_seconds': 7952.462965272367}
    original_path, files_path, source_path = [tmp_path / name for name in ('original.json', 'files.json', 'source.json')]
    original_path.write_text(json.dumps(original))
    files_path.write_text('{}')
    source_path.write_text('{}')
    late = dict(original, status='complete_saved_outputs_verified_late', original_driver_status='failed',
        original_driver_failure={'error_type': 'TypeError',
            'message': 'Object of type PosixPath is not JSON serializable',
            'phase': 'save report/files.private.json after full API return and saved-output checks',
            'original_runner_sha256': comparison.COLD_HELPER_SHA,
            'original_report_sha256': comparison.sha256(original_path)},
        full_api_seconds=None, returned_api_total_seconds=None, original_source_guard_passed=True,
        late_input_guards_equal=True, late_source_guards_equal=True, GIFTI_CIFTI_cortical_values_exact=True,
        late_input_sha256_before={'output': 'same'}, late_input_sha256_after={'output': 'same'},
        source_manifest_sha256=comparison.sha256(source_path), late_private_filemap_sha256=comparison.sha256(files_path),
        original_private_filemap_saved=False, saved_timing_seconds={'total_before_publication': 7948.804876163602},
        original_driver_failure_wall_seconds=original['driver_through_saved_output_validation_seconds'])
    return original, late, original_path, files_path, source_path


@pytest.mark.parametrize('key,value', [('full_api_seconds', 7952.0), ('returned_api_total_seconds', 7948.8),
    ('original_driver_status', 'complete'), ('original_private_filemap_saved', True),
    ('late_source_guards_equal', False), ('late_private_filemap_sha256', 'changed')])
def test_explicit_late_branch_rejects_invented_clock_or_provenance(tmp_path, key, value):
    original, late, *paths = late_bindings(tmp_path)
    comparison.verify_late_report(original, late, *paths)
    late[key] = value
    with pytest.raises(ValueError):
        comparison.verify_late_report(original, late, *paths)


def test_late_branch_keeps_original_failed_status_and_rejects_changed_filemap(tmp_path):
    original, late, original_path, files_path, source_path = late_bindings(tmp_path)
    comparison.verify_late_report(original, late, original_path, files_path, source_path)
    assert original['status'] == 'failed'
    assert late['full_api_seconds'] is None
    files_path.write_text('{"other":"path"}')
    with pytest.raises(ValueError):
        comparison.verify_late_report(original, late, original_path, files_path, source_path)
