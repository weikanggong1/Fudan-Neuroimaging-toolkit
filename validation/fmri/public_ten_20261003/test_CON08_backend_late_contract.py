"""Path reporter 晚复核合同；不执行 MRI，也不恢复未保存时钟。"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('late_saved_contract',
    Path(__file__).with_name('validate_CON08_backend_saved_late.py'))
late = importlib.util.module_from_spec(spec)
spec.loader.exec_module(late)


def original():
    return {'status': 'failed', 'error_type': 'TypeError', 'subject': 'CON08',
        'source_revision': late.REVISION, 'runner_sha256': late.RUNNER, 'backend': 'freesurfer',
        'surface_device': 'cpu', 'cpu_threads': 4, 'cuda_visible_devices': '',
        'torch_cuda_available': False, 'volume_executed': False, 'cold_reconstruction_preexisting': False,
        'raw_t1w_sha256': late.RAW_T1, 'raw_bold_sha256': late.RAW_BOLD,
        'readonly_input_guards_equal': True, 'frozen_source_guards_equal': True, 'binding_guard_equal': True}


FAILURE = ('line 292, in execute\n'
    'save(output / "files.private.json", {"result": dataclasses.asdict(result), "configuration": config})\n'
    'TypeError: ' + late.MESSAGE + '\n')


@pytest.mark.parametrize('key,value', [('status', 'complete'), ('error_type', 'ValueError'),
    ('binding_guard_equal', False), ('runner_sha256', 'other'), ('raw_bold_sha256', 'other'),
    ('volume_executed', True), ('full_api_seconds', 7952.0)])
def test_recovery_is_restricted_to_the_exact_preserved_reporter_failure(key, value):
    report = original()
    late.validate_failure(report, FAILURE)
    report[key] = value
    with pytest.raises(ValueError):
        late.validate_failure(report, FAILURE)


def test_failure_phase_must_be_after_api_return_not_an_arbitrary_type_error():
    with pytest.raises(ValueError):
        late.validate_failure(original(), FAILURE.replace('line 292, in execute', 'line 210, in execute'))


def test_late_filemap_never_accepts_external_output_symlinks(tmp_path):
    derivatives = tmp_path / 'derivatives'
    derivatives.mkdir()
    original = tmp_path / 'outside'
    original.write_text('original')
    (derivatives / 'escape').symlink_to(original)
    with pytest.raises(ValueError):
        late.output_path(derivatives, 'escape')
    valid = derivatives / 'saved'
    valid.write_text('saved')
    assert late.output_path(derivatives, 'saved') == valid


def test_late_validation_cannot_write_into_the_original_run(tmp_path):
    root = tmp_path / 'FNIT'
    original = root / 'runs/original'
    with pytest.raises(ValueError):
        late.protect_output(root, original / 'late', [original])
    late.protect_output(root, root / 'runs/independent', [original])
    assert not root.exists()
