"""只测试真实报告合同路由；不生成 MRI 或替代真实 benchmark。"""
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).with_name('render_surface_backend_comparison.py')
spec = importlib.util.spec_from_file_location('backend_renderer_contract', SOURCE)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


@pytest.mark.parametrize('guards', [
    ('readonly_input_guards_equal', 'frozen_source_guards_equal',
     'validation_helper_guard_equal', 'demo_runner_guard_equal', 'configuration_guard_equal'),
    ('raw_inputs_unchanged', 'source_unchanged_during_run',
     'configuration_unchanged', 'driver_unchanged', 'provenance_guards_passed'),
])
def test_actual_staged_and_formal_guards_must_all_be_true(guards):
    report = {'status': 'complete', 'source_revision': renderer.__dict__.get(
        'REVISION', '1128bc52c7a0233266e5b8a8d7dc0b382994e676')}
    report.update(dict.fromkeys(guards, True))
    assert renderer.completed_guarded_report(report)
    for key in guards:
        assert not renderer.completed_guarded_report({**report, key: False})
        assert not renderer.completed_guarded_report({k: v for k, v in report.items() if k != key})
    assert not renderer.completed_guarded_report({**report, 'status': 'running'})
    assert not renderer.completed_guarded_report({**report, 'source_revision': 'other'})


def test_unknown_guard_schema_is_rejected():
    assert not renderer.completed_guarded_report({
        'status': 'complete', 'source_revision': '1128bc52c7a0233266e5b8a8d7dc0b382994e676',
        'input_unchanged_during_run': True, 'source_unchanged_during_run': True})


@pytest.mark.parametrize('guard,key', [('readonly_input_guards_equal', 'outputs'),
                                      ('raw_inputs_unchanged', 'output_checks')])
def test_saved_cifti_binding_uses_actual_schema(guard, key):
    record = {'shape': [180, 91282], 'all_finite': True, 'sha256': 'a' * 64}
    assert renderer.saved_cifti_record({guard: True, key: {'dtseries': record}}) == record
    with pytest.raises(ValueError):
        renderer.saved_cifti_record({guard: True, key: {'dtseries': {**record, 'shape': [1, 91282]}}})
    with pytest.raises(ValueError):
        renderer.saved_cifti_record({guard: True, key: {'dtseries': {**record, 'all_finite': False}}})
