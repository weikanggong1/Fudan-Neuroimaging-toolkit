"""CON08 图的真实参考状态合同；不执行 MRI 或模拟 benchmark。"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('CON08_figure_contract',
    Path(__file__).with_name('render_CON08_backend_comparisons.py'))
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('subject', 'CON01'),
    ('container_exit_code', 1), ('source_unchanged_during_run', None),
    ('input_unchanged_during_run', False), ('corrective_saved_file_guards_equal', None),
    ('input_sha256', {'t1w': 'wrong', 'bold': 'same'}), ('frames', 179), ('repetition_time', 2.1)])
def test_plot_requires_exact_complete_corrected_reference_scope(key, value):
    raw = {'t1w': 'exact', 'bold': 'same'}
    report = dict(status='complete', subject='CON08', input_sha256=raw, frames=180,
        repetition_time=2.4, container_exit_code=0, source_unchanged_during_run=True,
        input_unchanged_during_run=True, corrective_saved_file_guards_equal=True)
    renderer.validate_reference_report(report, raw)
    report[key] = value
    with pytest.raises(ValueError):
        renderer.validate_reference_report(report, raw)
