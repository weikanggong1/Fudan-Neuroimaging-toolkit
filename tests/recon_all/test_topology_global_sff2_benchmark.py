"""实际 SFF2 输入域、缺失保留状态和 after-only 曲率比较的契约。"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def tool():
    import sys
    directory = Path(__file__).parents[2] / 'tools'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('global_sff2_benchmark', directory / 'benchmark_recon_topology_global_sff2_core.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def fixtures():
    vertex = np.zeros((3, 16), np.float32)
    vertex[:, :3] = vertex[:, 3:6] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    vertex[:, 6:9] = vertex[:, :3]
    vertex[:, 10:12] = 123  # after-K deliberately differs from independent input
    faces = np.array([[0, 1, 2]], np.int32)
    offsets = np.array([0, 2, 4, 6], np.int32)
    neighbors = np.array([1, 2, 0, 2, 0, 1], np.int32)
    statistics = dict(vertex_fields=vertex, faces=faces, neighbor_offsets=offsets, neighbors=neighbors)
    calibration = dict(header=np.array([3, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int32),
        topology=np.tile([2, 2, 2], (3, 1)).astype(np.int32), flags=np.zeros((3, 2), np.int32),
        vertex_fields=vertex[:, :9].copy(), faces=faces.copy(), neighbor_offsets=offsets.copy(), neighbors=neighbors.copy())
    return statistics, calibration


def test_after_k_never_initializes_input_or_reference_fields():
    statistics, calibration = fixtures()
    args, audit = tool().actual_inputs(statistics=statistics, calibration=calibration)
    assert np.array_equal(args['initial_k1'], np.zeros(3, np.float32))
    assert np.array_equal(args['initial_k2'], np.zeros(3, np.float32))
    statistics['vertex_fields'][:, 10:12] = -999
    assert not args['initial_k1'].any() and not args['initial_k2'].any()
    assert audit['reference_output_k_used_as_input'] is False


@pytest.mark.parametrize('wrong', ['phase', 'face_order', 'prefix', 'rip', 'nsize', 'original_xyz'])
def test_captured_topology_must_be_same_domain_and_unchanged(wrong):
    statistics, calibration = fixtures()
    if wrong == 'phase': calibration['header'][6] = 2
    elif wrong == 'face_order': calibration['faces'][0] = [1, 0, 2]
    elif wrong == 'prefix': calibration['neighbors'][:2] = [2, 1]
    elif wrong == 'rip': calibration['flags'][1, 1] = 1
    elif wrong == 'nsize': calibration['topology'][1, 2] = 1
    else: calibration['vertex_fields'][1, 0] += 1
    with pytest.raises(ValueError): tool().actual_inputs(statistics=statistics, calibration=calibration)


def test_initial_state_consumption_audit_not_filled_from_after():
    statistics, calibration = fixtures()
    statistics['vertex_fields'][0, 15] = calibration['flags'][0, 1] = 1
    args, audit = tool().actual_inputs(statistics=statistics, calibration=calibration)
    assert audit['rip_vertices'] == 1
    assert args['initial_k1'][0] == 0  # neither the source after-value nor an invented prior


def test_source_zero_sign_still_recorded_as_byte_difference():
    comparison = tool()._compare({'k1': np.array([0], np.float32)},
                                 {'k1': np.array([-0.0], np.float32)})['k1']
    assert comparison['differing_values'] == 0 and comparison['byte_equal'] is False


@pytest.mark.parametrize('retained', [False, True])
def test_full_core_gate_requires_no_unknown_initial_value_consumption(tmp_path, monkeypatch, retained):
    module = tool(); statistics, calibration = fixtures()
    if retained:
        statistics['vertex_fields'][0, 15] = calibration['flags'][0, 1] = 1
    paths = []
    for label in ('statistics', 'statistics_repeat', 'calibration', 'calibration_repeat', 'source'):
        path = tmp_path / label; path.write_text('{}'); paths.append(path)
    monkeypatch.setattr(module, 'load_statistics_capture', lambda **kwargs: statistics)
    monkeypatch.setattr(module, 'load_calibration_capture', lambda **kwargs: calibration)
    calls = []
    def own_producer(**inputs):
        assert not inputs['initial_k1'].any() and not inputs['initial_k2'].any()
        calls.append(1)
        return dict(k1=np.full(3, 123, np.float32), k2=np.full(3, 123, np.float32),
            fit_kind=np.array([-1 if retained else 1, 1, 1], np.int8), seconds=.001)
    monkeypatch.setattr(module, 'global_second_fundamental_form', own_producer)
    report = module.run(statistics=paths[0], statistics_repeat=paths[1],
        calibration=paths[2], calibration_repeat=paths[3], source_manifest=paths[4],
        output=tmp_path/'out', threads=2)
    assert len(calls) == 2
    assert report['strict_sff2_core_reproduction'] is (not retained)
    assert report['full_global_sff_state_reproduction'] == 'not_assessed'
    assert report['input_audit']['retained_initial_k_vertices'] == int(retained)
