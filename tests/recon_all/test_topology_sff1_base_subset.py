"""自产base映射失败不允许同索引K通过；参考只作after-only比较。"""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest


def tool():
    directory = Path(__file__).parents[2] / 'tools'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('sff1_subset', directory/'benchmark_recon_topology_sff1_base_subset.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def inputs():
    xyz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    state = dict(original_vertices=xyz.copy(), current_vertices=xyz.copy(),
        faces=np.array([[0, 1, 2]], np.int32), nfaces=1,
        k1=np.array([1, 2, 3], np.float32), k2=np.array([3, 2, 1], np.float32))
    return dict(candidate=state, reference=copy.deepcopy(state),
        candidate_maps=dict(source_order=np.array([1, 2, 0], np.int32),
                            vertex_translation=np.array([2, 0, 1], np.int32)),
        reference_selection=dict(vertex_translation=np.array([2, 0], np.int32)),
        reference_metadata=dict(phase=0))


def test_after_only_reference_cannot_rewrite_candidate_or_mapping():
    values = inputs(); before = copy.deepcopy(values)
    values['reference']['k1'][0] += 10
    result = tool().compare_subset(**values)
    assert result['same_index_domain_strict']
    assert not result['strict_sff1_base_k_reproduction']
    assert result['curvature']['k1']['max_abs'] == 10
    for key in ('k1', 'current_vertices'):
        np.testing.assert_array_equal(values['candidate'][key], before['candidate'][key])
    np.testing.assert_array_equal(values['candidate_maps']['source_order'], before['candidate_maps']['source_order'])


@pytest.mark.parametrize('changed', ['original_vertices', 'current_vertices', 'faces', 'mapping'])
def test_wrong_domain_never_passes_even_if_k_equal(changed):
    values = inputs()
    if changed == 'mapping': values['reference_selection']['vertex_translation'][0] = 1
    elif changed == 'faces': values['reference']['faces'][0] = [1, 0, 2]
    else: values['reference'][changed][0, 0] += 1
    result = tool().compare_subset(**values)
    assert not result['same_index_domain_strict']
    assert not result['strict_sff1_base_k_reproduction']
    assert result['curvature'] == 'not_assessed_same_index_domain_failed'


def test_map_prefix_need_not_include_complete_source_domain():
    result = tool().compare_subset(**inputs())
    assert result['mapping_reference_entries'] == 2 and result['vertices'] == 3
    assert result['strict_sff1_base_k_reproduction']
    assert result['H_K_E_reference'] == 'not_captured'


@pytest.mark.parametrize('wrong', ['duplicate', 'inverse', 'phase', 'invalid_face', 'wrong_dtype'])
def test_invalid_provenance_shape_or_mapping_fails(wrong):
    values = inputs()
    if wrong == 'duplicate': values['candidate_maps']['source_order'][0] = 0
    elif wrong == 'inverse': values['candidate_maps']['source_order'] = np.arange(3, dtype=np.int32)
    elif wrong == 'phase': values['reference_metadata']['phase'] = 2
    elif wrong == 'invalid_face': values['candidate']['faces'][0, 0] = 3
    else: values['candidate']['current_vertices'] = values['candidate']['current_vertices'].astype(np.float64)
    with pytest.raises(ValueError): tool().compare_subset(**values)


def test_inactive_face_capacity_not_a_logical_mesh_change():
    values = inputs()
    values['candidate']['faces'] = np.vstack([values['candidate']['faces'], np.array([[100, 100, 100]], np.int32)])
    assert tool().compare_subset(**values)['strict_sff1_base_k_reproduction']


def test_signed_zero_remains_strict_byte_diagnostic():
    values = inputs(); values['reference']['k1'][0] = np.float32(-0.)
    values['candidate']['k1'][0] = np.float32(0.)
    result = tool().compare_subset(**values)
    assert result['same_index_domain_strict']
    assert result['curvature']['k1']['differing_values'] == 0
    assert not result['strict_sff1_base_k_reproduction']
