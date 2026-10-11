"""真实 register_t1 接线合同；优化器精度另由完整协议测试验证。"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fnit.recon_all import mri_em_register_python as registration


def test_register_t1_consumes_stable_then_all_sample_complete_em(tmp_path):
    stable = SimpleNamespace(labels=np.ones(2, np.int32))
    all_samples = SimpleNamespace(labels=np.ones(3, np.int32))
    initial = np.eye(4, dtype=np.float32)
    stable_matrix = initial.copy()
    stable_matrix[0, 3] = 1
    final_matrix = initial.copy()
    final_matrix[0, 3] = 2
    calls = []

    def complete(objective, matrix, *, tolerance):
        calls.append((objective.samples, objective.clamp, matrix.copy(), tolerance))
        output = stable_matrix if len(calls) == 1 else final_matrix
        return output, {'passes': [], 'outer_cost_after': 0.0}

    def objective(atlas, samples, source, *, clamp):
        return SimpleNamespace(samples=samples, clamp=clamp)

    with patch.object(registration, 'read_gca', return_value=object()), \
            patch.object(registration, 'read_masked_input', return_value=object()), \
            patch.object(registration, 'estimate_image_white_matter_peak', return_value=(1, None, None)), \
            patch.object(registration, 'atlas_label_peak', return_value=1), \
            patch.object(registration, 'scale_input_intensity', return_value=object()), \
            patch.object(registration, 'find_stable_samples', return_value=stable), \
            patch.object(registration, 'find_all_samples', return_value=all_samples), \
            patch.object(registration, 'gca_mean_volume', return_value=None), \
            patch.object(registration, 'gca_centroid', return_value=None), \
            patch.object(registration, 'find_optimal_translation_source', return_value=(initial, [(0, (0, 0, 0), initial)])), \
            patch.object(registration, 'find_optimal_linear_transform_source', return_value=(initial, [])), \
            patch.object(registration, 'EMObjective', side_effect=objective), \
            patch.object(registration, 'align_em', side_effect=complete), \
            patch.object(registration, 'write_voxel_lta') as write:
        report = registration.register_t1(nu_path=tmp_path/'nu.mgz', mask_path=tmp_path/'mask.mgz',
            atlas_path=tmp_path/'declared.gca', output_path=tmp_path/'result.lta')
    assert len(calls) == 2
    assert calls[0][0] is stable and calls[0][1] == 0.0 and calls[0][3] == 1e-5
    assert calls[1][0] is all_samples and calls[1][1] == 6.0 and calls[1][3] == 1e-7
    np.testing.assert_array_equal(calls[0][2], initial)
    np.testing.assert_array_equal(calls[1][2], stable_matrix)
    np.testing.assert_array_equal(write.call_args.args[1], final_matrix)
    assert set(report['em_complete_protocol']) == {'stable', 'all_samples'}
