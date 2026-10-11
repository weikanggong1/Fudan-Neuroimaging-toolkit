"""校准输入边界、缺陷屏蔽与两个顶点索引空间的合同。"""
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.topology_defect_calibration import calibrated_defect_volume_priors


class CalibrationContracts(unittest.TestCase):
    def arguments(self):
        xyz = np.array([[5, 5, 5], [7, 5, 5], [5, 7, 5], [5, 5, 7]], np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
        x, y, z = np.indices((20, 20, 20))
        return dict(vertices=xyz, faces=faces, defect_vertices=np.array([True, False, False, False]),
            defect_hull=np.array([1, 2, 3], np.int32), candidate_edges=np.array([[0, 1], [1, 2]], np.int32),
            vertex_translation=np.arange(4, dtype=np.int32), volume=(x+y+z+40).astype(np.uint8),
            ras_to_voxel=np.eye(4, dtype=np.float32))

    def test_calibration_excludes_defects_and_preserves_zero_bin_metadata(self):
        args = self.arguments()
        before = {key: value.copy() for key, value in args.items()}
        result = calibrated_defect_volume_priors(**args)
        np.testing.assert_array_equal(result['calibration']['selected'], [1, 2, 3])
        self.assertEqual(result['calibration']['bin_size'].dtype, np.float32)
        self.assertEqual(result['calibration']['bin_size'], 0)
        np.testing.assert_array_equal(result['calibration']['bins'], np.arange(256))
        self.assertEqual(result['priors']['defect_intensity_volume'].dtype, np.uint8)
        self.assertEqual(result['priors']['white_log_probability'].dtype, np.float32)
        for key, value in before.items(): np.testing.assert_array_equal(value, args[key])

    def test_translation_preserves_geometry_instead_of_reinterpreting_indices(self):
        original = self.arguments()
        a = calibrated_defect_volume_priors(**original)
        translated = self.arguments()
        permutation = np.array([2, 0, 3, 1], np.int32)
        translated['vertex_translation'] = permutation
        translated['candidate_edges'] = permutation[original['candidate_edges']]
        b = calibrated_defect_volume_priors(**translated)
        for key in ('defect_intensity_volume', 'initial_edge_mask', 'white_log_probability',
                    'gray_log_probability', 'algorithm_volume_geometry'):
            self.assertEqual(a['priors'][key].tobytes(), b['priors'][key].tobytes())

    def test_invalid_state_rejected_before_mri_sampling(self):
        invalid = [('vertex_translation', np.array([0, 0, 2, 3], np.int32)),
                   ('defect_hull', np.array([], np.int32)),
                   ('defect_hull', np.array([4], np.int32)),
                   ('candidate_edges', np.array([[0, 4]], np.int32))]
        for key, value in invalid:
            args = self.arguments(); args[key] = value
            with patch('fnit.recon_all.topology_defect_calibration.prepare_topology_edge_inputs') as mocked:
                with self.assertRaises(ValueError): calibrated_defect_volume_priors(**args)
                mocked.assert_not_called()

    def test_empty_unmarked_calibration_is_explicit_failure(self):
        args = self.arguments(); args['defect_vertices'][:] = True
        with self.assertRaisesRegex(ValueError, 'no unmarked'):
            calibrated_defect_volume_priors(**args)


if __name__ == '__main__': unittest.main()
