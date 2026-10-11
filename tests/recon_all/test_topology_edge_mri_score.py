"""MRI 边评分的组织方向、完整采样及输入契约。"""
import unittest
import numpy as np
import nibabel as nib
from fnit.recon_all.topology_edge_mri_score import (source_mri_edge_scores,
    source_topology_ras_to_voxel, _median_twice, _sample)


class TopologyEdgeMRIScoreTests(unittest.TestCase):
    def test_topology_map_uses_mri_tkreg_and_not_scanner_center(self):
        image = nib.MGHImage(np.zeros((8, 10, 12), np.uint8), np.eye(4))
        image.header["Pxyz_c"] = [13.123456, -27.4, 8.2]
        matrix = source_topology_ras_to_voxel(volume_header=image.header)
        np.testing.assert_array_equal(matrix, np.array(
            [[-1, 0, 0, 4], [0, 0, -1, 5], [0, 1, 0, 6], [0, 0, 0, 1]], np.float32))
        with self.assertRaises(ValueError):
            source_topology_ras_to_voxel(volume_header={})

    def test_coordinate_row_accumulates_double_before_result_float(self):
        volume = np.broadcast_to(np.arange(8, dtype=np.uint8)[:, None, None], (8, 8, 8)).copy()
        matrix = np.eye(4, dtype=np.float32)
        # 中间F32归约会丢掉1，正确F64乘加最终得到1.5体素。
        matrix[0] = [16777216, 1, -16777216, .5]
        self.assertEqual(float(_sample(volume, matrix, 1., 1., 1.)), 1.5)

    def arguments(self):
        return dict(original_vertices=np.array([[2, 2, 3], [6, 2, 3]], np.float32),
                    original_normals=np.array([[0, 0, 1], [0, 0, 1]], np.float32),
                    edge_vertices=np.array([[0, 1]], np.int32),
                    white_values=np.array([75, 75], np.float32),
                    gray_values=np.array([85, 85], np.float32),
                    original_edge_present=np.array([True]),
                    volume=np.broadcast_to((50 + 10 * np.arange(9))[None, None, :], (9, 9, 9)).astype(np.uint8),
                    ras_to_voxel=np.eye(4, dtype=np.float32))

    def test_correct_gray_white_direction_and_source_edge_priority(self):
        args = self.arguments()
        self.assertEqual(source_mri_edge_scores(**args).dtype, np.float32)
        self.assertEqual(float(source_mri_edge_scores(**args)[0]), 0)
        args['original_edge_present'] = np.array([False])
        self.assertEqual(float(source_mri_edge_scores(**args)[0]), 100)
        args['original_normals'] *= -1
        self.assertEqual(float(source_mri_edge_scores(**args)[0]), 110)

    def test_interior_samples_not_only_endpoints(self):
        args = self.arguments()
        x = np.arange(9)
        args['volume'] = np.broadcast_to((x * x)[:, None, None] + 2 * np.arange(9)[None, None, :], (9, 9, 9)).astype(np.uint8)
        args['white_values'] = np.array([9, 41], np.float32)
        args['gray_values'] = np.array([11, 43], np.float32)
        positions = np.array([2., 6.] + [2. + .4 * s for s in range(1, 10)])
        expected = np.mean(np.abs(np.interp(positions, x, x * x) - 20.))
        self.assertAlmostEqual(float(source_mri_edge_scores(**args)[0]), expected, places=5)
        self.assertNotEqual(float(source_mri_edge_scores(**args)[0]), 16.)

    def test_opposite_normals_preserve_zero_direction_branch(self):
        args = self.arguments()
        args['original_normals'][1] *= -1
        self.assertEqual(float(source_mri_edge_scores(**args)[0]), 5.)

    def test_nearly_cancelling_normals_use_source_flt_epsilon(self):
        args = self.arguments()
        # 原 FZERO 是 FLT_EPSILON；5e-7 不能错误地当作零。
        args['original_normals'][:] = [0, 0, 5e-7]
        self.assertEqual(float(source_mri_edge_scores(**args)[0]), 0.)
        args['original_normals'][:] = [0, 0, 5e-8]
        self.assertAlmostEqual(float(source_mri_edge_scores(**args)[0]), 5., places=5)

    def test_existing_median_skips_ripped_neighbors_and_preserves_values(self):
        values = np.array([2, 4, 250], np.float32)
        ripped = np.array([False, False, True])
        neighbors = np.array([[1, 2], [0, 2], [0, 1]], np.int32)
        result = _median_twice(values, ripped, neighbors, np.array([2, 2, 2], np.int32))
        np.testing.assert_array_equal(result, np.array([3, 3, 250], np.float32))
        np.testing.assert_array_equal(values, np.array([2, 4, 250], np.float32))

    def test_empty_edges_keep_type_and_no_output_mutation(self):
        args = self.arguments(); original = args['original_vertices'].copy()
        args['edge_vertices'] = np.empty((0, 2), np.int32)
        args['original_edge_present'] = np.empty(0, bool)
        self.assertEqual(source_mri_edge_scores(**args).shape, (0,))
        np.testing.assert_array_equal(args['original_vertices'], original)

    def test_invalid_inputs_fail_before_scoring(self):
        for name, value in [('edge_vertices', np.array([[0., 1.]])),
                            ('edge_vertices', np.array([[0, 2]])),
                            ('edge_vertices', np.array([[1, 1]])),
                            ('original_edge_present', np.array([1], np.int32)),
                            ('white_values', np.array([np.nan, 75])),
                            ('volume', np.zeros((9, 9, 9), np.float32)),
                            ('ras_to_voxel', np.zeros((3, 4)))]:
            args = self.arguments(); args[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    source_mri_edge_scores(**args)


if __name__ == '__main__':
    unittest.main()
