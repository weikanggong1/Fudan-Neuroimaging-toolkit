"""unMRI 聚合：有序 float32、原索引、哨兵和重复顶点反馈契约。"""
import unittest
import numpy as np
from fnit.recon_all.topology_defect_volume_likelihood import defect_volume_likelihood


class TopologyDefectVolumeLikelihoodTests(unittest.TestCase):
    def arguments(self):
        shape = (9, 9, 9)
        return dict(defect_intensity_volume=np.zeros(shape, np.uint8),
            signed_distance_mm=np.full(shape, 1000, np.float32),
            white_log_probability=np.zeros(shape, np.float32),
            gray_log_probability=np.zeros(shape, np.float32),
            initial_white_mean_log_probability=1.0,
            initial_gray_mean_log_probability=2.0,
            initial_white_mean_intensity=100.0, initial_gray_mean_intensity=50.0,
            volume_origin_mm=np.zeros(3, np.float32),
            volume_index_scale_per_mm=np.ones(3, np.float32),
            vertex_sampling_positions_mm=np.array([[3, 3, 3]], np.float32),
            initial_vertex_curvbak=np.array([8], np.float32),
            translated_defect_vertices=np.array([0], np.int32),
            discarded_defect_vertices=np.array([False]), distance_unset_sentinel_mm=1000.0)

    def test_k_j_i_float32_order_and_initial_count_one(self):
        a = self.arguments()
        for i, logp in zip((3, 4, 5), (-1e8, 1e8, 1)):
            a['defect_intensity_volume'][i, 3, 3] = i
            a['signed_distance_mm'][i, 3, 3] = -.6
            a['white_log_probability'][i, 3, 3] = logp
        r = defect_volume_likelihood(**a)
        self.assertEqual(r['white_ll'], .25)
        self.assertEqual(r['white_mean_intensity'], 28.0)
        self.assertEqual((r['nwhite'], r['ngray']), (4, 1))
        self.assertEqual(r['unmri_ll'], 2.25)

    def test_three_voxel_margin_and_strict_half_mm_classification(self):
        a = self.arguments()
        for pos, distance in [((2, 3, 3), 100), ((3, 3, 3), .5), ((4, 3, 3), -.5), ((5, 3, 3), 1000)]:
            a['defect_intensity_volume'][pos] = 100
            a['signed_distance_mm'][pos] = distance
        r = defect_volume_likelihood(**a)
        self.assertEqual((r['nwhite'], r['ngray']), (1, 1))
        self.assertEqual(r['raw_max_distance_mm'], .5)
        self.assertEqual(r['max_distance_mm'], 1)
        np.testing.assert_array_equal(r['vertex_curvbak'], [4])

    def test_maximum_clip_to_two_and_feedback_cap(self):
        a = self.arguments()
        a['defect_intensity_volume'][5, 3, 3] = 100
        a['signed_distance_mm'][5, 3, 3] = 3
        a['signed_distance_mm'][3, 3, 3] = -.5
        r = defect_volume_likelihood(**a)
        self.assertEqual((r['raw_max_distance_mm'], r['max_distance_mm']), (3, 2))
        np.testing.assert_array_equal(r['vertex_curvbak'], [2])
        a['signed_distance_mm'][3, 3, 3] = -20
        np.testing.assert_array_equal(defect_volume_likelihood(**a)['vertex_curvbak'], [8])

    def test_sentinel_and_outside_volume_preserve_feedback(self):
        a = self.arguments()
        np.testing.assert_array_equal(defect_volume_likelihood(**a)['vertex_curvbak'], [8])
        a['vertex_sampling_positions_mm'][0] = [-20, 3, 3]
        np.testing.assert_array_equal(defect_volume_likelihood(**a)['vertex_curvbak'], [8])

    def test_duplicate_translation_multiplies_in_order_and_discard_only_skips(self):
        a = self.arguments()
        a['signed_distance_mm'][3, 3, 3] = .5
        a['translated_defect_vertices'] = np.array([0, 0, -999], np.int32)
        a['discarded_defect_vertices'] = np.array([False, False, True])
        before = a['initial_vertex_curvbak'].copy()
        np.testing.assert_array_equal(defect_volume_likelihood(**a)['vertex_curvbak'], [2])
        np.testing.assert_array_equal(a['initial_vertex_curvbak'], before)

    def test_iVOL_uses_float32_multiply_and_asymmetric_truncation(self):
        a = self.arguments()
        a['volume_index_scale_per_mm'][:] = 2
        a['vertex_sampling_positions_mm'][0] = [1.5, 1.5, 1.5]
        a['signed_distance_mm'][3, 3, 3] = .5
        np.testing.assert_array_equal(defect_volume_likelihood(**a)['vertex_curvbak'], [4])
        a['volume_index_scale_per_mm'][:] = 1
        a['vertex_sampling_positions_mm'][0] = [-.6, 0, 0]
        a['signed_distance_mm'][0, 0, 0] = .25
        np.testing.assert_array_equal(defect_volume_likelihood(**a)['vertex_curvbak'], [2])

    def test_finite_shape_status_and_index_validation(self):
        for key, value in [('signed_distance_mm', np.ones((1, 1, 1), np.float32)),
                           ('volume_index_scale_per_mm', np.array([0, 1, 1])),
                           ('initial_white_mean_intensity', np.nan),
                           ('translated_defect_vertices', np.array([1], np.int32)),
                           ('discarded_defect_vertices', np.array([1], np.int32))]:
            a = self.arguments(); a[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                defect_volume_likelihood(**a)


if __name__ == '__main__':
    unittest.main()
