"""几何分类不是质量豁免；模拟合同与真实pial分开。"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('pial_geometry_diagnostic', Path(__file__).resolve().parents[2]/'tools/diagnose_recon_pial_residual_geometry.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PialResidualGeometry(unittest.TestCase):
    def setUp(self): self.first = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]])

    def test_true_crossing_segment_has_nonzero_length_and_plane_depth(self):
        row = module.pair_geometry(self.first, np.array([[0., -.5, -1.], [0., -.5, 1.], [0., .5, 1.]]))
        self.assertEqual(row['classification'], 'noncoplanar_segment_overlap')
        self.assertAlmostEqual(row['intersection_segment_overlap_mm'], .5)
        self.assertGreater(row['second_plane_straddle_depth_mm'], .1)
        self.assertEqual(row['exact_duplicate_corner_pairs'], 0)

    def test_coplanar_same_triangle_has_full_area_and_duplicate_corners(self):
        row = module.pair_geometry(self.first, self.first)
        self.assertEqual(row['classification'], 'coplanar_area_overlap')
        self.assertAlmostEqual(row['coplanar_overlap_area_mm2'], 2.)
        self.assertEqual(row['exact_duplicate_corner_pairs'], 3)

    def test_diagnostic_area_uses_normalized_normal_for_scale_and_slope(self):
        rotation = np.array([[.8, 0., .6], [0., 1., 0.], [-.6, 0., .8]])
        for scale in (.25, 1., 4.):
            triangle = self.first @ rotation * scale + np.array([73.331, 12.717, -61.723])
            with self.subTest(scale=scale):
                row = module.pair_geometry(triangle, triangle)
                self.assertAlmostEqual(row['coplanar_overlap_area_mm2'], 2.*scale**2, places=9)

    def test_disjoint_exact_slices_do_not_become_a_true_crossing(self):
        second = np.array([[0., 4., -1.], [0., 4., 1.], [0., 5., 1.]])
        row = module.pair_geometry(self.first, second)
        self.assertEqual(row['classification'], 'near_contact_disjoint_exact_slices')
        self.assertLess(row['intersection_segment_overlap_mm'], 0)

    def test_degenerate_triangle_and_nonfinite_fail(self):
        for second in (np.zeros((3,3)), np.full((3,3), np.nan)):
            with self.subTest(shape=second.shape), self.assertRaises(ValueError): module.pair_geometry(self.first,second)


if __name__ == '__main__': unittest.main()
