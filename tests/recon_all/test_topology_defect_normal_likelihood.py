"""法向/曲率 GA 子项：有序均值、源索引、字段量化和失败契约。"""
import unittest

import numpy as np

from fnit.recon_all.topology_defect_normal_likelihood import (
    _nint, patch_curvature_likelihood, patch_normal_dot_likelihood,
)


class TopologyDefectNormalLikelihoodTests(unittest.TestCase):
    def normal_arguments(self):
        return dict(original_vertices=np.array([[0, 0, 0], [0, 0, .5], [0, 0, 2]], np.float32),
                    vertex_normals=np.array([[0, 0, 1]] * 3, np.float32),
                    vertex_areas_mm2=np.array([1, 3, 0], np.float32),
                    patch_vertices=np.array([0, 1], np.int32),
                    neighbor_offsets=np.array([0, 2, 3, 4], np.int32),
                    ordered_neighbors=np.array([1, 2, 0, 0], np.int32),
                    normal_dot_histogram=dict(bins=np.arange(3, dtype=np.float32),
                        counts=np.array([.1, .2, .7], np.float32), bin_size=np.float32(1)))

    def curvature_arguments(self):
        hist = dict(bins=np.arange(3, dtype=np.float32),
                    counts=np.array([.1, .2, .7], np.float32), bin_size=np.float32(1))
        return dict(principal_curvature_k1_mm_inv=np.array([.5, 1.9], np.float32),
                    principal_curvature_k2_mm_inv=np.array([0, 1.9], np.float32),
                    vertex_areas_mm2=np.array([1, 3], np.float32),
                    patch_vertices=np.array([0, 1], np.int32),
                    k1_histogram=hist, k2_histogram=hist,
                    joint_curvature_pdf=np.array([[.1, .2], [.3, .4]], np.float32),
                    joint_origin_mm_inv=np.zeros(2, np.float32),
                    joint_spacing_mm_inv=np.ones(2, np.float32))

    def test_nint_is_half_away_from_zero(self):
        for value, expected in [(-1.5, -2), (-.5, -1), (-.49, 0), (.5, 1), (1.5, 2)]:
            self.assertEqual(_nint(value), expected)

    def test_normal_mean_is_per_vertex_not_per_edge_and_keeps_area_order(self):
        args = self.normal_arguments()
        result = patch_normal_dot_likelihood(**args)
        counts = args['normal_dot_histogram']['counts'].astype(np.float64)
        a, b = (np.log(counts[1]) + np.log(counts[2])) / 2, np.log(counts[0])
        self.assertEqual(result['curv_ll'], float(np.float32((a + b) / 2)))
        self.assertEqual(result['cll'], float(np.float32((a + b * 3) / 4)))
        np.testing.assert_array_equal(result['per_vertex_log_likelihood'], [a, b])

    def test_current_normals_are_used_and_input_is_not_mutated(self):
        args = self.normal_arguments()
        before = {key: value.copy() for key, value in args.items() if isinstance(value, np.ndarray)}
        old = patch_normal_dot_likelihood(**args)
        for key, expected in before.items():
            np.testing.assert_array_equal(args[key], expected)
        args['vertex_normals'] = np.zeros((3, 3), np.float32)
        new = patch_normal_dot_likelihood(**args)
        self.assertNotEqual(old['curv_ll'], new['curv_ll'])
        self.assertEqual(new['curv_ll'], float(np.float32(np.log(np.float64(np.float32(.1))))))

    def test_joint_curvature_indices_truncate_and_final_mean_overwrites_marginal(self):
        args = self.curvature_arguments()
        result = patch_curvature_likelihood(**args)
        logs = np.log(np.array([np.float32(.1), np.float32(.4)], np.float64))
        self.assertEqual(result['qcurv_ll'], float(np.float32(logs.mean())))
        self.assertEqual(result['qcll'], float(np.float32((logs[0] + 3 * logs[1]) / 4)))
        counts = args['k1_histogram']['counts'].astype(np.float64)
        marginal = 2 * np.log(counts[1]) + np.log(counts[0]) + 3 * np.log(counts[2])
        self.assertEqual(result['marginal_qcurv_ll'], float(np.float32(marginal / 2)))
        self.assertNotEqual(result['qcurv_ll'], result['marginal_qcurv_ll'])

    def test_joint_axis_order_clamping_and_zero_individual_area(self):
        args = self.curvature_arguments()
        args['principal_curvature_k1_mm_inv'] = np.array([20, -20], np.float32)
        args['principal_curvature_k2_mm_inv'] = np.array([-20, 20], np.float32)
        args['vertex_areas_mm2'] = np.array([0, 3], np.float32)
        result = patch_curvature_likelihood(**args)
        self.assertEqual(result['qcll'], float(np.float32(np.log(np.float64(np.float32(.2))))))
        args['joint_curvature_pdf'] = args['joint_curvature_pdf'][:, :, None]
        self.assertEqual(patch_curvature_likelihood(**args)['qcurv_ll'], result['qcurv_ll'])

    def test_empty_patch_is_zero_without_division(self):
        for fn, args, fields in [(patch_normal_dot_likelihood, self.normal_arguments(), ['curv_ll', 'cll']),
                                 (patch_curvature_likelihood, self.curvature_arguments(), ['qcurv_ll', 'qcll', 'marginal_qcurv_ll'])]:
            args['patch_vertices'] = np.empty(0, np.int32)
            result = fn(**args)
            for field in fields:
                self.assertEqual(result[field], 0)
            self.assertEqual(result['per_vertex_log_likelihood'].size, 0)

    def test_invalid_csr_patch_area_and_pdf_are_not_silently_dropped(self):
        for key, value in [('patch_vertices', np.array([0, 0], np.int32)),
                           ('patch_vertices', np.array([0.1])),
                           ('neighbor_offsets', np.array([0, 0, 3, 4], np.int32)),
                           ('ordered_neighbors', np.array([1, 3, 0, 0], np.int32)),
                           ('vertex_areas_mm2', np.array([0, np.nan, 1], np.float32))]:
            args = self.normal_arguments(); args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                patch_normal_dot_likelihood(**args)
        args = self.curvature_arguments(); args['joint_curvature_pdf'][0, 0] = 0
        with self.assertRaises(ValueError):
            patch_curvature_likelihood(**args)

    def test_histogram_is_not_renormalized_or_nonfinite_accepted(self):
        args = self.normal_arguments()
        args['normal_dot_histogram']['counts'] *= np.float32(2)
        result = patch_normal_dot_likelihood(**args)
        self.assertAlmostEqual(result['curv_ll'] - patch_normal_dot_likelihood(**self.normal_arguments())['curv_ll'], np.log(2), places=6)
        args['normal_dot_histogram']['bin_size'] = np.nan
        with self.assertRaises(ValueError):
            patch_normal_dot_likelihood(**args)

    def test_signed_vertex_weights_are_preserved_for_both_components(self):
        for fn, arguments, weighted_name in [
                (patch_normal_dot_likelihood, self.normal_arguments, 'cll'),
                (patch_curvature_likelihood, self.curvature_arguments, 'qcll')]:
            for a, b in [(-1, 3), (-1, -3)]:
                args = arguments()
                args['vertex_areas_mm2'][:2] = [a, b]
                result = fn(**args)
                logp = result['per_vertex_log_likelihood']
                expected = np.float32((a * logp[0] + b * logp[1]) / (a + b))
                self.assertEqual(result[weighted_name], float(expected))

    def test_zero_denominator_keeps_finite_core_and_ieee_diagnostic(self):
        for fn, arguments, core, weighted in [
                (patch_normal_dot_likelihood, self.normal_arguments, 'curv_ll', 'cll'),
                (patch_curvature_likelihood, self.curvature_arguments, 'qcurv_ll', 'qcll')]:
            args = arguments()
            expected_core = fn(**args)[core]
            for areas in ([0, 0], [-3, 3]):
                args['vertex_areas_mm2'][:2] = areas
                result = fn(**args)
                self.assertEqual(result[core], expected_core)
                self.assertFalse(result[weighted + '_defined'])
                self.assertFalse(np.isfinite(result[weighted]))
                self.assertEqual(result[weighted + '_float32_bits'],
                                 int(np.asarray(result[weighted], np.float32).view(np.uint32)))
                if areas == [0, 0]:
                    self.assertTrue(np.isnan(result[weighted]))
                    self.assertEqual(result[weighted + '_float32_bits'], 0xffc00000)
        # Ordered double additions lose the small term; np.sum's pairwise
        # order is not a substitute for the denominator used by the kernel.
        args = self.curvature_arguments()
        args['principal_curvature_k1_mm_inv'] = np.zeros(3, np.float32)
        args['principal_curvature_k2_mm_inv'] = np.zeros(3, np.float32)
        args['vertex_areas_mm2'] = np.array([1e20, 1, -1e20], np.float32)
        args['patch_vertices'] = np.arange(3)
        result = patch_curvature_likelihood(**args)
        self.assertTrue(np.isfinite(result['qcurv_ll']))
        self.assertFalse(result['qcll_defined'])


if __name__ == '__main__':
    unittest.main()
