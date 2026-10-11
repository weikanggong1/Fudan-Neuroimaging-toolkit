"""局部投影的真实约束合同；模拟三角面不替代真实T1 benchmark。"""
import unittest
import numpy as np
from fnit.recon_all.place_surface_residual_cleanup import _face_normals, _pair_sets
from fnit.recon_all.place_surface_residual_projection import resolve_projected_intersections


class SurfaceResidualProjection(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                             [0., -.5, -.3], [0., -.5, .3], [0., .5, .3]], np.float32)
        self.faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)

    def test_fixed_opponent_and_full_affected_geometry_preserved(self):
        fixed = np.array([True, True, True, False, False, False])
        baseline = self.xyz.copy()
        result, report = resolve_projected_intersections(vertices=self.xyz, faces=self.faces,
            fixed_vertices=fixed, white_vertices=self.xyz, max_sweeps=1)
        self.assertEqual(report['status'], 'passed_zero_intersections')
        self.assertEqual(_pair_sets(result, self.faces), (set(), set()))
        self.assertTrue(np.array_equal(result[fixed], baseline[fixed]))
        self.assertTrue(np.array_equal(self.xyz, baseline))
        before = _face_normals(baseline, self.faces); after = _face_normals(result, self.faces)
        self.assertTrue(np.all(np.sum(before*after, axis=1) > 0))
        self.assertGreaterEqual((np.linalg.norm(after, axis=1)/np.linalg.norm(before, axis=1)).min(), .2)
        self.assertLessEqual(report['max_displacement_mm'], 1.)
        self.assertTrue(all(report['global_checks'][0].values()))
        self.assertEqual(report['overall_metric_equivalence'], 'not_assessed')
        self.assertTrue(all(row['decision']=='eligible' for row in report['accepted']))

    def test_fully_fixed_case_remains_a_visible_failure(self):
        result, report = resolve_projected_intersections(vertices=self.xyz, faces=self.faces,
            fixed_vertices=np.ones(6, bool), white_vertices=self.xyz, max_sweeps=1)
        self.assertEqual(report['status'], 'failed_residual')
        self.assertTrue(np.array_equal(result, self.xyz)); self.assertEqual(report['accepted'], [])

    def test_limited_diagnostic_does_not_claim_complete_processing(self):
        result, report = resolve_projected_intersections(vertices=self.xyz, faces=self.faces,
            fixed_vertices=np.ones(6, bool), white_vertices=self.xyz, diagnostic_pair_limit=1)
        self.assertEqual(report['scope'], 'limited_diagnosis')

    def test_free_plane_keeps_exact_fixed_and_same_dual_detector_gates(self):
        fixed = np.array([True, True, True, False, False, False])
        result, report = resolve_projected_intersections(vertices=self.xyz, faces=self.faces,
            fixed_vertices=fixed, white_vertices=self.xyz, max_sweeps=1,
            projection_mode='free_separating_plane')
        self.assertTrue(np.array_equal(result[fixed], self.xyz[fixed]))
        self.assertEqual(report['algorithm'], 'independent_free_plane_pair_solver_v1')
        self.assertEqual(report['status'], 'passed_zero_intersections')
        self.assertEqual(_pair_sets(result, self.faces), (set(), set()))
        self.assertTrue(all(report['global_checks'][0].values()))
        self.assertLessEqual(report['max_displacement_mm'], 1.)

    def test_initial_pair_limit_does_not_refill_after_acceptance(self):
        xyz = np.concatenate((self.xyz, self.xyz+np.array([10.,0.,0.],np.float32)))
        faces = np.concatenate((self.faces,self.faces+6))
        fixed = np.tile([True,True,True,False,False,False],2)
        result, report = resolve_projected_intersections(vertices=xyz,faces=faces,
            fixed_vertices=fixed,white_vertices=xyz,diagnostic_pair_limit=1,max_sweeps=4)
        self.assertEqual(report['limited_initial_pairs'], [[0,1]])
        self.assertEqual(len(report['accepted']),1)
        self.assertTrue(np.array_equal(result[6:],xyz[6:]))
        self.assertEqual(report['float64_pairs_after'],1)
        self.assertEqual(report['status'],'failed_residual')

    def test_invalid_parameters_fail_without_modifying_input(self):
        options = dict(vertices=self.xyz, faces=self.faces, fixed_vertices=np.zeros(6, bool), white_vertices=self.xyz)
        for change in (dict(clearance_mm=0), dict(max_sweeps=True), dict(optimizer_iterations=0),
                       dict(diagnostic_pair_limit=False), dict(diagnostic_pair_limit=0), dict(projection_mode='unknown'),
                       dict(crossing_guard='yes')):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                resolve_projected_intersections(**(options | change))


if __name__ == '__main__': unittest.main()
