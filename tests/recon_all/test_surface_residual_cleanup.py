"""局部独立清理的覆盖、约束与失败合同；真实benchmark另列。"""
import unittest
import numpy as np

from fnit.recon_all.place_surface_residual_cleanup import _pair_sets, resolve_residual_intersections
from fnit.recon_all.mris_remove_intersection_python import mark_intersections
from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections


class SurfaceResidualCleanup(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                             [0., -.5, -1.], [0., -.5, 1.], [0., .5, .5]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)

    def test_full_and_subset_pair_coverage_matches_existing_detectors(self):
        source, independent = _pair_sets(self.xyz, self.faces)
        self.assertEqual(independent, {(0, 1)})
        self.assertEqual(source, {(0, 1), (1, 0)})
        for ids in ([0], [1], [0, 1], [1, 1]):
            self.assertEqual(_pair_sets(self.xyz, self.faces, np.array(ids)), (source, independent))
        source_marks, source_count = mark_source_intersections(self.xyz, self.faces, predicate_backend="numba")
        f64_marks, f64_count = mark_intersections(self.xyz, self.faces)
        self.assertEqual(source_count, len({a for a, b in source}))
        self.assertEqual(f64_count, len({v for pair in independent for v in pair}))
        self.assertTrue(source_marks.all()); self.assertTrue(f64_marks.all())

    def test_shared_vertex_faces_are_excluded(self):
        faces = self.faces.copy(); faces[1, 0] = 0
        self.assertEqual(_pair_sets(self.xyz, faces), (set(), set()))

    def test_fixed_crossing_is_retained_and_reported_failed(self):
        baseline = self.xyz.copy()
        result, report = resolve_residual_intersections(
            vertices=self.xyz, faces=self.faces, fixed_vertices=np.ones(6, bool),
            white_vertices=self.xyz, max_displacement_mm=1., minimum_area_ratio=.2, max_passes=1)
        self.assertEqual(report["status"], "failed_residual")
        self.assertEqual(report["fixed_different_elements"], 0)
        self.assertTrue(np.array_equal(result, baseline)); self.assertTrue(np.array_equal(self.xyz, baseline))
        self.assertTrue(all(report["global_checks"][0].values()))

    def test_independent_candidate_respects_fixed_and_orientation_limits(self):
        white = self.xyz.copy(); white[3:, 0] = 2.
        fixed = np.array([True, True, True, False, False, False])
        result, report = resolve_residual_intersections(
            vertices=self.xyz, faces=self.faces, fixed_vertices=fixed,
            white_vertices=white, max_displacement_mm=1., minimum_area_ratio=.2, max_passes=1)
        self.assertTrue(np.array_equal(result[fixed], self.xyz[fixed]))
        self.assertLessEqual(report["max_displacement_mm"], 1.)
        self.assertEqual(report["status"], "passed_zero_intersections")
        self.assertTrue(report["accepted"])
        self.assertTrue(all(row['decision']=='eligible' for row in report['accepted']))
        self.assertTrue(all(report["global_checks"][0].values()))
        self.assertNotEqual(report["strict_reproduction"], "passed")

    def test_empty_and_no_intersection_do_not_move_coordinates(self):
        for faces in (self.faces[:1], np.empty((0, 3), np.int32)):
            result, report = resolve_residual_intersections(
                vertices=self.xyz, faces=faces, fixed_vertices=np.zeros(6, bool), white_vertices=self.xyz)
            self.assertEqual(report["status"], "passed_zero_intersections")
            self.assertTrue(np.array_equal(result, self.xyz))

    def test_invalid_parameters_or_domain_fail_before_geometry_change(self):
        options = dict(vertices=self.xyz, faces=self.faces, fixed_vertices=np.zeros(6, bool), white_vertices=self.xyz)
        for change in (dict(vertices=self.xyz.astype(np.float64)), dict(fixed_vertices=np.zeros(6, np.int32)),
                       dict(max_displacement_mm=0), dict(max_displacement_mm=np.nan), dict(minimum_area_ratio=0),
                       dict(max_passes=True), dict(max_passes=0), dict(faces=np.array([[0, 1, 6]])),
                       dict(faces=np.array([[0, 0, 1]])), dict(white_vertices=self.xyz[:5])):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                resolve_residual_intersections(**(options | change))


if __name__ == "__main__": unittest.main()
