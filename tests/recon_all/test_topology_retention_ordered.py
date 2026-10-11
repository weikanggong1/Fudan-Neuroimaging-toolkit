"""合并后的源顺序retention接口；测试输入边界，不替代真实基准。"""
import unittest
import numpy as np

from fnit.recon_all.topology_preflight_python import (
    defect_border_labels, defect_hull_labels, defect_regions,
    defect_retention_status,
)


class OrderedRetentionTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[0, 0, 10], [.01, 0, 10], [2, 0, 10], [0, 2, 10]], np.float32)
        self.orig = self.xyz.copy()
        self.faces = np.array([[0, 2, 3], [1, 2, 3], [0, 1, 3]], np.int32)
        self.labels = np.array([1, 1, 0, 0], np.int32)

    def status(self, regions):
        return defect_retention_status(self.xyz, self.orig, self.faces, self.labels,
                                       ordered_regions=regions)

    def test_default_bfs_and_explicit_source_rows_have_identical_full_outputs(self):
        rows = defect_regions(self.labels, self.faces)
        old = defect_retention_status(self.xyz, self.orig, self.faces, self.labels)
        np.testing.assert_array_equal(self.status(rows), old)
        np.testing.assert_array_equal(defect_border_labels(self.labels, self.faces, ordered_regions=rows),
                                      defect_border_labels(self.labels, self.faces))
        np.testing.assert_array_equal(defect_hull_labels(self.labels, self.faces, ordered_regions=rows),
                                      defect_hull_labels(self.labels, self.faces))

    def test_merge_concat_order_controls_which_close_inside_vertex_is_retained(self):
        forward = self.status([([0, 1], [2, 3])])
        reverse = self.status((dict(vertices=np.array([1, 0]), border=np.array([3, 2])),))
        np.testing.assert_array_equal(forward, [1, -1, 0, 0])
        np.testing.assert_array_equal(reverse, [-1, 1, 0, 0])
        # Source status conversion is explicit; outside0 is not DISCARD.
        np.testing.assert_array_equal((reverse[[1, 0]] == -1).astype(np.int32), [0, 1])

    def test_disconnected_merged_inside_union_is_supported_without_reordering(self):
        xyz = np.array([[0, 0, 10], [2, 0, 10], [0, 2, 10],
                        [4, 0, 10], [6, 0, 10], [4, 2, 10]], np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
        labels = np.array([1, 0, 0, 1, 0, 0], np.int32)
        result = defect_retention_status(xyz, xyz, faces, labels,
                                        ordered_regions=[([3, 0], [5, 4, 2, 1])])
        np.testing.assert_array_equal(result, [1, 0, 0, 1, 0, 0])

    def test_source_border_outer_loop_does_not_add_border_inside_distance_checks(self):
        labels = np.array([1, 0, 0, 0], np.int32)
        # The close border vertex1 is closer than either threshold. j<ninside
        # nevertheless leaves inside0 retained; this is the fixed source rule.
        result = defect_retention_status(self.xyz, self.orig,
            np.array([[0, 1, 3], [0, 2, 3]], np.int32), labels,
            ordered_regions=[([0], [1, 2, 3])])
        np.testing.assert_array_equal(result, [1, 0, 0, 0])

    def test_incomplete_inside_or_border_and_foreign_positive_border_fail(self):
        for rows in ([([0], [2, 3])], [([0, 1], [2])], [([0, 1], [1, 2, 3])]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.status(rows)

    def test_rows_must_be_integer_unique_and_in_range_before_conversion(self):
        for row in ([0., 1.], [0, 0, 1], np.array([0, 2**40], np.int64), [0, -1]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.status([(row, [2, 3])])
        with self.assertRaises(ValueError):
            defect_retention_status(self.xyz, self.orig, self.faces.astype(float), self.labels,
                                    ordered_regions=[([0, 1], [2, 3])])

    def test_region_count_and_label_identity_must_cover_same_complete_domain(self):
        for labels, rows in ((self.labels, []), (np.array([2, 2, 0, 0]), [([0, 1], [2, 3])]),
                              (np.array([1, 2, 0, 0]), [([0, 1], [2, 3]), ([1], [2, 3])])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                defect_retention_status(self.xyz, self.orig, self.faces, labels, ordered_regions=rows)

    def test_inputs_and_source_row_order_are_not_modified(self):
        row = [1, 0]; border = [3, 2]
        before = [a.copy() for a in (self.xyz, self.orig, self.faces, self.labels)]
        self.status([dict(vertices=row, border=border)])
        for actual, original in zip((self.xyz, self.orig, self.faces, self.labels), before):
            np.testing.assert_array_equal(actual, original)
        self.assertEqual(row, [1, 0]); self.assertEqual(border, [3, 2])

    def test_empty_defect_domain_accepts_empty_explicit_region_list(self):
        result = defect_retention_status(self.xyz, self.orig, self.faces,
                                        np.zeros(4, np.int32), ordered_regions=[])
        np.testing.assert_array_equal(result, np.zeros(4, np.int8))


if __name__ == '__main__':
    unittest.main()
