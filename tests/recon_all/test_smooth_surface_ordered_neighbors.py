"""有序邻接的角点方向、重复、孤立顶点及非法索引契约。"""
import unittest

import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors, average_positions


class OrderedNeighborTests(unittest.TestCase):
    def test_face_order_previous_before_following_and_duplicate_first_occurrence(self):
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2]], np.int32)
        self.assertEqual(ordered_neighbors(faces, 5),
                         [[2, 1, 3], [0, 2], [1, 0, 3], [2, 0], []])
        self.assertEqual(ordered_neighbors(faces[[1, 0, 2]], 5)[0], [3, 2, 1])

    def test_repeated_corner_self_neighbor_and_zero_degree_are_retained(self):
        self.assertEqual(ordered_neighbors(np.array([[0, 0, 1]], np.int64), 3),
                         [[1, 0], [0], []])
        self.assertEqual(ordered_neighbors(np.empty((0, 3), np.int32), 3), [[], [], []])
        self.assertEqual(ordered_neighbors(np.empty((0, 3), np.int32), 0), [])

    def test_returned_rows_are_independent_and_do_not_cache_changed_faces(self):
        faces = np.array([[0, 1, 2]], np.int32)
        first = ordered_neighbors(faces, 3)
        first[0].append(27)
        second = ordered_neighbors(faces, 3)
        self.assertEqual(second[0], [2, 1])
        faces[0] = [0, 2, 1]
        self.assertEqual(ordered_neighbors(faces, 3)[0], [1, 2])

    def test_averaging_preserves_float32_rounding_order_and_isolated_vertex(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        xyz = np.array([[2**24, 0, 0], [-2**24, 0, 0], [1, 0, 0],
                        [2, 0, 0], [5, 6, 7]], np.float32)
        result = average_positions(xyz, faces, iterations=1, device='cpu')
        # v0 adds neighbor 2 (rounded away), then 1, then 3; sorting would
        # instead add 1 first and produce .75 rather than .5.
        self.assertEqual(result[0, 0], np.float32(.5))
        np.testing.assert_array_equal(result[4], xyz[4])
        self.assertEqual(result.dtype, np.float32)

    def test_malformed_faces_and_indices_fail_before_compiled_access(self):
        for faces, count in [(np.array([[0., 1., 2.]]), 3),
                             (np.array([0, 1, 2]), 3),
                             (np.array([[0, 1, -1]]), 3),
                             (np.array([[0, 1, 3]]), 3),
                             (np.empty((0, 3), np.int32), True)]:
            with self.subTest(faces=faces, count=count), self.assertRaises(ValueError):
                ordered_neighbors(faces, count)


if __name__ == '__main__':
    unittest.main()
