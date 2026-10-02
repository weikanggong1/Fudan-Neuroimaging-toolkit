"""Compiled independent assignments retain sentinel and mixed-region rules."""
import unittest
import numpy as np
from fnit.recon_all.place_surface_collision import _assign_vertices


class PlacementSubvolumeTest(unittest.TestCase):
    def test_ripped_isolated_equal_and_mixed_regions(self):
        regions = np.array([7, 7, 63, 64, 7], np.int32)
        incident = np.array([0, 1, 0, 2, 3, 4], np.int64)
        offsets = np.array([0, 2, 4, 5, 6, 6], np.int64)
        ripped = np.array([False, False, False, True, False])
        result = _assign_vertices(regions, incident, offsets, ripped)
        np.testing.assert_array_equal(result, [7, 64, 64, -1, -1])
        self.assertEqual(result.dtype, np.int32)


if __name__ == '__main__':
    unittest.main()
