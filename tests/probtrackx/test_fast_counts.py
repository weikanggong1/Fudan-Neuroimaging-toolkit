"""Count semantics for the accelerated single-waypoint volume path."""

import numpy as np

from fnit.probtrackx._fast_counts import accumulate_single_waypoint_paths


def test_single_waypoint_avoid_counts_unique_voxels_and_discards_halves():
    forward = np.array([
        [5, 4, 2, -1, -1],
        [5, 4, 3, -1, -1],
        [5, 4, 2, 8, -1],
        [5, 4, 2, -1, -1],
        [5, 4, 2, 4, -1],
    ], dtype=np.int32)
    backward = np.array([
        [5, 6, 7, -1, -1],
        [5, 4, 2, -1, -1],
        [5, 6, 7, -1, -1],
        [5, 8, 2, -1, -1],
        [5, 6, 7, -1, -1],
    ], dtype=np.int32)
    avoid = np.zeros(10, dtype=bool)
    avoid[8] = True
    waypoint = np.zeros(10, dtype=bool)
    waypoint[2] = True
    density = np.zeros(10, dtype=np.float32)
    totals = np.zeros(1, dtype=np.int64)

    accumulate_single_waypoint_paths(
        forward, backward, avoid, waypoint, density, totals, 0)

    expected = np.zeros(10, dtype=np.float32)
    expected[[2, 4, 5, 6, 7]] = [4, 4, 4, 2, 2]
    np.testing.assert_array_equal(density, expected)
    assert totals[0] == 4
