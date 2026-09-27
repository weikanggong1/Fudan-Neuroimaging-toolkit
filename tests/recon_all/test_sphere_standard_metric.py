"""The native reciprocal pass updates the first reverse duplicate in order."""

import math

import numpy as np

from fnit.recon_all.sphere_standard_metric import _angle, average_standard_metric


def test_reciprocal_average_with_duplicate_neighbors() -> None:
    offsets = np.asarray([0, 3, 5], np.int64)
    neighbors = np.asarray([1, 1, 1, 0, 0], np.int32)
    distances = np.asarray([1, 2, 3, 4, 5], np.float32)
    result, matched, _ = average_standard_metric(offsets, neighbors, distances)
    np.testing.assert_array_equal(
        result, np.asarray([3.78125, 2.25, 2.625, 2.5625, 3.78125], np.float32))
    assert matched == 5
    np.testing.assert_array_equal(distances, [1, 2, 3, 4, 5])


def test_real_t1_angle_at_native_sampling_threshold() -> None:
    # 候选 LH smoothwm 的顶点 42548、45660、44218；单精度长度会误拒绝 45660。
    xyz = np.asarray([
        [-29.524288177490234, -19.210586547851562, 18.053237915039062],
        [-29.286697387695312, -15.171409606933594, 19.649944305419922],
        [-29.31351661682129, -17.090951919555664, 22.060016632080078],
    ], dtype=np.float32)
    threshold = np.float32(0.9 * 2.0 * math.pi / 8.0)
    observed = np.float32(_angle(xyz, 0, 1, 2))
    assert observed.view(np.uint32) == threshold.view(np.uint32)
