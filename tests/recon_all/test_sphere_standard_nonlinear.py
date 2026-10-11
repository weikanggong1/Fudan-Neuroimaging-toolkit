"""Regression for the native one-ring nonlinear-area transition."""

import math

import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.sphere_standard_nonlinear import (
    _nonlinear_area_sse, nonlinear_epoch_sse, one_ring_metric,
)


def test_one_ring_metric_keeps_full_metric_average_neighbor_count():
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], np.int32)
    rows = ordered_neighbors(faces, 4)
    full_rows = [np.r_[row, row[0]] for row in rows]
    full_neighbors = np.concatenate(full_rows).astype(np.int32)
    full_offsets = np.arange(0, len(full_neighbors) + 1, len(full_rows[0]), dtype=np.int64)
    full_distances = np.arange(len(full_neighbors), dtype=np.float32)

    offsets, neighbors, distances, old_average = one_ring_metric(
        faces, 4, full_offsets, full_neighbors, full_distances)

    assert np.array_equal(np.diff(offsets), [3, 3, 3, 3])
    assert np.array_equal(neighbors, np.concatenate(rows))
    assert np.array_equal(distances, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
    assert old_average == np.float32(4)


def test_nonlinear_area_sse_uses_native_logistic_area_penalty():
    area = np.array([-0.5, 0, 0.5], np.float32)
    expected = sum(math.log(1 + math.exp(10 * float(a))) / 10 - float(a)
                   for a in area)
    assert math.isclose(_nonlinear_area_sse(area, 1.0, 10.0), expected,
                        rel_tol=0, abs_tol=1e-12)


def test_nonlinear_sse_uses_native_float32_distance_weight():
    vertices = np.asarray([[100, 0, 0], [0, 100, 0]], np.float32)
    result = nonlinear_epoch_sse(
        vertices, np.empty((0, 3), np.int32),
        np.asarray([0, 1, 2], np.int64), np.asarray([1, 0], np.int32),
        np.zeros(2, np.float32), np.float32(4 * math.pi * 10000), 0.1)
    arc = float(np.float32(np.float32(math.pi / 2) * np.float32(100)))
    expected = float(np.float32(0.1)) * (2 * arc * arc)
    assert result["weighted_distance"] == expected
    assert result["total"] == expected


def test_nonlinear_distance_scale_uses_fixed_tu_float_sqrt():
    vertices = np.asarray([[100, 0, 0], [0, 100, 0]], np.float32)
    result = nonlinear_epoch_sse(
        vertices, np.empty((0, 3), np.int32),
        np.asarray([0, 1, 2], np.int64), np.asarray([1, 0], np.int32),
        np.zeros(2, np.float32), np.float32(float.fromhex("0x1.140f74p+16")))
    # Same expression independently compiled with fixed original TU headers.
    assert result["distance_scale"] == float.fromhex("0x1.7ff6p-1")
