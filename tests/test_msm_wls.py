"""Ordered WLS arithmetic and public native buffer boundary checks."""
import math

import numpy as np
import pytest

from fnit.msm import _fastpd_native


def _literal_cost(samples, sigma):
    # Python scalar libm is an independent CPU oracle for source arithmetic.
    total = 0.0
    for row in samples:
        weight_sum = value_sum = 0.0
        for distance, similarity, valid in row:
            if valid and distance > 0.0:
                weight = math.exp(-float(distance) / ((2.0*sigma)*sigma))
                weight_sum += weight
                value_sum += float(similarity)*weight
        total += value_sum / weight_sum if weight_sum else value_sum
    return total


@pytest.mark.parametrize("similarities", [
    [[-2., -0.5, -3., -1.], [-1., -4., -0.2, -8.]],  # sim1 SSD
    [[1., -1., 0., 1.], [-1., 1., 1., 0.]],  # sim2 single-feature Pearson
])
def test_wls_matches_literal_cpu_arithmetic(similarities):
    samples = np.stack(([[0., 0.7, 1.2, 2.], [0.3, 0.8, 0., 3.]],
                        similarities, [[1., 1., 1., 0.], [1., 0., 1., 1.]]), axis=-1)
    sigma = 1.8885289137246517
    actual = _fastpd_native.source_wls_cost(samples, 2, 4, sigma)
    expected = _literal_cost(samples, sigma)
    assert np.float64(actual).view(np.uint64) == np.float64(expected).view(np.uint64)


def test_wls_preserves_serial_vertex_sum():
    samples = np.array([[[1., 1., 1.]], [[1., 2.**-54, 1.]], [[1., -1., 1.]]])
    assert _fastpd_native.source_wls_cost(samples, 3, 1, 1.) == 0.0
    assert _fastpd_native.source_wls_cost(samples[[0, 2, 1]], 3, 1, 1.) == 2.**-54


def test_wls_skips_zero_distance_and_invalid_query_points():
    samples = np.array([[[0., -10., 1.], [1., -3., 0.], [2., -2., 1.]],
                        [[0., 1., 1.], [3., -1., 0.], [2., 1., 0.]]])
    assert _fastpd_native.source_wls_cost(samples, 2, 3, 1.) == -2.


def test_wls_accepts_unaligned_readonly_buffer():
    samples = np.array([[[0.2, -1., 1.], [0.7, 1., 1.]]])
    view = memoryview(b'x'+samples.tobytes())[1:]
    assert _fastpd_native.source_wls_cost(view, 1, 2, 1.) == _literal_cost(samples, 1.)


@pytest.mark.parametrize("field,value", [(0, -1.), (0, np.nan), (0, np.inf),
                                       (1, np.nan), (1, np.inf),
                                       (2, -1.), (2, 0.5), (2, np.nan)])
def test_wls_rejects_invalid_sample(field, value):
    sample = np.ones((1, 1, 3)); sample[0, 0, field] = value
    with pytest.raises(ValueError, match="WLS distance and similarity"):
        _fastpd_native.source_wls_cost(sample, 1, 1, 1.)


@pytest.mark.parametrize("sigma", [0., -1., np.nan, np.inf, 1e-300, 1e300])
def test_wls_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="finite positive sigma"):
        _fastpd_native.source_wls_cost(np.ones((1, 1, 3)), 1, 1, sigma)


@pytest.mark.parametrize("rows,width", [(-1, 1), (1, 0), (1, -1), (2, 1), (1, 2),
                                      (2**62, 2**62)])
def test_wls_rejects_invalid_dimensions_or_payload_length(rows, width):
    with pytest.raises(ValueError, match="float64"):
        _fastpd_native.source_wls_cost(np.ones((1, 1, 3)), rows, width, 1.)


def test_wls_empty_vertex_set_returns_zero():
    assert _fastpd_native.source_wls_cost(b'', 0, 1, 1.) == 0.
