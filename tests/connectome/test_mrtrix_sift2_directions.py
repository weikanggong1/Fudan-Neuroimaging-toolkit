"""Source-derived MRtrix SIFT2 grid and quadrature checks."""

import math
from importlib.resources import files

import numpy as np
from scipy.special import sph_harm_y


def test_mrtrix_sift2_direction_asset() -> None:
    asset = files("fnit.connectome").joinpath("data/mrtrix_sift2_1281.npz")
    with asset.open("rb") as stream, np.load(stream) as data:
        directions = data["directions"]
        indptr = data["adjacency_indptr"]
        indices = data["adjacency_indices"]
        weights = data["integration_weights"]

    assert directions.shape == (1281, 3) and directions.dtype == np.float64
    assert indptr.shape == (1282,) and indptr.dtype == np.int32
    assert indices.shape == (7680,) and indices.dtype == np.int16
    assert weights.shape == (1281,) and weights.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=3e-16)
    assert np.array_equal(np.unique(np.diff(indptr), return_counts=True)[1], [6, 1275])
    for row in range(1281):
        neighbors = indices[indptr[row] : indptr[row + 1]]
        assert row not in neighbors
        assert np.all(neighbors[:-1] < neighbors[1:])
        assert all(row in indices[indptr[n] : indptr[n + 1]] for n in neighbors)

    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    elevation = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    moments = []
    for degree in range(0, 25, 2):
        for order in range(-degree, degree + 1):
            harmonic = sph_harm_y(degree, abs(order), elevation, azimuth)
            value = (math.sqrt(2) * harmonic.imag if order < 0 else
                     math.sqrt(2) * harmonic.real if order > 0 else harmonic.real)
            moments.append(weights @ value)
    assert abs(sum(weights) - 4 * math.pi) < 2e-8
    assert abs(moments[0] - 2 * math.sqrt(math.pi)) < 4e-9
    assert max(abs(value) for value in moments[1:]) < 1e-6
    assert weights.min() > 0 and weights.max() / weights.min() > 2
