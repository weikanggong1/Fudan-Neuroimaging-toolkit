"""Independent SciPy controls for the shared spline sampler's short axes."""

import numpy as np
import pytest
import torch
from scipy.ndimage import map_coordinates

from fnit.eddy.fsl2111_strict.spline import (
    _pad_cubic_coefficients,
    sample_cubic_periodic_fast,
)


@pytest.mark.parametrize("shape", [(1, 2, 3), (2, 1, 2), (3, 2, 1), (4, 5, 6)])
@pytest.mark.parametrize("boundary,scipy_mode", [("mirror", "mirror"), ("periodic", "grid-wrap")])
def test_short_axis_spline_against_independent_scipy(shape, boundary, scipy_mode):
    rng = np.random.default_rng(731)
    coefficients = rng.normal(size=shape)
    # Nonnegative queries also exercise reflected/wrapped points outside the
    # FOV without changing the existing FSL negative-coordinate convention.
    coordinates = rng.uniform(size=(3, 37)) * np.asarray(shape)[:, None]
    coordinates[:, :3] = np.asarray([np.zeros(3), np.asarray(shape) - 1,
                                     np.asarray(shape)]).T
    expected = map_coordinates(coefficients, coordinates, order=3,
                               mode=scipy_mode, prefilter=False)
    tensor = torch.tensor(coefficients[None], dtype=torch.float64)
    query = torch.tensor(coordinates.reshape(1, 3, 37, 1, 1), dtype=torch.float64)
    actual = sample_cubic_periodic_fast(tensor, query, boundary=boundary)
    reused = sample_cubic_periodic_fast(
        tensor, query, boundary=boundary,
        padded_coeff=_pad_cubic_coefficients(tensor, boundary),
    )
    np.testing.assert_allclose(actual.numpy().ravel(), expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(actual, reused, rtol=0, atol=0)
