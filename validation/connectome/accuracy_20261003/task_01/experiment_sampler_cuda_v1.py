"""Independent boundary controls for the shared CUDA cubic sampler.

These compact fixtures verify other sampler callers; real-data oracle results
and runtime measurements are recorded separately in the task report.
"""
import numpy as np
import pytest
import torch
from scipy.ndimage import map_coordinates

from fnit.eddy.fsl2111_strict.spline import (
    _pad_cubic_coefficients,
    sample_cubic_periodic_fast,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(1, 2, 3), (2, 1, 2), (7, 8, 6)])
def test_periodic_cuda_batch_and_padding_match_independent_scipy(shape):
    generator = np.random.default_rng(731)
    coefficients = generator.normal(size=shape).astype(np.float32)
    # Include wrapped points and exact edge centres. Nonnegative queries use
    # SciPy's cubic convention; FSL's negative tap convention is unchanged.
    coordinates = (generator.random((2, 3, 7, 2, 3))
                   * np.asarray(shape)[None, :, None, None, None] * 2).astype(np.float32)
    coordinates[:, :, 0, 0, 0] = 0
    coordinates[:, :, 1, 0, 0] = np.asarray(shape) - 1
    coordinates[:, :, 2, 0, 0] = np.asarray(shape)
    expected = np.stack([
        map_coordinates(coefficients.astype(np.float64), grid.astype(np.float64),
                        order=3, mode="grid-wrap", prefilter=False)
        for grid in coordinates
    ])
    tensor = torch.from_numpy(coefficients).cuda()
    query = torch.from_numpy(coordinates).cuda()
    actual = sample_cubic_periodic_fast(tensor, query)
    padded = _pad_cubic_coefficients(tensor[None].expand(2, -1, -1, -1), "periodic")
    reused = sample_cubic_periodic_fast(tensor, query, padded_coeff=padded)
    assert actual.shape == query.shape[:1] + query.shape[2:]
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=4e-7, rtol=2e-6)
    torch.testing.assert_close(actual, reused, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_coefficient_gradient_fallback_remains_available():
    coefficients = torch.arange(24, dtype=torch.float32, device="cuda:0").reshape(2, 3, 4)
    coefficients.requires_grad_(True)
    coordinates = torch.tensor([.2, 1.2, 2.2], device="cuda:0").reshape(1, 3, 1, 1, 1)
    sampled = sample_cubic_periodic_fast(coefficients, coordinates)
    sampled.sum().backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    torch.testing.assert_close(coefficients.grad.sum(), torch.ones((), device="cuda:0"),
                               atol=2e-6, rtol=0)
