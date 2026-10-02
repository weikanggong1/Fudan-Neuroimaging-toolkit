"""Regression checks for sparse fixed-Gaussian mesh fitting (not a benchmark)."""

import numpy as np
import torch

from fnit.gems import GEMSAtlas, TorchGEMS
from fnit.gems.gaussian import GaussianParameters


def test_sparse_fixed_gaussians_and_interval_refresh_match_dense():
    vertices = np.array([[1., 1., 1.], [7., 1., 1.], [1., 7., 1.], [1., 1., 7.]])
    atlas = GEMSAtlas(vertices, vertices.copy(), np.array([[0, 1, 2, 3]]),
                      np.array([[.8, .2], [.2, .8], [.4, .6], [.7, .3]], np.float32),
                      .1, np.ones((4, 3), bool), np.array([0, 10]), ("Unknown", "ROI"))
    image = torch.zeros((10, 10, 10))
    image[2:4, 2:4, 2:4] = torch.tensor([1., 2.])[:, None, None]
    fixed = GaussianParameters(torch.tensor([[1.], [2.]]),
                               torch.full((2, 1, 1), .2))
    options = dict(fixed_gaussians=fixed, em_iterations=1, deform_iterations=3,
                   deform_optimizer="lbfgs", deform_lr=.1, deform_em_interval=1,
                   index_margin=2, adaptive_index=True)
    dense = TorchGEMS(atlas)(image, compact=False, **options)
    sparse = TorchGEMS(atlas)(image, compact=True, **options)
    torch.testing.assert_close(sparse.vertices, dense.vertices, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(sparse.posterior, dense.posterior, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(sparse.objective_history, dense.objective_history, atol=2e-5, rtol=2e-5)
    assert sparse.posterior.shape == (2, 10, 10, 10)
    assert sparse.min_jacobian > 0
