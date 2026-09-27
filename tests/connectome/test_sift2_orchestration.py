"""SIFT2 FMLS → track mapping → optimizer integration contract."""

import torch

from fnit.connectome.sift2 import estimate_sift2_weights


def test_sift2_small_fod_returns_finite_track_weights():
    fod = torch.zeros((5, 5, 5, 45), dtype=torch.float32)
    fod[..., 0] = 1
    five = torch.zeros((5, 5, 5, 5), dtype=torch.float32)
    five[..., 2] = 1
    paths = (
        torch.tensor([[0., 2., 2.], [1., 2., 2.], [2., 2., 2.],
                      [3., 2., 2.], [4., 2., 2.]]),
        torch.tensor([[0., 3., 2.], [1., 3., 2.], [2., 3., 2.],
                      [3., 3., 2.], [4., 3., 2.]]),
    )
    weights = estimate_sift2_weights(
        paths, fod, torch.eye(4), five, torch.eye(4), step_size_mm=.5,
        processing_mask=torch.ones((5, 5, 5), dtype=torch.float32),
    )
    assert weights.shape == (2,)
    assert weights.dtype == torch.float64
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()
