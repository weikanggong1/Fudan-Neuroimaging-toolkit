"""Small physical and density checks for the SIFT2-inspired approximation."""

import math

import pytest
import torch

from fnit.connectome.sift2 import estimate_sift2_weights


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_duplicate_streamlines_share_orientation_density(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    wm_sh = torch.full((7, 5, 3, 1), 1.0 / math.sqrt(4.0 * math.pi), device=device)
    affine = torch.eye(4, device=device)
    x = torch.arange(1, 6, device=device, dtype=torch.float32)
    crowded = torch.stack((x, torch.ones_like(x), torch.ones_like(x)), dim=1)
    sparse = torch.stack((x, torch.full_like(x, 3), torch.ones_like(x)), dim=1)
    outside = sparse + torch.tensor([0.0, 10.0, 0.0], device=device)

    weights = estimate_sift2_weights((crowded, crowded.clone(), sparse, outside), wm_sh, affine, lmax=0)

    assert weights.shape == (4,)
    assert weights.dtype == torch.float32 and weights.device.type == device_name
    assert torch.isfinite(weights).all()
    assert (weights[:3] > 0).all()
    torch.testing.assert_close(weights[0], weights[1], rtol=1e-5, atol=1e-5)
    # Two identical paths should share the same FOD-supported density that
    # one path receives in a separate, otherwise identical corridor.
    torch.testing.assert_close(weights[0] + weights[1], weights[2], rtol=0.03, atol=0.03)
    assert weights[3] == 0


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_world_mm_and_affine_scale_together(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    wm_sh = torch.full((7, 3, 3, 1), 1.0 / math.sqrt(4.0 * math.pi), device=device)
    x = torch.arange(1, 6, device=device, dtype=torch.float32)
    path = torch.stack((x, torch.ones_like(x), torch.ones_like(x)), dim=1)
    identity = torch.eye(4, device=device)
    scaled = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0], device=device))

    original = estimate_sift2_weights((path,), wm_sh, identity, lmax=0)
    twice_mm = estimate_sift2_weights((path * 2.0,), wm_sh, scaled, lmax=0)

    torch.testing.assert_close(original, twice_mm, rtol=1e-5, atol=1e-5)


def test_empty_tractogram() -> None:
    wm_sh = torch.zeros((2, 2, 2, 1), dtype=torch.float32)
    weights = estimate_sift2_weights((), wm_sh, torch.eye(4), lmax=0)
    assert weights.shape == (0,)
