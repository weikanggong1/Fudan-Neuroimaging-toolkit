"""Small physical-signal checks for fixed-response three-tissue CSD."""

import math

import pytest
import torch

from fnit.connectome.fod import fit_three_tissue_csd, real_sh


def _directions(n: int, device: str) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt(1.0 - z * z)
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fixed_response_three_tissue_signal(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    directions = _directions(32, device)
    bvecs = torch.cat((torch.zeros((1, 3), device=device), directions, directions))
    bvals = torch.cat((torch.zeros(1, device=device), torch.full((32,), 1000.0, device=device), torch.full((32,), 2000.0, device=device)))
    shell_bvals = torch.tensor([0.0, 1000.0, 2000.0], device=device)
    wm_response = torch.tensor([[1.0, 0.0, 0.0], [0.50, -0.055, 0.009], [0.28, -0.045, 0.007]], device=device)
    gm_response = torch.tensor([1.0, 0.30, 0.13], device=device)
    csf_response = torch.tensor([1.0, 0.08, 0.02], device=device)

    # Independent analytic signal for a unit-integral cos(theta)^4 WM lobe.
    # A direction represented exactly by the nonnegative angular dictionary.
    kernel_index = 63
    kernel_z = (kernel_index + 0.5) / 64.0
    kernel_phi = kernel_index * (math.pi * (3.0 - math.sqrt(5.0)))
    fibre = torch.tensor([
        math.sqrt(1.0 - kernel_z * kernel_z) * math.cos(kernel_phi),
        math.sqrt(1.0 - kernel_z * kernel_z) * math.sin(kernel_phi),
        kernel_z,
    ], device=device)
    cosine = bvecs @ fibre
    p2 = (3.0 * cosine.square() - 1.0) / 2.0
    p4 = (35.0 * cosine.square().square() - 30.0 * cosine.square() + 3.0) / 8.0
    shell = torch.cat((torch.zeros(1, dtype=torch.long, device=device), torch.ones(32, dtype=torch.long, device=device), torch.full((32,), 2, dtype=torch.long, device=device)))
    wm_signal = wm_response[shell, 0] + (20.0 / 7.0) * wm_response[shell, 1] * p2 + (8.0 / 7.0) * wm_response[shell, 2] * p4
    normalized = 0.62 * wm_signal + 0.23 * gm_response[shell] + 0.15 * csf_response[shell]
    signal = torch.zeros((3, 1, 1, bvals.numel()), device=device, dtype=torch.float32)
    signal[0, 0, 0] = 1200.0 * normalized
    signal[1, 0, 0] = 700.0 * normalized
    mask = torch.tensor([True, True, False], device=device).reshape(3, 1, 1)

    wm_sh, gm, csf = fit_three_tissue_csd(
        signal, bvals, bvecs, shell_bvals, wm_response, gm_response, csf_response, mask, lmax=4
    )
    assert wm_sh.shape == (3, 1, 1, 15)
    assert wm_sh.dtype == torch.float32 and wm_sh.device.type == device
    assert gm.shape == csf.shape == (3, 1, 1)
    assert torch.allclose(wm_sh[0], wm_sh[1], atol=0.025)
    assert torch.allclose(gm[0], gm[1], atol=0.025)
    assert torch.allclose(csf[0], csf[1], atol=0.025)
    assert bool((wm_sh[2] == 0).all() and (gm[2] == 0).all() and (csf[2] == 0).all())

    basis = real_sh(torch.where(bvals[:, None] == 0, fibre[None], bvecs), 4)
    degree_response = torch.cat((wm_response[shell, 0:1], wm_response[shell, 1:2].expand(-1, 5), wm_response[shell, 2:3].expand(-1, 9)), dim=1)
    predicted = 4.0 * math.pi * ((basis * degree_response) @ wm_sh[0, 0, 0]) + gm[0, 0, 0] * gm_response[shell] + csf[0, 0, 0] * csf_response[shell]
    assert float(torch.sqrt(torch.mean((predicted - normalized).square()))) < 0.02
    wm_fraction = math.sqrt(4.0 * math.pi) * wm_sh[0, 0, 0, 0]
    assert abs(float(wm_fraction + gm[0, 0, 0] + csf[0, 0, 0] - 1.0)) < 0.03
    assert abs(float(wm_fraction - 0.62)) < 0.03
    assert abs(float(gm[0, 0, 0] - 0.23)) < 0.03
    assert abs(float(csf[0, 0, 0] - 0.15)) < 0.03
    assert float(gm[0, 0, 0]) >= 0.0 and float(csf[0, 0, 0]) >= 0.0
    assert float((real_sh(_directions(128, device), 4) @ wm_sh[0, 0, 0]).min()) >= -1e-4


def test_real_sh_even_antipodal_and_input_validation() -> None:
    directions = _directions(12, "cpu")
    y = real_sh(directions, 4)
    assert y.shape == (12, 15)
    assert torch.allclose(y, real_sh(-directions, 4), atol=1e-6)
    assert torch.allclose(y[:, 0], torch.full((12,), 1.0 / math.sqrt(4.0 * math.pi)), atol=1e-6)
    with pytest.raises(ValueError):
        real_sh(torch.zeros((1, 3)), 4)
    with pytest.raises(ValueError):
        real_sh(directions, 3)


def test_requires_b0() -> None:
    with pytest.raises(ValueError, match="b=0"):
        fit_three_tissue_csd(
            torch.ones((1, 1, 1, 1), dtype=torch.float32),
            torch.tensor([1000.0]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([1000.0]),
            torch.tensor([[0.5, -0.05, 0.01]]),
            torch.tensor([0.3]),
            torch.tensor([0.1]),
        )
