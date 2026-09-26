"""Synthetic known-signal checks for mask-guided response estimation."""

import math

import pytest
import torch

from fnit.connectome.fod import fit_three_tissue_csd
from fnit.connectome.response import estimate_three_tissue_response


def _directions(n: int, device: str) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt(1.0 - z.square())
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=1)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_recovers_three_tissue_responses_from_known_signals(device):
    directions = _directions(48, device)
    bvecs = torch.cat((torch.zeros((2, 3), device=device), directions, directions))
    bvals = torch.cat((torch.zeros(2, device=device),
                       torch.tensor([980., 1020.], device=device).repeat(24),
                       torch.tensor([1980., 2020.], device=device).repeat(24)))
    shell = torch.cat((torch.zeros(2, dtype=torch.long, device=device),
                       torch.ones(48, dtype=torch.long, device=device),
                       torch.full((48,), 2, dtype=torch.long, device=device)))
    true_wm = torch.tensor([[1., 0., 0.], [.5, -.055, .009], [.28, -.045, .007]], device=device)
    true_gm = torch.tensor([1., .30, .13], device=device)
    true_csf = torch.tensor([1., .08, .02], device=device)

    def wm_signal(fibre):
        cosine = bvecs @ fibre
        p2 = (3 * cosine.square() - 1) / 2
        p4 = (35 * cosine.pow(4) - 30 * cosine.square() + 3) / 8
        return (true_wm[shell, 0] + 5 * true_wm[shell, 1] * p2
                + 9 * true_wm[shell, 2] * p4)

    pure_wm = [wm_signal(axis) for axis in torch.eye(3, device=device)]
    # The fourth WM-labelled voxel is a low-FA GM-like contaminant.
    normalized = torch.stack((*pure_wm, true_gm[shell], true_gm[shell], true_csf[shell]))
    scales = torch.tensor([800., 1000., 1200., 950., 700., 1400.], device=device)
    signal = (normalized * scales[:, None]).reshape(6, 1, 1, -1)
    labels = torch.tensor([2, 2, 2, 2, 1, 3], device=device).reshape(6, 1, 1)
    fa = torch.tensor([.7, .8, .9, .05, .15, .02], device=device).reshape(6, 1, 1)

    shell_bvals, wm, gm, csf = estimate_three_tissue_response(
        signal, bvals, bvecs, labels, fa=fa, lmax=4
    )
    assert shell_bvals.device.type == device
    assert wm.dtype == gm.dtype == csf.dtype == torch.float32
    torch.testing.assert_close(shell_bvals, torch.tensor([0., 1000., 2000.], device=device))
    torch.testing.assert_close(wm[0], torch.tensor([1., 0., 0.], device=device))
    assert gm[0] == csf[0] == 1
    torch.testing.assert_close(wm[1:], true_wm[1:], atol=0.012, rtol=0)
    torch.testing.assert_close(gm, true_gm, atol=1e-6, rtol=0)
    torch.testing.assert_close(csf, true_csf, atol=1e-6, rtol=0)
    clean_labels = labels.clone()
    clean_labels[3] = 0
    _, wm_without_fa, _, _ = estimate_three_tissue_response(
        signal, bvals, bvecs, clean_labels, lmax=4
    )
    torch.testing.assert_close(wm_without_fa[1:], true_wm[1:], atol=0.012, rtol=0)
    wm_fod, gm_fraction, csf_fraction = fit_three_tissue_csd(
        signal[:1], bvals, bvecs, shell_bvals, wm, gm, csf, lmax=4
    )
    assert bool(torch.isfinite(wm_fod).all())
    assert bool(torch.isfinite(gm_fraction).all() and torch.isfinite(csf_fraction).all())


def test_requires_tissue_voxels_and_identifiable_gradients():
    signal = torch.ones((3, 1, 1, 3), dtype=torch.float32)
    bvals = torch.tensor([0., 1000., 2000.])
    bvecs = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    labels = torch.tensor([2, 1, 3]).reshape(3, 1, 1)
    with pytest.raises(ValueError, match="two diffusion shells"):
        estimate_three_tissue_response(signal, torch.tensor([0., 1000., 1000.]), bvecs,
                                       torch.tensor([2, 1, 3]).reshape(3, 1, 1))
    with pytest.raises(ValueError, match="six low-shell"):
        estimate_three_tissue_response(signal, bvals, bvecs, labels)
    labels[2] = 0
    with pytest.raises(ValueError, match="tissue-label 3"):
        estimate_three_tissue_response(signal, bvals, bvecs, labels)
