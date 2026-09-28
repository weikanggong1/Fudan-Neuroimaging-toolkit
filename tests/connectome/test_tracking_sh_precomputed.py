"""Real ds004666 FOD samples against instrumented MRtrix iFOD2."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fnit.connectome.fod import tracking_sh_precomputed


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_ifod2_arc_amplitude_and_probability(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    fixture = Path(__file__).parent / "data/real_ifod2_arc_ds004666.npz"
    with np.load(fixture, allow_pickle=False) as real:
        directions = torch.as_tensor(real["directions"].copy(), device=device)
        coefficients = torch.as_tensor(real["fod_coefficients"].copy(), device=device)
        expected_start = torch.as_tensor(real["official_start_amplitude"].copy(), device=device)
        expected_probability = torch.as_tensor(real["official_probability"].copy(), device=device)
        cutoff = float(real["cutoff"])

    basis = tracking_sh_precomputed(directions, lmax=8)
    amplitude = (basis * coefficients).sum(-1)
    probability = torch.exp(.5 * (
        .5 * amplitude[:, 0].clamp_min(1e-20).log() +
        amplitude[:, 1].clamp_min(1e-20).log() +
        .5 * amplitude[:, 2].clamp_min(1e-20).log()
    )) * ((amplitude[:, 1] >= cutoff) & (amplitude[:, 2] >= cutoff))

    assert torch.equal(probability > 0, expected_probability > 0)
    assert float((amplitude[:, 0] - expected_start).abs().max()) < 2e-6
    assert float((probability - expected_probability).abs().max()) < 2e-6
