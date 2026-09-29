"""真实 ds004666 FOD 系数下的 iFOD2 连续初始方向规则。"""

from pathlib import Path

import numpy as np
import pytest
import torch

from fnit.connectome.fod import tracking_sh_precomputed
from fnit.connectome.tracking import _initial_directions


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_fod_initial_directions_are_legal_and_reproducible(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = Path(__file__).parent / "data/real_ifod2_arc_ds004666.npz"
    with np.load(path, allow_pickle=False) as real:
        coefficients = torch.as_tensor(real["fod_coefficients"][:, 0, :].copy(), device=device)
    first = _initial_directions(
        coefficients, torch.Generator(device=device).manual_seed(23), lmax=8, cutoff=0.1,
    )
    repeat = _initial_directions(
        coefficients, torch.Generator(device=device).manual_seed(23), lmax=8, cutoff=0.1,
    )
    for left, right in zip(first, repeat):
        assert torch.equal(left, right)
    directions, valid, attempts = first
    assert bool(valid.any())
    amplitude = (coefficients * tracking_sh_precomputed(directions, 8)).sum(-1)
    assert bool((amplitude[valid] > 0.1).all())
    assert bool(torch.allclose(torch.linalg.vector_norm(directions, dim=-1),
                               torch.ones(len(directions), device=device), atol=1e-6))
    assert bool(((attempts >= 1) & (attempts <= 1000)).all())
    assert bool((attempts[~valid] == 1000).all())


def test_initial_directions_report_exhausted_real_fod_when_cutoff_is_unreachable():
    path = Path(__file__).parent / "data/real_ifod2_arc_ds004666.npz"
    with np.load(path, allow_pickle=False) as real:
        coefficients = torch.as_tensor(real["fod_coefficients"][:2, 0, :].copy())
    directions, valid, attempts = _initial_directions(
        coefficients, torch.Generator().manual_seed(0), lmax=8, cutoff=10.0,
    )
    assert not bool(valid.any())
    assert torch.equal(attempts, torch.full((2,), 1000, dtype=torch.int32))
    torch.testing.assert_close(directions, torch.tensor([[1., 0., 0.], [1., 0., 0.]]))
