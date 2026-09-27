"""Numerical tie handling for the shared PyTorch SynthSeg implementation."""

import torch

from fnit.synthseg_parc.synthseg import (
    SYNTHSEG_TIE_EPSILON, _synthseg_index_with_numerical_ties,
)


def test_near_ties_prefer_first_channel_without_changing_clear_winners():
    posterior = torch.tensor([
        [0.5000000, 0.4999980, 0.3],
        [0.5000004, 0.5000000, 0.4],
        [0.5000008, 0.4,       0.5],
    ], dtype=torch.float32)
    assert SYNTHSEG_TIE_EPSILON == 2 ** -20
    assert posterior.argmax(0).tolist() == [2, 1, 2]
    assert _synthseg_index_with_numerical_ties(posterior).tolist() == [0, 1, 2]
