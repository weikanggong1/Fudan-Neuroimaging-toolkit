"""纯 PyTorch mri_em_register 候选内核的确定性回归。"""

import torch

from fnit.recon_all.mri_em_register_torch import (
    TorchAffineScoreConfig,
    evaluate_affine_candidates,
    sample_candidates,
)


def test_identity_sampling_is_chunk_invariant():
    source = torch.arange(5 * 6 * 7, dtype=torch.float32).reshape(5, 6, 7)
    identity = torch.eye(4)
    matrices = torch.stack((identity, identity), dim=0)
    one = sample_candidates(source, matrices, config=TorchAffineScoreConfig(candidate_chunk=1))
    two = sample_candidates(source, matrices, config=TorchAffineScoreConfig(candidate_chunk=2))
    assert one.shape == (2, 5, 6, 7)
    # ``grid_sample`` evaluates the identity grid through floating-point
    # interpolation; chunking must be bitwise stable, while the identity
    # sample itself is checked with the documented float32 tolerance.
    assert torch.equal(one, two)
    assert torch.allclose(one[0], source, atol=2e-5, rtol=0.0)


def test_candidate_selection_stays_on_torch_until_final_index():
    torch.manual_seed(11)
    source = torch.rand(6, 7, 8)
    target = source.clone()
    identity = torch.eye(4)
    shifted = identity.clone()
    shifted[0, 3] = 1.0
    matrices = torch.stack((shifted, identity), dim=0)
    _, scores, best = evaluate_affine_candidates(
        source,
        target,
        matrices,
        metric="ncc",
        config=TorchAffineScoreConfig(candidate_chunk=1),
    )
    assert scores.shape == (2,)
    assert best == 1
    assert scores[1] > scores[0]
