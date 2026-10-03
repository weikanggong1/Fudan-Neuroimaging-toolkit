"""Euler rotation accuracy is required independently of global TF32 mode."""
import pytest
import torch

from fnit.eddy.fsl2111_strict.geometry import fsl_rotation_matrix


def test_cpu_rotation_is_orthonormal_for_small_observed_motion():
    angles = torch.tensor([[.002, -.005, .004], [.010, .020, -.030]], dtype=torch.float64)
    rotation = fsl_rotation_matrix(angles)
    identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
    torch.testing.assert_close(rotation @ rotation.transpose(-1,-2), identity,
                               rtol=0, atol=3e-16)
    torch.testing.assert_close(torch.det(rotation), torch.ones(2,dtype=torch.float64),
                               rtol=0, atol=3e-16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("initial_tf32", [False, True])
def test_cuda_rotation_preserves_small_motion_and_restores_tf32(initial_tf32):
    angles = torch.tensor([[.002, -.005, .004], [.010, .020, -.030]], dtype=torch.float32)
    expected = fsl_rotation_matrix(angles)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = initial_tf32
        actual = fsl_rotation_matrix(angles.cuda()).cpu()
        assert torch.backends.cuda.matmul.allow_tf32 == initial_tf32
        torch.testing.assert_close(actual, expected, rtol=0, atol=1.2e-7)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
