"""完整 N4 的子段几何/失败合同；真实200次反馈另由 benchmark_full核验。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.n4_itk_torch_experimental import (
    N4CubicReconstruction, N4HistogramSharpening, correct_tensor, refine_lattice,
)


def test_reconstruction_order_and_constant_support():
    lattice = torch.ones((4, 4, 4), dtype=torch.float32)
    field = N4CubicReconstruction(control_shape=(4, 4, 4), output_shape=(5, 6, 7),
                                  spacing=(2, 3, 4), device="cpu")(lattice)
    assert field.shape == (5, 6, 7) and field.dtype == torch.float32
    assert float((field-1).abs().max()) < 1e-6
    # 开放边界不是环绕：仅末控制点的冲击在末端更大。
    impulse = torch.zeros_like(lattice); impulse[-1, -1, -1] = 1
    result = N4CubicReconstruction(control_shape=(4, 4, 4), output_shape=(5, 6, 7),
                                   spacing=(2, 3, 4), device="cpu")(impulse)
    assert result[0, 0, 0] == 0 and result[-1, -1, -1] > 0


def test_refinement_preserves_tiny_nonzero_source_coefficients():
    impulse = torch.zeros((4, 4, 4)); impulse[0, 0, 0] = 1
    result = refine_lattice(impulse)
    assert result.shape == (5, 5, 5)
    assert result[0, 0, 0] == .125
    assert result[1, 0, 0] > 1/32
    assert torch.isfinite(result).all()


def test_full_tensor_fails_without_fallback():
    with pytest.raises(ValueError, match="FP32"):
        correct_tensor(image=torch.ones((8, 8, 8), dtype=torch.float64))
    with pytest.raises(ValueError, match="nonnegative"):
        correct_tensor(image=-torch.ones((8, 8, 8)))
    with pytest.raises(ValueError, match="positive"):
        correct_tensor(image=torch.ones((8, 8, 8)), spacing=(1, 0, 1))
    with pytest.raises(ValueError, match="histogram range"):
        N4HistogramSharpening(device="cpu")(torch.ones((4, 4, 4)))
