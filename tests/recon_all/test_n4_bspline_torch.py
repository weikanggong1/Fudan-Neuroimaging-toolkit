"""有限小网格仅验证算子约定；真实 N4 回归另存于 validation。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.n4_bspline_torch import N4DenseBSplineFit, _cubic_kernel


def scalar_kernel(u):
    a = np.float32(abs(u))
    square = np.float32(a * a)
    if a < 1:
        return np.float32(np.float32(np.float32(4) - np.float32(6) * square +
                                        np.float32(3) * square * a) / np.float32(6))
    if a < 2:
        return np.float32(np.float32(np.float32(8) - np.float32(12) * a +
                                        np.float32(6) * square - square * a) / np.float32(6))
    return np.float32(0)


def test_cardinal_kernel_expression_and_support():
    u = np.array([-2, -1.999, -1, -.3, 0, .3, 1, 1.999, 2], np.float32)
    result = _cubic_kernel(torch.from_numpy(u)).numpy()
    reference = np.array([scalar_kernel(v) for v in u])
    assert np.array_equal(result.view(np.uint32), reference.view(np.uint32))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_single_level_linear_sign_and_repeatability(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available; real GPU gate runs on gpucw1")
    fit = N4DenseBSplineFit(field_shape=(6, 7, 8), control_shape=(4, 5, 6),
                          spacing=(2, 3, 4), origin=(1.5, -2, 7), device=device)
    values = torch.arange(6 * 7 * 8, dtype=torch.float32, device=device).reshape(6, 7, 8) / 1000
    a = fit.fit(values)
    b = fit.fit(-values)
    repeat = fit.fit(values)
    assert a.shape == (4, 5, 6) and a.dtype == torch.float32
    assert torch.equal(a, repeat)
    assert torch.equal(a, -b)
    assert torch.isfinite(a).all()
    assert fit.cache_bytes > 0


def test_contract_failures():
    with pytest.raises(ValueError, match="positive"):
        N4DenseBSplineFit(field_shape=(4, 4, 4), control_shape=(4, 4, 4), spacing=(1, 0, 1), device="cpu")
    with pytest.raises(ValueError, match="control_shape"):
        N4DenseBSplineFit(field_shape=(4, 4, 4), control_shape=(3, 4, 4), device="cpu")
    fit = N4DenseBSplineFit(field_shape=(4, 4, 4), control_shape=(4, 4, 4), device="cpu")
    with pytest.raises(TypeError, match="float32"):
        fit.fit(torch.zeros((4, 4, 4), dtype=torch.float64))
    with pytest.raises(ValueError, match="shape/device"):
        fit.fit(torch.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="finite"):
        fit.fit(torch.full((4, 4, 4), float("nan")))
