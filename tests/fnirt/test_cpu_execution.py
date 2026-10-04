"""CPU allocation reuse must preserve FSL's per-offset image roundings."""

import pytest
import torch

from fnit.fnirt.registration import (
    _fsl_gaussian_blur,
    _fsl_gaussian_blur_cpu,
    _fsl_gaussian_blur_reference,
    _fsl_masked_gaussian_blur,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1, 3, 4, 5), (2, 3, 17, 19, 23)])
@pytest.mark.parametrize("fwhm,zooms", [(8.0, (1, 2, 3)), (2.0, (2, 2, 2)), (1e-7, (1, 1, 1))])
def test_cpu_smoothing_keeps_each_assignment_bits(dtype, shape, fwhm, zooms):
    volume = torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(43))
    volume.flatten()[:6] = torch.tensor(
        [0.0, -0.0, 1.4e-45, -1.4e-45, 1e-35, -1e-35], dtype=dtype,
    )
    integer_dtype = torch.int32 if dtype == torch.float32 else torch.int64
    expected = _fsl_gaussian_blur_reference(volume, fwhm, zooms)
    actual = _fsl_gaussian_blur(volume, fwhm, zooms)
    assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))


def test_cpu_masked_smoothing_keeps_border_normalisation():
    volume = torch.randn((1, 1, 13, 15, 11), generator=torch.Generator().manual_seed(9))
    mask = torch.zeros(volume.shape[2:], dtype=torch.bool)
    mask[1:11, 2:12, 3:10] = True
    expected = _fsl_masked_gaussian_blur(volume, 4.5, (1, 2, 1.5), mask, execution="reference")
    actual = _fsl_masked_gaussian_blur(volume, 4.5, (1, 2, 1.5), mask, execution="optimized")
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_cpu_smoothing_preserves_zero_width_alias_and_thread_budget():
    volume = torch.arange(60.0).reshape(1, 1, 3, 4, 5)
    original_threads = torch.get_num_threads()
    assert _fsl_gaussian_blur_cpu(volume, 0, (1, 1, 1)) is volume
    noncontiguous = volume.transpose(2, 4)
    expected = _fsl_gaussian_blur_reference(noncontiguous, 3, (1, 1, 1))
    actual = _fsl_gaussian_blur_cpu(noncontiguous, 3, (1, 1, 1))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.get_num_threads() == original_threads


def test_cpu_smoothing_differentiable_call_retains_reference_graph():
    volume = torch.arange(60.0).reshape(1, 1, 3, 4, 5).requires_grad_()
    result = _fsl_gaussian_blur(volume, 3, (1, 1, 1))
    result.sum().backward()
    assert volume.grad is not None
    assert torch.isfinite(volume.grad).all()


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fastest", [0, 1, 2])
def test_cpu_fused_smoothing_keeps_strided_layout_and_thread_masks(threads, dtype, fastest, monkeypatch):
    import numba
    import fnit.fnirt._smoothing_cpu as smoothing_module
    prior_torch, prior_numba = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)
        monkeypatch.setattr(smoothing_module.config, "NUMBA_NUM_THREADS", threads)
        volume = torch.randn((2, 3, 5, 7, 9), dtype=dtype, generator=torch.Generator().manual_seed(311))
        if fastest == 0:
            volume = volume.permute(0, 1, 4, 3, 2).contiguous().permute(0, 1, 4, 3, 2)
        elif fastest == 1:
            volume = volume.permute(0, 1, 4, 2, 3).contiguous().permute(0, 1, 3, 4, 2)
        before = volume.clone()
        expected = _fsl_gaussian_blur_reference(volume, 4.5, (1.0, 1.5, 2.0))
        actual = _fsl_gaussian_blur_cpu(volume, 4.5, (1.0, 1.5, 2.0))
        integer_dtype = torch.int32 if dtype == torch.float32 else torch.int64
        assert torch.equal(actual.view(integer_dtype), expected.view(integer_dtype))
        assert torch.equal(volume, before)
        assert torch.get_num_threads() == threads
        assert numba.get_num_threads() == threads
    finally:
        torch.set_num_threads(prior_torch)
        numba.set_num_threads(prior_numba)


def test_cpu_smoothing_does_not_initialize_oversized_numba_pool(monkeypatch):
    import fnit.fnirt._smoothing_cpu as smoothing_module
    monkeypatch.setattr(smoothing_module.config, "NUMBA_NUM_THREADS", torch.get_num_threads() + 1)
    def unexpected_pool_initialization():
        raise AssertionError("Oversized Numba pool should not be initialized")
    monkeypatch.setattr(smoothing_module, "get_num_threads", unexpected_pool_initialization)
    volume = torch.ones((1, 1, 3, 4, 5), dtype=torch.float32)
    expected = _fsl_gaussian_blur_reference(volume, 3.0, (1.0, 1.0, 1.0))
    actual = _fsl_gaussian_blur_cpu(volume, 3.0, (1.0, 1.0, 1.0))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
