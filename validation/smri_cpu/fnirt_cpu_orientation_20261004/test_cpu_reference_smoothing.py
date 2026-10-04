"""NEWIMAGE internal orientation must not alter the public input grid."""
import numpy as np
import pytest
import torch

from fnit._nib import FNITNifti1Image
from fnit.fnirt import GMFNIRTConfig, TorchFNIRT
from fnit.fnirt import registration


def config():
    return GMFNIRTConfig(subsampling=(1,), maximum_iterations=(0,),
                         input_fwhm_mm=(2.,), reference_fwhm_mm=(2.,),
                         regularization=(0.,), estimate_intensity=(False,),
                         apply_reference_mask=(False,), implicit_input_mask=False,
                         implicit_reference_mask=False)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("execution", ["reference", "optimized"])
def test_world_equivalent_storage_directions_keep_identical_smoothing(masked, execution):
    generator = torch.Generator().manual_seed(73)
    storage = torch.rand((1, 1, 26, 18, 22), generator=generator) * 137.3
    volume = storage[:, :, ::2, ::2, ::2]  # Noncontiguous input with real strides.
    mask = torch.rand(volume.shape[2:], generator=generator) > .3 if masked else None
    original, original_mask = volume.clone(), None if mask is None else mask.clone()
    positive = registration._fsl_cpu_reference_smoothing(
        volume, 6., (.8, .7777778, .9), mask, flip_x=True, execution=execution)
    negative = registration._fsl_cpu_reference_smoothing(
        volume.flip(2), 6., (.8, .7777778, .9),
        None if mask is None else mask.flip(0), flip_x=False, execution=execution)
    assert torch.equal(positive.flip(2), negative)
    assert positive.shape == volume.shape and positive.dtype == volume.dtype
    assert torch.equal(volume, original)
    if mask is not None:
        assert torch.equal(mask, original_mask)
        assert positive[0, 0][~mask].count_nonzero() == 0
    else:
        # This fixture contains the FP32 order difference the fix addresses.
        old = registration._fsl_gaussian_blur_reference(volume, 6., (.8, .7777778, .9))
        assert not torch.equal(old, positive)


def test_zero_smoothing_preserves_storage_and_values():
    volume = torch.arange(240, dtype=torch.float32).reshape(1,1,10,6,4)[:, :, ::2]
    result = registration._fsl_cpu_reference_smoothing(
        volume, 0., (1.,1.,1.), None, flip_x=True)
    assert result is volume


@pytest.mark.parametrize("moving_positive,fixed_positive", [(True,False), (False,True), (True,True), (False,False)])
def test_cpu_uses_actual_moving_and_template_affine_signs(monkeypatch, moving_positive, fixed_positive):
    data = np.arange(512, dtype=np.float32).reshape(8,8,8) + 1
    positive = np.eye(4)
    negative = np.diag([-1.,1.,1.,1.]); negative[0,3] = 7
    moving = FNITNifti1Image(data if moving_positive else data[::-1], positive if moving_positive else negative)
    fixed = FNITNifti1Image(data if fixed_positive else data[::-1], positive if fixed_positive else negative)
    calls = []
    original = registration._fsl_cpu_reference_smoothing

    def record(*args, **kwargs):
        calls.append(kwargs["flip_x"])
        return original(*args, **kwargs)

    monkeypatch.setattr(registration, "_fsl_cpu_reference_smoothing", record)
    fit = TorchFNIRT(device="cpu", config=config())(moving, fixed, np.eye(4))
    assert calls == [moving_positive, fixed_positive]
    assert fit.moved.shape == fixed.shape
    np.testing.assert_array_equal(fit.moved.affine, fixed.affine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_does_not_enter_cpu_orientation_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA must keep the existing smoothing path")

    monkeypatch.setattr(registration, "_fsl_cpu_reference_smoothing", forbidden)
    data = np.arange(512, dtype=np.float32).reshape(8,8,8) + 1
    image = FNITNifti1Image(data, np.eye(4))
    fit = TorchFNIRT(device="cuda", config=config())(image, image, np.eye(4))
    assert fit.moved.shape == image.shape
