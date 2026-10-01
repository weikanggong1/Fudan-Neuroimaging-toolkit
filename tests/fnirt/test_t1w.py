"""T1-specific FNIRT configuration and intensity-map regression checks."""

from types import SimpleNamespace
from dataclasses import replace

import nibabel as nib
import numpy as np
import torch

from fnit.fmri import normalization
from fnit.fnirt import T1FNIRTConfig
from fnit.fnirt.registration import _JointT1System, _LevelSystem, _pack_t1, _unpack_t1
from fnit.fnirt.spline import BendingOperator, fsl_control_shape, spline_bases


def test_t1_config_uses_six_level_t1_schedule():
    config = T1FNIRTConfig()
    assert config.subsampling == (4, 4, 2, 2, 1, 1)
    assert config.maximum_iterations == (5, 5, 5, 5, 5, 10)
    assert config.intensity_model == "global_non_linear_with_bias"
    assert config.intensity_order == 5
    assert config.bias_resolution_mm == (50, 50, 50)
    assert config.jacobian_range == (0.01, 100)


def test_t1_joint_gradient_includes_deformation_polynomial_and_bias():
    shape = (6, 6, 6)
    x, y, z = torch.meshgrid(*[torch.arange(size) for size in shape], indexing="ij")
    reference = (30 + 6 * x + 4 * y + 2 * z).to(torch.float32)
    moving = (8 + 0.9 * reference + 0.001 * reference.square()).to(torch.float32)
    deformation_spacing = (3, 3, 3)
    bias_spacing = (4, 4, 4)
    dtype = torch.float64
    coordinates = torch.stack((x, y, z)).to(torch.float32)
    interior_mask = torch.zeros(shape, dtype=torch.bool)
    interior_mask[1:-1, 1:-1, 1:-1] = True
    deformation = _LevelSystem(
        moving, reference, interior_mask, None, torch.eye(4), coordinates,
        torch.eye(4),
        spline_bases(shape, deformation_spacing, (1, 1, 1), device="cpu", dtype=dtype),
        BendingOperator(shape, deformation_spacing, (1, 1, 1), device="cpu", dtype=dtype),
        0.0, False, False,
    )
    system = _JointT1System(
        deformation,
        spline_bases(shape, bias_spacing, (1, 1, 1), device="cpu", dtype=dtype),
        BendingOperator(shape, bias_spacing, (1, 1, 1), device="cpu", dtype=dtype),
        10.0, 5,
    )
    coefficients = torch.zeros((3, *fsl_control_shape(shape, deformation_spacing)), dtype=dtype)
    bias = torch.ones((1, *fsl_control_shape(shape, bias_spacing)), dtype=dtype)
    polynomial = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
    _, gradient, _, _ = system.linearize(coefficients, polynomial, bias, fit_intensity=True)
    initial = _pack_t1(coefficients, bias, polynomial)
    for index in (0, coefficients.numel(), initial.numel() - 2):
        perturbation = torch.zeros_like(initial)
        perturbation[index] = 1.0
        epsilon = 1e-3
        positive = _unpack_t1(initial + epsilon * perturbation, coefficients.shape[1:], bias.shape[1:])
        negative = _unpack_t1(initial - epsilon * perturbation, coefficients.shape[1:], bias.shape[1:])
        numerical = (
            system.evaluate(positive[0], positive[2], positive[1])["cost"]
            - system.evaluate(negative[0], negative[2], negative[1])["cost"]
        ) / (2 * epsilon)
        torch.testing.assert_close(2 * gradient[index], numerical, atol=0.15, rtol=0.06)


def test_fmri_fnirt_uses_t1_config_and_writes_pull(tmp_path, monkeypatch):
    moving = tmp_path / "t1.nii.gz"
    fixed = tmp_path / "mni.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    for path in (moving, fixed, mask):
        nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4)), path)

    class FakeFLIRT:
        def __init__(self, *, device):
            assert device == "cpu"

        def __call__(self, source, target):
            return SimpleNamespace(matrix=np.eye(4), moving_to_fixed_world=np.eye(4))

    seen_configs = []

    class FakeFNIRT:
        def __init__(self, *, device, config, execution="optimized"):
            assert device == "cpu"
            assert execution == "optimized"
            seen_configs.append(config)

        def __call__(self, source, target, initial, *, reference_mask):
            assert reference_mask == mask
            np.testing.assert_array_equal(initial, np.eye(4))
            return SimpleNamespace(pull_transform=nib.Nifti1Image(
                np.zeros((*target.shape, 3), dtype=np.float32), target.affine
            ))

    monkeypatch.setattr(normalization, "TorchFLIRT", FakeFLIRT)
    monkeypatch.setattr("fnit.fnirt.TorchFNIRT", FakeFNIRT)
    result = normalization.register_t1_to_mni(
        t1_brain=moving,
        mni_brain=fixed,
        output_dir=tmp_path / "output",
        backend="fnirt",
        reference_mask=mask,
        device="cpu",
    )
    assert result.backend == "fnirt"
    assert nib.load(str(result.pull_ras)).shape == (4, 5, 6, 3)
    assert seen_configs == [T1FNIRTConfig()]
    custom = replace(T1FNIRTConfig(), regularization=(250, 125, 90, 45, 35, 25))
    normalization.register_t1_to_mni(
        t1_brain=moving,
        mni_brain=fixed,
        output_dir=tmp_path / "custom",
        backend="fnirt",
        reference_mask=mask,
        fnirt_config=custom,
        device="cpu",
    )
    assert seen_configs[-1] is custom
