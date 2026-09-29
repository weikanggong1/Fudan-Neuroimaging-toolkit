"""T1-specific FNIRT configuration and intensity-map regression checks."""

from types import SimpleNamespace
from dataclasses import replace

import nibabel as nib
import numpy as np
import torch

from fnit.fmri import normalization
from fnit.fnirt import T1FNIRTConfig
from fnit.fnirt.registration import _apply_t1_intensity, _t1_intensity_map


def test_t1_config_uses_six_level_t1_schedule():
    config = T1FNIRTConfig()
    assert config.subsampling == (4, 4, 2, 2, 1, 1)
    assert config.maximum_iterations == (5, 5, 5, 5, 5, 10)
    assert config.intensity_model == "global_non_linear_with_bias"
    assert config.intensity_order == 5
    assert config.bias_resolution_mm == (50, 50, 50)
    assert config.jacobian_range == (0.01, 100)


def test_t1_mapping_reduces_masked_intensity_error():
    axis = torch.linspace(0, 1, 12, dtype=torch.float32)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    reference = 30 + 100 * x + 20 * y
    warped = (12 + 0.85 * reference + 0.0008 * reference.square()) * (0.9 + 0.2 * z)
    mask = torch.ones_like(reference, dtype=torch.bool)
    polynomial, bias, report = _t1_intensity_map(
        reference, warped, mask, (2, 2, 2), T1FNIRTConfig()
    )
    mapped = _apply_t1_intensity(reference, polynomial, bias)
    assert report.converged
    assert torch.mean((mapped - warped).square()) < torch.mean((reference - warped).square())
    assert bool(torch.isfinite(mapped).all())


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
        def __init__(self, *, device, config):
            assert device == "cpu"
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
