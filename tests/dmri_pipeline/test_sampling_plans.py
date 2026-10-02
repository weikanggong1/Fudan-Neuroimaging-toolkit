"""TBSS groups exact native geometries without changing FA/skeleton semantics."""
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import pytest

from fnit.dmri_pipeline import tbss as tbss_module
from fnit.dmri_pipeline.pipeline import STANDARD_MAP_NAMES
from fnit.applywarp import TorchApplyWarp


@pytest.mark.parametrize("different_grid", [False, True])
def test_tbss_prepares_once_per_grid_and_keeps_map_masks(monkeypatch, tmp_path, different_grid):
    shape = (8, 9, 10)
    affine = np.diag([-1.0, 1.5, 2.0, 1.0])
    data = np.zeros(shape, dtype=np.float32)
    data[1:-1, 1:-1, 1:-1] = 0.7
    fa = nib.Nifti1Image(data, affine)
    maps = {name: nib.Nifti1Image(data * (1 + index * 0.1), affine)
            for index, name in enumerate(STANDARD_MAP_NAMES)}
    maps["FA"] = fa
    maps["ICVF"] = nib.Nifti1Image(data[..., None], affine)
    if different_grid:
        shifted_affine = affine.copy()
        shifted_affine[1, 3] += 0.3
        maps["OD"] = nib.Nifti1Image(data * 0.4, shifted_affine)
    reference = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    skeleton_data = np.zeros(shape, dtype=np.float32)
    skeleton_data[::2] = 3000
    skeleton = nib.Nifti1Image(skeleton_data, affine)
    reference_path = tmp_path / "reference.nii.gz"
    skeleton_path = tmp_path / "skeleton.nii.gz"
    nib.save(reference, reference_path)
    nib.save(skeleton, skeleton_path)
    warp = nib.Nifti1Image(np.full((*shape, 3), (0.2, -0.1, 0.3), dtype=np.float32), affine)
    warp.header["intent_code"] = 2006
    preparations = []

    class FakeFLIRT:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(moving_to_fixed_world=np.eye(4), matrix=np.eye(4), qc={})

    class FakeFNIRT:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(coefficient_image=warp, nonlinear_jacobian=reference, qc={})

    class CountingApplyWarp(TorchApplyWarp):
        def prepare(self, source, target, **kwargs):
            preparations.append(source.shape)
            return super().prepare(source, target, **kwargs)

    monkeypatch.setattr(tbss_module, "TorchFLIRT", FakeFLIRT)
    monkeypatch.setattr(tbss_module, "TorchFNIRT", FakeFNIRT)
    monkeypatch.setattr(tbss_module, "TorchApplyWarp", CountingApplyWarp)
    actual = tbss_module.TorchTBSS(device="cpu")(maps, reference_path, skeleton_path)
    assert len(preparations) == (2 if different_grid else 1)
    preprocessed, _ = tbss_module.preprocess_fa(fa)
    expected = {}
    for name, source in maps.items():
        source = preprocessed if name == "FA" else source
        warped = TorchApplyWarp("cpu")(
            source, reference, warp=warp, interpolation="trilinear", warp_convention="relative"
        ).image
        values = np.asarray(warped.dataobj, dtype=np.float32)
        expected[name] = values[..., 0] if values.ndim == 4 else values
    valid = (expected["FA"] != 0) & (np.asarray(reference.dataobj) != 0)
    expected["FA"] *= valid
    skeleton_mask = (skeleton_data >= 2000) & valid
    for name in STANDARD_MAP_NAMES:
        np.testing.assert_array_equal(np.asarray(actual.standard_maps[name].dataobj), expected[name])
        np.testing.assert_array_equal(np.asarray(actual.skeleton_maps[name].dataobj), expected[name] * skeleton_mask)
