"""MMORF plans capture geometry while retaining image-specific interpolation."""
from dataclasses import FrozenInstanceError
import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.mmorf.core as core
from fnit.mmorf import MMORFWarpPlan, apply_mmorf_warp, prepare_mmorf_warp


def _image(data, affine):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.set_qform(affine, 2)
    image.set_sform(affine, 4)
    return image


@pytest.mark.parametrize("interpolation", ["linear", "nearest", "cubic"])
@pytest.mark.parametrize("handedness", [-1, 1])
def test_prepared_mmorf_retains_scalar_and_channel_sampling(monkeypatch, interpolation, handedness):
    shape = (6, 7, 8)
    affine = np.array([[handedness * 1.2, 0.2, 0, 3], [0, 1.5, 0.1, -2], [0, 0, 2, 4], [0, 0, 0, 1]])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image = _image(data, affine)
    reference = _image(np.zeros(shape, dtype=np.float32), affine)
    field = _image(np.full((*shape, 3), (0.13, -0.27, 0.32), dtype=np.float32), affine)
    matrix = np.eye(4)
    matrix[:3, 3] = (0.2, 0.1, -0.3)
    other = _image(np.stack((data * 0.5, data * 1.2), axis=-1), affine)
    options = dict(affine=matrix, device="cpu", interpolation=interpolation)
    expected = apply_mmorf_warp(other, reference, field, **options)
    plan = prepare_mmorf_warp(image, reference, field, **options)
    assert isinstance(plan, MMORFWarpPlan)
    assert plan._coordinates.dtype == torch.float32
    monkeypatch.setattr(core, "_affine_coordinates", lambda *a, **k: pytest.fail("affine grid rebuilt"))
    monkeypatch.setattr(core, "_mm_to_voxel_axes_rotation", lambda *a, **k: pytest.fail("rotation recomputed"))
    actual = plan.apply(other, reference=reference)
    np.testing.assert_array_equal(np.asarray(actual.dataobj), np.asarray(expected.dataobj))
    assert actual.header.binaryblock == expected.header.binaryblock
    np.testing.assert_array_equal(actual.affine, expected.affine)


@pytest.mark.parametrize("change", ["shape", "affine", "pixdim", "handedness", "reference_header"])
def test_prepared_mmorf_rejects_geometry_drift(change):
    shape = (5, 6, 7)
    affine = np.diag([-1.0, 2.0, 3.0, 1.0])
    image = _image(np.ones(shape, dtype=np.float32), affine)
    reference = _image(np.zeros(shape, dtype=np.float32), affine)
    field = _image(np.zeros((*shape, 3), dtype=np.float32), affine)
    plan = prepare_mmorf_warp(image, reference, field, device="cpu")
    changed = _image(np.ones(shape if change != "shape" else (6, 6, 7)), affine.copy())
    if change == "affine":
        changed.affine[0, 3] += 1e-8
    elif change == "pixdim":
        changed.header.set_zooms((1.25, 2.0, 3.0))
    elif change == "handedness":
        changed.affine[0, 0] *= -1
    elif change == "reference_header":
        reference.header["sform_code"] = 2
    with pytest.raises(ValueError, match="prepared sampling plan"):
        plan.apply(changed, reference=reference)


def test_prepared_mmorf_is_transform_snapshot_and_cannot_change_mode():
    image = _image(np.arange(60).reshape(4, 5, 3), np.eye(4))
    field = _image(np.zeros((*image.shape, 3), dtype=np.float32), image.affine)
    plan = prepare_mmorf_warp(image, image, field, device="cpu")
    expected = np.asarray(plan.apply(image).dataobj).copy()
    field.dataobj[...] = 100
    np.testing.assert_array_equal(np.asarray(plan.apply(image).dataobj), expected)
    with pytest.raises(FrozenInstanceError):
        plan.interpolation = "nearest"
