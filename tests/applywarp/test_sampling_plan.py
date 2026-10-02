"""Prepared geometry must preserve sampling and reject stale NIfTI grids."""
from dataclasses import FrozenInstanceError
import nibabel as nib
import numpy as np
import pytest
import torch

import fnit.applywarp.core as core
from fnit.applywarp import ApplyWarpPlan, TorchApplyWarp
from .test_applywarp import _cubic_coefficients


def _image(data, affine):
    image = nib.Nifti1Image(np.asarray(data), affine)
    image.set_qform(affine, 2)
    image.set_sform(affine, 4)
    return image


def _equal_output(left, right):
    np.testing.assert_array_equal(np.asarray(left.dataobj), np.asarray(right.dataobj))
    assert left.header.binaryblock == right.header.binaryblock
    np.testing.assert_array_equal(left.affine, right.affine)


@pytest.mark.parametrize("interpolation", ["trilinear", "nearest"])
@pytest.mark.parametrize("handedness", [-1, 1])
@pytest.mark.parametrize("kind", ["relative", "absolute", "coefficient"])
def test_prepared_warp_reuses_transform_but_preserves_per_image_contract(
    monkeypatch, interpolation, handedness, kind
):
    shape = (7, 6, 5)
    affine = np.diag([handedness * 1.5, 2.0, 2.5, 1.0])
    image = _image(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), affine)
    reference = _image(np.zeros(shape, dtype=np.float32), affine)
    matrix = np.eye(4)
    matrix[:3, 3] = (0.3, -0.2, 0.1)
    if kind == "coefficient":
        warp = _cubic_coefficients(shape, (2, 2, 2), (0.2, -0.1, 0.1), matrix)
        options = dict(warp=warp, interpolation=interpolation)
    else:
        field = np.full((*shape, 3), (0.2, -0.1, 0.1), dtype=np.float32)
        if kind == "absolute":
            fsl = core._fsl_voxel_matrix(reference)
            positions = np.indices(shape, dtype=np.float64).reshape(3, -1)
            field += (fsl[:3, :3] @ positions + fsl[:3, 3:4]).T.reshape(*shape, 3)
        warp = _image(field, affine)
        options = dict(warp=warp, interpolation=interpolation,
                       warp_convention=kind, premat=matrix, postmat=matrix)
    warper = TorchApplyWarp("cpu")
    other = _image((np.asarray(image.dataobj) * 2.0)[..., None].astype(np.float64), affine)
    other.header.set_zooms((*other.header.get_zooms()[:3], 2.7))
    expected = warper(other, reference, **options)
    plan = warper.prepare(image, reference, **options)
    assert isinstance(plan, ApplyWarpPlan)
    assert plan._coordinates.dtype == torch.float64
    if plan._grid is not None:
        assert plan._grid.dtype == torch.float32
    # A prepared apply must not decode the same warp or reconstruct its grid.
    monkeypatch.setattr(core, "_spatial_grid", lambda *a, **k: pytest.fail("grid rebuilt"))
    monkeypatch.setattr(core, "_expand_cubic_coefficients", lambda *a, **k: pytest.fail("coefficients expanded"))
    actual = plan.apply(other, reference=reference)
    _equal_output(actual.image, expected.image)
    np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)
    assert actual.qc == expected.qc
    assert actual.image.get_data_dtype() == np.dtype("float64")
    assert actual.image.header.get_zooms()[3] == other.header.get_zooms()[3]
    actual.valid_mask[:] = False
    np.testing.assert_array_equal(plan.apply(other).valid_mask, expected.valid_mask)


@pytest.mark.parametrize("change", ["shape", "affine", "pixdim", "handedness"])
def test_plan_rejects_different_native_coordinates(change):
    shape = (5, 6, 7)
    affine = np.diag([-1.0, 2.0, 3.0, 1.0])
    image = _image(np.ones(shape, dtype=np.float32), affine)
    plan = TorchApplyWarp("cpu").prepare(image, image)
    changed = _image(np.ones(shape if change != "shape" else (6, 6, 7), dtype=np.float32), affine.copy())
    if change == "affine":
        changed.affine[0, 3] += 1e-8
    elif change == "pixdim":
        changed.header.set_zooms((1.25, 2.0, 3.0))
    elif change == "handedness":
        changed.affine[0, 0] *= -1
    with pytest.raises(ValueError, match="prepared sampling plan"):
        plan.apply(changed)


@pytest.mark.parametrize("change", ["pixdim", "qform", "header", "extensions"])
def test_plan_rejects_changed_reference_header_even_with_same_affine(change):
    image = _image(np.ones((4, 5, 6), dtype=np.float32), np.eye(4))
    reference = _image(np.zeros(image.shape, dtype=np.float32), image.affine)
    plan = TorchApplyWarp("cpu").prepare(image, reference)
    if change == "pixdim":
        reference.header.set_zooms((1.2, 1.0, 1.0))
    elif change == "qform":
        reference.header["qform_code"] = 3
    elif change == "header":
        reference.header["descrip"] = b"different reference"
    else:
        reference.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"new metadata"))
    with pytest.raises(ValueError, match="reference"):
        plan.apply(image)


def test_interpolation_and_transform_are_fixed_when_plan_is_prepared():
    image = _image(np.arange(60, dtype=np.float32).reshape(4, 5, 3), np.eye(4))
    field = _image(np.zeros((*image.shape, 3), dtype=np.float32), image.affine)
    plan = TorchApplyWarp("cpu").prepare(image, image, warp=field, warp_convention="relative")
    expected = plan.apply(image)
    field.dataobj[...] = 100
    _equal_output(plan.apply(image).image, expected.image)
    with pytest.raises(FrozenInstanceError):
        plan.interpolation = "nearest"
    with pytest.raises(TypeError):
        plan.apply(image, interpolation="nearest")
