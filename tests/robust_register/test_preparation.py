"""Geometry, discrete sampling and boundary contracts; not an MRI benchmark."""

import sys

import nibabel as nib
from nibabel.freesurfer.mghformat import MGHHeader, MGHImage
import numpy as np
import pytest
from scipy import ndimage
import torch

from fnit.robust_register import prepare_subregion_alignment_target, reflect_atlas_header
from fnit.robust_register.preparation import _geometry, _nearest_mask


def mgh(data, sizes=(1, 1, 1), center=(12, -8, 5), rotation=None):
    header = MGHHeader()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    header["delta"] = sizes
    header["Pxyz_c"] = center
    if rotation is not None:
        header["Mdc"] = np.asarray(rotation).T
    return MGHImage(data, header.get_affine(), header)


def test_crop_keeps_ras_origin_and_mgh_storage(tmp_path):
    labels = np.zeros((13, 14, 15), np.int32)
    labels[4:7, 5:8, 6:9] = 53
    source = mgh(labels)
    source.header["tr"] = 2100
    prepared = prepare_subregion_alignment_target(source, (53, 54), bbox_margin_voxels=2, smoothing=None)
    assert prepared.image.shape == (7, 7, 7)
    assert prepared.report["bbox_lower"] == [2, 3, 4]
    assert prepared.report["bbox_upper_exclusive"] == [9, 10, 11]
    np.testing.assert_array_equal(prepared.image.dataobj, (labels[2:9, 3:10, 4:11] > 0).astype(np.float32) * 255)
    expected = source.affine.copy()
    expected[:3, 3] = (source.affine @ [2, 3, 4, 1])[:3]
    np.testing.assert_array_equal(prepared.image.affine, expected)
    assert prepared.image.header["tr"] == 2100
    assert prepared.image.header["fov"] == 7
    assert prepared.image.header["dof"] == 1
    path = tmp_path / "target.mgz"
    nib.save(prepared.image, path)
    loaded = nib.load(path)
    for key in ("delta", "Mdc", "Pxyz_c", "dims", "type", "dof", "tr", "fov"):
        np.testing.assert_array_equal(loaded.header[key], prepared.image.header[key])
    np.testing.assert_array_equal(loaded.dataobj, prepared.image.dataobj)


@pytest.mark.parametrize("mode", [None, "forward", "backward"])
@pytest.mark.parametrize("kind", ["interior", "boundary", "empty"])
def test_radius_one_morphology_and_border_one(mode, kind):
    labels = np.zeros((9, 8, 7), np.int32)
    if kind == "interior":
        labels[2:7, 2:6, 2:5] = 53
        labels[4, 3, 3] = 0
        labels[1, 3, 3] = 54
    elif kind == "boundary":
        labels[:4, :4, :4] = 53
        labels[1, 1, 1] = 0
    expected = labels > 0
    sphere = ndimage.generate_binary_structure(3, 1)
    if mode == "forward":
        expected = ndimage.binary_erosion(ndimage.binary_dilation(expected, sphere), sphere, border_value=1)
    elif mode == "backward":
        expected = ndimage.binary_dilation(ndimage.binary_erosion(expected, sphere, border_value=1), sphere)
    result = prepare_subregion_alignment_target(mgh(labels), (53, 54), bbox_margin_voxels=20, smoothing=mode)
    np.testing.assert_array_equal(np.asarray(result.image.dataobj) > 0, expected)
    assert result.image.dataobj.dtype == np.float32
    assert not torch.cuda.is_initialized()
    assert "surfa" not in sys.modules


@pytest.mark.parametrize("coordinate,expected", [
    (-.01, False), (0, False), (.49999997, False), (.5, True),
    (1.5, False), (2.99, False), (3.0, False),
])
def test_nearest_domain_and_away_half_ties(coordinate, expected):
    mask = torch.tensor([False, True, False]).reshape(3, 1, 1)
    pull = np.zeros((4, 4), np.float32)
    pull[3, 3] = 1
    pull[0, 3] = coordinate
    assert bool(_nearest_mask(mask, (1, 1, 1), pull, 1).item()) is expected


def test_rounding_at_last_half_voxel_clamps_after_validity():
    mask = torch.tensor([False, False, True]).reshape(3, 1, 1)
    pull = np.eye(4)
    pull[0, 3] = 2.75
    assert bool(_nearest_mask(mask, (1, 1, 1), pull, 1).item())
    pull[0, 3] = 3.0
    assert not bool(_nearest_mask(mask, (1, 1, 1), pull, 1).item())


def test_resize_center_phase_ceil_and_chunk_invariance():
    labels = np.zeros((9, 10, 11), np.int32)
    labels[::2] = 53
    source = mgh(labels, (.8, .9, 1.0))
    a = prepare_subregion_alignment_target(source, (53,), bbox_margin_voxels=30, smoothing=None, spatial_chunk_size=7)
    b = prepare_subregion_alignment_target(source, (53,), bbox_margin_voxels=30, smoothing=None, spatial_chunk_size=8192)
    assert a.report["resize_applied"]
    assert a.image.shape == (8, 9, 11)
    np.testing.assert_array_equal(a.image.dataobj, b.image.dataobj)
    np.testing.assert_array_equal(a.image.affine, b.image.affine)
    np.testing.assert_array_equal(a.image.header["Pxyz_c"], source.header["Pxyz_c"])
    # Source center is shape/2, giving x=(u-4)/.8+4.5 = 1.25*u-.5.
    # Pull of target x=0 is -.5 (outside), x=1=.75 rounds to 1, x=2=2.
    assert not np.asarray(a.image.dataobj)[0].any()
    assert not np.asarray(a.image.dataobj)[1].any()
    assert np.asarray(a.image.dataobj)[2, 4, 5] == 255
    assert a.image.header["delta"].tolist() == [1, 1, 1]


@pytest.mark.parametrize("voxel_mm,resize", [(.98, True), (.99, False), (1., False)])
def test_resize_trigger_is_strict_mean_099(voxel_mm, resize):
    source = mgh(np.ones((6, 7, 8), np.int32) * 53, (voxel_mm,) * 3)
    result = prepare_subregion_alignment_target(source, (53,), smoothing=None)
    assert result.report["resize_triggered"] is resize


def test_resize_matching_requested_size_is_copy_not_ceil():
    source = mgh(np.ones((6, 7, 8), np.int32) * 53, (.98,) * 3)
    result = prepare_subregion_alignment_target(source, (53,), target_voxel_mm=.98, smoothing=None)
    assert result.report["resize_triggered"]
    assert not result.report["resize_applied"]
    assert result.image.shape == source.shape


def test_mgh_fields_are_promoted_before_product():
    source = mgh(np.zeros((5, 6, 7), np.int32), (.7, .8, .9))
    angle = np.float32(.37)
    source.header["Mdc"] = np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
    geometry = _geometry(source)
    expected_linear = source.header["Mdc"].T.astype(np.float64) @ np.diag(source.header["delta"].astype(np.float64))
    np.testing.assert_array_equal(geometry.affine[:3, :3], expected_linear)
    assert np.max(np.abs(expected_linear - source.header.get_affine()[:3, :3])) > 0


def test_reflection_changes_header_only_and_preserves_footer(tmp_path):
    source = mgh(np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7))
    source.header["flip_angle"] = 13
    reflected = reflect_atlas_header(source)
    np.testing.assert_array_equal(reflected.dataobj, source.dataobj)
    expected = _geometry(source).affine.copy()
    expected[0] *= -1
    np.testing.assert_array_equal(reflected.affine, expected)
    assert reflected.header["flip_angle"] == 13
    path = tmp_path / "flipped.mgz"
    nib.save(reflected, path)
    reloaded = nib.load(path)
    np.testing.assert_array_equal(reloaded.dataobj, source.dataobj)
    np.testing.assert_array_equal(reloaded.affine, expected)
    np.testing.assert_array_equal(reflect_atlas_header(reflected).dataobj, source.dataobj)


def test_large_nonmatching_label_does_not_wrap_int32():
    source = mgh(np.ones((2, 2, 2), np.int32) * 53)
    result = prepare_subregion_alignment_target(source, (2**32 + 53,), smoothing=None)
    assert result.report["output_selected_voxels"] == 0


@pytest.mark.parametrize("option", [
    {"bbox_margin_voxels": 1.5}, {"bbox_margin_voxels": -1},
    {"smoothing": "closing"}, {"target_voxel_mm": 0},
    {"spatial_chunk_size": 0}, {"memory_budget_gb": 0},
])
def test_invalid_options_rejected(option):
    with pytest.raises(ValueError):
        prepare_subregion_alignment_target(mgh(np.zeros((3, 4, 5), np.int32)), (53,), **option)


def test_workspace_budget_checked_before_device_allocation():
    with pytest.raises(MemoryError):
        prepare_subregion_alignment_target(mgh(np.zeros((3, 4, 5), np.int32)), (53,), memory_budget_gb=1e-9)


def test_nifti_mm_input_and_invalid_spatial_units():
    source = nib.Nifti1Image(np.ones((5, 6, 7), np.int32) * 53, np.diag([1, 1, 1, 1]))
    source.header.set_xyzt_units("mm")
    prepared = prepare_subregion_alignment_target(source, (53,), smoothing=None)
    np.testing.assert_array_equal(prepared.image.dataobj, np.ones((5, 6, 7), np.float32) * 255)
    source.header.set_xyzt_units("meter")
    with pytest.raises(ValueError, match="coordinates in mm"):
        prepare_subregion_alignment_target(source, (53,))
