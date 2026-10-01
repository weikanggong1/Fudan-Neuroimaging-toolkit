"""SynthStrip geometry contracts that differ from a generic NIfTI conform."""

import nibabel as nib
import numpy as np

from fnit.synthstrip.pipeline import (
    _conform_lia_1mm, _crop_nonzero, _reshape_center, _resample_affine,
)


def test_resize_preserves_full_field_of_view_center():
    affine = np.array([[-2, 0, 0, 10], [0, 0, 2, -12], [0, -2, 0, 20], [0, 0, 0, 1]], dtype=float)
    image = nib.Nifti1Image(np.ones((5, 6, 7), dtype=np.float32), affine)
    resized = _conform_lia_1mm(image)
    assert resized.shape == (10, 12, 14)
    old_center = affine[:3, :3] @ (np.array(image.shape) / 2) + affine[:3, 3]
    new_center = resized.affine[:3, :3] @ (np.array(resized.shape) / 2) + resized.affine[:3, 3]
    np.testing.assert_array_equal(new_center, old_center)


def test_resize_extent_uses_header_voxel_sizes():
    affine = np.diag([-2.38636363, -2.4, 2.38636363, 1.0])
    image = nib.Nifti1Image(np.ones((88, 64, 88), dtype=np.float32), affine)
    image.header.set_zooms((np.float32(2.3863637), np.float32(2.4), np.float32(2.3863637)))
    resized = _conform_lia_1mm(image)
    assert resized.shape == (211, 211, 154)


def test_positive_bounding_box_ignores_negative_background():
    data = np.full((9, 9, 9), -1, dtype=np.float32)
    data[3:7, 2:6, 1:5] = 8
    cropped = _crop_nonzero(nib.Nifti1Image(data, np.eye(4)))
    assert cropped.shape == (4, 4, 4)
    np.testing.assert_array_equal(cropped.affine[:3, 3], [3, 2, 1])


def test_odd_cropping_keeps_low_voxel_when_extra_removed_high():
    data = np.arange(7 * 5 * 3, dtype=np.float32).reshape(7, 5, 3)
    resized = _reshape_center(nib.Nifti1Image(data, np.eye(4)), (4, 4, 4))
    np.testing.assert_array_equal(resized.data[..., :3], data[1:5, :4])
    np.testing.assert_array_equal(resized.affine[:3, 3], [1, 0, 0])


def test_nearest_half_voxel_and_last_voxel_contract():
    source = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    target_affine = np.eye(4)
    target_affine[0, 3] = 0.5
    sampled = _resample_affine(source, np.eye(4), (3, 1, 1), target_affine, nearest=True, fill=100)
    np.testing.assert_array_equal(sampled[:, 0, 0], [1, 2, 2])


def test_linear_last_voxel_clamps_high_neighbour():
    source = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    target_affine = np.eye(4)
    target_affine[0, 3] = 0.5
    sampled = _resample_affine(source, np.eye(4), (4, 1, 1), target_affine, fill=100)
    np.testing.assert_array_equal(sampled[:, 0, 0], [0.5, 1.5, 2, 100])
