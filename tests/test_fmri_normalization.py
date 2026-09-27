"""Coordinate-composition checks for EPI-to-T1-to-MNI resampling."""

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.normalization import resample_world


def test_world_affine_and_pull_compose_before_one_4d_interpolation(tmp_path):
    x, y, z = np.indices((7, 7, 7))
    source_data = np.stack((x + 10 * y + 100 * z,
                            1000 + x + 10 * y + 100 * z), axis=-1).astype(np.float32)
    source = nib.Nifti1Image(source_data, np.eye(4))
    source.header.set_zooms((1, 1, 1, 0.8))
    source.header.set_xyzt_units(t="sec")
    source_path = tmp_path / "source.nii.gz"
    nib.save(source, str(source_path))
    reference = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    reference_path = tmp_path / "reference.nii.gz"
    nib.save(reference, str(reference_path))
    pull_data = np.zeros((4, 4, 4, 3), dtype=np.float32)
    pull_data[..., 1] = 1
    pull_path = tmp_path / "pull.nii.gz"
    nib.save(nib.Nifti1Image(pull_data, np.eye(4)), str(pull_path))
    output_mask = np.ones((4, 4, 4), dtype=np.uint8)
    output_mask[0] = 0
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(output_mask, np.eye(4)), str(mask_path))
    affine = np.eye(4)
    affine[0, 3] = 1
    output = resample_world(
        source_path, reference_path, affine, tmp_path / "output.nii.gz",
        pre_affine_pull_ras=pull_path, output_mask=mask_path,
        batch_size=1, device="cpu",
    )
    actual = nib.load(str(output))
    expected = source_data[1:5, 1:5, 0:4].copy()
    expected[0] = 0
    np.testing.assert_allclose(np.asarray(actual.dataobj), expected, atol=1e-5)
    assert actual.shape == (4, 4, 4, 2)
    assert actual.header.get_zooms()[3] == pytest.approx(0.8)


def test_world_pull_rejects_wrong_reference_grid(tmp_path):
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    pull = tmp_path / "pull.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), str(source))
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), str(reference))
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 4, 3)), np.eye(4)), str(pull))
    with pytest.raises(ValueError, match="reference grid"):
        resample_world(source, reference, np.eye(4), tmp_path / "out.nii.gz",
                       pre_affine_pull_ras=pull, device="cpu")
