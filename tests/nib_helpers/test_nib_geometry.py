"""Image replacement preserves authored NIfTI geometry and decoded values."""
import nibabel as nib
import numpy as np
import pytest

from fnit._nib import FNITNifti1Image, new_image


@pytest.mark.parametrize("frames", [None, 2])
def test_frame_replacement_preserves_field_of_view_boundary(tmp_path, frames):
    shape = (8, 288, 4) if frames is None else (8, 288, 4, frames)
    affine = np.diag([0.8, np.float32(0.77777773), 0.9, 1.0])
    reference = nib.Nifti1Image(np.ones(shape, np.int16), affine)
    reference.header["pixdim"][2] = np.float32(0.77777779)
    source_path = tmp_path / "source.nii.gz"
    nib.save(reference, source_path)
    reference = nib.load(source_path)
    data = np.asanyarray(reference.dataobj)
    frame = data if frames is None else data[..., 0]
    result = new_image(frame.astype(np.float32), reference)
    assert isinstance(result, FNITNifti1Image)
    np.testing.assert_array_equal(result.header["pixdim"][:4],
                                  reference.header["pixdim"][:4])
    assert np.ceil(result.shape[1] * float(result.header["pixdim"][2])) == 225
    for field in ("qform_code", "sform_code", "quatern_b", "quatern_c",
                  "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z",
                  "srow_x", "srow_y", "srow_z"):
        np.testing.assert_array_equal(result.header[field], reference.header[field])
    path = tmp_path / "frame.nii.gz"
    result.save(path)
    reloaded = nib.load(path)
    np.testing.assert_array_equal(reloaded.affine, reference.affine)
    np.testing.assert_array_equal(reloaded.header["pixdim"][:4],
                                  reference.header["pixdim"][:4])
    np.testing.assert_array_equal(reloaded.dataobj, frame)


def test_separate_qform_and_sform_are_retained():
    reference = nib.Nifti1Image(np.zeros((3, 4, 5), np.float32), np.eye(4))
    qform = np.diag([0.8, 0.9, 1.1, 1.0])
    qform[:3, 3] = [1, 2, 3]
    reference.set_qform(qform, 1)
    reference.set_sform(np.eye(4), 4)
    result = new_image(np.ones(reference.shape, np.uint8), reference)
    np.testing.assert_array_equal(result.get_qform(), reference.get_qform())
    np.testing.assert_array_equal(result.get_sform(), reference.get_sform())
    assert result.get_qform(coded=True)[1] == 1
    assert result.get_sform(coded=True)[1] == 4


def test_explicit_new_grid_updates_forms_and_zooms():
    reference = nib.Nifti1Image(np.zeros((3, 4, 5), np.float32), np.eye(4))
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    result = new_image(np.ones((2, 2, 2), np.float32), reference, affine=affine)
    np.testing.assert_array_equal(result.affine, affine)
    np.testing.assert_allclose(result.get_qform(), affine)
    np.testing.assert_array_equal(result.get_sform(), affine)
    np.testing.assert_array_equal(result.header.get_zooms(), [2, 3, 4])


def test_mgh_reference_retains_world_affine_after_save(tmp_path):
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [7, 9, 11]
    reference = nib.MGHImage(np.zeros((3, 4, 5), np.float32), affine)
    result = new_image(np.ones(reference.shape, np.uint8), reference)
    path = tmp_path / "from_mgh.nii.gz"
    result.save(path)
    np.testing.assert_array_equal(nib.load(path).affine, reference.affine)
    assert int(result.header["sform_code"]) > 0
