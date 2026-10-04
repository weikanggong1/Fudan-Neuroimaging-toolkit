"""Preserve real oblique input pixdim when marking qform inactive."""

import nibabel as nib
import numpy as np

from fnit.synthseg_parc.synthseg import _segmentation_image


def test_keep_geometry_preserves_header_voxel_sizes_through_save(tmp_path):
    # Affine/spacing taken from the raw case01 NIfTI. Column norms differ
    # slightly from pixdim, so re-encoding qform silently changes spacing.
    affine = np.array([
        [0.7984692454338074, 0.04435047507286072, -0.01887112855911255, -95.93505859375],
        [-0.04570585489273071, 0.7765015959739685, -0.0029988891910761595, -120.73804473876953],
        [0.01920272968709469, 0.0040712677873671055, 0.7775430083274841, -128.53616333007812],
        [0, 0, 0, 1]])
    reference = nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), affine)
    reference.header.set_zooms([0.800000011920929, 0.7777777910232544, 0.7777777910232544])
    image = _segmentation_image(np.ones(reference.shape), reference, affine)
    path = tmp_path / "labels.nii.gz"
    image.save(path)
    restored = nib.load(path)
    assert restored.shape == reference.shape
    assert np.array_equal(restored.affine, reference.affine)
    assert restored.header.get_zooms() == reference.header.get_zooms()
    assert int(restored.header["qform_code"]) == 0
    assert int(restored.header["sform_code"]) == 2
    assert restored.get_data_dtype() == np.dtype(np.int32)
