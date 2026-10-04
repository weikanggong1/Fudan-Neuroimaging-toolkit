"""Check T1 orientation, intensity scaling, and padding without FreeSurfer."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.synthseg_parc.preprocess import preprocess_t1, _resample_1mm


def test_preprocess_orients_and_center_pads_nifti(tmp_path):
    data = np.arange(16 * 18 * 20, dtype=np.float32).reshape(16, 18, 20)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    path = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)

    result = preprocess_t1(path, min_pad=32)
    assert result.image.shape == (32, 32, 32)
    assert result.original_shape == data.shape
    assert result.content_slices == (slice(8, 24), slice(7, 25), slice(6, 26))
    low, high = np.percentile(data, [0.5, 99.5])
    expected = (np.clip(data[::-1], low, high) - low) / (high - low)
    np.testing.assert_allclose(result.image[result.content_slices].numpy(),
                               expected, rtol=0, atol=2e-7)
    assert result.image[0].count_nonzero() == 0
    assert np.allclose(result.aligned_affine[:3, :3], np.eye(3))


def test_preprocess_rejects_multiframe_input(tmp_path):
    path = tmp_path / "four_frames.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 4)), np.eye(4)), path)
    with pytest.raises(ValueError, match="single 3-D"):
        preprocess_t1(path)


def test_resampling_preserves_reference_floating_endpoint_plane():
    # This affine factor is from the public ds003138 raw T1 control. The
    # reference retains 225 output samples on a 288-plane axis; fixed ceil
    # length truncates it to 224 and changes the U-Net's center padding.
    affine = np.diag([1.0, 0.7777777591500216, 1.0, 1.0])
    image = torch.zeros((2, 288, 2), dtype=torch.float64)
    image[:, -1, :] = 1
    resampled, output_affine = _resample_1mm(image, affine)
    assert resampled.shape == (2, 225, 2)
    assert resampled[:, -1].min() > 0.99
    np.testing.assert_allclose(output_affine[:3, :3], np.eye(3), rtol=0, atol=0)
