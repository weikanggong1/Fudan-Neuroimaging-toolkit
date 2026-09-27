import nibabel as nib
import numpy as np
import pytest

from fnit.applywarp import TorchApplyWarp
from fnit.fmri.mask import epi_brain_mask
from fnit.fmri.motion import estimate_motion
from fnit.fmri.spatial import apply_motion_warp


def test_per_frame_resampling_matches_validated_single_frame_applywarp():
    rng = np.random.default_rng(8)
    affine = np.diag([-2.0, 2.1, 2.2, 1.0])
    data = rng.normal(size=(11, 12, 10, 3)).astype(np.float32)
    bold = nib.Nifti1Image(data, affine)
    ref = nib.Nifti1Image(np.zeros(data.shape[:3], dtype=np.float32), affine)
    matrices = np.repeat(np.eye(4)[None], 3, axis=0)
    matrices[1, 0, 3] = 0.6
    matrices[2, 1, 3] = -0.8
    actual = apply_motion_warp(bold, ref, matrices, device="cpu", batch_size=2)
    expected = np.stack([
        TorchApplyWarp(device="cpu")(
            nib.Nifti1Image(data[..., frame], affine), ref,
            premat=matrices[frame], interpolation="trilinear",
        ).image.get_fdata(dtype=np.float32)
        for frame in range(3)
    ], axis=-1)
    np.testing.assert_allclose(actual.get_fdata(dtype=np.float32), expected, atol=2e-5)


def test_motion_recovers_known_translation():
    axes = np.meshgrid(*[np.arange(25) for _ in range(3)], indexing="ij")
    fixed = sum(np.exp(-sum((axis - center) ** 2 for axis, center in zip(axes, centers)) / 10)
                for centers in ((8, 10, 13), (16, 14, 9), (13, 19, 16)))
    fixed = fixed.astype(np.float32)
    from scipy.ndimage import shift
    moving = shift(fixed, (0.8, -0.5, 0.3), order=1)
    image = nib.Nifti1Image(np.stack((fixed, moving), axis=-1), np.eye(4))
    reference = nib.Nifti1Image(fixed, np.eye(4))
    result = estimate_motion(image, reference, device="cpu", batch_size=2,
                             iterations=(30, 30, 30))
    np.testing.assert_allclose(result.parameters[0, 3:], 0, atol=0.2)
    np.testing.assert_allclose(result.parameters[1, 3:], (0.8, -0.5, 0.3), atol=0.35)


def test_epi_mask_is_same_grid_and_nonempty():
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[4:16, 5:15, 5:15] = 500
    data[8:12, 8:12, 8:12] = 1500
    image = nib.Nifti1Image(data, np.diag((2, 2, 2, 1)))
    mask = epi_brain_mask(image)
    assert mask.shape == image.shape
    np.testing.assert_allclose(mask.affine, image.affine)
    assert 0 < np.asarray(mask.dataobj).sum() < data.size


def test_warp_reference_grid_mismatch_requires_postmat():
    bold = nib.Nifti1Image(np.zeros((11, 12, 10, 2), dtype=np.float32), np.eye(4))
    reference = nib.Nifti1Image(np.zeros((11, 12, 10), dtype=np.float32), np.eye(4))
    warp = nib.Nifti1Image(np.zeros((10, 12, 10, 3), dtype=np.float32), np.eye(4))
    matrices = np.repeat(np.eye(4)[None], 2, axis=0)
    with pytest.raises(ValueError, match="warp reference grid differs"):
        apply_motion_warp(bold, reference, matrices, warp=warp,
                          warp_convention="relative", device="cpu")


def test_resampled_bold_header_uses_reference_spatial_zooms():
    bold = nib.Nifti1Image(np.ones((11, 12, 10, 2), dtype=np.float32),
                           np.diag((2.0, 2.0, 2.0, 1.0)))
    bold.header.set_zooms((2.0, 2.0, 2.0, 0.735))
    bold.header.set_xyzt_units(t="sec")
    reference = nib.Nifti1Image(np.zeros((8, 9, 7), dtype=np.float32),
                                np.diag((3.0, 3.0, 3.0, 1.0)))
    matrices = np.repeat(np.eye(4)[None], 2, axis=0)
    result = apply_motion_warp(bold, reference, matrices, device="cpu")
    assert result.shape == (8, 9, 7, 2)
    np.testing.assert_allclose(result.header.get_zooms(), (3.0, 3.0, 3.0, 0.735), rtol=1e-6)
    assert result.header.get_xyzt_units()[1] == "sec"


def test_cubic_spline_resampling_reduces_error_on_smooth_signal():
    axes = np.meshgrid(*(np.arange(17) for _ in range(3)), indexing="ij")
    signal = (np.sin(0.25 * axes[0]) + 0.5 * np.sin(0.35 * axes[1])
              + 0.3 * np.cos(0.2 * axes[2])).astype(np.float32)
    affine = np.diag((-1.0, 1.0, 1.0, 1.0))
    bold = nib.Nifti1Image(signal[..., None], affine)
    reference = nib.Nifti1Image(np.zeros(signal.shape, dtype=np.float32), affine)
    matrix = np.eye(4)[None]
    matrix[0, :3, 3] = (0.35, -0.42, 0.27)
    shifted = (axes[0] - 0.35, axes[1] + 0.42, axes[2] - 0.27)
    expected = (np.sin(0.25 * shifted[0]) + 0.5 * np.sin(0.35 * shifted[1])
                + 0.3 * np.cos(0.2 * shifted[2]))[3:-3, 3:-3, 3:-3]
    linear = apply_motion_warp(bold, reference, matrix, device="cpu",
                               interpolation="linear").get_fdata(dtype=np.float32)
    spline = apply_motion_warp(bold, reference, matrix, device="cpu",
                               interpolation="spline").get_fdata(dtype=np.float32)
    linear_error = np.mean(np.abs(linear[3:-3, 3:-3, 3:-3, 0] - expected))
    spline_error = np.mean(np.abs(spline[3:-3, 3:-3, 3:-3, 0] - expected))
    assert spline_error < linear_error / 5
