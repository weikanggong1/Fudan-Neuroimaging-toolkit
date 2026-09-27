"""Numerical properties of FEAT intensity scaling and temporal filtering."""

import nibabel as nib
import numpy as np
import pytest

from fnit.feat.temporal import (
    gaussian_highpass,
    gaussian_highpass_matrix,
    grand_mean_scale,
    highpass_nifti,
    scale_nifti,
)


def _direct_weighted_line_filter(series, sigma):
    """Small reference computed by fitting one local line at a time."""
    fitted = np.empty_like(series, dtype=np.float64)
    radius = int(3 * sigma)
    for t in range(series.size):
        locations = np.arange(max(0, t - radius), min(series.size, t + radius + 1))
        distance = locations - t
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
        design = np.stack((np.ones(distance.size), distance), axis=1)
        fitted[t] = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            series[locations] * np.sqrt(weights), rcond=None,
        )[0][0]
    residual = series - fitted
    return residual - residual.mean() + series.mean()


def test_grand_mean_uses_all_masked_timepoints_and_fsl_upper_median():
    data = np.array([[[[10, 30, 70, 90]]], [[[100, 200, 300, 400]]]], dtype=np.float32)
    mask = np.array([[[1]], [[0]]], dtype=np.float32)
    scaled, factor = grand_mean_scale(data, mask, target_median=10000.0)
    assert factor == pytest.approx(10000 / 70)
    np.testing.assert_allclose(scaled, data * np.float32(factor))


def test_highpass_local_line_and_boundary_weights():
    time = np.arange(21, dtype=np.float64)
    series = 4000 + 13 * time + 50 * np.sin(time * 0.7)
    expected = _direct_weighted_line_filter(series, sigma=2.3)
    observed = gaussian_highpass(series.reshape(1, 1, 1, -1), sigma_volumes=2.3,
                                 voxel_chunk=1)[0, 0, 0]
    np.testing.assert_allclose(observed, expected, atol=7e-4)
    assert observed.mean() == pytest.approx(series.mean(), abs=5e-4)


def test_highpass_removes_linear_drift_and_preserves_original_mean():
    time = np.arange(33, dtype=np.float32)
    data = np.stack((8000 + 3 * time, 12000 - 8 * time), axis=0).reshape(2, 1, 1, -1)
    filtered = gaussian_highpass(data, sigma_volumes=5.0, voxel_chunk=1)
    expected_mean = np.broadcast_to(data.mean(axis=-1, keepdims=True), data.shape)
    np.testing.assert_allclose(filtered, expected_mean, atol=0.02)
    demeaned = gaussian_highpass(data, sigma_volumes=5.0,
                                 voxel_chunk=1, preserve_mean=False)
    np.testing.assert_allclose(demeaned, 0, atol=0.02)


def test_matrix_never_uses_samples_outside_three_sigma_window():
    matrix = gaussian_highpass_matrix(20, 1.1)
    # Impulse at time 10 cannot affect the fitted line at time 0; only the
    # whole-run mean-restoration term connects them.
    assert matrix[0, 10] == pytest.approx(matrix[1, 10], abs=1e-7)


def test_nifti_wrappers_preserve_geometry_and_tr(tmp_path):
    data = np.arange(1, 1 + 2 * 2 * 2 * 12, dtype=np.float32).reshape(2, 2, 2, 12)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    bold = nib.Nifti1Image(data, affine)
    bold.header.set_zooms((2.0, 2.0, 2.0, 0.8))
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), affine)
    input_path = tmp_path / "input.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    scaled_path = tmp_path / "scaled.nii.gz"
    filtered_path = tmp_path / "filtered.nii.gz"
    nib.save(bold, input_path)
    nib.save(mask, mask_path)
    factor = scale_nifti(input_path, mask_path, scaled_path)
    assert factor == pytest.approx(10000 / 49)
    assert highpass_nifti(scaled_path, filtered_path, cutoff_seconds=16.0) == filtered_path
    result = nib.load(filtered_path)
    assert result.shape == data.shape
    np.testing.assert_allclose(result.affine, affine)
    assert result.header.get_zooms()[3] == pytest.approx(0.8)
    assert result.get_data_dtype() == np.dtype(np.float32)


def test_highpass_nifti_converts_header_milliseconds(tmp_path):
    time = np.arange(30, dtype=np.float32)
    data = (3000 + 25 * np.sin(time / 4)).reshape(1, 1, 1, 30)
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_zooms((1, 1, 1, 735))
    image.header.set_xyzt_units(t="msec")
    source = tmp_path / "milliseconds.nii.gz"
    output = tmp_path / "filtered.nii.gz"
    nib.save(image, source)
    highpass_nifti(source, output, cutoff_seconds=10.0)
    expected = gaussian_highpass(data, sigma_volumes=10 / (2 * 0.735))
    np.testing.assert_allclose(nib.load(output).get_fdata(dtype=np.float32), expected, atol=1e-4)
