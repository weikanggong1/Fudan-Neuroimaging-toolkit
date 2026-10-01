"""Formula regressions for normalization and identifiable OLS, not benchmarks."""

import json

import h5py
import nibabel as nib
import numpy as np
import pytest
from scipy.stats import norm, t as student_t

import fnit.bigflica.streaming as streaming_module
from fnit.bigflica.pipeline import (_NORMALIZATION_VERSION, _file_record,
                                    _signature, _spatial_z, _standardize)
from fnit.bigflica.stats_torch import SpatialRegression
from fnit.bigflica.streaming import prepare_modalities


def _signed_rows():
    # All three nonzero rows sum to zero; the fourth is genuinely empty.
    return np.array([[1, -1, 3, -3], [2, -2, -4, 4],
                     [.25, -.25, 2.5, -2.5], [0, 0, 0, 0]], dtype=np.float32)


def _write_images(tmp_path):
    root = tmp_path / "subjects"
    subjects = ["first", "second", "third", "empty"]
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), np.eye(4)), mask)
    for subject, values in zip(subjects, _signed_rows()):
        directory = root / subject
        directory.mkdir(parents=True)
        nib.save(nib.Nifti1Image(values.reshape(2, 2, 1), np.eye(4)),
                 directory / "signed.nii.gz")
    return root, subjects, {"signed": {"image": "signed.nii.gz", "mask": str(mask)}}


def test_dense_normalization_keeps_nonzero_signed_rows_and_float64_statistics():
    values = _signed_rows()
    normalized, mean, std = _standardize(values)
    expected_mean = np.array([13 / 12, -13 / 12, .5, -.5])
    expected_std = np.sqrt(np.array([37 / 72, 37 / 72, 61 / 6, 61 / 6]))
    expected = np.vstack(((values[:3] - expected_mean) / expected_std, np.zeros(4)))
    assert normalized.dtype == mean.dtype == std.dtype == np.dtype("float64")
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=2e-16)
    np.testing.assert_allclose(std, expected_std, rtol=2e-15, atol=0)
    np.testing.assert_allclose(normalized, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(normalized[-1], np.zeros(4))


def test_saved_dense_statistics_reproduce_first_normalization_exactly(tmp_path):
    values = _signed_rows()[:3]
    normalized, mean, std = _standardize(values)
    # Repeating thirds and square roots expose the former float32-stat rounding.
    assert np.any(mean != mean.astype(np.float32).astype(np.float64))
    np.save(tmp_path / "mean.npy", mean)
    np.save(tmp_path / "std.npy", std)
    restored_mean = np.load(tmp_path / "mean.npy")
    restored_std = np.load(tmp_path / "std.npy")
    assert restored_mean.dtype == restored_std.dtype == np.dtype("float64")
    np.testing.assert_array_equal((values - restored_mean) / restored_std, normalized)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_streaming_statistics_match_dense_and_reuse_without_reading_images(
        tmp_path, monkeypatch, dtype):
    root, subjects, specs = _write_images(tmp_path)
    destination = tmp_path / "normalized"
    prepare_modalities(root, specs, subjects, destination, feature_block=2,
                       normalized_dtype=dtype)
    normalized, mean, std = _standardize(_signed_rows())
    with h5py.File(destination / "signed.h5", "r") as file:
        assert file["mean"].dtype == file["std"].dtype == np.dtype("float64")
        assert file.attrs["normalization_version"] == _NORMALIZATION_VERSION
        np.testing.assert_array_equal(file["valid_rows"][:], [True, True, True, False])
        np.testing.assert_allclose(file["mean"][:], mean, rtol=0, atol=2e-16)
        np.testing.assert_allclose(file["std"][:], std, rtol=2e-15, atol=0)
        np.testing.assert_allclose(file["data"][:], normalized.astype(dtype),
                                   rtol=2e-15 if dtype == "float64" else 2e-7,
                                   atol=2e-15 if dtype == "float64" else 2e-7)
        before = {key: file[key][:] for key in ("data", "mean", "std", "valid_rows")}
    manifest_before = (destination / "manifest.json").read_text()
    assert json.loads(manifest_before)["normalization_version"] == _NORMALIZATION_VERSION

    def unexpected_read(*args):
        raise AssertionError("A valid normalization cache must not reread NIfTI")

    monkeypatch.setattr(streaming_module, "_read_vector", unexpected_read)
    prepare_modalities(root, specs, subjects, destination, feature_block=2,
                       normalized_dtype=dtype)
    assert (destination / "manifest.json").read_text() == manifest_before
    with h5py.File(destination / "signed.h5", "r") as file:
        for key, expected in before.items():
            np.testing.assert_array_equal(file[key][:], expected)


def test_previous_zero_sum_normalization_cache_is_rebuilt(tmp_path, monkeypatch):
    root, subjects, specs = _write_images(tmp_path)
    destination = tmp_path / "normalized"
    prepare_modalities(root, specs, subjects, destination, feature_block=2)
    old_signature = _signature({
        "subjects": subjects, "normalized_dtype": "float64-v2-stats64",
        "modalities": {"signed": {"image": "signed.nii.gz",
                                    "mask": _file_record(tmp_path / "mask.nii.gz")}},
        "images": {"signed": [_file_record(root / subject / "signed.nii.gz")
                                for subject in subjects]},
    })
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["signature"] != old_signature
    manifest["signature"] = old_signature
    manifest.pop("normalization_version")
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with h5py.File(destination / "signed.h5", "r+") as file:
        file["valid_rows"][:] = False
        file["data"][:] = 999

    original_read = streaming_module._read_vector
    calls = []

    def record_read(path, *args):
        calls.append(path)
        return original_read(path, *args)

    monkeypatch.setattr(streaming_module, "_read_vector", record_read)
    prepare_modalities(root, specs, subjects, destination, feature_block=2)
    assert len(calls) == len(subjects)
    updated = json.loads((destination / "manifest.json").read_text())
    assert updated["signature"] != old_signature
    assert updated["normalization_version"] == _NORMALIZATION_VERSION
    with h5py.File(destination / "signed.h5", "r") as file:
        np.testing.assert_array_equal(file["valid_rows"][:], [True, True, True, False])
        assert not np.any(file["data"][:] == 999)


@pytest.mark.parametrize("case", ["component_with_intercept", "duplicate_component"])
def test_regressions_reject_nonidentifiable_design(case):
    slope = np.arange(7, dtype=np.float64)
    h = np.column_stack((slope, np.ones(7) if case == "component_with_intercept"
                         else 2 * slope))
    if case == "component_with_intercept":
        assert np.linalg.matrix_rank(h) == h.shape[1]
    projected = np.arange(21, dtype=np.float64).reshape(3, 7)
    with pytest.raises(ValueError, match="full column rank"):
        _spatial_z(h, projected)
    with pytest.raises(ValueError, match="full column rank"):
        SpatialRegression(h, device="cpu")


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_regressions_reject_nonfinite_design(nonfinite):
    h = np.arange(7, dtype=np.float64)[:, None]
    h[0, 0] = nonfinite
    with pytest.raises(ValueError, match="finite"):
        _spatial_z(h, np.ones((3, 7)))
    with pytest.raises(ValueError, match="finite"):
        SpatialRegression(h, device="cpu")


def test_identifiable_regressions_preserve_analytic_t_z_and_degrees_of_freedom():
    # Orthogonal columns and residual give coefficients/standard errors directly.
    h = np.column_stack(([-3, -2, -1, 0, 0, 1, 2, 3],
                         [1, -1, 1, -1, -1, 1, -1, 1])).astype(np.float64)
    residual = np.array([1, 1, -1, -1, -1, -1, 1, 1], dtype=np.float64)
    beta = np.array([[2, -1], [.25, .5], [-3, 2]], dtype=np.float64)
    amplitude = np.array([.5, 2, 1.5])
    intercept = np.array([10, -2, 7])
    projected = (h @ beta.T + residual[:, None] * amplitude + intercept).T
    df = 5
    sigma = amplitude * np.sqrt(8 / df)
    expected_t = beta * np.sqrt([28, 8]) / sigma[:, None]
    regression = SpatialRegression(h, device="cpu")
    assert regression.df == df
    np.testing.assert_allclose(regression.t(projected), expected_t,
                               rtol=2e-14, atol=2e-14)
    expected_z = np.sign(expected_t) * norm.isf(student_t.sf(np.abs(expected_t), df))
    np.testing.assert_array_equal(_spatial_z(h, projected), expected_z.astype(np.float32))
