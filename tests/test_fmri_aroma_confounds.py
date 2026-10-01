from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.fmri.aroma import (
    classify_components,
    denoise_aroma,
    motion_correlation_feature,
    spatial_features,
)
from fnit.fmri.confounds import clean_confounds, motion_regressors


def _nifti(path, array):
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), np.eye(4)), path)
    return path


def test_motion_models_follow_matlab_and_aroma_classifier():
    six = np.arange(24, dtype=float).reshape(4, 6) / 10
    twelve = motion_regressors(six, 12)
    twenty_four = motion_regressors(six, 24)
    np.testing.assert_array_equal(twelve[0, 6:], np.zeros(6))
    np.testing.assert_array_equal(twelve[1, 6:], six[1] - six[0])
    np.testing.assert_array_equal(twenty_four[0, 6:12], np.zeros(6))
    np.testing.assert_array_equal(twenty_four[1, 6:12], six[0])
    np.testing.assert_array_equal(twenty_four[:, 12:18], six**2)
    with pytest.raises(ValueError, match="6, 12, or 24"):
        motion_regressors(six, 18)

    noise = classify_components(
        max_rp_corr=np.array([0, 0, 0, 1]),
        edge_fraction=np.array([0, 0, 0, 0.5]),
        high_freq_content=np.array([0, 0, 0.36, 0]),
        csf_fraction=np.array([0, 0.11, 0, 0]),
    )
    np.testing.assert_array_equal(noise, [1, 2, 3])


def test_aroma_spatial_feature_and_motion_correlation(tmp_path):
    maps = np.zeros((2, 2, 2, 1), dtype=np.float32)
    maps[0, 0, 0, 0] = -2  # CSF
    maps[0, 0, 1, 0] = 3   # brain edge
    maps[0, 1, 0, 0] = 4   # outside brain
    maps[1, 1, 1, 0] = 1   # interior
    csf = np.zeros((2, 2, 2)); csf[0, 0, 0] = 1
    edge = np.zeros((2, 2, 2)); edge[0, 0, 1] = 1
    outside = np.zeros((2, 2, 2)); outside[0, 1, 0] = 1
    edge_fraction, csf_fraction = spatial_features(
        _nifti(tmp_path / "maps.nii.gz", maps),
        _nifti(tmp_path / "csf.nii.gz", csf),
        _nifti(tmp_path / "edge.nii.gz", edge),
        _nifti(tmp_path / "outside.nii.gz", outside),
    )
    np.testing.assert_allclose(edge_fraction, [7 / 8])
    np.testing.assert_allclose(csf_fraction, [2 / 10])

    rng = np.random.default_rng(7)
    motion = rng.normal(size=(100, 6))
    mix = np.column_stack((motion[:, 0], rng.normal(size=100)))
    scores = motion_correlation_feature(mix, motion, n_splits=20, random_state=13)
    assert scores[0] > 0.999
    assert scores[1] < 0.6


def test_aroma_nonaggressive_retains_shared_signal(tmp_path):
    time = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    noise = np.sin(3 * time)
    signal = 0.55 * noise + np.cos(5 * time)
    mix = np.column_stack((noise, signal))
    observed = 100 + 1.5 * noise + 2 * signal
    data = observed.reshape(1, 1, 1, -1)
    source = _nifti(tmp_path / "in.nii.gz", data)
    nonaggr = denoise_aroma(source, mix, [0], tmp_path / "nonaggr.nii.gz", device="cpu")
    aggr = denoise_aroma(source, mix, [0], tmp_path / "aggr.nii.gz", mode="aggr", device="cpu")
    np.testing.assert_allclose(np.asarray(nib.load(nonaggr).dataobj).reshape(-1), 100 + 2 * signal, atol=1e-4)
    assert not np.allclose(np.asarray(nib.load(aggr).dataobj), np.asarray(nib.load(nonaggr).dataobj))


def test_aroma_uses_fsl_default_mean_intensity_mask(tmp_path):
    time = np.arange(30)
    mixture = np.sin(2 * np.pi * time / 10)[:, None]
    data = np.zeros((2, 1, 1, 30), dtype=np.float32)
    data[0, 0, 0] = 100 + mixture[:, 0]
    data[1, 0, 0] = 0.2 + 0.1 * mixture[:, 0]
    source = _nifti(tmp_path / "in.nii.gz", data)
    output = denoise_aroma(source, mixture, [0], tmp_path / "out.nii.gz", device="cpu")
    result = np.asarray(nib.load(output).dataobj)
    np.testing.assert_allclose(result[0, 0, 0], 100, atol=1e-5)
    np.testing.assert_array_equal(result[1, 0, 0], 0)


def test_joint_nuisance_projection_and_bandpass(tmp_path):
    nt = 100
    time = np.arange(nt)
    signal = np.sin(2 * np.pi * 0.05 * time)
    nuisance = np.sin(2 * np.pi * 0.07 * time)
    stopband = np.sin(2 * np.pi * 0.2 * time)
    data = np.zeros((2, 1, 1, nt), dtype=np.float32)
    data[0, 0, 0] = 100 + signal + 3 * nuisance + 4 * stopband + 0.002 * time**2
    data[1, 0, 0] = 60 + nuisance
    wm = np.zeros((2, 1, 1)); wm[1, 0, 0] = 1
    source = _nifti(tmp_path / "in.nii.gz", data)
    wm_file = _nifti(tmp_path / "wm.nii.gz", wm)
    output = clean_confounds(
        source,
        tmp_path / "clean.nii.gz",
        wm_mask=wm_file,
        bandpass=(0.03, 0.09),
        tr=1.0,
        device="cpu",
    )
    result = np.asarray(nib.load(output).dataobj)[0, 0, 0]
    # Quadratic detrending also removes the signal's small projection on drift.
    assert np.corrcoef(result, signal)[0, 1] > 0.9
    assert abs(np.dot(result, nuisance)) < 1e-2
    assert abs(np.dot(result, stopband)) < 1e-2


def test_confounds_rejects_unregistered_tissue_mask(tmp_path):
    source = _nifti(tmp_path / "in.nii.gz", np.ones((2, 1, 1, 10)))
    wm = _nifti(tmp_path / "wm.nii.gz", np.ones((3, 1, 1)))
    with pytest.raises(ValueError, match="aligned"):
        clean_confounds(source, tmp_path / "out.nii.gz", wm_mask=wm, device="cpu")


@pytest.mark.parametrize(("time_unit", "header_tr"), [("msec", 735.0), ("usec", 735000.0)])
def test_confounds_reads_nifti_time_units(tmp_path, time_unit, header_tr):
    time = np.arange(100) * 0.735
    data = (100 + np.sin(2 * np.pi * 0.05 * time)).reshape(1, 1, 1, -1).astype(np.float32)
    header = nib.Nifti1Header()
    header.set_data_shape(data.shape)
    header.set_zooms((1, 1, 1, header_tr))
    header.set_xyzt_units("mm", time_unit)
    source = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4), header), source)
    inferred = clean_confounds(source, tmp_path / "inferred.nii.gz", bandpass=(0.01, 0.1), device="cpu")
    explicit = clean_confounds(source, tmp_path / "explicit.nii.gz", bandpass=(0.01, 0.1), tr=0.735, device="cpu")
    np.testing.assert_allclose(nib.load(inferred).get_fdata(), nib.load(explicit).get_fdata(), atol=1e-5)


def _motion_tissue_fixture(tmp_path):
    random_generator = np.random.default_rng(412)
    frame_count = 160
    motion = random_generator.normal(size=(frame_count, 6))
    motion[:, :3] *= 0.003  # radians; squared rotations are much smaller than tissue baselines
    motion[:, 3:] *= 0.03
    data = np.empty((3, 1, 1, frame_count), dtype=np.float32)
    data[0, 0, 0] = (
        1000 + random_generator.normal(size=frame_count)
        + 10 * (motion[:, 0] / 0.003) ** 2
    )
    # Integer tissue signals remain exactly representable after affine scaling.
    data[1, 0, 0] = 10000 + 128 * random_generator.integers(-3, 4, size=frame_count)
    data[2, 0, 0] = 20000 + 256 * random_generator.integers(-3, 4, size=frame_count)
    wm_mask = np.zeros((3, 1, 1)); wm_mask[1, 0, 0] = 1
    csf_mask = np.zeros((3, 1, 1)); csf_mask[2, 0, 0] = 1
    return (
        _nifti(tmp_path / "motion_tissue_bold.nii.gz", data), data, motion,
        _nifti(tmp_path / "wm_motion.nii.gz", wm_mask),
        _nifti(tmp_path / "csf_motion.nii.gz", csf_mask),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bandpass", [None, (0.02, 0.2)])
def test_confounds_motion_unit_invariance(tmp_path, device, bandpass):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    source, _, motion_radians, wm_mask, csf_mask = _motion_tissue_fixture(tmp_path)
    motion_degrees = motion_radians.copy()
    motion_degrees[:, :3] *= 180 / np.pi
    radians_output = clean_confounds(
        source, tmp_path / "radians.nii.gz", wm_mask=wm_mask, csf_mask=csf_mask,
        motion=motion_radians, bandpass=bandpass, tr=1.0, device=device,
    )
    degrees_output = clean_confounds(
        source, tmp_path / "degrees.nii.gz", wm_mask=wm_mask, csf_mask=csf_mask,
        motion=motion_degrees, bandpass=bandpass, tr=1.0, device=device,
    )
    # Regression depends on a column's direction, not its radian/degree units.
    np.testing.assert_allclose(
        nib.load(radians_output).get_fdata(), nib.load(degrees_output).get_fdata(),
        atol=2e-5, rtol=1e-6,
    )


def test_confounds_tissue_baseline_and_scale_invariance(tmp_path):
    source, data, motion, wm_mask, csf_mask = _motion_tissue_fixture(tmp_path)
    changed_data = data.copy()
    changed_data[1, 0, 0] = data[1, 0, 0] * 8 + 10000000
    changed_data[2, 0, 0] = data[2, 0, 0] * 4 + 5000000
    changed_source = _nifti(tmp_path / "changed_tissue_baselines.nii.gz", changed_data)
    original_output = clean_confounds(
        source, tmp_path / "original_baselines.nii.gz", wm_mask=wm_mask,
        csf_mask=csf_mask, motion=motion, device="cpu",
    )
    changed_output = clean_confounds(
        changed_source, tmp_path / "changed_baselines.nii.gz", wm_mask=wm_mask,
        csf_mask=csf_mask, motion=motion, device="cpu",
    )
    np.testing.assert_allclose(
        nib.load(original_output).get_fdata()[0, 0, 0],
        nib.load(changed_output).get_fdata()[0, 0, 0], atol=2e-5, rtol=1e-6,
    )


def test_confounds_duplicate_and_zero_columns_match_independent_projection(tmp_path):
    random_generator = np.random.default_rng(55)
    frame_count = 120
    nuisance = random_generator.integers(-3, 4, size=frame_count).astype(float)
    data = np.empty((3, 1, 1, frame_count), dtype=np.float32)
    data[0, 0, 0] = 500 + 3 * nuisance + random_generator.normal(size=frame_count)
    data[1, 0, 0] = 10000 + nuisance
    data[2, 0, 0] = 20000 + 2 * nuisance  # same nuisance direction, different scale and baseline
    wm_mask = np.zeros((3, 1, 1)); wm_mask[1, 0, 0] = 1
    csf_mask = np.zeros((3, 1, 1)); csf_mask[2, 0, 0] = 1
    output = clean_confounds(
        _nifti(tmp_path / "duplicate_bold.nii.gz", data), tmp_path / "duplicate_clean.nii.gz",
        wm_mask=_nifti(tmp_path / "duplicate_wm.nii.gz", wm_mask),
        csf_mask=_nifti(tmp_path / "duplicate_csf.nii.gz", csf_mask),
        motion=np.zeros((frame_count, 6)), device="cpu",
    )
    time = np.linspace(-1.0, 1.0, frame_count)
    independent_design = np.column_stack((np.ones(frame_count), time, time**2, nuisance))
    series = data[0, 0, 0].astype(float)
    expected = series - independent_design @ np.linalg.lstsq(independent_design, series, rcond=None)[0]
    result = nib.load(output).get_fdata()
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[0, 0, 0], expected, atol=2e-5, rtol=1e-6)


def test_confounds_drops_stopband_roundoff_instead_of_fitting_it(tmp_path):
    frame_count = 128
    time = np.arange(frame_count)
    signal = np.cos(2 * np.pi * 10 * time / frame_count) + np.sin(2 * np.pi * 9 * time / frame_count)
    source = _nifti(tmp_path / "stopband_bold.nii.gz", (100 + signal).reshape(1, 1, 1, -1))
    motion = np.zeros((frame_count, 6))
    motion[:, 3] = np.sin(2 * np.pi * 40 * time / frame_count)
    baseline = clean_confounds(
        source, tmp_path / "bandpass_baseline.nii.gz", bandpass=(0.04, 0.1),
        tr=1.0, device="cpu",
    )
    with_motion = clean_confounds(
        source, tmp_path / "bandpass_stopband_motion.nii.gz", motion=motion, motion_model=6,
        bandpass=(0.04, 0.1), tr=1.0, device="cpu",
    )
    baseline_data = nib.load(baseline).get_fdata()[0, 0, 0]
    result = nib.load(with_motion).get_fdata()[0, 0, 0]
    np.testing.assert_allclose(result, baseline_data, atol=2e-5, rtol=1e-6)
    frequencies = np.fft.rfftfreq(frame_count, 1.0)
    stopband = (frequencies < 0.04) | (frequencies > 0.1)
    assert np.max(np.abs(np.fft.rfft(result)[stopband])) < 1e-5
