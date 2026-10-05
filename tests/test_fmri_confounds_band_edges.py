"""Regress the extra passband-edge bin observed in the real 490-frame run."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.fmri.confounds import _passband_bins, clean_confounds


def test_490_frame_afni_passband_regressor_count():
    keep = _passband_bins(490, 0.735, 0.01, 0.1)
    np.testing.assert_array_equal(np.flatnonzero(keep), np.arange(4, 36))
    # AFNI_24.2.02 reports 425 stopband columns plus a separate intercept.
    assert 490 - 2 * keep.sum() - 1 == 425
    assert not keep[36]  # Bin center 0.09996 Hz was incorrectly retained.


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_edge_frequency_is_removed_by_joint_projection(tmp_path, device):
    frame_count = 490
    sample = np.arange(frame_count)
    retained = np.sin(2 * np.pi * 20 * sample / frame_count)
    removed = np.sin(2 * np.pi * 36 * sample / frame_count)
    data = (100 + retained + 5 * removed).astype(np.float32).reshape(1, 1, 1, -1)
    source = tmp_path / "edge.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    output = clean_confounds(source, tmp_path / "clean.nii.gz", tr=0.735,
                             bandpass=(0.01, 0.1), device=device, projection="afni")
    result = np.asarray(nib.load(output).dataobj).ravel()
    assert abs(result @ removed) / (removed @ removed) < 1e-4
    assert np.corrcoef(result, retained)[0, 1] > 0.99


def test_odd_run_keeps_both_top_frequency_components():
    # An odd-length run has no Nyquist-only cosine bin.
    keep = _passband_bins(101, 1.0, 0.02, 0.2)
    np.testing.assert_array_equal(np.flatnonzero(keep), np.arange(3, 20))


def test_default_strict_projection_preserves_old_edge_convention(tmp_path):
    frame_count = 490
    sample = np.arange(frame_count)
    retained = np.sin(2 * np.pi * 36 * sample / frame_count)
    source = tmp_path / "edge.nii.gz"
    nib.save(nib.Nifti1Image((100 + retained).astype(np.float32).reshape(1, 1, 1, -1),
                           np.eye(4)), source)
    output = clean_confounds(source, tmp_path / "strict.nii.gz", tr=0.735,
                             bandpass=(0.01, 0.1), device="cpu")
    result = np.asarray(nib.load(output).dataobj).ravel()
    assert np.corrcoef(result, retained)[0, 1] > 0.99
