"""Small numerical checks for the independent spatial ICA decomposition."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from fnit.melodic.ica import decompose_spatial_ica
from fnit.melodic import run_melodic_bids


def test_two_skewed_components_recover_temporal_modes(tmp_path):
    rng = np.random.default_rng(9)
    shape = (12, 10, 8)
    nt = 120
    nvox = np.prod(shape)
    # Skewed spatial sources are identifiable by MELODIC's pow3 contrast.
    sources = rng.exponential(size=(nvox, 2))
    sources -= sources.mean(axis=0)
    time = np.arange(nt)
    mixing = np.column_stack((
        np.sin(2 * np.pi * 3 * time / nt),
        np.cos(2 * np.pi * 11 * time / nt),
    ))
    bold = (100 + sources @ mixing.T + rng.normal(0, 0.01, (nvox, nt))).reshape(*shape, nt)
    affine = np.diag((2, 2, 2, 1))
    input_path = tmp_path / "bold.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(bold.astype(np.float32), affine), input_path)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine), mask_path)

    result = decompose_spatial_ica(
        input_path, mask_path, tmp_path / "ica", n_components=2,
        device="cpu", voxel_batch_size=256, random_state=7, mm_threshold=0.5,
    )
    component_image = nib.load(result.component_maps)
    assert component_image.header.get_zooms()[3] == 1.0
    assert component_image.header.get_xyzt_units()[1] == "unknown"
    np.testing.assert_allclose(component_image.affine, affine)
    maps = np.asarray(component_image.dataobj).reshape(nvox, 2)
    estimated = np.loadtxt(result.mixing)
    spectral = np.loadtxt(result.frequency_power)
    assert np.isfinite(maps).all()
    correlation = np.abs(np.corrcoef(estimated.T, mixing.T)[:2, 2:])
    assert (correlation.max(axis=1) > 0.9).all()
    assert (correlation.max(axis=0) > 0.9).all()
    assert estimated.shape == (nt, 2)
    assert spectral.shape == (nt // 2, 2)
    np.testing.assert_allclose(spectral, np.abs(np.fft.rfft(estimated, axis=0)[1:]) ** 2, rtol=1e-6, atol=1e-5)
    assert result.converged
    assert result.final_decorrelation_change < 1e-3
    assert result.pca_variance_explained > 0.98
    assert np.count_nonzero(np.asarray(nib.load(result.thresholded_maps).dataobj)) > 0
    probabilities = np.asarray(nib.load(result.posterior_maps).dataobj)
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    assert result.model_order_method == "fixed"
    assert result.estimated_resels is None


def test_component_count_requires_temporal_rank_and_aligned_mask(tmp_path):
    bold = tmp_path / "bold.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 4), dtype=np.float32), np.eye(4)), bold)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.diag((2, 1, 1, 1))), mask)
    with pytest.raises(ValueError, match="aligned"):
        decompose_spatial_ica(bold, mask, tmp_path / "out", n_components=2, device="cpu")
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), mask)
    with pytest.raises(ValueError, match="smaller than the number of time points"):
        decompose_spatial_ica(bold, mask, tmp_path / "out", n_components=4, device="cpu")


def test_pica_mixture_probability_separates_null_and_signal():
    import torch
    from fnit.melodic.ica import _mixture_posterior

    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, 9000)
    positive = rng.gamma(3, 1, 5000) + 1
    negative = -rng.gamma(3, 1, 5000) - 1
    adjusted, posterior = _mixture_posterior(torch.as_tensor(np.r_[null, positive, negative]))
    probability = posterior.numpy()
    assert np.isfinite(adjusted.numpy()).all()
    assert (probability[:9000] >= 0.5).mean() < 0.2
    assert (probability[9000:14000] >= 0.5).mean() > 0.9
    assert (probability[14000:] >= 0.5).mean() > 0.9


def test_standalone_melodic_writes_bids_derivatives(tmp_path):
    import json

    raw = tmp_path / "raw"
    source = tmp_path / "preproc"
    output = tmp_path / "fnit"
    raw.mkdir()
    (raw / "dataset_description.json").write_text(json.dumps({
        "Name": "Test raw BIDS", "BIDSVersion": "1.11.1",
    }))
    source.mkdir()
    (source / "dataset_description.json").write_text(json.dumps({
        "Name": "Test preprocessing", "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "DatasetLinks": {"raw": "../raw"},
    }))
    folder = source / "sub-01" / "func"
    folder.mkdir(parents=True)
    bold = folder / "sub-01_task-rest_desc-preproc_bold.nii.gz"
    mask = folder / "sub-01_task-rest_desc-brain_mask.nii.gz"
    rng = np.random.default_rng(42)
    spatial = rng.exponential(size=(8 * 8 * 8, 2))
    times = rng.normal(size=(60, 2))
    data = (spatial @ times.T + rng.normal(scale=0.1, size=(512, 60)))
    nib.save(nib.Nifti1Image(data.reshape(8, 8, 8, 60).astype(np.float32), np.eye(4)), bold)
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)), mask)

    result = run_melodic_bids(
        source_derivatives_root=source, derivatives_root=output,
        input_bold=bold, brain_mask=mask, n_components=2,
        device="cpu", voxel_batch_size=256, random_state=0,
    )
    assert nib.load(result.components).shape == (8, 8, 8, 2)
    assert nib.load(result.posterior).shape == (8, 8, 8, 2)
    assert nib.load(result.thresholded).shape == (8, 8, 8, 2)
    assert result.mixing.read_text().splitlines()[0] == "melodic_0\tmelodic_1"
    assert json.loads(result.metadata.read_text())["Sources"] == [
        "bids:preproc:sub-01/func/sub-01_task-rest_desc-preproc_bold.nii.gz",
        "bids:preproc:sub-01/func/sub-01_task-rest_desc-brain_mask.nii.gz",
    ]
    description = json.loads((output / "dataset_description.json").read_text())
    assert description["DatasetLinks"] == {"raw": "../raw", "preproc": "../preproc"}
