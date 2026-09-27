"""Synthetic fibre field checks for volume tractography and ROI counting."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.probtrackx import TorchProbtrackX


def _field(tmp_path, *, neurological=False):
    samples = tmp_path / "bedpostX"
    samples.mkdir()
    affine = np.diag([2.0 if neurological else -2.0, 2.0, 2.0, 1.0])
    mask = np.ones((9, 5, 5), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, affine), samples / "nodif_brain_mask.nii.gz")
    for name, value in (("th", np.pi / 2), ("ph", np.pi), ("f", 1.0)):
        data = np.full((*mask.shape, 3), value, dtype=np.float32)
        nib.save(nib.Nifti1Image(data, affine), samples / f"merged_{name}1samples.nii.gz")
    return samples, affine


def _roi(path, affine, x):
    data = np.zeros((9, 5, 5), dtype=np.uint8)
    data[x, 2, 2] = 1
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_seed_to_voxel_follows_straight_posterior_and_preserves_geometry(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, 4)
    result = TorchProbtrackX(nsamples=8, nsteps=40, steplength=1,
                             batch_size=4, seed=7).run(samples, tmp_path / "paths", seed=seed)
    output = nib.load(result.paths)
    density = np.asarray(output.dataobj)
    assert result.accepted_streamlines == 8
    assert density[4, 2, 2] == 8
    assert density[2, 2, 2] == 8
    assert density[6, 2, 2] == 8
    assert np.count_nonzero(density[:, 1, :]) == 0
    np.testing.assert_allclose(output.affine, affine)
    assert int(result.waytotal.read_text().strip()) == 8


def test_network_counts_each_target_once_per_streamline_and_is_directed(tmp_path):
    samples, affine = _field(tmp_path)
    left = _roi(tmp_path / "left.nii.gz", affine, 2)
    right = _roi(tmp_path / "right.nii.gz", affine, 6)
    disconnected = np.zeros((9, 5, 5), dtype=np.uint8)
    disconnected[4, 1, 1] = 1
    third = tmp_path / "third.nii.gz"
    nib.save(nib.Nifti1Image(disconnected, affine), third)
    result = TorchProbtrackX(nsamples=6, nsteps=40, steplength=1,
                             batch_size=3, seed=3).run(
                                 samples, tmp_path / "network", regions=[left, right, third])
    np.testing.assert_array_equal(np.loadtxt(result.network_matrix, dtype=int),
                                  [[0, 6, 0], [6, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(np.loadtxt(result.waytotal, dtype=int), [6, 6, 0])
    assert result.accepted_streamlines == 12
    density = np.asarray(nib.load(result.paths).dataobj)
    assert density[0, 2, 2] == 6
    assert density[8, 2, 2] == 6
    with pytest.raises(FileExistsError):
        TorchProbtrackX(nsamples=1, nsteps=2).run(samples, tmp_path / "network",
                                                     regions=[left, right])


def test_neurological_storage_flip_and_roi_geometry_check(tmp_path):
    samples, affine = _field(tmp_path, neurological=True)
    seed = _roi(tmp_path / "seed.nii.gz", affine, 2)
    result = TorchProbtrackX(nsamples=2, nsteps=20, steplength=1).run(
        samples, tmp_path / "paths", seed=seed)
    output = nib.load(result.paths)
    assert np.asarray(output.dataobj)[2, 2, 2] == 2
    np.testing.assert_allclose(output.affine, affine)
    bad = _roi(tmp_path / "bad.nii.gz", np.eye(4), 2)
    with pytest.raises(ValueError, match="geometry"):
        TorchProbtrackX(nsamples=1, nsteps=2).run(samples, tmp_path / "bad-out", seed=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fused_walk_straight_field_and_network(tmp_path):
    pytest.importorskip("triton")
    samples, affine = _field(tmp_path)
    left = _roi(tmp_path / "left.nii.gz", affine, 2)
    right = _roi(tmp_path / "right.nii.gz", affine, 6)
    tracker = TorchProbtrackX(device="cuda:0", nsamples=6, nsteps=40,
                              steplength=1, batch_size=4, seed=3)
    result = tracker.run(samples, tmp_path / "cuda-network", regions=[left, right])
    np.testing.assert_array_equal(np.loadtxt(result.network_matrix, dtype=int),
                                  [[0, 6], [6, 0]])
    np.testing.assert_array_equal(np.loadtxt(result.waytotal, dtype=int), [6, 6])
    density = np.asarray(nib.load(result.paths).dataobj)
    assert density[4, 2, 2] == 12
