"""Count-mode sparse connectomes and per-seed target classification."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.probtrackx import TorchProbtrackX


def _field(tmp_path):
    samples = tmp_path / "bedpostX"
    samples.mkdir()
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    shape = (9, 5, 5)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.uint8), affine),
             samples / "nodif_brain_mask.nii.gz")
    for name, value in (("th", np.pi / 2), ("ph", np.pi), ("f", 1.0)):
        nib.save(nib.Nifti1Image(np.full((*shape, 3), value, dtype=np.float32), affine),
                 samples / f"merged_{name}1samples.nii.gz")
    return samples, affine


def _roi(path, affine, xs):
    data = np.zeros((9, 5, 5), dtype=np.uint8)
    data[list(xs), 2, 2] = 1
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _dot(path):
    triples = np.loadtxt(path, ndmin=2)
    rows, cols = triples[-1, :2].astype(int)
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for row, col, value in triples[:-1]:
        matrix[int(row) - 1, int(col) - 1] = value
    return matrix


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sparse_matrices_and_seed_to_target_counts(tmp_path, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [2, 6])
    line = _roi(tmp_path / "line.nii.gz", affine, range(9))
    left = _roi(tmp_path / "left.nii.gz", affine, [0])
    right = _roi(tmp_path / "right.nii.gz", affine, [8])
    target_list = tmp_path / "targets.txt"
    target_list.write_text("left.nii.gz\nright.nii.gz\n")
    result = TorchProbtrackX(device=device, nsamples=3, nsteps=40, steplength=1,
                             batch_size=2, seed=7).run(
        samples, tmp_path / "out", seed=seed, matrix1=True,
        target2=line, target3=line, targetmasks=target_list)
    np.testing.assert_array_equal(_dot(result.matrix1), [[0, 3], [3, 0]])
    np.testing.assert_array_equal(_dot(result.matrix2), np.full((2, 9), 3))
    expected = np.triu(np.full((9, 9), 6), 1)
    np.testing.assert_array_equal(_dot(result.matrix3), expected)
    np.testing.assert_array_equal(np.loadtxt(result.seed_to_targets_matrix),
                                  np.full((2, 2), 3))
    assert [path.name for path in result.seed_to_targets] == [
        "seeds_to_left.nii.gz", "seeds_to_right.nii.gz"]
    for path in result.seed_to_targets:
        values = np.asarray(nib.load(path).dataobj)
        np.testing.assert_array_equal(values[:, 2, 2], [0, 0, 3, 0, 0, 0, 3, 0, 0])
    lookup = np.asarray(nib.load(result.matrix2_lookup).dataobj)
    np.testing.assert_array_equal(lookup[:, 2, 2], np.arange(1, 10))
    assert len(result.matrix1_coords.read_text().splitlines()) == 2
    assert len(result.matrix2_target_coords.read_text().splitlines()) == 9


def test_matrix3_two_masks_and_independent_distance_threshold(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [4])
    left = _roi(tmp_path / "left.nii.gz", affine, [0, 1])
    right = _roi(tmp_path / "right.nii.gz", affine, [7, 8])
    result = TorchProbtrackX(nsamples=2, nsteps=40, steplength=1,
                             seed=3).run(samples, tmp_path / "out", seed=seed,
                                         target3=left, lrtarget3=right)
    np.testing.assert_array_equal(_dot(result.matrix3), np.full((2, 2), 2))
    assert result.matrix3_target_coords is not None
    two_seeds = _roi(tmp_path / "two_seeds.nii.gz", affine, [2, 6])
    threshold = TorchProbtrackX(nsamples=2, nsteps=40, steplength=1,
                                seed=3).run(samples, tmp_path / "threshold", seed=two_seeds,
                                            matrix1=True, target3=left,
                                            distthresh1=1000, distthresh3=1000)
    assert np.count_nonzero(_dot(threshold.matrix1)) == 0
    assert np.count_nonzero(_dot(threshold.matrix3)) == 0
    assert int(threshold.waytotal.read_text()) == 4


def test_network_probabilities_normalize_source_roi_size(tmp_path):
    samples, affine = _field(tmp_path)
    left = _roi(tmp_path / "left.nii.gz", affine, [2, 3])
    right = _roi(tmp_path / "right.nii.gz", affine, [6])
    result = TorchProbtrackX(nsamples=4, nsteps=40, steplength=1,
                             seed=7).run(samples, tmp_path / "network",
                                         regions=[left, right])
    np.testing.assert_array_equal(np.loadtxt(result.network_matrix), [[0, 8], [4, 0]])
    np.testing.assert_array_equal(np.loadtxt(result.network_probability), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(np.loadtxt(result.network_symmetric), [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="count mode only"):
        TorchProbtrackX(nsamples=1, nsteps=2, pathdist=True).run(
            samples, tmp_path / "invalid", seed=left, matrix1=True)
