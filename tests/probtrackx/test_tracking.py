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
    tracking = np.zeros((9, 5, 5), dtype=np.uint8)
    tracking[:6] = 1
    tracking_path = tmp_path / "cuda-mask.nii.gz"
    nib.save(nib.Nifti1Image(tracking, affine), tracking_path)
    clipped = tracker.run(samples, tmp_path / "cuda-clipped", seed=left,
                          mask=tracking_path)
    assert np.asarray(nib.load(clipped.paths).dataobj)[6, 2, 2] == 6


def test_tracking_mask_and_distthresh_match_half_path_rules(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, 4)
    tracking = np.zeros((9, 5, 5), dtype=np.uint8)
    tracking[:6] = 1
    tracking_path = tmp_path / "tracking.nii.gz"
    nib.save(nib.Nifti1Image(tracking, affine), tracking_path)
    result = TorchProbtrackX(nsamples=8, nsteps=40, steplength=1,
                             seed=7).run(samples, tmp_path / "clipped",
                                         seed=seed, mask=tracking_path)
    density = np.asarray(nib.load(result.paths).dataobj)
    assert density[4, 2, 2] == 8
    assert density[6, 2, 2] == 8  # FSL records one boundary point before stopping
    assert not np.any(density[7:])
    rejected = TorchProbtrackX(nsamples=8, nsteps=40, steplength=1,
                                distthresh=100, seed=7).run(
                                    samples, tmp_path / "short", seed=seed,
                                    mask=tracking_path)
    assert rejected.accepted_streamlines == 0
    assert not np.any(np.asarray(nib.load(rejected.paths).dataobj))



@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stop_and_avoid_match_fsl_straight_field(tmp_path, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, 4)
    barrier = _roi(tmp_path / "barrier.nii.gz", affine, 6)
    for option, last_x in (("stop", 6), ("avoid", 4)):
        result = TorchProbtrackX(device=device, nsamples=8, nsteps=40,
                                 steplength=1, seed=7).run(
                                     samples, tmp_path / option, seed=seed,
                                     **{option: barrier})
        density = np.asarray(nib.load(result.paths).dataobj)
        np.testing.assert_array_equal(density[:, 2, 2],
                                      [8 if x <= last_x else 0 for x in range(9)])
        assert int(result.waytotal.read_text().strip()) == 8



@pytest.mark.parametrize("option,first,last_x", [
    ("stop", False, 8),
    ("stop", True, 8),
    ("avoid", False, -1),
    ("avoid", True, 8),
])
def test_constraint_at_seed_matches_fsl_first_step_rules(tmp_path, option, first, last_x):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, 4)
    result = TorchProbtrackX(nsamples=8, nsteps=40, steplength=1,
                             seed=7).run(samples, tmp_path / "paths", seed=seed,
                                         forcefirststep=first, **{option: seed})
    density = np.asarray(nib.load(result.paths).dataobj)
    if option == "stop" and not first:
        expected = [8 if x >= 4 else 0 for x in range(9)]
    elif option == "avoid" and first:
        expected = [8 if x >= 4 else 0 for x in range(9)]
    else:
        expected = [8 if x <= last_x else 0 for x in range(9)]
    np.testing.assert_array_equal(density[:, 2, 2], expected)
    assert int(result.waytotal.read_text().strip()) == (0 if last_x < 0 else 8)


def test_sampvox_draws_seed_offsets_inside_physical_sphere():
    tracker = TorchProbtrackX(sampvox=1.2, seed=11)
    tracker._voxel_size = torch.tensor([2.0, 1.5, 2.5])
    starts = torch.zeros((100, 3), dtype=torch.float32)
    a = tracker._jitter_seed(starts, torch.Generator().manual_seed(11))
    b = tracker._jitter_seed(starts, torch.Generator().manual_seed(11))
    torch.testing.assert_close(a, b)
    assert torch.all(torch.linalg.vector_norm(a * tracker._voxel_size, dim=1) <= 1.2)
    assert torch.any(a != 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fibst_selects_second_posterior_fibre(tmp_path, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    samples, affine = _field(tmp_path)
    for key, value in (("th", np.pi / 2), ("ph", np.pi / 2), ("f", 0.3)):
        data = np.full((9, 5, 5, 3), value, dtype=np.float32)
        nib.save(nib.Nifti1Image(data, affine),
                 samples / f"merged_{key}2samples.nii.gz")
    seed = _roi(tmp_path / "seed.nii.gz", affine, 4)
    result = TorchProbtrackX(device=device, nsamples=8, nsteps=40, steplength=1,
                             fibst=2, seed=7).run(samples, tmp_path / "fiber2", seed=seed)
    density = np.asarray(nib.load(result.paths).dataobj)
    assert density[4, 2, 2] == 8
    assert density[4, 1, 2] == 8
    assert density[2, 2, 2] == 0
    with pytest.raises(ValueError, match="fibst"):
        TorchProbtrackX(fibst=3, nsamples=1, nsteps=2).run(
            samples, tmp_path / "invalid-fibst", seed=seed)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_usef_zero_fraction_stops_after_seed_record(tmp_path, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        pytest.importorskip("triton")
    samples, affine = _field(tmp_path)
    zero = np.zeros((9, 5, 5, 3), dtype=np.float32)
    nib.save(nib.Nifti1Image(zero, affine), samples / "merged_f1samples.nii.gz")
    seed = _roi(tmp_path / "seed.nii.gz", affine, 4)
    result = TorchProbtrackX(device=device, nsamples=6, nsteps=40,
                             usef=True, seed=7).run(samples, tmp_path / "usef", seed=seed)
    density = np.asarray(nib.load(result.paths).dataobj)
    assert result.accepted_streamlines == 6
    assert density[4, 2, 2] == 6
    assert np.count_nonzero(density) == 1


@pytest.mark.parametrize("mode,expected_second", [(1, 0.5), (2, 0.2), (3, 0.5)])
def test_randfib_starting_population_weights(mode, expected_second):
    tracker = TorchProbtrackX(randfib=mode)
    fraction = torch.tensor([[0.8, 0.2]]).repeat(10000, 1)
    selected = tracker._starting_fibres(fraction, torch.Generator().manual_seed(17))
    assert abs((selected == 1).float().mean().item() - expected_second) < 0.03
    forced = TorchProbtrackX(randfib=mode, fibst=2)
    assert torch.all(forced._starting_fibres(fraction,
                     torch.Generator().manual_seed(17)) == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_randfib_population_weights(tmp_path):
    pytest.importorskip("triton")
    samples, _ = _field(tmp_path)
    first = nib.load(samples / "merged_f1samples.nii.gz")
    nib.save(nib.Nifti1Image(np.full(first.shape, 0.8, np.float32), first.affine),
             samples / "merged_f1samples.nii.gz")
    for key, value in (("th", np.pi / 2), ("ph", np.pi / 2), ("f", 0.2)):
        nib.save(nib.Nifti1Image(np.full(first.shape, value, np.float32), first.affine),
                 samples / f"merged_{key}2samples.nii.gz")
    for mode, expected in ((1, 0.5), (2, 0.2), (3, 0.5)):
        tracker = TorchProbtrackX(device="cuda:0", nsamples=1000, nsteps=20,
                                  randfib=mode)
        tracker._load_samples(samples)
        starts = torch.tensor([[4., 2., 2.]], device="cuda:0").repeat(1000, 1)
        _, first_direction = tracker._walk(
            starts, torch.Generator(device="cuda:0").manual_seed(17))
        selected_second = (first_direction[:, 1].abs() >
                           first_direction[:, 0].abs()).float().mean().item()
        assert abs(selected_second - expected) < 0.07
