"""Volume waypoint and exit-stop rules in the FSL half-path model."""

import nibabel as nib
import numpy as np
import pytest

from fnit.probtrackx import TorchProbtrackX


def _field(tmp_path):
    samples = tmp_path / "bedpostX"
    samples.mkdir()
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    shape = (9, 5, 5)
    nib.save(nib.Nifti1Image(np.ones(shape, np.uint8), affine),
             samples / "nodif_brain_mask.nii.gz")
    for name, value in (("th", np.pi / 2), ("ph", np.pi), ("f", 1.0)):
        nib.save(nib.Nifti1Image(np.full((*shape, 3), value, np.float32), affine),
                 samples / f"merged_{name}1samples.nii.gz")
    return samples, affine


def _roi(path, affine, xs):
    data = np.zeros((9, 5, 5), np.uint8)
    data[list(xs), 2, 2] = 1
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _dot(path):
    triples = np.loadtxt(path, ndmin=2)
    rows, cols = triples[-1, :2].astype(int)
    matrix = np.zeros((rows, cols), dtype=int)
    for row, col, value in triples[:-1]:
        matrix[int(row) - 1, int(col) - 1] = value
    return matrix


def test_waypoint_and_accumulates_across_halves_and_filters_matrix(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [4])
    _roi(tmp_path / "left.nii.gz", affine, [2])
    _roi(tmp_path / "right.nii.gz", affine, [6])
    target = _roi(tmp_path / "all.nii.gz", affine, range(9))
    waylist = tmp_path / "waypoints.txt"
    waylist.write_text("left.nii.gz\nright.nii.gz\n")
    model = TorchProbtrackX(nsamples=3, nsteps=40, steplength=1, seed=7)
    shared = model.run(samples, tmp_path / "shared", seed=seed,
                       waypoints=waylist, target2=target)
    assert shared.accepted_streamlines == 3
    np.testing.assert_array_equal(np.asarray(nib.load(shared.paths).dataobj)[:, 2, 2],
                                  np.full(9, 3))
    np.testing.assert_array_equal(_dot(shared.matrix2), np.full((1, 9), 3))
    separate = model.run(samples, tmp_path / "separate", seed=seed,
                         waypoints=[tmp_path / "left.nii.gz", tmp_path / "right.nii.gz"],
                         onewaycondition=True, target2=target)
    assert separate.accepted_streamlines == 0
    assert not np.any(np.asarray(nib.load(separate.paths).dataobj))
    assert not np.any(_dot(separate.matrix2))


def test_waypoint_or_keeps_only_half_that_crosses_mask(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [4])
    left = _roi(tmp_path / "left.nii.gz", affine, [2])
    result = TorchProbtrackX(nsamples=2, nsteps=40, steplength=1, seed=7).run(
        samples, tmp_path / "or", seed=seed, waypoints=left,
        waycond="OR", onewaycondition=True)
    assert result.accepted_streamlines == 2
    np.testing.assert_array_equal(np.asarray(nib.load(result.paths).dataobj)[:, 2, 2],
                                  [2, 2, 2, 2, 2, 0, 0, 0, 0])


def test_waypoint_order_checks_crossing_sequence():
    masks = np.zeros((2, 9), dtype=bool)
    masks[0, 2] = True
    masks[1, 6] = True
    passed = np.zeros(2, dtype=bool)
    assert TorchProbtrackX._waypoint_status(
        np.array([4, 3, 2]), masks, passed, "AND", True) == 2
    assert TorchProbtrackX._waypoint_status(
        np.array([4, 5, 6]), masks, passed, "AND", True) == 0
    passed[:] = False
    assert TorchProbtrackX._waypoint_status(
        np.array([4, 5, 6]), masks, passed, "AND", True) == 1
    assert TorchProbtrackX._waypoint_status(
        np.array([4, 3, 2]), masks, passed, "AND", True) == 0


def test_wtstop_discards_exit_point_and_allows_seed_exit_once(tmp_path):
    mask = np.zeros((1, 9), dtype=bool)
    mask[0, 3:5] = True
    np.testing.assert_array_equal(TorchProbtrackX._filter_half(
        np.array([2, 3, 4, 5, 6]), None, None, False, mask), [2, 3, 4])
    np.testing.assert_array_equal(TorchProbtrackX._filter_half(
        np.array([3, 4, 5, 3, 4, 5]), None, None, False, mask), [3, 4, 5, 3, 4])
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [4])
    exit_mask = _roi(tmp_path / "exit.nii.gz", affine, [6, 7])
    result = TorchProbtrackX(nsamples=2, nsteps=40, steplength=1, seed=7).run(
        samples, tmp_path / "exit", seed=seed, wtstop=exit_mask)
    np.testing.assert_array_equal(np.asarray(nib.load(result.paths).dataobj)[:, 2, 2],
                                  [2, 2, 2, 2, 2, 2, 2, 2, 0])


def test_wtstop_exit_still_tests_avoid_and_waypoint(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [4])
    wtstop = _roi(tmp_path / "wtstop.nii.gz", affine, [5])
    exit_mask = _roi(tmp_path / "exit.nii.gz", affine, [6])
    model = TorchProbtrackX(nsamples=2, nsteps=40, steplength=1, seed=7)
    rejected = model.run(samples, tmp_path / "avoid_exit", seed=seed,
                         wtstop=wtstop, avoid=exit_mask)
    np.testing.assert_array_equal(np.asarray(nib.load(rejected.paths).dataobj)[:, 2, 2],
                                  [2, 2, 2, 2, 2, 0, 0, 0, 0])
    selected = model.run(samples, tmp_path / "way_exit", seed=seed,
                         wtstop=wtstop, waypoints=exit_mask, onewaycondition=True)
    assert selected.accepted_streamlines == 2
    np.testing.assert_array_equal(np.asarray(nib.load(selected.paths).dataobj)[:, 2, 2],
                                  [0, 0, 0, 0, 2, 2, 0, 0, 0])
    flat_wtstop = np.zeros((1, 9), dtype=bool)
    flat_wtstop[0, 5] = True
    flat_avoid = np.zeros(9, dtype=bool)
    flat_avoid[6] = True
    kept, events = TorchProbtrackX._filter_half(
        np.array([4, 5, 6, 7]), None, None, False, flat_wtstop,
        return_events=True)
    np.testing.assert_array_equal(kept, [4, 5])
    np.testing.assert_array_equal(events, [4, 5, 6])
    assert not len(TorchProbtrackX._filter_half(
        np.array([4, 5, 6, 7]), flat_avoid, None, False, flat_wtstop))


def test_waypoint_options_reject_invalid_combinations(tmp_path):
    samples, affine = _field(tmp_path)
    seed = _roi(tmp_path / "seed.nii.gz", affine, [4])
    model = TorchProbtrackX(nsamples=1, nsteps=2)
    with pytest.raises(ValueError, match="wayorder requires AND"):
        model.run(samples, tmp_path / "bad_order", seed=seed,
                  waycond="OR", wayorder=True)
    with pytest.raises(ValueError, match="waypoints must contain"):
        model.run(samples, tmp_path / "empty", seed=seed, waypoints=[])
