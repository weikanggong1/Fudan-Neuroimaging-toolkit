"""BBR geometry and optimization checks with controlled 3D contrast."""

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri.bbr import _BBRCost, _boundary, register_bbr


def _sphere_case(shift=0):
    grid = np.indices((40, 40, 40), dtype=np.float32)
    radius = np.sqrt(sum((grid[axis] - (20 + (shift if axis == 0 else 0))) ** 2 for axis in range(3)))
    white = (radius < 10).astype(np.uint8)
    epi = np.where(radius < 10, 80.0, 100.0).astype(np.float32)
    epi[radius > 17] = 0.0
    affine = np.diag((1.0, 1.0, 1.0, 1.0))
    return nib.Nifti1Image(epi, affine), nib.Nifti1Image(white, affine)


def test_boundary_cost_prefers_correct_alignment():
    epi, _ = _sphere_case(shift=2)
    t1, wm = _sphere_case()
    gm, white, centre = _boundary(wm, t1)
    objective = _BBRCost(epi, gm, white, "cpu")
    correct = np.eye(4)
    correct[0, 3] = 2.0  # neurological NIfTI x is reversed in FSL scaled-mm coordinates
    assert len(gm) > 100
    assert objective(correct) < objective(np.eye(4))


@pytest.mark.parametrize("execution", ["batched", "reference"])
def test_bbr_recovers_shift_and_saves_matrix(tmp_path, execution):
    epi, _ = _sphere_case(shift=2)
    t1, wm = _sphere_case()
    t1 = nib.Nifti1Image(np.asarray(t1.dataobj, dtype=np.int16), t1.affine)
    result = register_bbr(epi, t1, wm, init=np.eye(4), device="cpu", grid_search=False,
                          execution=execution)
    assert result.final_cost < result.initial_cost
    assert abs(result.matrix[0, 3] - 2.0) < 1.5
    assert result.moved.shape == t1.shape
    out = tmp_path / "bbr.nii.gz"
    mat = tmp_path / "bbr.mat"
    result.save(output=out, omat=mat)
    assert nib.load(out).shape == t1.shape
    assert nib.load(out).get_data_dtype() == np.dtype(np.float32)
    np.testing.assert_allclose(np.loadtxt(mat), result.matrix, atol=1e-9)


def test_bbr_requires_matching_wmseg_grid():
    epi, wm = _sphere_case()
    moved = nib.Nifti1Image(np.asarray(wm.dataobj), np.diag((2.0, 1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="same voxel grid"):
        register_bbr(epi, epi, moved, init=np.eye(4), device="cpu")
