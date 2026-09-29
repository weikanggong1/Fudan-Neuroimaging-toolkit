import nibabel as nib
import numpy as np
import pytest

from fnit.eddy import EDDYConfig, EDDYResult, TorchEDDY
from fnit.eddy.topup_field import _load_topup_field


def test_public_eddy_uses_source_aligned_backend():
    model = TorchEDDY(device="cpu")
    assert isinstance(model.config, EDDYConfig)
    assert model.config.niter == 8
    assert model.config.fwhm_mm == (10, 8, 4, 2, 0, 0, 0, 0)


def test_eddy_accepts_absent_topup_field():
    field, derivative, voxel_sizes = _load_topup_field(None, (4, 5, 6), "cpu", 1)
    assert field.shape == (4, 5, 6)
    assert not field.any() and not derivative.any()
    assert voxel_sizes is None


def test_eddy_rejects_mask_on_different_grid(tmp_path):
    dwi = tmp_path / "dwi.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), np.eye(4)), dwi)
    nib.save(nib.Nifti1Image(np.ones((3, 4, 4), np.float32), np.eye(4)), mask)
    np.savetxt(tmp_path / "acqp.txt", [[0, -1, 0, 0.05]])
    np.savetxt(tmp_path / "index.txt", [[1, 1]], fmt="%d")
    np.savetxt(tmp_path / "bvecs", np.zeros((3, 2)))
    np.savetxt(tmp_path / "bvals", [[0, 1000]])
    with pytest.raises(ValueError, match="mask and DWI grids"):
        TorchEDDY(device="cpu")(
            imain=dwi, mask=mask, acqp=tmp_path / "acqp.txt",
            index=tmp_path / "index.txt", bvecs=tmp_path / "bvecs",
            bvals=tmp_path / "bvals", topup=None,
        )


def test_eddy_result_saves_fsl_sidecars(tmp_path):
    image = nib.Nifti1Image(np.zeros((2, 2, 3, 4), np.float32), np.eye(4))
    result = EDDYResult(
        corrected=image,
        rotated_bvecs=np.zeros((3, 4)),
        parameters=np.zeros((4, 16)),
        movement_rms=np.zeros((4, 2)),
        restricted_movement_rms=np.zeros((4, 2)),
        outlier_map=np.zeros((4, 3), dtype=int),
        outlier_n_stdev_map=np.zeros((4, 3)),
        outlier_n_sqr_stdev_map=np.zeros((4, 3)),
        outlier_report_lines=[],
        qc={},
    )
    paths = result.save(tmp_path / "data")
    assert (tmp_path / "data.nii.gz") in paths
    assert np.loadtxt(tmp_path / "data.eddy_parameters").shape == (4, 16)
    assert np.loadtxt(tmp_path / "data.eddy_outlier_map", skiprows=1).shape == (4, 3)
