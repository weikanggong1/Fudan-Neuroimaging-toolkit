"""The UKB good-voxel rule excludes high-variance ribbon voxels."""

from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pytest

from fnit.fmri import surface_qc


def test_ribbon_goodvoxels_excludes_high_cov_voxel(tmp_path, monkeypatch):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    bold = np.full((9, 9, 9, 8), 100.0, dtype=np.float32)
    bold += np.array([-1, 1] * 4, dtype=np.float32)
    bold[4, 4, 4, :] = np.array([20, 180] * 4, dtype=np.float32)
    bold[0, 0, 0, :] = -100 + np.array([-1, 1] * 4, dtype=np.float32)
    bold_path = tmp_path / "bold.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(bold, affine), bold_path)
    nib.save(nib.Nifti1Image(bold[..., 0], affine), reference)
    native = tmp_path / "surface.surf.gii"
    native.touch()

    def signed_distance(command, **kwargs):
        assert command[1] == "-create-signed-distance-volume"
        data = np.zeros((9, 9, 9), dtype=np.float32)
        if ".white." in Path(command[-1]).name:
            data[2:7, 2:7, 2:7] = 1
        else:
            data[2:7, 2:7, 2:7] = -1
        nib.save(nib.Nifti1Image(data, affine), command[-1])

    monkeypatch.setattr(surface_qc.subprocess, "run", signed_distance)
    result = surface_qc.make_ribbon_goodvoxels(
        clean_bold=bold_path,
        reference=reference,
        left_white=native,
        left_pial=native,
        right_white=native,
        right_pial=native,
        output_dir=tmp_path / "qc",
    )
    ribbon = np.asarray(nib.load(result.ribbon).dataobj)
    good = np.asarray(nib.load(result.goodvoxels).dataobj)
    assert ribbon.sum() == 125
    assert good[4, 4, 4] == 0
    assert good[3, 3, 3] == 1
    assert good[0, 0, 0] == 1  # fslmaths -bin treats a negative mean as nonzero
    assert np.array_equal(nib.load(result.goodvoxels).affine, affine)
    report = json.loads(result.report.read_text())
    expected_cov = np.std(bold[2:7, 2:7, 2:7], axis=3, ddof=1) / 100.0
    assert np.isclose(report["ribbon_cov_mean"], expected_cov.mean(), rtol=1e-6)



def test_fsl_bin_keeps_negative_mean_and_excludes_equal_upper(tmp_path, monkeypatch):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    bold = np.full((9, 9, 9, 8), 100.0, dtype=np.float32)
    bold += np.array([-1, 1] * 4, dtype=np.float32)
    bold[0, 0, 0, :] = -100 + np.array([-1, 1] * 4, dtype=np.float32)
    bold_path = tmp_path / "bold.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(bold, affine), bold_path)
    nib.save(nib.Nifti1Image(bold[..., 0], affine), reference)
    surface = tmp_path / "surface.surf.gii"
    surface.touch()

    def signed_distance(command, **kwargs):
        data = np.zeros((9, 9, 9), dtype=np.float32)
        data[2:7, 2:7, 2:7] = 1 if ".white." in Path(command[-1]).name else -1
        nib.save(nib.Nifti1Image(data, affine), command[-1])

    monkeypatch.setattr(surface_qc.subprocess, "run", signed_distance)
    result = surface_qc.make_ribbon_goodvoxels(
        clean_bold=bold_path, reference=reference,
        left_white=surface, left_pial=surface,
        right_white=surface, right_pial=surface,
        output_dir=tmp_path / "qc", neighborhood_sigma_mm=0.001,
        threshold_factor=0.0,
    )
    good = np.asarray(nib.load(result.goodvoxels).dataobj)
    assert good[0, 0, 0] == 1  # FSL -bin means nonzero, including negative
    assert good[4, 4, 4] == 0  # FSL -thr Upper marks equality as bad
    assert np.isclose(json.loads(result.report.read_text())["normalized_upper_threshold"], 1.0)


def test_ribbon_goodvoxels_rejects_mismatched_grids(tmp_path):
    affine = np.eye(4)
    bold = tmp_path / "bold.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3, 8), dtype=np.float32), affine), bold)
    nib.save(nib.Nifti1Image(np.ones((4, 3, 3), dtype=np.float32), affine), reference)
    with pytest.raises(ValueError, match="same MNI grid"):
        surface_qc.make_ribbon_goodvoxels(
            bold, reference, bold, bold, bold, bold, tmp_path / "qc"
        )
