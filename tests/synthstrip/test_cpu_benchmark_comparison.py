"""The benchmark must reject incompatible grids and distinguish mask errors."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "validation/smri_cpu/strip_sr_20261004/compare_outputs.py"
SPEC = importlib.util.spec_from_file_location("strip_sr_cpu_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def save_image(path, data, affine=None):
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)


def test_changed_geometry_prevents_voxel_metrics(tmp_path):
    reference = tmp_path / "reference.nii.gz"
    candidate = tmp_path / "candidate.nii.gz"
    data = np.ones((2, 3, 4), dtype=np.float32)
    save_image(reference, data)
    affine = np.eye(4)
    affine[0, 3] = 1
    save_image(candidate, data, affine)
    report = comparison.compare(reference, candidate)
    assert not report["valid_direct_comparison"]
    assert "whole_grid" not in report


def test_empty_masks_are_not_reported_as_perfect_brain_extraction(tmp_path):
    reference = tmp_path / "reference.nii.gz"
    candidate = tmp_path / "candidate.nii.gz"
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    save_image(reference, data)
    save_image(candidate, data)
    report = comparison.compare(reference, candidate, mask=True)
    assert not report["both_nonempty"]
    assert report["dice"] is None


def test_mask_boundary_changes_are_visible_in_volume_and_dice(tmp_path):
    reference = tmp_path / "reference.nii.gz"
    candidate = tmp_path / "candidate.nii.gz"
    first = np.zeros((2, 3, 4), dtype=np.uint8)
    second = first.copy()
    first[0] = 1
    second[0] = 1
    second[1, 0, 0] = 1
    save_image(reference, first, np.diag([2., 2., 2., 1.]))
    save_image(candidate, second, np.diag([2., 2., 2., 1.]))
    report = comparison.compare(reference, candidate, mask=True)
    assert report["whole_grid"]["different_voxels"] == 1
    assert report["dice"] == 24 / 25
    assert np.isclose(report["relative_volume_difference"], 1 / 12)
    assert np.isclose(report["candidate_volume_mm3"] - report["reference_volume_mm3"], 8)


def test_npz_float_error_is_not_lost_to_uint8_quantization(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez_compressed(reference, vol_data=np.full((2, 3, 4), 1.1))
    np.savez_compressed(candidate, vol_data=np.full((2, 3, 4), 1.2))
    report = comparison.compare(reference, candidate)
    assert report["valid_direct_comparison"]
    assert np.isclose(report["whole_grid"]["mae"], .1)
    assert report["whole_grid"]["different_voxels"] == 24
