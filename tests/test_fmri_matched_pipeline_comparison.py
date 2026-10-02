"""单例对照报告的数值、坐标、网格和隐私控制；不是影像 benchmark。"""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def comparison():
    path = Path(__file__).resolve().parents[1] / "validation/fmri/compare_matched_pipeline.py"
    specification = importlib.util.spec_from_file_location("matched_comparison", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def image(path, data, affine=None):
    result = nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine)
    result.header.set_xyzt_units("mm", "sec")
    if result.ndim == 4:
        result.header.set_zooms((*result.header.get_zooms()[:3], 0.75))
    nib.save(result, path)
    return path


def test_temporal_errors_include_constants_but_correlations_do_not(comparison):
    first = np.array([[1., 2., 3.], [7., 7., 7.]])
    second = np.array([[3., 2., 1.], [10., 10., 10.]])
    metrics = comparison.correlations(first, second, temporal=True, chunk=1)
    assert metrics["valid_voxel_temporal_r"] == 1
    assert metrics["mean_voxel_temporal_r"] == pytest.approx(-1)
    assert metrics["pooled_time_demeaned_r"] == pytest.approx(-1)
    assert metrics["mae"] == pytest.approx(np.abs(first - second).mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.square(first - second).mean()))
    assert metrics["pearson_r"] == pytest.approx(np.corrcoef(first.ravel(), second.ravel())[0, 1])


def test_constant_series_are_null_and_json_finite(comparison):
    metrics = comparison.correlations(np.ones((2, 5)), np.ones((2, 5)), temporal=True)
    assert metrics["valid_voxel_temporal_r"] == 0
    assert metrics["pearson_r"] is None
    assert metrics["pooled_time_demeaned_r"] is None
    assert metrics["median_voxel_temporal_r"] is None
    json.dumps(metrics, allow_nan=False)


def test_nonfinite_outside_comparison_region_is_rejected(comparison, tmp_path):
    array = np.ones((2, 2, 2))
    array[0, 0, 0] = np.nan
    first = image(tmp_path / "first.nii.gz", array)
    second = image(tmp_path / "second.nii.gz", np.ones((2, 2, 2)))
    mask = np.ones((2, 2, 2))
    mask[0, 0, 0] = 0
    mask_path = image(tmp_path / "mask.nii.gz", mask)
    with pytest.raises(ValueError, match="nonfinite"):
        comparison.image_pair({"kind": "scalar", "candidate": first,
                               "reference": second, "mask": mask_path}, tmp_path)


def test_mask_overlap_and_grid_mismatch(comparison, tmp_path):
    left = np.zeros((3, 3, 3))
    right = left.copy()
    left[:2, 0, 0] = 1
    right[1:, 0, 0] = 1
    first = image(tmp_path / "a.nii.gz", left)
    second = image(tmp_path / "b.nii.gz", right)
    specification = {"kind": "mask", "candidate": first, "reference": second}
    report = comparison.image_pair(specification, tmp_path)
    assert report["dice"] == pytest.approx(0.5)
    assert report["jaccard"] == pytest.approx(1 / 3)
    assert report["candidate_only_voxels"] == report["reference_only_voxels"] == 1
    image(second, right, np.diag([2., 1., 1., 1.]))
    with pytest.raises(ValueError, match="grids differ"):
        comparison.image_pair(specification, tmp_path)


def test_fsl_affine_physical_displacement_uses_stored_pixdim(comparison, tmp_path):
    affine = np.array([[2., 0.8, 0., 10.], [0., 3., 0., 20.], [0., 0., 4., -5.], [0., 0., 0., 1.]])
    moving_path = image(tmp_path / "moving.nii.gz", np.ones((3, 4, 5)), affine)
    reference_path = image(tmp_path / "reference.nii.gz", np.ones((3, 4, 5)), affine)
    for path in (moving_path, reference_path):
        nifti = nib.load(path)
        nifti.header.set_zooms((2., 3., 4.))
        nib.save(nifti, path)
    candidate, reference = np.eye(4), np.eye(4)
    reference[1, 3] = 3.
    np.savetxt(tmp_path / "a.mat", candidate)
    np.savetxt(tmp_path / "b.mat", reference)
    report = comparison.affine_pair({"candidate_matrix": "a.mat", "reference_matrix": "b.mat",
                                     "moving": moving_path, "reference": reference_path,
                                     "convention": "fsl"}, tmp_path)
    assert report["inverse_world_displacement"]["rms_mm"] == pytest.approx(np.sqrt(3**2 + 0.8**2))


def test_motion_sequence_checks_all_frames(comparison, tmp_path):
    moving = image(tmp_path / "moving.nii.gz", np.ones((3, 4, 5, 2)))
    reference = image(tmp_path / "ref.nii.gz", np.ones((3, 4, 5)))
    mask = image(tmp_path / "mask.nii.gz", np.ones((3, 4, 5)))
    for name in ("candidate", "reference"):
        (tmp_path / name).mkdir()
        for frame in range(2):
            matrix = np.eye(4)
            if name == "reference" and frame == 1:
                matrix[0, 3] = 2
            np.savetxt(tmp_path / name / f"MAT_{frame:04d}", matrix)
    specification = {"candidate_matrices": "candidate", "reference_matrices": "reference",
                     "moving": moving, "reference": reference, "mask": mask}
    report = comparison.motion_pair(specification, tmp_path)
    assert report["frames"] == 2
    assert report["mean_frame_rms_mm"] == pytest.approx(1)
    assert report["max_frame_rms_mm"] == pytest.approx(2)
    (tmp_path / "reference/MAT_0001").unlink()
    with pytest.raises(ValueError, match="full BOLD frame count"):
        comparison.motion_pair(specification, tmp_path)


def test_pull_semantics_and_composed_world_coordinates(comparison, tmp_path):
    reference = image(tmp_path / "ref.nii.gz", np.ones((3, 4, 5)))
    mask = image(tmp_path / "mask.nii.gz", np.ones((3, 4, 5)))
    first = image(tmp_path / "a.nii.gz", np.zeros((3, 4, 5, 3)))
    displacement = np.zeros((3, 4, 5, 3))
    displacement[..., 1] = 2
    second = image(tmp_path / "b.nii.gz", displacement)
    specification = {"candidate": first, "reference": second, "mni_template": reference, "mask": mask}
    with pytest.raises(ValueError, match="explicit ras_mm"):
        comparison.pull_pair(specification, tmp_path)
    specification["convention"] = "ras_mm_pull_displacement"
    report = comparison.pull_pair(specification, tmp_path)
    assert report["mni_to_t1_world_displacement"]["rms_mm"] == pytest.approx(2)
    transform = np.eye(4)
    transform[1, 1] = 3
    np.savetxt(tmp_path / "pull.txt", transform)
    specification.update(candidate_to_epi_world="pull.txt", reference_to_epi_world="pull.txt")
    report = comparison.pull_pair(specification, tmp_path)
    assert report["mni_to_epi_world_displacement"]["rms_mm"] == pytest.approx(6)


def test_input_pin_and_public_privacy(comparison, tmp_path):
    paths = {"bold": image(tmp_path / "private-sub-123456_bold.nii.gz", np.ones((2, 2, 2, 3))),
             "t1w": image(tmp_path / "private-t1.nii.gz", np.ones((2, 2, 2))),
             "mni_template": image(tmp_path / "template.nii.gz", np.ones((2, 2, 2)))}
    manifest = {"candidate_revision": "abcdef0", "reference_revision": "123abcd",
                "input_files": {key: str(path) for key, path in paths.items()},
                "reference_input_files": {key: str(path) for key, path in paths.items()}}
    report = comparison.compare_manifest(manifest, tmp_path)
    text = json.dumps(report, allow_nan=False)
    assert "private-sub" not in text and str(tmp_path) not in text
    assert all(report["same_raw_inputs_and_resources"].values())
    manifest["expected_input_sha256"] = {"bold": "0" * 64}
    with pytest.raises(ValueError, match="pinned manifest"):
        comparison.compare_manifest(manifest, tmp_path)


def test_component_version_comes_from_actual_binary_record(comparison, tmp_path):
    executable = tmp_path / "melodic"
    executable.write_bytes(b"component-executable")
    version_log = tmp_path / "private-version.log"
    version_log.write_text("MELODIC 2601.1-dirty")
    report = comparison.oracle_provenance({"melodic": {"executable": executable,
                                          "version": "2601.1-dirty", "version_record": version_log}}, tmp_path)
    assert report["melodic"]["version"] == "2601.1-dirty"
    assert report["melodic"]["executable_sha256"] == comparison.sha256(executable)
    assert report["melodic"]["version_record_sha256"] == comparison.sha256(version_log)
    assert str(tmp_path) not in json.dumps(report)


def test_cross_controls_hold_sampler_fixed(comparison, tmp_path):
    array = np.arange(3 * 3 * 3 * 4, dtype=np.float32).reshape(3, 3, 3, 4)
    candidate = image(tmp_path / "candidate.nii.gz", array)
    reference = image(tmp_path / "reference.nii.gz", 2 * array)
    template = image(tmp_path / "template.nii.gz", np.ones((3, 3, 3)))
    mask = image(tmp_path / "mask.nii.gz", np.ones((3, 3, 3)))
    first_pull = image(tmp_path / "first_pull.nii.gz", np.zeros((3, 3, 3, 3)))
    shifted = np.zeros((3, 3, 3, 3))
    shifted[..., 0] = 1
    second_pull = image(tmp_path / "second_pull.nii.gz", shifted)
    transform = tmp_path / "world.txt"
    np.savetxt(transform, np.eye(4))
    specification = {"convention": "ras_mm_pull_displacement",
                     "candidate_native": candidate, "reference_native": reference,
                     "candidate_pull": first_pull, "reference_pull": second_pull,
                     "candidate_to_epi_world": transform, "reference_to_epi_world": transform,
                     "template": template, "mask": mask}
    report = comparison.cross_controls(specification, tmp_path, tmp_path / "private_controls", "cpu")
    assert len(report["pairs"]) == 4
    same_warp = report["pairs"]["same_candidate_warp_cleaning_difference"]["metrics"]
    assert same_warp["mean_voxel_temporal_r"] == pytest.approx(1)
    assert same_warp["rmse"] == pytest.approx(np.sqrt(np.square(array.astype(np.float64)).mean()), rel=1e-6)
    same_clean = report["pairs"]["same_candidate_native_map_difference"]["metrics"]
    assert same_clean["rmse"] > 0
    assert str(tmp_path) not in json.dumps(report, allow_nan=False)
