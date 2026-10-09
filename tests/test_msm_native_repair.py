"""Final-native repair contracts; small mathematical fixtures, not benchmarks."""

import inspect

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.msm._native_repair import _guarded_repair, repair_native_sphere
from fnit.msm.msmsulc import _ico, _native_output_qc, _unfold


def orientation_ratios(points, faces, reference):
    """Independent determinant calculation, also used on the float32 cast."""
    def signed(coordinates):
        xyz = np.asarray(coordinates, dtype=np.float64)[faces]
        return np.einsum("ij,ij->i", np.cross(xyz[:, 1], xyz[:, 2]), xyz[:, 0])
    return signed(points) / signed(reference)


def local_fold(level=1):
    reference, faces = _ico(level)
    points = reference.copy()
    first, second, third = faces[0]
    points[first] = .55 * points[second] + .55 * points[third] - .1 * points[first]
    points[first] *= 100 / np.linalg.norm(points[first])
    assert np.count_nonzero(orientation_ratios(points.astype(np.float32), faces, reference) <= 0) == 1
    return points, faces, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clean_native_is_exact_clone_without_unfold_or_guard(monkeypatch, dtype):
    points, faces = _ico(1)
    # Noncontiguous/read-only inputs exercise the caller ownership contract.
    packed = torch.empty((len(points), 6), dtype=dtype)
    packed[:, ::2] = torch.as_tensor(points, dtype=dtype)
    vertices = packed[:, ::2]
    faces.setflags(write=False)
    points.setflags(write=False)
    original = vertices.clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("a saved-float32 clean native mesh must not be repaired")
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", forbidden)
    monkeypatch.setattr("fnit.msm._native_repair._guarded_repair", forbidden)
    monkeypatch.setattr("fnit.msm._native_patch.harmonic_repair", forbidden)
    monkeypatch.setattr("fnit.msm._native_patch.quantized_joint_repair", forbidden)
    repaired, report = repair_native_sphere(vertices, faces, points)
    assert torch.equal(repaired, original) and torch.equal(vertices, original)
    assert repaired.data_ptr() != vertices.data_ptr()
    assert repaired.dtype == dtype and repaired.device == vertices.device
    assert report["success"] and not report["applied"]
    assert report["attempts"] == []
    assert report["moved_vertices"] == report["unfold_updates"] == report["guard_updates"] == 0
    np.testing.assert_array_equal(points, _ico(1)[0])
    np.testing.assert_array_equal(faces, _ico(1)[1])


def test_guard_each_accepted_move_preserves_good_star_and_improves_local_score():
    points, faces, reference = local_fold()
    saved = points.astype(np.float32).astype(np.float64)
    original_points, original_faces = points.copy(), faces.copy()
    baseline_xyz = reference[faces]
    baseline = np.einsum("ij,ij->i", np.cross(baseline_xyz[:, 1], baseline_xyz[:, 2]), baseline_xyz[:, 0])
    observations = []
    result, report = _guarded_repair(points, faces, baseline,
                                   audit=lambda before, after: observations.append((before, after)))
    assert observations and report["updates"] == len(observations)
    for before, after in observations:
        assert np.isfinite(before).all() and np.isfinite(after).all()
        assert np.all(after[before > 0] > 0)
        before_score = (np.count_nonzero(before <= 0), -float(before.min()))
        after_score = (np.count_nonzero(after <= 0), -float(after.min()))
        assert after_score < before_score
    initial_ratios = orientation_ratios(saved, faces, reference)
    final_ratios = orientation_ratios(result.astype(np.float32), faces, reference)
    assert np.all(final_ratios[initial_ratios > 0] > 0)
    assert np.all(final_ratios > 0) and report["success"]
    assert report["sweeps"] <= report["maximum_sweeps"] == 64
    assert report["newly_folded_previously_positive_faces"] == 0
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(faces, original_faces)


def test_native_repair_preserves_inputs_and_is_idempotent():
    points, faces, reference = local_fold()
    vertices = torch.as_tensor(points)
    copies = vertices.clone(), faces.copy(), reference.copy()
    result, report = repair_native_sphere(vertices, faces, reference)
    final = result.numpy().astype(np.float32)
    assert report["success"] and np.all(orientation_ratios(final, faces, reference) > 0)
    changed = np.count_nonzero(np.any(final != points.astype(np.float32), axis=1))
    assert report["moved_vertices"] == changed > 0
    assert report["attempts"][0]["stage"] == "direct_guard"
    assert report["unfold_updates"] == 0
    assert torch.equal(vertices, copies[0])
    np.testing.assert_array_equal(faces, copies[1])
    np.testing.assert_array_equal(reference, copies[2])
    again, repeated = repair_native_sphere(result, faces, reference)
    assert torch.equal(again, result) and repeated["success"]
    assert not repeated["applied"] and repeated["attempts"] == []
    assert repeated["moved_vertices"] == repeated["unfold_updates"] == repeated["guard_updates"] == 0


def test_float32_fold_cannot_be_certified_by_double_precision(monkeypatch):
    reference, faces = _ico(2)
    points = reference.copy()
    first, second, third = 43, 14, 44
    points[first] = (reference[second] + reference[third]) / 2 + 1e-10 * reference[first]
    points[first] *= 100 / np.linalg.norm(points[first])
    assert np.all(orientation_ratios(points, faces, reference) > 0)
    assert np.count_nonzero(orientation_ratios(points.astype(np.float32), faces, reference) <= 0) == 1
    def no_change(vertices, triangles, *, maximum_sweeps):
        return vertices.clone(), 0
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", no_change)
    tensor = torch.as_tensor(points)
    failed, failure = repair_native_sphere(tensor, faces, reference,
                                           unfold_budgets=(0,), maximum_guard_sweeps=0,
                                           maximum_patch_rings=0, maximum_quantized_sweeps=0)
    assert failure["applied"] and not failure["success"]
    assert failure["remaining_folded_faces"] == 1
    assert np.count_nonzero(orientation_ratios(failed.numpy().astype(np.float32), faces, reference) <= 0) == 1
    repaired, success = repair_native_sphere(tensor, faces, reference, unfold_budgets=(0,))
    assert success["success"] and success["guard_updates"] > 0
    assert np.all(orientation_ratios(repaired.numpy().astype(np.float32), faces, reference) > 0)
    np.testing.assert_array_equal(tensor.numpy(), points)


def test_bounded_attempts_restart_input_and_stop_without_false_success(monkeypatch):
    points, faces, reference = local_fold()
    calls = []
    def no_change(vertices, triangles, *, maximum_sweeps):
        calls.append((maximum_sweeps, vertices.numpy().copy()))
        return vertices.clone(), 0
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", no_change)
    result, report = repair_native_sphere(torch.as_tensor(points), faces, reference,
                                          maximum_guard_sweeps=0, maximum_patch_rings=0,
                                          maximum_quantized_sweeps=0)
    assert [budget for budget, _ in calls] == [128, 512, 1000]
    for _, coordinates in calls:
        np.testing.assert_array_equal(coordinates, points)
    assert not report["success"] and report["remaining_folded_faces"] == 1
    assert report["guard_updates"] == 0
    assert all(attempt["guard"]["sweeps"] == 0 for attempt in report["attempts"])
    np.testing.assert_array_equal(result.numpy(), points.astype(np.float32).astype(np.float64))
    assert inspect.signature(_unfold).parameters["maximum_sweeps"].default == 1000


def test_nonfinite_coarse_outputs_are_failed_attempts_not_success(monkeypatch):
    points, faces, reference = local_fold()
    def nonfinite(vertices, triangles, *, maximum_sweeps):
        invalid = vertices.clone()
        invalid[0, 0] = float("inf")
        return invalid, 1
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", nonfinite)
    result, report = repair_native_sphere(torch.as_tensor(points), faces, reference,
                                          maximum_guard_sweeps=0, maximum_patch_rings=0,
                                          maximum_quantized_sweeps=0)
    assert not report["success"] and report["remaining_folded_faces"] == 1
    assert len(report["attempts"]) == 3
    assert all(attempt["termination"] == "nonfinite_unfold_output" for attempt in report["attempts"])
    assert np.isfinite(result.numpy()).all()
    np.testing.assert_array_equal(result.numpy(), points.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("case", ["wrong_shape", "empty", "float_indices", "negative", "out_of_range", "repeated_corner", "unreferenced_vertex"])
def test_invalid_faces_are_rejected_without_changing_input(case):
    points, faces = _ico(1)
    invalid = faces.copy()
    if case == "wrong_shape": invalid = invalid[:, :2]
    elif case == "empty": invalid = invalid[:0]
    elif case == "float_indices": invalid = invalid.astype(float) + .25
    elif case == "negative": invalid[0, 0] = -1
    elif case == "out_of_range": invalid[0, 0] = len(points)
    elif case == "repeated_corner": invalid[0, 0] = invalid[0, 1]
    else: invalid = invalid[~np.any(invalid == 0, axis=1)]
    vertices = torch.as_tensor(points.copy())
    before = vertices.clone(), invalid.copy(), points.copy()
    with pytest.raises(ValueError):
        repair_native_sphere(vertices, invalid, points)
    assert torch.equal(vertices, before[0])
    np.testing.assert_array_equal(invalid, before[1])
    np.testing.assert_array_equal(points, before[2])


@pytest.mark.parametrize("case", ["shape", "nan", "inf", "degenerate"])
def test_invalid_orientation_reference_is_rejected(case):
    points, faces = _ico(1)
    reference = points.copy()
    if case == "shape": reference = reference[:-1]
    elif case == "nan": reference[0, 0] = np.nan
    elif case == "inf": reference[0, 0] = np.inf
    else: reference[:] = reference[0]
    before = points.copy(), reference.copy()
    with pytest.raises(ValueError):
        repair_native_sphere(torch.as_tensor(points), faces, reference)
    np.testing.assert_array_equal(points, before[0])
    np.testing.assert_array_equal(reference, before[1])


def test_consistent_reversed_face_winding_is_a_valid_reference():
    points, faces = _ico(1)
    reversed_faces = faces[:, ::-1].copy()
    result, report = repair_native_sphere(torch.as_tensor(points), reversed_faces, points)
    assert report["success"] and not report["applied"]
    np.testing.assert_array_equal(result.numpy(), points)


def test_existing_input_fold_is_repaired_and_reported_with_mixed_reference(monkeypatch):
    reference, faces, _ = local_fold()
    vertices = torch.as_tensor(reference.copy())
    copies = vertices.clone(), faces.copy(), reference.copy()
    def no_change(points, triangles, *, maximum_sweeps):
        return points.clone(), 0
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", no_change)
    result, report = repair_native_sphere(vertices, faces, reference, unfold_budgets=(0,))
    assert report["orientation_reference_sign"] == 1
    assert report["reference_folded_faces"] == 1
    assert report["applied"] and report["success"] and report["guard_updates"] > 0
    # Correcting an inward reference face is a relative sign change, while
    # the true absolute output orientation must become consistently outward.
    final = result.numpy().astype(np.float32)
    xyz = np.asarray(final, dtype=np.float64)[faces]
    absolute_signed = np.einsum("ij,ij->i", np.cross(xyz[:, 1], xyz[:, 2]), xyz[:, 0])
    assert np.all(absolute_signed > 0)
    qc = _native_output_qc(result.numpy(), faces, reference)
    assert qc["folded_output_faces"] == 1
    assert qc["absolute_folded_output_faces"] == qc["new_relative_folded_output_faces"] == 0
    assert torch.equal(vertices, copies[0])
    np.testing.assert_array_equal(faces, copies[1])
    np.testing.assert_array_equal(reference, copies[2])


def test_corrected_reference_preserves_relative_metrics_but_passes_absolute_qc():
    folded_reference, faces, correct = local_fold()
    qc = _native_output_qc(correct, faces, folded_reference)
    assert qc["folded_output_faces"] == qc["folded_solver_faces"] == 1
    assert qc["absolute_folded_input_faces"] == 1
    assert qc["absolute_folded_output_faces"] == qc["absolute_folded_solver_faces"] == 0
    assert qc["new_relative_folded_output_faces"] == qc["new_relative_folded_solver_faces"] == 0
    assert qc["minimum_output_orientation_ratio"] < 0
    assert qc["minimum_absolute_output_orientation_ratio"] > 0


def test_corrected_reference_saved_gifti_and_all_stages_pass(tmp_path):
    from fnit.fmri.surface_pipeline import _orientation_chain, _orientation_stage, _saved_native_sphere_qc
    reference, faces, correct = local_fold()
    def save(path, points):
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(points.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"),
        ]), path)
        return path
    reference_path = save(tmp_path / "reference.surf.gii", reference)
    correct_path = save(tmp_path / "corrected.surf.gii", correct)
    inherited = _saved_native_sphere_qc(reference_path, reference_path, baseline="input native sphere")
    assert inherited["folded_output_faces"] == 0
    assert inherited["absolute_folded_output_faces"] == 1
    assert inherited["orientation_qc"] == "warning"
    corrected = _saved_native_sphere_qc(correct_path, reference_path, baseline="input native sphere")
    assert corrected["folded_output_faces"] == 1
    assert corrected["absolute_folded_input_faces"] == 1
    assert corrected["absolute_folded_output_faces"] == corrected["new_relative_folded_output_faces"] == 0
    assert corrected["orientation_qc"] == "pass"
    assert corrected["coordinates"] == "saved GIFTI coordinates"
    assert "absolute_folded_solver_faces" not in corrected
    assert "new_relative_folded_solver_faces" not in corrected
    assert "minimum_absolute_solver_orientation_ratio" not in corrected
    stage = {hemisphere: corrected for hemisphere in "LR"}
    assert _orientation_stage(stage)["status"] == "pass"
    assert _orientation_chain(stage, None, stage)["all_stages"] == "pass"
    # Reports without the additive absolute field retain the legacy meaning.
    legacy = {hemisphere: {"folded_output_faces": 1} for hemisphere in "LR"}
    assert _orientation_stage(legacy)["status"] == "warning"


def test_reference_without_majority_winding_is_rejected():
    points, faces = _ico(1)
    mixed = faces.copy()
    mixed[:len(faces) // 2] = mixed[:len(faces) // 2, ::-1]
    with pytest.raises(ValueError, match="majority winding"):
        repair_native_sphere(torch.as_tensor(points), mixed, points)


@pytest.mark.parametrize("case", ["nan", "inf", "zero_radius", "float32_overflow"])
def test_invalid_vertices_including_saved_precision_overflow_are_rejected(case):
    reference, faces = _ico(1)
    points = reference.copy()
    if case == "nan": points[0, 0] = np.nan
    elif case == "inf": points[0, 0] = np.inf
    elif case == "zero_radius": points[0] = 0
    else: points[0, 0] = 1e40
    before = points.copy()
    with pytest.raises(ValueError):
        repair_native_sphere(torch.as_tensor(points), faces, reference)
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16])
def test_unsupported_tensor_precision_is_rejected(dtype):
    points, faces = _ico(1)
    with pytest.raises(TypeError, match="float32 or float64"):
        repair_native_sphere(torch.tensor(points, dtype=dtype), faces, points)


@pytest.mark.parametrize("options", [{"unfold_budgets": ()}, {"unfold_budgets": (-1,)},
                                      {"unfold_budgets": (1.5,)}, {"maximum_guard_sweeps": -1},
                                      {"maximum_guard_sweeps": 1.5}, {"maximum_patch_rings": -1},
                                      {"maximum_patch_rings": 1.5}, {"maximum_quantized_sweeps": -1},
                                      {"maximum_quantized_sweeps": 1.5}])
def test_invalid_budgets_fail_before_computation(monkeypatch, options):
    points, faces = _ico(1)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid budgets must fail before unfolding")
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", forbidden)
    with pytest.raises(ValueError, match="budgets"):
        repair_native_sphere(torch.as_tensor(points), faces, points, **options)
