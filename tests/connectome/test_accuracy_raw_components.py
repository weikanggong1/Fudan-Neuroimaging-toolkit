"""CPU tests for provenance gates and complete-volume comparison definitions."""
import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

MODULE_PATH = (Path(__file__).resolve().parents[2] /
               "validation/connectome/accuracy_20261003/root/compare_raw_components.py")
SPEC = importlib.util.spec_from_file_location("accuracy_raw_components", MODULE_PATH)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def image(tmp_path, name, array, affine=None):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(np.asarray(array), np.eye(4) if affine is None else affine), path)
    return record(path)


def gradients(tmp_path, prefix, vectors, bvalues):
    paths = {}
    for role, data in (("bvecs", vectors), ("bvals", bvalues)):
        path = tmp_path / f"{prefix}_{role}.txt"
        np.savetxt(path, data, fmt="%.17g")
        paths[role] = record(path)
    return paths


def test_streamed_statistics_equal_complete_definition():
    x = np.arange(120, dtype=np.float64).reshape(10, 12)
    y = x * 1.03 - 2
    complete, chunks = compare.Statistics(), compare.Statistics()
    complete.add(x, y)
    for xx, yy in zip(x, y):
        chunks.add(xx, yy)
    a, b = complete.report(), chunks.report()
    assert a["count_all_values"] == b["count_all_values"] == 120
    for key in ("mae", "rmse", "pearson", "relative_rmse_reference_rms", "signed_mean_error"):
        assert a["metrics_all_values"][key] == pytest.approx(b["metrics_all_values"][key], abs=1e-14)
    assert a["metrics_all_values"]["relative_rmse_reference_rms"] == pytest.approx(
        np.linalg.norm(x - y) / np.linalg.norm(y))


def test_nonfinite_never_disappears_from_primary_counts():
    statistics = compare.Statistics()
    statistics.add([1, np.nan, np.inf, -np.inf, 0], [3, np.nan, -np.inf, 2, np.nan])
    row = statistics.report()
    assert row["count_all_values"] == 5
    assert row["finite_pair_count"] == 1
    assert row["nonfinite_state_mismatch"] == 3
    assert row["candidate_nan"] == 1 and row["reference_nan"] == 2
    assert row["candidate_posinf"] == row["candidate_neginf"] == 1
    assert all(value is None for value in row["metrics_all_values"].values())
    assert row["finite_pair_diagnostic"]["rmse"] == 2
    json.dumps(row, allow_nan=False)


def test_zero_denominator_does_not_invent_relative_error():
    statistics = compare.Statistics()
    statistics.add([0, 0], [0, 0])
    assert statistics.report()["metrics_all_values"]["relative_rmse_reference_rms"] is None
    assert statistics.report()["metrics_all_values"]["pearson"] is None


def test_all_frames_and_outside_brain_errors_are_retained(tmp_path):
    reference = np.zeros((2, 3, 4, 5), dtype=np.float32)
    candidate = reference.copy()
    candidate[1, 2, 3, 4] = 7
    candidate[0, 0, 0, 2] = -2
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    mask[0, 0, 0] = 1
    result = compare.compare_image(image(tmp_path, "c.nii.gz", candidate),
                                   image(tmp_path, "r.nii.gz", reference),
                                   image(tmp_path, "m.nii.gz", mask), expected_ndim=4)
    assert result["frames_retained"] == 5
    assert result["whole_volume"]["count_all_values"] == 120
    assert result["whole_volume"]["metrics_all_values"]["max_absolute"] == 7
    assert result["whole_volume"]["metrics_all_values"]["rmse"] == pytest.approx(np.sqrt(53 / 120))
    assert result["official_brain_mask"]["count_all_values"] == 5
    assert result["official_brain_mask"]["metrics_all_values"]["rmse"] == pytest.approx(np.sqrt(4 / 5))
    assert result["per_frame"][4]["max_finite_error_voxel_ijk"] == [1, 2, 3]
    assert result["per_frame"][2]["whole_volume"]["candidate_negative"] == 1


def test_fa_nonfinite_mask_cell_is_counted_and_tail_null(tmp_path):
    reference = np.ones((2, 2, 2), dtype=np.float32)
    reference[1, 1, 1] = np.nan
    candidate = np.ones_like(reference)
    result = compare.compare_image(image(tmp_path, "c.nii", candidate),
                                   image(tmp_path, "r.nii", reference),
                                   image(tmp_path, "m.nii", np.ones_like(reference, dtype=np.uint8)), expected_ndim=3)
    assert result["official_brain_mask"]["count_all_values"] == 8
    assert result["official_brain_mask"]["reference_nan"] == 1
    assert result["official_brain_mask"]["nonfinite_state_mismatch"] == 1
    assert result["per_frame"][0]["official_brain_mask"]["absolute_error_percentiles_all_values"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("difference", ["shape", "affine"])
def test_grid_mismatch_is_not_resampled(tmp_path, difference):
    reference = np.ones((2, 2, 2), dtype=np.float32)
    candidate = np.ones((3, 2, 2), dtype=np.float32) if difference == "shape" else reference
    affine = np.eye(4)
    if difference == "affine":
        affine[0, 3] = 0.01
    result = compare.compare_image(image(tmp_path, "c.nii", candidate, affine),
                                   image(tmp_path, "r.nii", reference),
                                   image(tmp_path, "m.nii", np.ones_like(reference, dtype=np.uint8)), expected_ndim=3)
    assert result["status"] == "not_comparable_grid"
    assert "whole_volume" not in result


def test_official_mask_cannot_be_replaced_by_intersection_or_probability(tmp_path):
    array = np.ones((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="not binary"):
        compare.compare_image(image(tmp_path, "c.nii", array), image(tmp_path, "r.nii", array),
                              image(tmp_path, "mask.nii", array * .5), expected_ndim=3)


def test_all_gradient_frames_zero_support_and_norm_are_checked(tmp_path):
    x = np.zeros((3, 102), dtype=np.float64)
    y = x.copy()
    x[0, 1:] = y[0, 1:] = 1
    x[:, 101] = [0, 2, 0]
    x[:, 1] = 0
    bvals = np.full(102, 1000.0)
    bvals[0] = 0
    result = compare.compare_gradients(gradients(tmp_path, "c", x, bvals),
                                       gradients(tmp_path, "r", y, bvals), 102)
    assert len(result["all_frames"]) == result["frames_retained"] == 102
    assert result["zero_vector_support_mismatch"] == 1
    assert result["all_frames"][101]["candidate_norm"] == 2
    assert result["all_frames"][101]["directed_angle_degrees"] == 90
    assert result["bvec_components"]["count_all_values"] == 306


def test_antipodal_angle_does_not_replace_original_vector_comparison(tmp_path):
    x = np.zeros((3, 4))
    x[0] = 1
    result = compare.compare_gradients(gradients(tmp_path, "c", x, [1000] * 4),
                                       gradients(tmp_path, "r", -x, [1000] * 4), 4)
    assert result["all_frames"][0]["directed_angle_degrees"] == 180
    assert result["all_frames"][0]["antipodal_equivalent_angle_degrees"] == 0
    assert result["bvec_components"]["metrics_all_values"]["max_absolute"] == 2


def test_nonfinite_gradient_frame_retained(tmp_path):
    x = np.zeros((3, 4))
    y = x.copy()
    x[0, 3] = np.nan
    result = compare.compare_gradients(gradients(tmp_path, "c", x, [0] * 4),
                                       gradients(tmp_path, "r", y, [0] * 4), 4)
    assert len(result["all_frames"]) == 4
    assert result["all_frames"][3]["finite_all_components_bvals"] is False
    assert result["bvec_components"]["candidate_nan"] == 1
    assert result["bvec_components"]["metrics_all_values"]["rmse"] is None
    json.dumps(result, allow_nan=False)


def test_wrong_gradient_frame_count_rejected(tmp_path):
    files = gradients(tmp_path, "a", np.zeros((3, 5)), [0] * 5)
    with pytest.raises(ValueError, match="exactly every DWI frame"):
        compare.compare_gradients(files, files, 4)


def test_sha_conflict_and_after_read_mutation_rejected(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("abc")
    initial = record(path)
    audit = compare.Audit()
    audit.check(initial)
    path.write_text("def")
    with pytest.raises(ValueError, match="conflicting SHA"):
        audit.check(record(path))
    with pytest.raises(ValueError, match="changed during comparison"):
        audit.finish()


def test_wrong_namespace_rejected_before_reading_candidate(tmp_path):
    config = {"run_root": str(tmp_path)}
    row = {"gpu_report": str(tmp_path / "unrelated/gpu_report.json"),
           "wall_report": str(tmp_path / "unrelated/raw_bids_wall.json")}
    with pytest.raises(ValueError, match="outside the explicit planned job"):
        compare.candidate_inputs(config, {"case_id": "sub-CON03"}, {}, "candidate", row, compare.Audit())


def test_actual_official_manifest_uses_state_execution_completed(tmp_path):
    path = tmp_path / "reference.json"
    # This is a protocol unit fixture, not a scientific producer or result.
    path.write_text(json.dumps({"state": "completed", "execution_completed": True,
                                "case_id": "sub-CON03", "commands": [], "completed_commands": []}))
    with pytest.raises(ValueError, match="commands are absent"):
        compare.official_inputs({"official_reference_manifest": record(path)},
                                {"case_id": "sub-CON03", "input_files": []}, {}, {}, compare.Audit())


def test_path_alias_with_conflicting_hashes_rejected(tmp_path):
    actual = tmp_path / "file.txt"
    actual.write_text("source")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(actual)
    audit = compare.Audit()
    audit.check(record(actual))
    with pytest.raises(ValueError, match="conflicting SHA"):
        audit.check({"path": str(alias), "sha256": "0" * 64})


def test_pending_and_unscheduled_pairs_never_become_scientific_results(tmp_path, monkeypatch):
    # Protocol-only fixture: there are no completed producers, MRI files or solvers.
    phase = tmp_path / "new_phase"
    frozen, runs = phase / "frozen", phase / "actual_runs"
    frozen.mkdir(parents=True)
    runs.mkdir()
    case = {"case_id": "sub-CON03", "input_files": []}
    manifest = frozen / "manifest.json"
    manifest.write_text("{}")
    bindings = frozen / "bindings.json"
    bindings.write_text(json.dumps({"cases": {"sub-CON03": {
        "anatomy": {"directory": str(tmp_path / "provided_anatomy")},
        "official_reference_manifest": {"path": str(tmp_path / "official/reference.json"), "sha256": "0" * 64},
    }}}))
    config = {"run_root": str(runs), "raw_manifest": record(manifest), "input_bindings": record(bindings),
              "sources": {name: str(frozen / name) for name in ("baseline", "candidate")},
              "declared_source_manifests": {},
              "execution_order": [{"version": "candidate", "case_id": "sub-CON03"}]}
    configuration = frozen / "accuracy_configuration.json"
    configuration.write_text(json.dumps(config))
    (runs / "configuration.json").write_text(json.dumps({**config, "frozen_sources": {}}))
    state = {"configuration": record(configuration), "execution_order": config["execution_order"],
             "raw_manifest": config["raw_manifest"], "input_bindings": config["input_bindings"], "cases": {}}
    (runs / "status.json").write_text(json.dumps(state))
    monkeypatch.setattr(compare.cohort, "validate_manifest", lambda _: [case])
    result = compare.execute(configuration, record(configuration)["sha256"], phase / "new_comparison",
                             ["sub-CON03"], ["baseline", "candidate"])
    assert result["status"] == "compared_subset"
    assert result["scientific_parity"] == "not_assessed"
    assert result["cases"] == {}
    pending = result["coverage"]["not_compared"]
    assert pending["candidate/sub-CON03"]["status"] == "not_launched"
    assert pending["baseline/sub-CON03"]["status"] == "not_scheduled"
    assert result["coverage"]["actually_compared"] == []


def test_exact_frozen_empty_source_marker_is_valid(tmp_path):
    source = tmp_path / "__init__.py"
    source.write_bytes(b"")
    source_record = record(source)
    audit = compare.Audit()
    actual = audit.check(source_record, allow_empty_source=True)
    assert actual["size_bytes"] == 0
    assert actual["sha256"] == hashlib.sha256(b"").hexdigest()
    audit.finish()


def test_allow_empty_source_does_not_allow_wrong_sha_or_missing_file(tmp_path):
    source = tmp_path / "__init__.py"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="file bytes changed"):
        compare.Audit().check({"path": str(source), "sha256": "0" * 64}, allow_empty_source=True)
    with pytest.raises(ValueError, match="missing absolute bound file"):
        compare.Audit().check({"path": str(tmp_path / "absent.py"), "sha256": hashlib.sha256(b"").hexdigest()},
                              allow_empty_source=True)


def test_empty_producer_stays_rejected_even_after_source_audit(tmp_path):
    source = tmp_path / "__init__.py"
    source.write_bytes(b"")
    source_record = record(source)
    audit = compare.Audit()
    with pytest.raises(ValueError, match="empty/missing producer"):
        audit.check(source_record)
    audit.check(source_record, allow_empty_source=True)
    with pytest.raises(ValueError, match="empty/missing producer"):
        audit.check(source_record)


def test_empty_source_after_read_mutation_stays_rejected(tmp_path):
    source = tmp_path / "__init__.py"
    source.write_bytes(b"")
    audit = compare.Audit()
    audit.check(record(source), allow_empty_source=True)
    source.write_text("# mutated\n")
    with pytest.raises(ValueError, match="changed during comparison"):
        audit.finish()


def test_original_undefined_channel_spacing_is_json_safe_metadata(tmp_path):
    array = np.ones((2, 3, 4, 5), dtype=np.float32)
    original = nib.Nifti1Image(array, np.eye(4))
    original.header["pixdim"][4] = np.nan
    path = tmp_path / "original_5tt.nii"
    nib.save(original, path)
    image = nib.load(path)
    result = compare.geometry(image, nonspatial_axis_type="tissue_channel")
    assert result["spacing"] == [1.0, 1.0, 1.0, None]
    assert result["nonfinite_spacing_axes"] == [3]
    assert result["undefined_spacing"] == [{"axis": 3, "axis_type": "tissue_channel", "stored_header_value": "nan"}]
    assert result["affine_nonfinite_count"] == 0
    assert compare.geometry_check(image, image)["same_grid"] is True
    assert np.isnan(image.header.get_zooms()[3])
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), array)
    json.dumps(result, allow_nan=False)


def test_invalid_affine_stays_explicitly_not_comparable_and_json_safe():
    from types import SimpleNamespace
    affine = np.eye(4)
    affine[0, 3] = np.inf
    image = SimpleNamespace(shape=(2, 3, 4), affine=affine,
                            header=SimpleNamespace(get_zooms=lambda: (1.0, 1.0, 1.0)),
                            get_data_dtype=lambda: np.dtype("float32"))
    result = compare.geometry_check(image, image)
    assert result["same_grid"] is False
    assert result["candidate"]["affine"][0][3] is None
    assert result["candidate"]["affine_nonfinite_count"] == 1
    assert result["affine_max_absolute_mm"] is None
    json.dumps(result, allow_nan=False)
