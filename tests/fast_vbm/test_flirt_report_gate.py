"""Tiny saved artifacts test the gate, not registration accuracy or speed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "tools/check_flirt_gpu_parity.py"
_SPEC = importlib.util.spec_from_file_location("flirt_report_gate", _SCRIPT)
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


@pytest.fixture
def paired_directories(tmp_path):
    directories = (tmp_path / "baseline", tmp_path / "optimized")
    qc = {"schedule": "FSL default 8/4/2/1 mm", "angular_search": True,
          "degrees_of_freedom": 12, "cost": "FSL correlation ratio",
          "search_cost": "FSL correlation ratio", "optimizer": "MISCMATHS Brent coordinate search",
          "matrix_coordinate_system": "FSL scaled-mm",
          "matrix_direction": "moving/input-to-fixed/reference",
          "cost_value": 0.25, "cost_evaluations": 9000}
    report = {"schema_version": 1,
              "input_sha256": dict(zip(("moving", "reference", "fsl_matrix", "fsl_moved"),
                                       (value * 64 for value in "abcd"))),
              "dof": 12, "cost": "corratio", "runs": []}
    paired = {
        "world_grid_displacement_mm": {"mean": .1, "median": .1, "p95": .2,
                                        "minimum": .01, "maximum": .3, "rms": .12},
        "warped": {"comparison_mask": "official output foreground; unchanged across paths",
                   "voxels": 26, "pearson_r": .99, "mae": 1., "rmse": 2., "support_dice": .98},
        "grid_header": {"shape_equal": True, "affine_max_abs": 0., "dtype_equal": True,
                        "qform_code_equal": True, "sform_code_equal": True},
        "cost": .25, "cost_evaluations": 9000,
    }
    for condition in ("cold", "warm"):
        report["runs"].append({"condition": condition, "qc": deepcopy(qc),
                               "paired": deepcopy(paired)})
    for directory in directories:
        directory.mkdir()
        (directory / "report.private.json").write_text(json.dumps(report))
        for name in ("cold_0", "warm_1"):
            np.savetxt(directory / f"{name}.mat", np.eye(4), fmt="%.12g")
            image = nib.Nifti1Image(np.arange(27, dtype=np.float32).reshape(3, 3, 3),
                                    np.diag([2, 2, 2, 1]))
            nib.save(image, directory / f"{name}.nii.gz")
    return directories


def _edit_report(directory, change):
    path = directory / "report.private.json"
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))


def test_matching_outputs_pass_and_absent_search_trace_is_explicit(paired_directories):
    result = gate.check_parity(*paired_directories)

    assert result["status"] == "pass"
    assert result["conditions_equal"]
    assert result["angular_search"]["status"] == "not captured"
    assert all(all(checks.values()) for checks in result["runs"].values())


@pytest.mark.parametrize("field,value", (("dof", 6), ("cost", "normmi")))
def test_changed_registration_profile_fails(paired_directories, field, value):
    _edit_report(paired_directories[1], lambda report: report.update({field: value}))

    assert gate.check_parity(*paired_directories)["status"] == "fail"


def test_input_or_initial_matrix_fingerprints_must_match(paired_directories):
    _edit_report(paired_directories[1],
                 lambda report: report["input_sha256"].update(initial_matrix="e" * 64))

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "fail"
    assert not result["conditions_equal"]


@pytest.mark.parametrize("field,value", (("schedule", "fast 4/2 mm"), ("angular_search", False),
                                       ("optimizer", "Adam"), ("matrix_direction", "reference-to-input")))
def test_reduced_or_changed_schedule_is_rejected(paired_directories, field, value):
    _edit_report(paired_directories[1],
                 lambda report: report["runs"][0]["qc"].update({field: value}))

    assert gate.check_parity(*paired_directories)["status"] == "fail"


def test_one_saved_matrix_bit_change_is_rejected(paired_directories):
    matrix = np.eye(4)
    matrix[0, 3] = 1e-12
    np.savetxt(paired_directories[1] / "warm_1.mat", matrix, fmt="%.12g")

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "fail"
    assert not result["runs"]["warm_1"]["matrix_values_bitwise_equal"]


def test_positive_and_negative_zero_voxels_are_distinct(paired_directories):
    path = paired_directories[1] / "cold_0.nii.gz"
    image = nib.load(path)
    data = np.array(image.dataobj)
    data[0, 0, 0] = -0.0
    nib.save(nib.Nifti1Image(data, image.affine, image.header), path)

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "fail"
    assert not result["runs"]["cold_0"]["voxels_bitwise_equal"]


def test_header_change_is_rejected_even_with_equal_values_and_affine(paired_directories):
    path = paired_directories[1] / "cold_0.nii.gz"
    image = nib.load(path)
    image.header["descrip"] = b"changed output header"
    nib.save(image, path)

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "fail"
    assert result["runs"]["cold_0"]["voxels_bitwise_equal"]
    assert result["runs"]["cold_0"]["affine_bitwise_equal"]
    assert not result["runs"]["cold_0"]["header_bitwise_equal"]


@pytest.mark.parametrize("field,value", (("cost_value", .25000000001), ("cost_evaluations", 8999)))
def test_cost_and_evaluation_count_are_gated(paired_directories, field, value):
    _edit_report(paired_directories[1],
                 lambda report: report["runs"][1]["qc"].update({field: value}))

    assert gate.check_parity(*paired_directories)["status"] == "fail"


@pytest.mark.parametrize("group,field,value", (
    ("world_grid_displacement_mm", "mean", .100000001),
    ("world_grid_displacement_mm", "median", .100000001),
    ("world_grid_displacement_mm", "p95", .200000001),
    ("world_grid_displacement_mm", "rms", .120000001),
    ("warped", "pearson_r", .9900000001),
    ("warped", "mae", 1.000000001),
    ("warped", "voxels", 25),
    ("warped", "comparison_mask", "changed mask"),
    ("grid_header", "qform_code_equal", False),
    ("grid_header", "affine_max_abs", 1e-14),
))
def test_changed_reported_fsl_metrics_fail_even_when_artifacts_match(
    paired_directories, group, field, value,
):
    _edit_report(paired_directories[1], lambda report:
                 report["runs"][0]["paired"][group].update({field: value}))

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "fail"
    assert not result["runs"]["cold_0"]["fsl_paired_metrics_unchanged"]
    assert result["runs"]["cold_0"]["voxels_bitwise_equal"]


@pytest.mark.parametrize("group,field", (("world_grid_displacement_mm", "p95"),
                                      ("warped", "pearson_r"), ("grid_header", "dtype_equal")))
def test_missing_paired_metrics_on_both_sides_are_not_an_equality_pass(
    paired_directories, group, field,
):
    for directory in paired_directories:
        _edit_report(directory, lambda report: report["runs"][1]["paired"][group].pop(field))

    assert gate.check_parity(*paired_directories)["status"] == "fail"


def test_reporting_tolerance_is_not_a_runtime_or_wall_time_gate(paired_directories):
    def change(report):
        paired = report["runs"][0]["paired"]
        paired["world_grid_displacement_mm"]["p95"] += 5e-11
        paired["warped"]["pearson_r"] += 5e-13
        paired["warped"]["mae"] += 5e-11
        report["runs"][0]["wall_seconds"] = 12345
    _edit_report(paired_directories[1], change)

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "pass"
    assert "reported benchmark summaries" in result["fsl_paired_scope"]


def test_captured_candidate_hash_order_and_pruning_must_match(paired_directories):
    trace = [{"optimized_candidates": 3, "preoptimized_candidates": 4,
              "optimized_costs": [.25, .26, .27], "matrix_sha256": "f" * 64}]
    for directory in paired_directories:
        _edit_report(directory, lambda report: report.update(profile={"angular_search": trace}))
    assert gate.check_parity(*paired_directories)["status"] == "pass"
    _edit_report(paired_directories[1],
                 lambda report: report["profile"]["angular_search"][0].update(preoptimized_candidates=3))

    assert gate.check_parity(*paired_directories)["status"] == "fail"


@pytest.mark.parametrize("missing", ("report.private.json", "warm_1.mat", "cold_0.nii.gz"))
def test_missing_artifacts_fail_with_json_record(paired_directories, missing):
    (paired_directories[1] / missing).unlink()

    result = gate.check_parity(*paired_directories)

    assert result["status"] == "fail"
    assert result["errors"]


def test_cli_writes_summary_and_nonzero_exit_on_regression(paired_directories, tmp_path, capsys):
    output = tmp_path / "gate.json"
    arguments = ["--baseline-dir", str(paired_directories[0]),
                 "--optimized-dir", str(paired_directories[1]), "--output-json", str(output)]

    assert gate.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert json.loads(output.read_text())["status"] == "pass"
    _edit_report(paired_directories[1], lambda report: report["runs"].pop())
    assert gate.main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
