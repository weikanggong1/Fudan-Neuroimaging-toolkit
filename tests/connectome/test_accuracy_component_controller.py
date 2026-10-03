"""CPU protocol tests: pending producers cannot become component results."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[2] / "validation/connectome/accuracy_20261003/root/control_raw_components.py"
SPEC = importlib.util.spec_from_file_location("accuracy_component_controller", PATH)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def configuration():
    cases = ["sub-CON01", *[f"sub-CON{index:02}" for index in range(3, 12)]]
    return {"execution_order": [{"version": "baseline", "case_id": "sub-CON01"},
                                *[{"version": "candidate", "case_id": case} for case in cases],
                                {"version": "baseline", "case_id": "sub-CON03"}]}


def report(tmp_path, key, helper, config):
    directory = tmp_path / key.replace("/", "_")
    directory.mkdir()
    gpu = directory / "gpu.json"
    wall = directory / "wall.json"
    gpu.write_text('{"status":"completed"}\n')
    wall.write_text('{"status":"completed"}\n')
    row = {"status": "completed", "gpu_report": str(gpu), "wall_report": str(wall)}
    version, case = key.split("/")
    nonfinite = {"metrics_all_values": {"rmse": None}, "finite_pair_count": 3,
                 "finite_pair_diagnostic": {"count": 3, "rmse": .04}, "candidate_nan": 1, "reference_nan": 2}
    image = {"status": "compared", "whole_volume": nonfinite, "official_brain_mask": nonfinite,
             "per_frame": [{name: {"absolute_error_percentiles_all_values": None}
                            for name in ("whole_volume", "official_brain_mask")} ]}
    data = {"status": "requested_completed_cases_compared", "scientific_parity": "not_assessed",
            "script": helper, "configuration": config,
            "coverage": {"actually_compared": [key], "not_compared": {}, "requested_cases": [case], "requested_versions": [version]},
            "controller_snapshot": {"selected_completed_rows": {key: row}},
            "total_CPU_wall_seconds": .01, "immutable_files_verified_before_after": [record(gpu), record(wall)],
            "cases": {key: {"case_id": case, "version": version, "status": "compared", "scientific_parity": "not_assessed",
                             "comparison_scope": "raw full-chain", "candidate_producer": {"gpu_report": record(gpu), "wall_report": record(wall)},
                             "official_producer": {}, "corrected_dwi": image, "FA": image,
                             "rotated_gradients": {"status": "not_comparable_frame_count"},
                             "ancillary_geometry": {"five_tissue": {"status": "not_comparable_grid"}}}}}
    path = directory / "components.json"
    path.write_text(json.dumps(data) + "\n")
    return record(path), row, data


def test_duplicate_actual_plan_is_rejected():
    config = configuration()
    config["execution_order"].append(config["execution_order"][0])
    with pytest.raises(ValueError, match="duplicate"):
        controller.plan(config)


def test_partial_or_failed_ten_candidates_cannot_generate_summary():
    config = configuration()
    rows = {f"candidate/{item['case_id']}": {"status": "completed"}
            for item in config["execution_order"] if item["version"] == "candidate"}
    rows["candidate/sub-CON11"] = {"status": "analysis_failed"}
    # This deliberately lacks report paths; partial coverage must return before
    # attempting to read files or invent a placeholder numerical result.
    assert controller.collect_all_candidates(config, rows, {}, {}, {}) is None


def test_all10_summary_requires_every_real_completed_report_and_row(tmp_path):
    config = configuration()
    helper = {"path": str(tmp_path / "helper.py"), "sha256": "a" * 64}
    cfg = {"path": str(tmp_path / "configuration.json"), "sha256": "b" * 64}
    rows, producer = {}, {"cases": {}}
    for version, case in controller.plan(config):
        if version == "candidate":
            key = f"{version}/{case}"
            report_record, row, _ = report(tmp_path, key, helper, cfg)
            rows[key] = {"status": "completed", "report": report_record}
            producer["cases"][key] = row
    results = controller.collect_all_candidates(config, rows, producer, helper, cfg)
    assert len(results) == 10 and all(row["components"]["scientific_parity"] == "not_assessed" for row in results.values())
    producer["cases"]["candidate/sub-CON11"]["status"] = "failed"
    with pytest.raises(ValueError, match="completed producer row"):
        controller.collect_all_candidates(config, rows, producer, helper, cfg)


def test_reused_report_requires_original_bytes_and_unchanged_immutable_files(tmp_path):
    helper = {"path": str(tmp_path / "helper.py"), "sha256": "a" * 64}
    cfg = {"path": str(tmp_path / "configuration.json"), "sha256": "b" * 64}
    identity, row, _ = report(tmp_path, "baseline/sub-CON01", helper, cfg)
    assert controller.reuse_report(identity, "baseline/sub-CON01", helper, cfg, row)["scientific_parity"] == "not_assessed"
    Path(row["gpu_report"]).write_text('{"status":"failed"}\n')
    with pytest.raises(ValueError, match="bound bytes changed"):
        controller.reuse_report(identity, "baseline/sub-CON01", helper, cfg, row)


def test_report_helper_identity_cannot_be_relabeled(tmp_path):
    helper = {"path": str(tmp_path / "helper.py"), "sha256": "a" * 64}
    cfg = {"path": str(tmp_path / "configuration.json"), "sha256": "b" * 64}
    identity, row, _ = report(tmp_path, "baseline/sub-CON01", helper, cfg)
    wrong = {**helper, "sha256": "c" * 64}
    with pytest.raises(ValueError, match="helper/configuration identity"):
        controller.checked_report(identity, "baseline/sub-CON01", wrong, cfg, row)


def test_summary_keeps_nonfinite_primary_and_finite_pair_diagnostic_separate(tmp_path):
    helper = {"path": str(tmp_path / "helper.py"), "sha256": "a" * 64}
    cfg = {"path": str(tmp_path / "configuration.json"), "sha256": "b" * 64}
    _, _, result = report(tmp_path, "baseline/sub-CON01", helper, cfg)
    summary = controller.summary_case(result, "baseline/sub-CON01")
    values = summary["FA"]["official_brain_mask"]
    assert values["metrics_all_values"]["rmse"] is None
    assert values["finite_pair_diagnostic"] == {"count": 3, "rmse": .04}
    assert values["candidate_nan"] == 1 and values["reference_nan"] == 2
    assert summary["ancillary_geometry"]["five_tissue"]["status"] == "not_comparable_grid"


def test_controller_rejects_nonempty_cuda_environment_before_reading_files(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES empty"):
        controller.run(SimpleNamespace())
