"""CPU source/receipt tests; fixtures are protocol records, not MRI results."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("component_collect", ROOT / "validation/connectome/accuracy_20261003/root/collect_raw_component_evidence.py")
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)
FIXTURE_SPEC = importlib.util.spec_from_file_location("component_protocol_fixture", Path(__file__).with_name("test_accuracy_component_controller.py"))
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=3) + "\n")
    return collect.control.file_record(path)


@pytest.fixture
def completed_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    phase = tmp_path / "phase"
    phase.mkdir()
    helper_path = phase / "runtime/validation/connectome/accuracy/root/compare_raw_components.py"
    helper_path.parent.mkdir(parents=True)
    helper_path.write_text("# protocol fixture helper\n")
    helper = collect.control.file_record(helper_path)
    controller = collect.control.file_record(collect.control.__file__)
    config = fixtures.configuration()
    config.update(run_root=str(phase / "raw_run"), sources={"baseline": str(phase / "source_baseline"), "candidate": str(phase / "source_candidate")},
                  raw_manifest={"path": str(phase / "raw_manifest.json"), "sha256": "a" * 64},
                  input_bindings={"path": str(phase / "bindings.json"), "sha256": "b" * 64})
    config_path = phase / "frozen/configuration.json"
    configuration = save(config_path, config)
    analysis = phase / "analysis"
    analysis.mkdir()
    key = "candidate/sub-CON03"
    report_record, producer_row, report = fixtures.report(phase, key, helper, configuration)
    summary = save(analysis / "case_summary.json", {"original_report": report_record, "components": collect.control.summary_case(report, key)})
    row = {"key": key, "status": "completed", "returncode": 0, "error": None,
           "environment": {"CUDA_VISIBLE_DEVICES": ""}, "configuration": configuration, "controller": controller,
           "helper_before": helper, "helper_after": helper, "producer_completed_before": producer_row,
           "producer_completed_after": producer_row, "report": report_record, "summary": summary}
    receipt = save(analysis / "receipt.json", row)
    row["receipt_file"] = receipt
    state = {"configuration": configuration, "helper": helper, "controller": controller, "cases": {key: row}, "all10_summary": None}
    save(analysis / "analysis_configuration.json", {"configuration": configuration, "helper": helper, "controller": controller})
    save(analysis / "status.json", state)
    producer_state = {"configuration": configuration, "execution_order": config["execution_order"],
                      "raw_manifest": config["raw_manifest"], "input_bindings": config["input_bindings"], "cases": {key: producer_row}}
    save(phase / "raw_run/status.json", producer_state)
    return {"phase": phase, "analysis": analysis, "configuration": configuration, "helper": helper,
            "state": state, "producer": producer_state, "key": key, "row": row}


def execute(fixture, *, case_ids=("sub-CON03",), destination=None):
    return collect.collect(fixture["analysis"], fixture["configuration"]["path"], fixture["configuration"]["sha256"],
                           fixture["helper"]["sha256"], destination or fixture["phase"] / "collected", case_ids)


def test_completed_original_reports_are_copied_byte_for_byte(completed_protocol):
    f = completed_protocol
    result = execute(f)
    assert result["coverage"] == ["candidate/sub-CON03"] and result["scientific_parity"] == "not_assessed"
    for copy in result["byte_copies"]:
        assert Path(copy["original"]["path"]).read_bytes() == Path(copy["collected"]["path"]).read_bytes()
        assert copy["original"]["sha256"] == copy["collected"]["sha256"]
    fa = result["cases"][f["key"]]["components"]["FA"]["official_brain_mask"]
    assert fa["metrics_all_values"]["rmse"] is None
    assert fa["finite_pair_diagnostic"] == {"count": 3, "rmse": .04}


def test_unfinished_analysis_cannot_create_an_output(completed_protocol):
    f = completed_protocol
    f["state"]["cases"][f["key"]]["status"] = "running"
    save(f["analysis"] / "status.json", f["state"])
    with pytest.raises(ValueError, match="analysis not completed"):
        execute(f)
    assert not (f["phase"] / "collected").exists()


def test_failed_actual_producer_cannot_reuse_a_completed_analysis(completed_protocol):
    f = completed_protocol
    f["producer"]["cases"][f["key"]]["status"] = "failed"
    save(f["phase"] / "raw_run/status.json", f["producer"])
    with pytest.raises(ValueError, match="actual producer"):
        execute(f)
    assert not (f["phase"] / "collected").exists()


def test_modified_case_summary_cannot_be_reinterpreted_as_original(completed_protocol):
    f = completed_protocol
    path = Path(f["row"]["summary"]["path"])
    summary = json.loads(path.read_text())
    summary["components"]["FA"]["official_brain_mask"]["metrics_all_values"]["rmse"] = 0.0
    f["row"]["summary"] = save(path, summary)
    f["row"]["receipt_file"] = save(f["analysis"] / "receipt.json", {k: v for k, v in f["row"].items() if k != "receipt_file"})
    save(f["analysis"] / "status.json", f["state"])
    with pytest.raises(ValueError, match="saved case summary differs"):
        execute(f)
    assert not (f["phase"] / "collected").exists()


def test_all_ten_collection_requires_real_all_ten_controller_summary(completed_protocol):
    f = completed_protocol
    with pytest.raises(ValueError, match="all-ten candidate summary is not completed"):
        execute(f, case_ids=None)
    assert not (f["phase"] / "collected").exists()


def test_collection_cannot_overwrite_actual_producer_namespace(completed_protocol):
    f = completed_protocol
    with pytest.raises(ValueError, match="overlaps"):
        execute(f, destination=f["phase"] / "raw_run")


def test_wrong_helper_sha_rejects_before_reading_numerical_reports(completed_protocol):
    f = completed_protocol
    with pytest.raises(ValueError, match="unexpected helper SHA"):
        collect.collect(f["analysis"], f["configuration"]["path"], f["configuration"]["sha256"],
                        "0" * 64, f["phase"] / "collected", ["sub-CON03"])
    assert not (f["phase"] / "collected").exists()
