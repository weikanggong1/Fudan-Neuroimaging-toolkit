"""CPU status protocol fixtures; these are not MRI or performance benchmarks."""
import hashlib
import json

import pytest

from tools import analyze_connectome_accuracy_cohort as analysis


def baseline_report(tmp_path, status, case_id="sub-CON03"):
    path = tmp_path / "baseline" / case_id / "gpu_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": status, "version": "baseline", "case_id": case_id}))
    return path


def plan(tmp_path, scheduled=True):
    case = {"case_id": "sub-CON03"}
    return {"run_root": str(tmp_path), "execution_order":
            [{"version": "baseline", "case_id": case["case_id"]}] if scheduled else []}, case


def forbid_completed_run(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("optional incomplete baseline reached wall/source/timing reads")
    monkeypatch.setattr(analysis, "completed_run", forbidden)


@pytest.mark.parametrize("status", ["running", "failed"])
def test_incomplete_baseline_does_not_read_missing_wall(tmp_path, monkeypatch, status):
    config, case = plan(tmp_path)
    path = baseline_report(tmp_path, status)
    if status == "failed":
        (path.parent / "raw_bids_wall.json").write_text("must not read this incomplete wall")
    forbid_completed_run(monkeypatch)
    result = analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)
    assert result["baseline_timing_status"] == "not_assessed"
    assert result["baseline_status"] == status
    assert result["baseline_observation"] == {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert "baseline_raw_dwi_cli_seconds" not in result
    assert "candidate_to_baseline_ratio" not in result
    if status == "running":
        assert not (path.parent / "raw_bids_wall.json").exists()


def test_unscheduled_baseline_ignores_even_an_unrelated_completed_file(tmp_path, monkeypatch):
    config, case = plan(tmp_path, scheduled=False)
    path = baseline_report(tmp_path, "completed")
    path.write_text("this unscheduled receipt must not be read")
    forbid_completed_run(monkeypatch)
    result = analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)
    assert result["baseline_status"] == "not_scheduled"
    assert result["baseline_timing_status"] == "not_assessed"
    assert "baseline_observation" not in result


def test_missing_same_case_baseline_does_not_use_another_case(tmp_path, monkeypatch):
    config, case = plan(tmp_path)
    baseline_report(tmp_path, "completed", case_id="sub-CON01")
    forbid_completed_run(monkeypatch)
    result = analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)
    assert result["baseline_status"] == "not_started"
    assert result["baseline_timing_status"] == "not_assessed"


def test_completed_baseline_uses_strict_same_case_validation(tmp_path, monkeypatch):
    config, case = plan(tmp_path)
    path = baseline_report(tmp_path, "completed")
    calls = []
    def completed(actual_config, actual_case, version, anatomy):
        calls.append((actual_config, actual_case, version, anatomy))
        return {"raw_dwi_cli_total_runtime_seconds": 10.0, "memory_budget": {"status": "observed"}}, {}, path
    monkeypatch.setattr(analysis, "completed_run", completed)
    result = analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)
    assert calls == [(config, case, "baseline", "/anatomy")]
    assert result["baseline_timing_status"] == "assessed"
    assert result["baseline_scope"] == "this_phase_same_raw_case_same_parameters"
    assert result["baseline_raw_dwi_cli_seconds"] == 10.0
    assert result["candidate_to_baseline_ratio"] == 0.8


@pytest.mark.parametrize("error", [ValueError("source differs"), FileNotFoundError("missing completed wall")])
def test_invalid_completed_baseline_is_not_downgraded(tmp_path, monkeypatch, error):
    config, case = plan(tmp_path)
    baseline_report(tmp_path, "completed")
    def invalid(*args):
        raise error
    monkeypatch.setattr(analysis, "completed_run", invalid)
    with pytest.raises(type(error), match=str(error)):
        analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)


def test_malformed_scheduled_report_is_not_downgraded(tmp_path, monkeypatch):
    config, case = plan(tmp_path)
    baseline_report(tmp_path, "running").write_text("invalid json")
    forbid_completed_run(monkeypatch)
    with pytest.raises(json.JSONDecodeError):
        analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)


@pytest.mark.parametrize("payload", [[], {}, {"status": "unknown"}])
def test_missing_or_unknown_status_is_not_downgraded(tmp_path, monkeypatch, payload):
    config, case = plan(tmp_path)
    baseline_report(tmp_path, "running").write_text(json.dumps(payload))
    forbid_completed_run(monkeypatch)
    with pytest.raises(ValueError, match="invalid status"):
        analysis.optional_baseline_timing(config, case, "/anatomy", 8.0)
