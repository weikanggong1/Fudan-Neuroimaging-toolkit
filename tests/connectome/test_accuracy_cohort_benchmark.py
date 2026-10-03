"""Reject incomplete or duplicated accuracy cohorts before starting MRI work."""
import hashlib
import json

import pytest


def tool():
    from tools import benchmark_connectome_accuracy_cohort as module
    return module


def plan(module, monkeypatch):
    cases = [{"case_id": f"case-{number}"} for number in range(10)]
    monkeypatch.setattr(module.cohort, "validate_manifest", lambda manifest: manifest["cases"])
    order = [{"version": "candidate", "case_id": case["case_id"]} for case in cases]
    order.insert(0, {"version": "baseline", "case_id": "case-0"})
    return {"sources": {"baseline": "/frozen/old", "candidate": "/frozen/new"},
            "execution_order": order}, {"cases": cases}, {"cases": {case["case_id"]: {} for case in cases}}


def test_accuracy_plan_covers_ten_candidates_and_matched_timing(monkeypatch):
    module = tool()
    config, manifest, bindings = plan(module, monkeypatch)
    assert len(module.validate_plan(config, manifest, bindings)) == 10
    config["execution_order"].pop()
    with pytest.raises(ValueError, match="all ten"):
        module.validate_plan(config, manifest, bindings)


def test_accuracy_plan_rejects_duplicate_as_independent_repeat(monkeypatch):
    module = tool()
    config, manifest, bindings = plan(module, monkeypatch)
    config["execution_order"].append(dict(config["execution_order"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_plan(config, manifest, bindings)


def test_accuracy_plan_rejects_wrong_case_or_source(monkeypatch):
    module = tool()
    config, manifest, bindings = plan(module, monkeypatch)
    config["execution_order"][0]["case_id"] = "unrelated"
    with pytest.raises(ValueError, match="unknown"):
        module.validate_plan(config, manifest, bindings)
    config["sources"].pop("baseline")
    with pytest.raises(ValueError, match="baseline"):
        module.validate_plan(config, manifest, bindings)


def test_accuracy_bound_report_rejects_mutated_bytes(tmp_path):
    module = tool()
    path = tmp_path / "producer.json"
    path.write_text(json.dumps({"status": "completed"}))
    identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert module.bound(identity)["status"] == "completed"
    path.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="identity differs"):
        module.bound(identity)


def test_analysis_counts_undefined_as_unassessed():
    from tools import analyze_connectome_accuracy_cohort as module
    from tools.connectome_repeat_common import envelope
    result = module.decisions([envelope([0.1, 0.2], [0.15, 0.3, None])])
    assert result == {"passed": 1, "failed": 1, "not_assessed": 1, "total": 3}


def test_analysis_refuses_changed_same_run_export(tmp_path):
    from tools import analyze_connectome_accuracy_cohort as module
    exported = tmp_path / "tracks.tck"
    exported.write_bytes(b"actual-output")
    identity = {"size_bytes": exported.stat().st_size,
                "sha256": hashlib.sha256(exported.read_bytes()).hexdigest()}
    wall = {"post_timing_result_export": {"status": "completed", "files": {str(exported): identity}}}
    assert module.verified_export(wall, "tracks.tck") == exported
    exported.write_bytes(b"changed-output")
    with pytest.raises(ValueError, match="changed"):
        module.verified_export(wall, "tracks.tck")


def test_analysis_does_not_fabricate_missing_export():
    from tools import analyze_connectome_accuracy_cohort as module
    with pytest.raises(ValueError, match="incomplete"):
        module.verified_export({}, "tracks.tck")
    with pytest.raises(ValueError, match="exactly one"):
        module.verified_export({"post_timing_result_export": {"status": "completed", "files": {}}}, "tracks.tck")
