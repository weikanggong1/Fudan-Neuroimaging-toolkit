"""Scheduler/finalizer contracts; no synthetic imaging or timing benchmark."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "validation/dmri_pipeline/public10_20261002/finish_cohort.py"
spec = importlib.util.spec_from_file_location("public10_finalizer", SOURCE)
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def reports_and_row(tmp_path):
    reports = {}
    for side in ("candidate", "reference"):
        report = tmp_path / side / "report.json"
        write_json(report, {"status": "complete"})
        (report.parent / "exitcode").write_text("0\n")
        reports[side] = report
    row = {"case_id": "case01", "backend": "mmorf",
           "candidate_report": str(reports["candidate"]), "reference_report": str(reports["reference"]),
           "candidate_dir": str(reports["candidate"].parent), "reference_dir": str(reports["reference"].parent),
           "comparison_report": str(tmp_path / "comparisons/case01.mmorf.json")}
    return reports, row


def queue(tmp_path, jobs, outcomes):
    plan = tmp_path / "queue/fixed_plan.json"
    write_json(plan, {"jobs": jobs})
    state_file = plan.with_name("cohort_status.json")
    state = {"plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
             "planned_jobs": len(jobs), "outcomes": outcomes}
    write_json(state_file, state)
    return state_file, state


def arguments(tmp_path, manifest, status_file):
    return ["--manifest", str(manifest), "--cohort-status", str(status_file),
            "--comparator", str(tmp_path / "comparator.py"),
            "--reference-roi", str(tmp_path / "roi.nii.gz"), "--fa-skeleton", str(tmp_path / "skeleton.nii.gz"),
            "--inputs-root", str(tmp_path / "inputs"), "--output-dir", str(tmp_path / "summaries"),
            "--poll-seconds", ".01"]


def stub_commands(monkeypatch, *, comparison_exit=0, aggregate_status="complete"):
    calls = []
    def run(command, **kwargs):
        mode = command[2]
        calls.append(mode)
        destination = Path(command[command.index("--report") + 1])
        if mode == "compare":
            if kwargs.get("stdout") is not None:
                kwargs["stdout"].write(b"retained fixture comparison log\n")
            if comparison_exit not in (0, 2):
                return SimpleNamespace(returncode=comparison_exit)
            status = "complete" if comparison_exit == 0 else "required_map_gate_failed"
        else:
            status = aggregate_status
        write_json(destination, {"status": status})
        return SimpleNamespace(returncode=0 if status == "complete" else 2)
    monkeypatch.setattr(finalizer.subprocess, "run", run)
    return calls


def test_queue_completion_requires_all_terminal_outcomes():
    assert not finalizer.queues_finished([])
    assert not finalizer.queues_finished([{"planned_jobs": 1, "outcomes": []}])
    assert not finalizer.queues_finished([{"planned_jobs": 1, "outcomes": [{"status": "running"}]}])
    assert not finalizer.queues_finished([{"planned_jobs": 0, "outcomes": []}])
    assert finalizer.queues_finished([{"planned_jobs": 1, "outcomes": [{"status": "scheduler_failed"}]},
                                     {"planned_jobs": 1, "outcomes": [{"status": "complete"}]}])


def test_scheduler_failure_identity_requires_exact_selected_report_and_plan_hash(tmp_path):
    reports, row = reports_and_row(tmp_path)
    (reports["candidate"].parent / "exitcode").unlink()
    old_job = {"case_id": "case01", "backend": "mmorf", "implementation": "fnit",
               "report": str(tmp_path / "initial_failed/report.json")}
    outcome = {"case_id": "case01", "backend": "mmorf", "implementation": "fnit",
               "status": "scheduler_failed", "exception_type": "FileNotFoundError"}
    status_file, state = queue(tmp_path, [old_job], [outcome])
    failures = finalizer.bound_scheduler_failures(status_file, state)
    assert len(failures) == 1
    assert not finalizer.selected_terminal(row, "candidate", failures)
    state["plan_sha256"] = "0" * 64
    assert finalizer.bound_scheduler_failures(status_file, state) == []
    selected_job = {**old_job, "report": row["candidate_report"]}
    status_file, state = queue(tmp_path, [selected_job], [outcome])
    failures = finalizer.bound_scheduler_failures(status_file, state)
    assert finalizer.selected_terminal(row, "candidate", failures)
    wrong_implementation = [{**failures[0], "implementation": "official"}]
    assert not finalizer.selected_terminal(row, "candidate", wrong_implementation)


def test_bound_scheduler_failure_without_exitcode_ends_incomplete(tmp_path, monkeypatch):
    reports, row = reports_and_row(tmp_path)
    reports["candidate"].unlink()
    (reports["candidate"].parent / "exitcode").unlink()
    job = {"case_id": "case01", "backend": "mmorf", "implementation": "fnit", "report": row["candidate_report"]}
    outcome = {"case_id": "case01", "backend": "mmorf", "implementation": "fnit",
               "status": "scheduler_failed", "exception_type": "FileNotFoundError"}
    status_file, _ = queue(tmp_path, [job], [outcome])
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"planned_cases": [row]})
    calls = stub_commands(monkeypatch, aggregate_status="complete")
    monkeypatch.setattr(finalizer.time, "sleep", lambda value: pytest.fail("must terminate, not poll forever"))
    assert finalizer.main(arguments(tmp_path, manifest, status_file)) == 2
    assert calls == ["aggregate"]
    aggregate = json.loads((tmp_path / "summaries/aggregate.final.json").read_text())
    assert aggregate["status"] == "incomplete" and not aggregate["all_planned_results_complete"]
    assert aggregate["selected_scheduler_failures"] == [{
        "case_id": "case01", "backend": "mmorf", "implementation": "fnit", "exception_type": "FileNotFoundError"}]
    assert str(tmp_path) not in json.dumps(aggregate)
    assert (tmp_path / "summaries/finalizer.exitcode").read_text().strip() == "2"


def test_final_scan_drains_pair_that_finished_after_initial_scan(tmp_path, monkeypatch):
    reports, row = reports_and_row(tmp_path)
    write_json(reports["candidate"], {"status": "running"})
    (reports["candidate"].parent / "exitcode").unlink()
    job = {"case_id": "case01", "backend": "mmorf", "implementation": "fnit", "report": row["candidate_report"]}
    status_file, state = queue(tmp_path, [job], [])
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"planned_cases": [row]})
    original_complete = finalizer.complete_job
    checked = []
    def finish_during_scan(path):
        checked.append(path)
        if len(checked) == 1:
            write_json(reports["candidate"], {"status": "complete"})
            (reports["candidate"].parent / "exitcode").write_text("0\n")
            state["outcomes"] = [{"case_id": "case01", "backend": "mmorf", "implementation": "fnit", "status": "complete"}]
            write_json(status_file, state)
            return False
        return original_complete(path)
    monkeypatch.setattr(finalizer, "complete_job", finish_during_scan)
    calls = stub_commands(monkeypatch)
    assert finalizer.main(arguments(tmp_path, manifest, status_file)) == 0
    assert calls == ["compare", "aggregate"]
    assert Path(row["comparison_report"]).exists()


@pytest.mark.parametrize("comparison_exit", [2, 7])
def test_comparison_gate_or_execution_failure_keeps_log_and_finishes(tmp_path, monkeypatch, comparison_exit):
    _, row = reports_and_row(tmp_path)
    job = {"case_id": "case01", "backend": "mmorf", "implementation": "fnit", "report": row["candidate_report"]}
    status_file, _ = queue(tmp_path, [job], [{"status": "complete"}])
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"planned_cases": [row]})
    calls = stub_commands(monkeypatch, comparison_exit=comparison_exit, aggregate_status="incomplete")
    assert finalizer.main(arguments(tmp_path, manifest, status_file)) == 2
    assert calls == ["compare", "aggregate"]
    comparison = json.loads(Path(row["comparison_report"]).read_text())
    expected_status = "required_map_gate_failed" if comparison_exit == 2 else "comparison_execution_failed"
    assert comparison["status"] == expected_status
    assert (tmp_path / "summaries/case01_mmorf_compare.log").read_bytes() == b"retained fixture comparison log\n"
