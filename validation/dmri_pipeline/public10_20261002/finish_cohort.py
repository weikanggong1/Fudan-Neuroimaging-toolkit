#!/usr/bin/env python3
"""Compare finished independent jobs and retain the full ten-person denominator."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def complete_job(report_path):
    report_path = Path(report_path)
    try:
        return read_json(report_path).get("status") == "complete" and (
            report_path.parent / "exitcode").read_text().strip() == "0"
    except OSError:
        return False


def queues_finished(states):
    terminal = {"complete", "failed", "scheduler_failed"}
    return bool(states) and all(type(state.get("planned_jobs")) is int and state["planned_jobs"] > 0
               and isinstance(state.get("outcomes"), list)
               and len(state["outcomes"]) == state["planned_jobs"]
               and all(isinstance(outcome, dict) and outcome.get("status") in terminal
                       for outcome in state["outcomes"])
               for state in states)


def bound_scheduler_failures(status_path, state):
    """Bind a scheduler failure to the frozen plan's exact report, not just case ID."""
    outcomes = state.get("outcomes", [])
    if not any(isinstance(item, dict) and item.get("status") == "scheduler_failed" for item in outcomes):
        return []
    digest = state.get("plan_sha256")
    if not isinstance(digest, str):
        return []
    plan = None
    # run_cohort records the exact plan hash but older running schedulers do
    # not include report paths in outcomes. Recover that identity from its
    # unchanged sibling plan without modifying a running scheduler.
    for path in sorted(Path(status_path).parent.glob("*.json")):
        if path == Path(status_path):
            continue
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == digest:
                candidate = json.loads(raw)
                if isinstance(candidate, dict) and isinstance(candidate.get("jobs"), list):
                    plan = candidate
                    break
        except (OSError, ValueError):
            continue
    if plan is None:
        return []
    failures = []
    for index, outcome in enumerate(outcomes):
        if outcome.get("status") != "scheduler_failed" or index >= len(plan["jobs"]):
            continue
        job = plan["jobs"][index]
        if (not isinstance(job, dict) or not isinstance(job.get("report"), str)
                or any(outcome.get(key) != job.get(key) for key in ("case_id", "backend", "implementation"))):
            continue
        failures.append({"case_id": job["case_id"], "backend": job["backend"],
                         "implementation": job["implementation"],
                         "report_identity": str(Path(job["report"]).resolve()),
                         "exception_type": outcome.get("exception_type")})
    return failures


def selected_scheduler_failure(row, role, failures):
    implementation = "fnit" if role == "candidate" else "official"
    identity = str(Path(row[role + "_report"]).resolve())
    return next((failure for failure in failures
                 if failure["case_id"] == row["case_id"] and failure["backend"] == row["backend"]
                 and failure["implementation"] == implementation
                 and failure["report_identity"] == identity), None)


def selected_terminal(row, role, failures):
    return ((Path(row[role + "_report"]).parent / "exitcode").is_file()
            or selected_scheduler_failure(row, role, failures) is not None)


@contextmanager
def admission(path):
    if path is None:
        yield
    else:
        with path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield


def compare_ready_rows(rows, args):
    for row in rows:
        destination = Path(row["comparison_report"])
        if destination.exists() or not all(complete_job(row[role + "_report"])
                                            for role in ("candidate", "reference")):
            continue
        command = [sys.executable, str(args.comparator), "compare", "--case-id", row["case_id"],
                   "--backend", row["backend"], "--candidate-dir", row["candidate_dir"],
                   "--reference-dir", row["reference_dir"], "--reference-roi", str(args.reference_roi),
                   "--fa-skeleton", str(args.fa_skeleton), "--bvals",
                   str(args.inputs_root / row["case_id"] / "raw/AP.bval"),
                   "--report", str(destination)]
        with admission(args.gpu_lock):
            with (args.output_dir / f"{row['case_id']}_{row['backend']}_compare.log").open("wb") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode not in (0, 2) or not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "case_id": row["case_id"], "registration_backend": row["backend"],
                "status": "comparison_execution_failed", "exit_code": result.returncode,
                "numerical_equivalence_claimed": False}, indent=2) + "\n")
        print(json.dumps({"event": "comparison_saved", "case_id": row["case_id"],
                          "backend": row["backend"], "exit_code": result.returncode}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort-status", type=Path, required=True)
    parser.add_argument("--additional-cohort-status", type=Path, action="append", default=[],
                        help="additional fixed queues sharing one GPU lock; all must finish")
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--reference-roi", type=Path, required=True)
    parser.add_argument("--fa-skeleton", type=Path, required=True)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--gpu-lock", type=Path,
                        help="serialize CPU image comparisons with measured jobs to avoid I/O contention")
    parser.add_argument("--report-renderer", type=Path,
                        help="optional standalone renderer of Chinese Markdown/CSV summaries")
    parser.add_argument("--dataset-manifest", type=Path,
                        help="public fixed input manifest to bind rendered reports")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    previous_done = -1
    while True:
        # A documented fresh rerun may replace a failed reference path while
        # retaining its initial failure. The fixed subjects/branches do not change.
        rows = read_json(args.manifest)["planned_cases"]
        compare_ready_rows(rows, args)
        state_paths = (args.cohort_status, *args.additional_cohort_status)
        states = [read_json(path) for path in state_paths]
        failures = [failure for path, state in zip(state_paths, states)
                    for failure in bound_scheduler_failures(path, state)]
        finished = (queues_finished(states)
                    and all(selected_terminal(row, role, failures)
                            for row in rows for role in ("candidate", "reference")))
        if finished:
            # A final job can finish after the first scan. Drain again before
            # publishing a terminal aggregate; failed jobs need no comparison.
            compare_ready_rows(rows, args)
        done = sum(Path(row["comparison_report"]).exists() for row in rows)
        if done != previous_done or finished:
            destination = args.output_dir / ("aggregate.final.json" if finished else f"aggregate.progress_{done:02d}.json")
            if not destination.exists():
                result = subprocess.run([sys.executable, str(args.comparator), "aggregate",
                                         "--manifest", str(args.manifest), "--report", str(destination)])
                if result.returncode not in (0, 2):
                    raise RuntimeError("cohort aggregation failed")
            selected_failures = [selected_scheduler_failure(row, role, failures)
                                 for row in rows for role in ("candidate", "reference")]
            selected_failures = [failure for failure in selected_failures if failure is not None]
            if finished and selected_failures:
                summary = read_json(destination)
                summary.update(status="incomplete", all_planned_results_complete=False,
                               selected_scheduler_failures=[{
                                   key: failure.get(key) for key in
                                   ("case_id", "backend", "implementation", "exception_type")}
                                   for failure in selected_failures])
                destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            if args.report_renderer is not None:
                rendered_dir = args.output_dir if finished else args.output_dir / destination.stem
                command = [sys.executable, str(args.report_renderer),
                           "--aggregate", str(destination), "--output-dir", str(rendered_dir)]
                if args.dataset_manifest is not None:
                    command += ["--dataset-manifest", str(args.dataset_manifest)]
                rendered = subprocess.run(command)
                if rendered.returncode not in (0, 2):
                    raise RuntimeError("cohort report rendering failed")
            previous_done = done
        if finished:
            summary = read_json(args.output_dir / "aggregate.final.json")
            (args.output_dir / "finalizer.exitcode").write_text(
                "0\n" if summary.get("status") == "complete" else "2\n")
            return 0 if summary.get("status") == "complete" else 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
