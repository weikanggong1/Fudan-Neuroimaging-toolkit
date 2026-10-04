"""Publish completed weighted/batched CPU followups with their actual source."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


def module_at(path):
    spec = importlib.util.spec_from_file_location("flirt_cpu_report_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_suite(run_root, original_records, helper, expected_records):
    path = run_root / "suite" / "suite.private.json"
    if not path.is_file():
        return {"status": "not_started", "expected_records": expected_records, "records": []}
    suite = json.loads(path.read_text())
    source = Path(suite["candidate_root"])
    launch = json.loads((run_root / "launch.private.json").read_text())
    actual = {relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
              for relative in launch["source_sha256"]}
    assert actual == launch["source_sha256"], "immutable source changed"
    records = []
    for private in suite["records"]:
        public = helper.public_record(private)
        previous = next((record for record in original_records
                         if record["case_id"] == private["case_id"]
                         and record["threads"] == private["threads"]), None)
        if previous is not None:
            before = helper.public_record(previous)
            old_output = before["candidate_output_metadata"][0]
            new_output = public["candidate_output_metadata"][0]
            old_qc, new_qc = before["fnit_diagnostics"][0], public["fnit_diagnostics"][0]
            checks = {
                "saved_output_files_sha256_identical": all(
                    old_output[key]["sha256"] == new_output[key]["sha256"]
                    for key in old_output),
                "final_cost_identical": old_qc["cost_value"] == new_qc["cost_value"],
                "cost_evaluations_identical": old_qc["cost_evaluations"] == new_qc["cost_evaluations"],
                "phase_counts_identical": old_qc["phase_cost_evaluations"] == new_qc["phase_cost_evaluations"],
            }
            assert all(checks.values()), (private["case_id"], private["threads"], checks)
            public["vs_v16_same_budget"] = checks
        else:
            public["vs_v16_same_budget"] = {"status": "original_complete_observation_pending"}
        records.append(public)
    return {
        "status": suite["status"], "expected_records": expected_records,
        "run_id": run_root.name, "candidate_snapshot": source.name,
        "source_sha256": actual,
        "candidate_archive_sha256": launch.get("candidate_archive_sha256") or {
            "candidate_all_v21": "6fe4ae95447729e68b94b8d7796f1d056237e679f95b44fcaf66f70a40957a72",
        }.get(source.name),
        "benchmark_sha256": {relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
                             for relative in ("tools/benchmark_multimodal_cpu.py",
                                              "tools/benchmark_multimodal_cpu_flirt.py")},
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--weighted-run-root", type=Path, required=True)
    parser.add_argument("--batched-run-root", type=Path)
    parser.add_argument("--batched-statistics-run-root", type=Path)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables", type=Path)
    args = parser.parse_args()
    helper = module_at(args.helper)
    original = json.loads((args.original_run_root / "functions/suite.private.json").read_text())["records"]
    groups = {"weighted": completed_suite(args.weighted_run_root, original, helper, 6)}
    if args.batched_run_root:
        groups["batched"] = completed_suite(args.batched_run_root, original, helper, 4)
    if args.batched_statistics_run_root:
        groups["batched_statistics"] = completed_suite(
            args.batched_statistics_run_root, original, helper, 4)
    measurements_complete = all(group["status"] == "executed_with_numeric_comparisons"
                                and len(group["records"]) == group["expected_records"]
                                for group in groups.values())
    previous_source_checks_complete = all(
        all("status" not in record["vs_v16_same_budget"] for record in group["records"])
        and len(group["records"]) == group["expected_records"] for group in groups.values())
    complete = measurements_complete and previous_source_checks_complete
    report = {
        "schema_version": 1, "date": "2026-10-04",
        "status": "complete" if complete else "in_progress",
        "measurements_complete": measurements_complete,
        "previous_source_checks_complete": previous_source_checks_complete,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "OpenNeuro ds000114 v1.0.2; FNIT defaced public T1w examples",
        "dataset_license": "CC0", "official_version": "FSL 6.0.7.4",
        "hardware": json.loads((args.original_run_root / "hardware.public.json").read_text()),
        "timing_scope": "One fresh complete official CLI and FNIT API worker per case/thread ceiling; includes startup, imports, read, full computation and save; no warmup or stable speed ratio",
        "previous_report": "cpu_20261004.public.json",
        "previous_comparison": "Same thread budget, complete outputs and search counts compared against v16; previous and current timing runs use different reserved physical-core groups",
        "weighted_exact_gate_report": "cpu_weighted_exact_20261004.public.json",
        "batched_exact_gate_report": "cpu_batched_exact_20261004.public.json" if args.batched_run_root else None,
        "batched_statistics_exact_gate_report": "cpu_batched_statistics_exact_20261004.public.json" if args.batched_statistics_run_root else None,
        "groups": groups,
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    assert all(value not in serialized for value in ("/cwStorage/", "/mnt/c/Users/", ".sock"))
    args.output.write_text(serialized)
    if args.tables:
        args.tables.write_text("\n\n".join("## " + name + "\n\n" + helper.table(group["records"], False)
                                           for name, group in groups.items()) + "\n")
    print(json.dumps({"status": report["status"], "records": {name: len(group["records"])
                                                               for name, group in groups.items()}}))


if __name__ == "__main__":
    main()
