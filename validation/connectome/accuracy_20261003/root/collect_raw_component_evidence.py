"""Collect original completed CPU reports without solving or changing MRI data.

Default scope requires all ten actual candidate reports and the controller's
genuine all-ten summary. Explicit case/version subsets must also all be complete.
Original reports, summaries, receipts and logs are copied byte for byte; a
collection manifest records original and collected SHA, exact coverage, and
unmodified nonfinite-aware metrics. No missing case gets a numerical placeholder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import time


SPEC = importlib.util.spec_from_file_location("component_control", Path(__file__).with_name("control_raw_components.py"))
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)
require = control.require


def selected_pairs(config, case_ids=None, versions=None):
    pairs = control.plan(config)
    versions = set(versions or ("candidate",))
    require(versions <= {"baseline", "candidate"}, "unknown comparison version")
    if case_ids:
        wanted = {(version, case) for version in versions for case in case_ids}
        require(wanted <= set(pairs), "unknown or unscheduled comparison pair")
        return [pair for pair in pairs if pair in wanted], False
    require(versions == {"candidate"}, "baseline collection requires explicit case IDs")
    return [pair for pair in pairs if pair[0] == "candidate"], True


def receipt_for(row, key, configuration, helper, controller, producer_row):
    require(row.get("status") == "completed", f"analysis not completed: {key}")
    receipt = control.bound_json(row["receipt_file"])
    require(receipt == {name: value for name, value in row.items() if name != "receipt_file"},
            f"receipt bytes do not bind the actual completed analysis row: {key}")
    require(receipt["key"] == key and receipt["returncode"] == 0 and receipt["error"] is None,
            f"unsuccessful actual comparison receipt: {key}")
    require(receipt["environment"]["CUDA_VISIBLE_DEVICES"] == "", "comparison receipt did not disable CUDA")
    require(control.same_file(receipt["configuration"], configuration)
            and control.same_file(receipt["controller"], controller)
            and control.same_file(receipt["helper_before"], helper)
            and control.same_file(receipt["helper_after"], helper), "comparison source/configuration identity differs")
    require(producer_row.get("status") == "completed"
            and receipt["producer_completed_before"] == receipt["producer_completed_after"] == producer_row,
            f"actual producer was not the unchanged completed row: {key}")
    return receipt


def collect(analysis_dir, configuration, configuration_sha256, helper_sha256, output_dir,
            case_ids=None, versions=None):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU collector requires CUDA_VISIBLE_DEVICES empty")
    started = time.perf_counter()
    collector_source = control.file_record(__file__)
    local_control_source = control.file_record(control.__file__)
    config_record = {"path": str(Path(configuration).resolve()), "sha256": configuration_sha256}
    config = control.bound_json(config_record)
    analysis = Path(analysis_dir).resolve()
    metadata_record = control.file_record(analysis / "analysis_configuration.json")
    metadata = control.bound_json(metadata_record)
    require(control.same_file(metadata["configuration"], config_record), "analysis configuration differs from the frozen input")
    helper = metadata["helper"]
    require(helper["sha256"] == helper_sha256, "unexpected helper SHA")
    controller = metadata["controller"]
    require(local_control_source["sha256"] == controller["sha256"], "local source gates differ from the bound actual controller")
    control.bound(helper)
    control.bound(controller)
    pairs, whole_ten = selected_pairs(config, case_ids, versions)
    phase = Path(config["run_root"]).resolve().parent
    destination = Path(output_dir).resolve()
    require(destination.parent == phase and destination != phase, "fresh isolated collection sibling required")
    protected = [analysis, Path(config["run_root"]).resolve(), Path(configuration).resolve().parent,
                 Path(helper["path"]).resolve().parents[4], *[Path(path).resolve() for path in config["sources"].values()]]
    require(not any(destination == path or destination.is_relative_to(path) or path.is_relative_to(destination)
                    for path in protected), "collection overlaps source, analysis or producer namespace")
    require(not destination.exists(), "fresh collection directory required; no prior evidence overwritten")
    state_path = analysis / "status.json"
    state = json.loads(state_path.read_bytes())
    require(control.same_file(state["configuration"], config_record)
            and control.same_file(state["helper"], helper)
            and control.same_file(state["controller"], controller), "actual analysis state has another source/configuration")
    raw_status = Path(config["run_root"]) / "status.json"
    producer_state = json.loads(raw_status.read_bytes())
    control.check_producer_state(producer_state, config, config_record)
    aggregate_record = state.get("all10_summary")
    aggregate = None
    if whole_ten:
        require(aggregate_record is not None, "actual all-ten candidate summary is not completed")
        aggregate = control.bound_json(aggregate_record)
        require(aggregate["status"] == "all_ten_actual_candidate_reports_compared"
                and aggregate["scientific_parity"] == "not_assessed"
                and control.same_file(aggregate["configuration"], config_record)
                and control.same_file(aggregate["helper"], helper), "all-ten aggregate identity or scientific scope differs")
        keys = sorted(f"{version}/{case}" for version, case in pairs)
        require(aggregate["coverage"] == keys and sorted(aggregate["cases"]) == keys,
                "all-ten aggregate does not cover exactly the actual ten candidate cases")
    audit = {}

    def verify(record):
        path = control.bound(record)
        actual = control.file_record(path)
        previous = audit.get(actual["path"])
        require(previous is None or previous == actual, "conflicting original file identities")
        audit[actual["path"]] = actual
        return actual

    for record in (config_record, metadata_record, helper, controller, collector_source, local_control_source):
        verify(record)
    if aggregate_record is not None and whole_ten:
        verify(aggregate_record)
    pending_copies, summaries, completed_rows = [], {}, {}
    for version, case in pairs:
        key = f"{version}/{case}"
        row = state["cases"].get(key, {})
        producer = producer_state["cases"].get(key, {})
        receipt = receipt_for(row, key, config_record, helper, controller, producer)
        report = control.checked_report(row["report"], key, helper, config_record, producer)
        summary = control.bound_json(row["summary"])
        metrics = control.summary_case(report, key)
        require(summary == {"original_report": row["report"], "components": metrics},
                f"saved case summary differs from the bound original report: {key}")
        if aggregate is not None:
            require(aggregate["cases"][key] == {"report": row["report"], "components": metrics},
                    f"all-ten summary differs from the original report: {key}")
        completed_rows[key] = {"analysis": row, "producer": producer}
        # Revalidate every originally audited source/input/output, including
        # valid exact-hash empty package markers. No MRI arrays are loaded.
        for record in report["immutable_files_verified_before_after"]:
            verify(record)
        for name, record in (("components.json", row["report"]), ("component_summary.json", row["summary"]),
                             ("receipt.json", row["receipt_file"])):
            pending_copies.append((Path(version) / case / name, verify(record)))
        for role in ("stdout", "stderr"):
            if role in receipt:
                pending_copies.append((Path(version) / case / (role + ".log"), verify(receipt[role])))
        summaries[key] = {"original_report": row["report"], "original_summary": row["summary"],
                          "original_receipt": row["receipt_file"], "components": metrics}
    if aggregate is not None:
        pending_copies.append((Path("all10_candidate_summary.json"), verify(aggregate_record)))
    # No output directory is created until all requested producer/receipt/SHA
    # gates have passed. Failures remain failures; the caller must use a fresh
    # namespace for a later retry.
    destination.mkdir()
    copied = []
    for relative, original in pending_copies:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = Path(original["path"]).read_bytes()
        require(control.sha256(original["path"]) == original["sha256"], "original changed before byte copy")
        with target.open("xb") as stream:
            stream.write(data)
        collected = control.file_record(target)
        require(collected["sha256"] == original["sha256"] and collected["size_bytes"] == original["size_bytes"],
                "byte copy differs from the original report")
        copied.append({"original": original, "collected": collected})
    for record in audit.values():
        control.bound(record)
    current = json.loads(state_path.read_bytes())
    current_producers = json.loads(raw_status.read_bytes())
    control.check_producer_state(current_producers, config, config_record)
    require(all(current["cases"].get(key) == rows["analysis"]
                and current_producers["cases"].get(key) == rows["producer"] for key, rows in completed_rows.items()),
            "selected completed analysis/producer rows changed during collection")
    if aggregate is not None:
        require(current["all10_summary"] == aggregate_record, "actual all-ten summary binding changed")
    result = {"schema_version": 1, "status": "all_ten_actual_candidate_evidence_collected" if whole_ten else "requested_actual_completed_pairs_collected",
              "scientific_parity": "not_assessed", "UTC": datetime.now(timezone.utc).isoformat(),
              "hostname": socket.gethostname(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "configuration": config_record, "helper": helper, "controller": controller,
              "collector": collector_source, "coverage": sorted(summaries),
              "scope": "CPU original-byte collection only; no GPU/official solver or MRI resampling; no scientific acceptance implied",
              "original_rows_verified_before_after": completed_rows, "original_files_verified_before_after": list(audit.values()),
              "byte_copies": copied, "cases": summaries, "CPU_collection_wall_seconds": time.perf_counter() - started}
    control.atomic_json(destination / "collection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--version", action="append", choices=("candidate", "baseline"))
    args = parser.parse_args()
    result = collect(args.analysis_dir, args.configuration, args.configuration_sha256, args.helper_sha256,
                     args.output_dir, args.case_id, args.version)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "output": str(args.output_dir / "collection.json")}))


if __name__ == "__main__":
    main()
