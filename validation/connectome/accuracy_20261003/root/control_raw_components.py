"""CPU-only controller for actually completed, provenance-bound raw comparisons.

This private validation controller never runs MRI solvers or changes scientific
inputs. A failed or unfinished producer has no component metrics. Each completed
pair has an independent output directory and receipt. An all-ten candidate
summary is written only after all ten reports and their producer rows are real.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def bound(record):
    path = Path(record["path"])
    require(path.is_absolute() and path.is_file(), f"missing bound file: {path}")
    require(sha256(path) == record["sha256"], f"bound bytes changed: {path}")
    require(path.stat().st_size == record.get("size_bytes", path.stat().st_size),
            f"bound size changed: {path}")
    return path


def bound_json(record):
    return json.loads(bound(record).read_bytes())


def same_file(first, second):
    return (Path(first["path"]).resolve() == Path(second["path"]).resolve()
            and first["sha256"] == second["sha256"])


def atomic_json(path, value):
    path = Path(path)
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plan(config):
    order = config["execution_order"]
    pairs = [(item["version"], item["case_id"]) for item in order]
    require(len(pairs) == len(set(pairs)), "duplicate actual execution pair")
    candidates = [case for version, case in pairs if version == "candidate"]
    require(len(candidates) == len(set(candidates)) == 10, "exactly ten candidate cases required")
    require({case for version, case in pairs if version == "baseline"} == {"sub-CON01", "sub-CON03"},
            "paired baseline cases differ from the actual frozen plan")
    require(all(version in ("baseline", "candidate") for version, case in pairs), "unknown version")
    return pairs


def check_producer_state(state, config, configuration_record):
    require(same_file(state["configuration"], configuration_record), "controller configuration differs")
    require(state["execution_order"] == config["execution_order"]
            and state["raw_manifest"] == config["raw_manifest"]
            and state["input_bindings"] == config["input_bindings"], "producer plan/input binding differs")


def checked_report(report_record, key, helper_record, configuration_record, producer_row):
    report = bound_json(report_record)
    require(report["status"] == "requested_completed_cases_compared"
            and report["scientific_parity"] == "not_assessed", "comparison report did not complete its scope")
    require(report["coverage"]["actually_compared"] == [key]
            and report["coverage"]["not_compared"] == {}, "comparison coverage differs")
    version, case = key.split("/")
    require(report["coverage"]["requested_cases"] == [case]
            and report["coverage"]["requested_versions"] == [version], "requested pair differs")
    require(same_file(report["script"], helper_record)
            and same_file(report["configuration"], configuration_record), "report helper/configuration identity differs")
    require(producer_row.get("status") == "completed"
            and report["controller_snapshot"]["selected_completed_rows"] == {key: producer_row},
            "actual completed producer row differs from comparison snapshot")
    require(set(report["cases"]) == {key} and report["cases"][key]["status"] == "compared", "case result missing")
    # Retain producer identity, including the real GPU and wall receipt bytes.
    for role in ("gpu_report", "wall_report"):
        identity = report["cases"][key]["candidate_producer"][role]
        require(Path(identity["path"]).resolve() == Path(producer_row[role]).resolve(), "producer receipt path differs")
        bound(identity)
    return report


def summary_case(report, key):
    case = report["cases"][key]
    summary = {"case_id": case["case_id"], "version": case["version"], "status": case["status"],
               "scientific_parity": case["scientific_parity"], "comparison_scope": case["comparison_scope"],
               "candidate_producer": case["candidate_producer"], "official_producer": case["official_producer"],
               "CPU_comparison_wall_seconds": report["total_CPU_wall_seconds"],
               "immutable_files_before_after_count": len(report["immutable_files_verified_before_after"])}
    for role in ("corrected_dwi", "FA"):
        image = case[role]
        summary[role] = {name: value for name, value in image.items() if name not in ("per_frame",)}
        if role == "FA" and image["status"] == "compared":
            summary[role]["absolute_error_percentiles_all_values"] = {
                region: image["per_frame"][0][region]["absolute_error_percentiles_all_values"]
                for region in ("whole_volume", "official_brain_mask")}
    gradients = case["rotated_gradients"]
    summary["rotated_gradients"] = {name: value for name, value in gradients.items() if name != "all_frames"}
    if gradients["status"] == "compared":
        rows = gradients["all_frames"]
        summary["rotated_gradients"]["all_frame_count"] = len(rows)
        summary["rotated_gradients"]["nonfinite_frame_count"] = sum(not row["finite_all_components_bvals"] for row in rows)
        angles = [row["directed_angle_degrees"] for row in rows if row["directed_angle_degrees"] is not None]
        summary["rotated_gradients"]["max_directed_angle_degrees"] = max(angles) if angles else None
        summary["rotated_gradients"]["directed_angle_defined_frame_count"] = len(angles)
    summary["ancillary_geometry"] = case["ancillary_geometry"]
    return summary


def reuse_report(report_record, key, helper_record, configuration_record, producer_row):
    report = checked_report(report_record, key, helper_record, configuration_record, producer_row)
    # Reuse the saved numerical result; independently re-check every immutable
    # file, including valid zero-byte source markers, and the original report.
    for record in report["immutable_files_verified_before_after"]:
        bound(record)
    bound(report_record)
    return report


def collect_all_candidates(config, analysis_rows, producer_state, helper_record, configuration_record):
    keys = [f"{version}/{case}" for version, case in plan(config) if version == "candidate"]
    if any(analysis_rows.get(key, {}).get("status") != "completed" for key in keys):
        return None
    reports = {}
    for key in keys:
        row = analysis_rows[key]
        producer_row = producer_state["cases"].get(key, {})
        reports[key] = checked_report(row["report"], key, helper_record, configuration_record, producer_row)
    return {key: {"report": analysis_rows[key]["report"], "components": summary_case(report, key)}
            for key, report in reports.items()}


def run(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "controller must start with CUDA_VISIBLE_DEVICES empty")
    require(30 <= args.poll_seconds <= 60, "independent polling must be between 30 and 60 seconds")
    configuration_record = {"path": str(args.configuration.resolve()), "sha256": args.configuration_sha256}
    helper_record = {"path": str(args.helper.resolve()), "sha256": args.helper_sha256}
    controller_record = {"path": str(Path(__file__).resolve()), "sha256": args.controller_sha256}
    config = bound_json(configuration_record)
    bound(helper_record)
    bound(controller_record)
    pairs = plan(config)
    phase = Path(config["run_root"]).resolve().parent
    output = args.output_dir.resolve()
    require(output.parent == phase and output.name == "root_component_analysis_v1", "isolated private controller namespace required")
    require(not output.exists(), "fresh controller namespace required; no prior evidence overwritten")
    for source in config["sources"].values():
        path = Path(source).resolve()
        require(not (output == path or output.is_relative_to(path) or path.is_relative_to(output)), "output overlaps a scientific source")
    output.mkdir(parents=True)
    environment = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command_environment = {**os.environ, **environment}
    state = {"schema_version": 1, "status": "running", "scientific_parity": "not_assessed",
             "hostname": socket.gethostname(), "PID": os.getpid(), "python": sys.executable,
             "controller": controller_record, "helper": helper_record, "configuration": configuration_record,
             "environment": environment, "poll_seconds": args.poll_seconds,
             "execution_order": [{"version": version, "case_id": case} for version, case in pairs],
             "cases": {}, "all10_summary": None,
             "policy": "only genuine completed producers; full-volume diagnostics; no GPU/official solver; no scientific acceptance implied"}
    atomic_json(output / "analysis_configuration.json", {key: value for key, value in state.items() if key != "cases"})
    status_path = output / "status.json"
    raw_status = Path(config["run_root"]) / "status.json"
    all10_path = output / "all10_candidate_summary.json"
    while True:
        bound(helper_record)
        bound(configuration_record)
        bound(controller_record)
        producer_state = json.loads(raw_status.read_bytes())
        check_producer_state(producer_state, config, configuration_record)
        for version, case_id in pairs:
            key = f"{version}/{case_id}"
            existing = state["cases"].get(key, {})
            if existing.get("status") in ("completed", "analysis_failed"):
                continue
            producer_row = producer_state["cases"].get(key)
            if producer_row is None or producer_row.get("status") != "completed":
                state["cases"][key] = {"status": "not_assessed", "producer_status": None if producer_row is None else producer_row.get("status"),
                                       "reason": "no genuinely completed producer; no MRI arrays or metrics read"}
                continue
            result_dir = output / "results" / version / case_id
            case_dir = output / "receipts" / version / case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            receipt = {"schema_version": 1, "key": key, "status": "running", "scientific_parity": "not_assessed",
                       "started_UTC": datetime.now(timezone.utc).isoformat(), "environment": environment,
                       "helper_before": file_record(bound(helper_record)), "configuration": configuration_record,
                       "controller": controller_record, "producer_completed_before": producer_row}
            state["cases"][key] = receipt
            atomic_json(status_path, state)
            started = time.perf_counter()
            try:
                if key == "baseline/sub-CON01":
                    report_record = {"path": str(args.reuse_CON01_baseline.resolve()), "sha256": args.reuse_CON01_baseline_sha256}
                    receipt["mode"] = "reuse_original_successful_report_with_full_SHA_recheck"
                    receipt["command"] = None
                    report = reuse_report(report_record, key, helper_record, configuration_record, producer_row)
                    receipt["returncode"] = 0
                else:
                    receipt["mode"] = "CPU_component_comparison"
                    command = [sys.executable, str(args.helper.resolve()), "--configuration", str(args.configuration.resolve()),
                               "--configuration-sha256", args.configuration_sha256, "--case-id", case_id,
                               "--version", version, "--output-dir", str(result_dir)]
                    receipt["command"] = command
                    atomic_json(case_dir / "started.json", receipt)
                    with (case_dir / "stdout.log").open("x") as stdout, (case_dir / "stderr.log").open("x") as stderr:
                        process = subprocess.run(command, env=command_environment, cwd=args.helper.resolve().parents[4],
                                                 stdout=stdout, stderr=stderr, check=False)
                    receipt["returncode"] = process.returncode
                    receipt["stdout"] = file_record(case_dir / "stdout.log")
                    receipt["stderr"] = file_record(case_dir / "stderr.log")
                    require(process.returncode == 0, f"comparison helper failed; exit {process.returncode}")
                    report_record = file_record(result_dir / "components.json")
                    report = checked_report(report_record, key, helper_record, configuration_record, producer_row)
                receipt["helper_after"] = file_record(bound(helper_record))
                bound(controller_record)
                bound(configuration_record)
                current = json.loads(raw_status.read_bytes())
                check_producer_state(current, config, configuration_record)
                require(current["cases"].get(key) == producer_row, "completed producer row changed during analysis")
                receipt["producer_completed_after"] = current["cases"][key]
                summary_path = case_dir / "component_summary.json"
                summary = {"original_report": report_record, "components": summary_case(report, key)}
                atomic_json(summary_path, summary)
                receipt.update(status="completed", report=report_record, summary=file_record(summary_path), error=None)
            except Exception:
                receipt.update(status="analysis_failed", error=traceback.format_exc())
            receipt["CPU_process_wall_seconds"] = time.perf_counter() - started
            receipt["finished_UTC"] = datetime.now(timezone.utc).isoformat()
            atomic_json(case_dir / "receipt.json", receipt)
            receipt["receipt_file"] = file_record(case_dir / "receipt.json")
            state["cases"][key] = receipt
            atomic_json(status_path, state)
            print(json.dumps({"UTC": receipt["finished_UTC"], "key": key, "status": receipt["status"],
                              "CPU_process_wall_seconds": receipt["CPU_process_wall_seconds"],
                              "report": receipt.get("report"), "error": receipt.get("error")}), flush=True)
        current = json.loads(raw_status.read_bytes())
        check_producer_state(current, config, configuration_record)
        if state["all10_summary"] is None:
            summaries = collect_all_candidates(config, state["cases"], current, helper_record, configuration_record)
            if summaries is not None:
                bound(helper_record)
                bound(configuration_record)
                bound(controller_record)
                final_producers = json.loads(raw_status.read_bytes())
                check_producer_state(final_producers, config, configuration_record)
                require(all(final_producers["cases"].get(key) == current["cases"].get(key) for key in summaries),
                        "completed candidate rows changed during all-ten collection")
                for key in summaries:
                    bound(state["cases"][key]["report"])
                atomic_json(all10_path, {"schema_version": 1, "status": "all_ten_actual_candidate_reports_compared",
                                        "scientific_parity": "not_assessed", "configuration": configuration_record,
                                        "helper": helper_record, "coverage": sorted(summaries), "cases": summaries})
                state["all10_summary"] = file_record(all10_path)
        done = all(state["cases"].get(f"{version}/{case}", {}).get("status") == "completed" for version, case in pairs)
        state["status"] = "all_twelve_actual_pairs_compared" if done else "waiting_for_actual_completed_producers"
        state["updated_UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, state)
        if done:
            return
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-CON01-baseline", dest="reuse_CON01_baseline", type=Path, required=True)
    parser.add_argument("--reuse-CON01-baseline-sha256", dest="reuse_CON01_baseline_sha256", required=True)
    parser.add_argument("--poll-seconds", type=int, default=45)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
