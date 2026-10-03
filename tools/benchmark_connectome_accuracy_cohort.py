"""Run an accuracy phase from raw DWI with already completed official anatomy.

This stdlib coordinator reuses the existing real raw-BIDS wall worker. It
does not read reference arrays into FNIT, change seeds, or reconstruct T1.
Run on the GPU host: python TOOL --configuration FROZEN_CONFIGURATION.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

try:
    from . import benchmark_connectome_raw_cohort as cohort
except ImportError:
    import benchmark_connectome_raw_cohort as cohort


def bound(record):
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file() or cohort.sha256(path) != record["sha256"]:
        raise ValueError(f"frozen input identity differs: {path}")
    return json.loads(path.read_bytes())


def validate_plan(config, manifest, bindings):
    cases = cohort.validate_manifest(manifest)
    by_case = {case["case_id"]: case for case in cases}
    if set(bindings["cases"]) != set(by_case):
        raise ValueError("actual prior case bindings do not cover the raw cohort")
    versions = set(config["sources"])
    if versions != {"baseline", "candidate"}:
        raise ValueError("accuracy phase needs explicit baseline and candidate sources")
    seen = set()
    for item in config["execution_order"]:
        if set(item) != {"version", "case_id"} or item["version"] not in versions or item["case_id"] not in by_case:
            raise ValueError("unknown execution-order fields, source or case")
        key = (item["version"], item["case_id"])
        if key in seen:
            raise ValueError("duplicate run cannot be treated as a repeat")
        seen.add(key)
    if {case for version, case in seen if version == "candidate"} != set(by_case):
        raise ValueError("candidate must cover all ten raw cases")
    if any(version == "baseline" and ("candidate", case) not in seen for version, case in seen):
        raise ValueError("timing baseline has no matched candidate")
    return by_case


def execute(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    manifest, bindings = bound(config["raw_manifest"]), bound(config["input_bindings"])
    cases = validate_plan(config, manifest, bindings)
    if cohort.sha256(__file__) != config["accuracy_coordinator_sha256"]:
        raise ValueError("accuracy coordinator changed after freeze")
    root = cohort.require_fresh(config["run_root"])
    state = {
        "status": "running", "start_utc": cohort.utc(), "cases": {},
        "scope": "fresh raw-DWI accuracy phase; completed official recon-all reused explicitly",
        "raw_manifest": config["raw_manifest"], "input_bindings": config["input_bindings"],
        "configuration": {"path": str(config_path), "sha256": cohort.sha256(config_path)},
        "execution_order": config["execution_order"],
        "scientific_parity": "not_assessed", "speed_acceptance": "not_assessed",
    }
    state_path = root / "status.json"
    cohort.atomic_json(root / "configuration.json", config)
    cohort.atomic_json(state_path, state)
    config["frozen_sources"] = {name: cohort.source_manifest(path) for name, path in config["sources"].items()}
    config["wall_script_sha256"] = cohort.sha256(config["wall_script"])
    if config["frozen_sources"] != config["declared_source_manifests"]:
        raise ValueError("actual source inventories differ from frozen phase declaration")
    cohort.atomic_json(root / "frozen_sources.json", config["frozen_sources"])

    def supplied_anatomy(actual_config, case, version, job):
        record = bindings["cases"][case["case_id"]]
        prior = bound(record["prior_gpu_report"])
        if prior.get("status") != "completed" or prior.get("anatomy") != record["anatomy"]["files"]:
            raise ValueError("prior completed official anatomy contract differs")
        return {"anatomy": record["anatomy"]["files"]}

    def subject_directory(actual_config, case, job):
        return bindings["cases"][case["case_id"]]["anatomy"]["directory"]

    started = time.perf_counter()
    failed = False
    for item in config["execution_order"]:
        name, case_id = item["version"], item["case_id"]
        key = f"{name}/{case_id}"
        record = bindings["cases"][case_id]
        state["cases"][key] = {"status": "running", "start_utc": cohort.utc()}
        cohort.atomic_json(state_path, state)
        before = time.perf_counter()
        extra = ("--result-export-dir", str(root / name / case_id / "returned_result"))
        try:
            report = cohort.worker({"action": "gpu", "config": config, "case": cases[case_id], "version": name},
                                   anatomy_loader=supplied_anatomy, anatomy_subject=subject_directory,
                                   extra_wall_arguments=extra)
            state["cases"][key] = {
                "status": report["status"], "gpu_report": str(root / name / case_id / "gpu_report.json"),
                "wall_report": report.get("wall_report"),
                "actual_source_fingerprint": report.get("source_before", {}).get("source_fingerprint"),
                "supplied_anatomy": record["anatomy"],
                "raw_dwi_cli_total_runtime_seconds": report.get("raw_dwi_cli_total_runtime_seconds"),
                "gpu_lock_queue_seconds": report.get("gpu_lock_queue_seconds"),
                "driver_run_seconds_including_queue_and_exports": time.perf_counter() - before,
                "memory_budget": report.get("memory_budget"), "error": report.get("error"),
            }
            if report["status"] != "completed":
                failed = True
        except Exception as error:
            state["cases"][key] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            failed = True
        cohort.atomic_json(state_path, state)
    state.update(status="failed" if failed else "execution_completed", end_utc=cohort.utc(),
                 driver_makespan_seconds=time.perf_counter() - started)
    cohort.atomic_json(state_path, state)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    return execute(parser.parse_args().configuration)


if __name__ == "__main__":
    raise SystemExit(main())
