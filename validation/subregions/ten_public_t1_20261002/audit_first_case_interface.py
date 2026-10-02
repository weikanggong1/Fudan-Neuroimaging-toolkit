"""Read-only real first-case comparison to catch interfaces before cohort publication."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ANALYZER_SHA256 = "ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc"


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "first_case_interface_audit.json"
    if output.exists():
        raise ValueError("Preserve the existing first-case audit")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    path = root / "analyze_cohort.py"
    if identity(path)["sha256"] != ANALYZER_SHA256:
        raise ValueError("Frozen analyzer identity differs")
    spec = importlib.util.spec_from_file_location("frozen_first_case_interface", path)
    analyzer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = analyzer
    spec.loader.exec_module(analyzer)
    manifest_path = root / "cohort_manifest.json"
    manifest = analyzer.read_json(manifest_path)
    case = next(case for case in manifest["cases"] if case["id"] == "sub-01")
    canonical = analyzer.canonical_labels(manifest, root)
    runtime, source_audit = analyzer.audit_source(manifest, root)
    queue = analyzer.merged_queues([root / name for name in ("fnit_queue.json", "fnit_stage_queue.json", "official_queue.json")])
    result = {"state": "auditing", "case_id": case["id"], "planned_subjects": 10,
              "audited_subjects": 1, "final_full_cohort_benchmark": False,
              "scope": "First real completed official and FNIT raw/stage results only; interface validation, not final cohort outcome.",
              "cpu_only": True, "thread_limit": 2, "started_unix": time.time(),
              "script": identity(__file__), "frozen_analyzer": identity(path),
              "manifest": identity(manifest_path), "source_audit": source_audit}
    analyzer.save_json(output, result)
    started = time.monotonic()
    try:
        record, rows, groups = analyzer.audit_case(case, queue, root, canonical, runtime)
        if record["status"] != "completed" or len(rows) != 440 or len(groups) != 24:
            raise ValueError("First case does not have all 440 ROI-space and 24 family-space results")
        if not all(row["measurement_status"] == "evaluated" for row in rows):
            raise ValueError("Missing first-case measurements")
        result.update(state="passed", record=record, roi_rows=rows, family_rows=groups,
                      completed_seconds=time.monotonic() - started, finished_unix=time.time())
        analyzer.save_json(output, result)
    except BaseException as error:
        result.update(state="failed", failure=repr(error), completed_seconds=time.monotonic() - started,
                      finished_unix=time.time())
        analyzer.save_json(output, result)
        raise
    print(json.dumps({"state": result["state"], "roi_space_rows": len(rows), "family_space_rows": len(groups),
                      "seconds": result["completed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
