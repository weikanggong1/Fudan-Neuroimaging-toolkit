"""Generate CPU-only step tables and brain figures after final cohort analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPECTED = {
    "analyze_cohort.py": "ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc",
    "extract_cohort_steps.py": "ca08aad54b81b3d6e7f6e826d205c7043081db6dc5136cb455b39374729e9eaa",
    "plot_cohort.py": "f135af099a67d543bfbefbbabbe8a71f1c6ba044133a7b5767d8368787e5b4ab",
}


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20 * 3600)
    args = parser.parse_args()
    root = args.root.resolve()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if hasattr(os, "sched_setaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, affinity[-2:])
    identities = {name: identity(root / name) for name in EXPECTED}
    if any(identities[name]["sha256"] != expected for name, expected in EXPECTED.items()):
        raise ValueError("Frozen analysis/step/plot source identity differs")
    status_path = root / "postprocess_status.json"
    if status_path.exists():
        raise ValueError("Preserve previous attempts; do not overwrite postprocess status")
    status = {"state": "waiting_for_final_analysis", "started_unix": time.time(),
              "script": identity(__file__), "sources": identities, "components": {},
              "cpu_only": True, "thread_limit": 2,
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    save(status_path, status)
    start = time.monotonic()
    try:
        while True:
            path = root / "analysis/analysis_status.json"
            analysis_status = json.loads(path.read_text()) if path.exists() else {}
            analysis_state = analysis_status.get("state")
            if analysis_state in ("completed", "completed_with_failures_or_unavailable"):
                break
            if analysis_state in ("failed", "partial_snapshot"):
                raise ValueError(f"Analysis is not final: {analysis_state}")
            if time.monotonic() - start > args.timeout_seconds:
                raise TimeoutError("Final cohort analysis wait expired")
            time.sleep(10)
        result_path = root / "analysis/cohort_analysis.json"
        result = json.loads(result_path.read_text())
        if not result.get("final_outcome_ready") or result.get("not_completed_attempts"):
            raise ValueError("Final cohort outcomes are not ready")
        status.update(state="processing", final_analysis=identity(result_path),
                      final_analysis_status=identity(path), wait_seconds=time.monotonic() - start)
        save(status_path, status)
        commands = (
            ("steps", [sys.executable, str(root / "extract_cohort_steps.py"), "--root", str(root), "--analysis", str(result_path)]),
            ("plots", [sys.executable, str(root / "plot_cohort.py"), "--root", str(root)]),
        )
        for name, command in commands:
            log_path = root / f"postprocess_{name}.log"
            record = {"command": command, "state": "running", "started_unix": time.time()}
            status["components"][name] = record
            save(status_path, status)
            begun = time.monotonic()
            with log_path.open("x") as output:
                child = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            record.update(exit_code=child.returncode, process_wall_seconds=time.monotonic() - begun,
                          finished_unix=time.time(), log=identity(log_path))
            if child.returncode:
                record["state"] = "failed"
                raise RuntimeError(f"CPU postprocess {name} failed; see preserved log")
            record["state"] = "completed"
            save(status_path, status)
        status.update(state="completed", finished_unix=time.time())
        save(status_path, status)
    except BaseException as error:
        status.update(state="failed", failure=repr(error), finished_unix=time.time())
        save(status_path, status)
        raise
    print(json.dumps({"state": status["state"], "status": str(status_path)}), flush=True)


if __name__ == "__main__":
    main()
