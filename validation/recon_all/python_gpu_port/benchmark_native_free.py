"""Summarize one native-free run against an archived same-T1 FreeSurfer log."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


_FSTIME = re.compile(r"^@#@FSTIME\s+\S+\s+(\S+)\s+.*?\be\s+([0-9.]+)\b")
_HOURS = re.compile(r"#@#%# recon-all-run-time-hours\s+([0-9.]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(run_file: Path, official_log: Path, comparison: Path,
              input_t1: Path, source: Path | None = None) -> dict:
    run = json.loads(run_file.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"candidate is not complete: {run.get('status')}")
    compared = json.loads(comparison.read_text())
    native_steps: dict[str, dict] = {}
    total_hours = None
    for line in official_log.read_text(errors="replace").splitlines():
        match = _FSTIME.search(line)
        if match:
            row = native_steps.setdefault(match[1], {"calls": 0, "seconds": 0.0})
            row["calls"] += 1
            row["seconds"] += float(match[2])
        match = _HOURS.search(line)
        if match:
            total_hours = float(match[1])
    if total_hours is None:
        raise ValueError("official log lacks recon-all-run-time-hours")
    candidate_seconds = float(run["total_seconds"])
    official_seconds = total_hours * 3600
    return {
        "input_t1": str(input_t1), "input_sha256": _sha256(input_t1),
        "candidate_source_sha256": _sha256(source) if source else None,
        "official_log": str(official_log), "official_log_sha256": _sha256(official_log),
        "candidate_run": str(run_file), "candidate_run_sha256": _sha256(run_file),
        "comparison": str(comparison), "comparison_sha256": _sha256(comparison),
        "official_total_seconds": official_seconds,
        "candidate_total_seconds": candidate_seconds,
        "candidate_over_official_time": candidate_seconds / official_seconds,
        "candidate_stages": run["stages"],
        "official_executable_elapsed_seconds": native_steps,
        "compared_files": compared["compared_files"],
        "missing_candidate_files": len(compared["missing_candidate"]),
        "caveat": "Archived official run and current candidate are not a paired cold-start trial; reconstructed outputs differ.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--official-log", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--input-t1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    report = summarize(args.candidate_run, args.official_log,
                       args.comparison, args.input_t1, args.source)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"candidate {report['candidate_total_seconds']:.1f}s; "
          f"official {report['official_total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
