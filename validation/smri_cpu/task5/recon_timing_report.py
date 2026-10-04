"""Export CPU recon timing without private paths or command arguments.

FreeSurfer command timers may be nested. Their rows are kept individually and
are never summed as the official end-to-end wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(path):
    value = json.loads(path.read_text())
    return {"status": value["status"], "hostname": value["hostname"],
            "threads_budget": value["max_cpu_threads"],
            "cpu_affinity": value["cpu_affinity"], "wall_seconds": value.get("wall_seconds"),
            "returncode": value.get("returncode"),
            "maximum_sampled_tree_rss_bytes": value.get("maximum_sampled_tree_rss_bytes"),
            "maximum_sampled_tree_threads": value.get("maximum_sampled_tree_threads"),
            "load_before": value.get("load_before"), "load_after": value.get("load_after"),
            "started_utc": value.get("started_utc"), "finished_utc": value.get("finished_utc"),
            "receipt_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-record", type=Path, required=True)
    parser.add_argument("--official-log", type=Path, required=True)
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official, candidate = receipt(args.official_record), receipt(args.candidate_record)
    report = json.loads(args.candidate_report.read_text())
    rows = []
    for index, line in enumerate(args.official_log.read_text().splitlines()):
        if "@#@FSTIME" not in line:
            continue
        tokens = line.split()
        match = re.search(r"\be\s+([0-9.]+)\s+S\s+([0-9.]+)\s+U\s+([0-9.]+)\b", line)
        if not match or len(tokens) < 3:
            raise ValueError("unrecognized official timer at line " + str(index + 1))
        rows.append({"log_line": index + 1, "timestamp": tokens[1],
                     "program": Path(tokens[2]).name,
                     "elapsed_seconds": float(match[1]),
                     "system_cpu_seconds": float(match[2]),
                     "user_cpu_seconds": float(match[3])})
    if not rows:
        raise ValueError("official log contains no command timers")
    stages = []
    keys = ("name", "seconds", "function_seconds", "parent_cpu_seconds", "child_cpu_seconds",
            "cuda_pre_sync_seconds", "cuda_post_sync_seconds", "cuda_synchronized")
    for value in report["stages"]:
        stages.append({key: value[key] for key in keys if key in value})
    result = {"source_label": args.source_label, "source_archive_sha256": args.archive_sha256,
              "input_sha256": "eb2bc2ff1f30441b0aff54685cfdd7f196bd02dccad4698100a8bad40421de22",
              "official_version": "FreeSurfer8.2.0-1 FS_V8_XOPTS=1", "official": official,
              "candidate": candidate, "candidate_report_status": report.get("status"),
              "candidate_report_sha256": digest(args.candidate_report),
              "official_log_sha256": digest(args.official_log),
              "official_command_timers": rows, "candidate_stages": stages,
              "official_command_timer_scope": "individual command timers may include child commands; no sum or one-to-one stage equivalence inferred",
              "wall_time_scope": "cold process including imports, validation, computation, data/model reads and output writes; excludes queue and posthoc scoring",
              "candidate_api_seconds": report.get("total_seconds"),
              "output_validation": report.get("output_validation"),
              "mesh_validation": report.get("mesh_validation"),
              "numerical_equivalence": "not_assessed", "equivalent_reconstruction_speedup": None,
              "shared_node": True}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
