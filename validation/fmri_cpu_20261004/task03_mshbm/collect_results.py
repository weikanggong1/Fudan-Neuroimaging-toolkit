"""Collect completed public reports and private clocks into aggregate CSV rows."""

import argparse
import csv
import io
import json
from pathlib import Path


def process_clock(path):
    fields = {}
    for line in path.read_text().splitlines():
        if ": " in line:
            key, value = line.strip().rsplit(": ", 1)
            fields[key] = value
    elapsed = fields.get("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    seconds = None
    if elapsed:
        seconds = 0.0
        for part in elapsed.split(":"):
            seconds = 60 * seconds + float(part)
    return {"process_wall_seconds": seconds,
            "process_user_seconds": float(fields["User time (seconds)"]),
            "process_system_seconds": float(fields["System time (seconds)"])}


def collect(root):
    rows = []
    for report_path in sorted(root.glob("*/report.public.json")):
        report = json.loads(report_path.read_text())
        if report.get("profiling_run"):
            continue
        name = report_path.parent.name
        clock = root / (name + ".clock.private.txt")
        row = {"scenario": name, "kind": report["kind"], "device": report["device"],
               "threads": report["threads"], "complete_source_frames": 490,
               "full_vertices": 64984, "cortex_vertices": 59412,
               "sessions": report.get("sessions", 2), "w": report["w"], "c": report["c"],
               "function_chain_seconds": report["function_chain_seconds"],
               "maximum_rss_kib": report["maximum_rss_kib"],
               "actual_core_sha256": report["source_sha256"]["src/fnit/mshbm/core.py"],
               "actual_volume_sha256": report["source_sha256"]["src/fnit/mshbm/volume.py"],
               "outer_em_iterations": json.dumps([[x["outer"], x["em"]] for x in report["history"]])}
        if clock.exists():
            row.update(process_clock(clock))
            row["non_function_chain_process_seconds"] = row["process_wall_seconds"] - row["function_chain_seconds"]
        row.update({"stage_" + key + "_seconds": value for key, value in report["stages_seconds"].items()})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for root in args.run_root for row in collect(root)]
    text = io.StringIO()
    keys = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(text, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    args.output.write_text(text.getvalue())
    print(json.dumps({"rows": len(rows), "source_frames": 490,
                      "non_function_chain_process_seconds_includes_import_validation_and_report_io": True}))


if __name__ == "__main__":
    main()
