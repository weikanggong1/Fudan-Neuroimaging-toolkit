"""Export anonymous completed CLI evidence and fixed same-input precision gates.

No inference is performed. Source plans and full execution records stay private;
only anonymous case labels, resource settings, hashes and numerical results are
exported. An incomplete case is explicitly listed as pending.
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--feature", choices=("synthstrip", "synthsr"), required=True)
    parser.add_argument("--comparison-driver", type=Path, default=Path(__file__).with_name("compare_outputs.py"))
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error("use a fresh output directory")
    args.output_directory.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    report = {
        "schema": "fnit.smri.cpu.strip_sr.cli_summary.v1",
        "source_revision": plan.get("source_revision"),
        "feature": args.feature,
        "scope": "Cold complete CLI including imports, loading, inference and specified output writes; lock wait excluded",
        "startup_variation": "GPFS imports and file cache vary; internal network profiles are reported separately",
        "cases": [], "pending_cases": [],
    }
    jobs = {row["id"]: row for row in plan["jobs"]}
    for case in plan["cases"]:
        if case["feature"] != args.feature:
            continue
        # Exact case matching avoids mixing v1 with v1_precedence, or no_flip
        # with no_flip_no_sharpen. Each execution has one index and one arm.
        case_jobs = [row for key, row in jobs.items()
                     if re.fullmatch(re.escape(case["id"]) + r"_\d+_(reference|baseline)", key)]
        records = []
        for job in case_jobs:
            record_path = args.runs_root / job["id"] / "record.json"
            if not record_path.is_file():
                break
            record = json.loads(record_path.read_text())
            if record["status"] != "complete" or record.get("returncode") != 0:
                break
            if record["hostname"].split(".")[0] != "nodecw10":
                raise ValueError("CPU timing was recorded on the wrong host")
            if record["max_cpu_threads"] != 8 or record["cpu_affinity"] != "0,4,8,12,16,20,24,28":
                raise ValueError("CPU timing resources differ from the fixed protocol")
            records.append(record)
        if len(records) != len(case_jobs):
            report["pending_cases"].append(case["id"])
            continue
        row = {"case": case["id"], "data_description": case.get("data_description"),
               "candidate_options": case.get("candidate_options", []), "executions": [], "comparisons": []}
        for job, record in zip(case_jobs, records):
            arm = "reference" if job["id"].endswith("_reference") else "fnit"
            row["executions"].append({"id": job["id"], "arm": arm,
                                      **{key: record.get(key) for key in ("hostname", "max_cpu_threads", "cpu_affinity", "wall_seconds",
                                            "maximum_sampled_tree_rss_bytes", "maximum_sampled_tree_threads", "returncode")}})
        reference_jobs = [job for job in case_jobs if job["id"].endswith("_reference")]
        reference_job = reference_jobs[0]
        comparisons = [job for job in case_jobs if job != reference_job]
        for other_job in comparisons:
            comparison_path = args.output_directory / (other_job["id"] + ".public.json")
            command = [sys.executable, str(args.comparison_driver),
                       "--feature", args.feature, "--case", other_job["id"],
                       "--reference-image", reference_job["expected_outputs"][0],
                       "--candidate-image", other_job["expected_outputs"][0],
                       "--gate", "strip_official" if args.feature == "synthstrip" else "sr_official",
                       "--report", str(comparison_path)]
            if args.feature == "synthstrip":
                command += ["--reference-mask", reference_job["expected_outputs"][1],
                            "--candidate-mask", other_job["expected_outputs"][1],
                            "--reference-distance", reference_job["expected_outputs"][2],
                            "--candidate-distance", other_job["expected_outputs"][2]]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            result = json.loads(comparison_path.read_text())
            row["comparisons"].append({"candidate": other_job["id"],
                                       "role": "official_repeat" if other_job in reference_jobs else "fnit_vs_official",
                                       "result": result})
        row["all_fixed_gates_pass"] = all(item["result"]["fixed_gate"]["passes"] for item in row["comparisons"])
        reference_times = [item["wall_seconds"] for item in row["executions"] if item["arm"] == "reference"]
        candidate_times = [item["wall_seconds"] for item in row["executions"] if item["arm"] == "fnit"]
        row["reference_median_seconds"] = statistics.median(reference_times)
        row["fnit_median_seconds"] = statistics.median(candidate_times)
        row["observed_complete_process_ratio"] = row["reference_median_seconds"] / row["fnit_median_seconds"]
        report["cases"].append(row)
    report["all_planned_cases_complete"] = not report["pending_cases"]
    report["all_completed_cases_pass"] = all(row["all_fixed_gates_pass"] for row in report["cases"])
    (args.output_directory / "summary.public.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed": len(report["cases"]), "pending": report["pending_cases"],
                      "gates_pass": report["all_completed_cases_pass"]}))


if __name__ == "__main__":
    main()
