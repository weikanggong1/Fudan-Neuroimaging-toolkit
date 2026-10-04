"""Export scalar/hash evidence from the two completed CPU API runs.

Never exports source/input paths, argv, credentials, images, arrays, subject IDs,
process IDs or raw logs. This script performs no inference or array loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.server_root / "workspaces/smri_cpu_20261004"
    code = workspace / "task1_memory_v1"
    run = args.server_root / "runs/smri_cpu_20261004/task1_memory_v1"
    result = {
        "schema": "fnit.smri.memory_api.public.v1", "date": "2026-10-04",
        "scope": "one real fully materialized case01 default CPU API per feature; archived references reused",
        "case_alias": "case01", "source_dataset": "OpenNeuro ds003138 v1.0.1",
        "privacy": "allowlisted scalars, public resource names and hashes only; no images, arrays, private paths, subject IDs, argv, process IDs or raw logs",
        "code_sha256": {name: sha256(code / name) for name in ("worker.py", "prepare_jobs.py", "compare_outputs.py")},
        "collector_sha256": sha256(__file__),
        "runner_sha256": sha256(workspace / "queue_runner_v3.py"),
        "private_plan_sha256": sha256(code / "jobs.private.json"),
        "features": {}}
    allow = ("schema", "status", "feature", "case", "worker_sha256", "config_sha256",
             "source", "resources", "comparator_sha256", "saved_reference_sha256",
             "canonical_index_sha256_before", "runtime", "materialized_input", "model",
             "network_bindings", "output_sha256", "stages", "unchanged_after",
             "comparisons_complete", "all_fixed_gates_passed",
             "overall_numerical_equivalence", "worker_seconds_including_posthoc", "timing_scope")
    for feature in ("synthstrip", "synthsr"):
        private_path = run / feature / "report.private.json"
        private = json.loads(private_path.read_text())
        if private["status"] != "complete":
            raise ValueError("retain failed job privately; scalar collection requires completion: " + feature)
        report = {key: private[key] for key in allow if key in private}
        report["private_report_sha256"] = sha256(private_path)
        report["comparisons"] = {arm: {key: row[key] for key in
                                 ("returncode", "seconds", "report_sha256", "result", "scientific_geometry")
                                 if key in row} for arm, row in private["comparisons"].items()}
        record_path = run / "queue" / (feature + "_memory_default_cpu8") / "record.json"
        record = json.loads(record_path.read_text())
        if record["status"] != "complete" or record["returncode"] != 0:
            raise ValueError("runner receipt is incomplete: " + feature)
        report["runner"] = {key: record[key] for key in (
            "job_sha256", "hostname", "max_cpu_threads", "cpu_affinity", "status",
            "started_utc", "finished_utc", "wall_seconds", "returncode",
            "maximum_sampled_tree_rss_bytes", "maximum_sampled_tree_threads",
            "resource_samples", "load_before", "load_after")}
        report["runner"]["receipt_sha256"] = sha256(record_path)
        report["runner"]["thread_environment"] = {key: value for key, value in record["environment"].items()
                                                        if key != "PYTHONPATH"}
        report["runner"]["expected_outputs"] = [{"name": Path(row["path"]).name,
                                                   "exists": row["exists"], "bytes": row["size"]}
                                                  for row in record["outputs"]]
        time_path = record_path.parent / "time.txt"
        report["runner"]["gnu_time_sha256"] = sha256(time_path)
        fields = ("User time (seconds)", "System time (seconds)", "Percent of CPU this job got",
                  "Elapsed (wall clock) time (h:mm:ss or m:ss)", "Maximum resident set size (kbytes)",
                  "Exit status")
        times = {}
        for line in time_path.read_text().splitlines():
            for field in fields:
                marker = field + ":"
                if line.strip().startswith(marker):
                    times[field] = line.strip()[len(marker):].strip()
        report["runner"]["gnu_time"] = times
        stages = {row["name"]: row["seconds"] for row in report["stages"]}
        report["timing_derived"] = {
            "metadata_decode_copy_construct_seconds": sum(stages[name] for name in (
                "load_image_metadata", "decode_original_values", "copy_decoded_array_K_order",
                "construct_materialized_spatial_image")),
            "api_seconds": stages["default_api_call"],
            "save_same_call_outputs_seconds": stages["save_same_call_outputs"],
            "api_and_save_seconds": stages["default_api_call"] + stages["save_same_call_outputs"],
            "archived_reference_comparison_seconds": stages["compare_saved_reference_outputs"],
            "basis": "nonoverlapping named phases; observed process wall includes posthoc work and is not a new paired speed comparison"}
        result["features"][feature] = report
    encoded = json.dumps(result, indent=2, allow_nan=False)
    prohibited = ("/cwStorage/", "/home/", "/mnt/", "10.190.", "gongwk@", "private_paths", '"argv"', '"pid"')
    if any(token in encoded for token in prohibited):
        raise ValueError("public report failed path/identity allowlist check")
    print(encoded)


if __name__ == "__main__":
    main()
