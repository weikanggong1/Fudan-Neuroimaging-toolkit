"""Export measured task2 results without credentials or private run arguments."""

import argparse
import hashlib
import json
from pathlib import Path


def collect(run_root, workspace):
    report = {"schema": "fnit_smri_cpu_seg_evidence/v1", "records": [],
              "comparisons": {}, "preprocessing": {}, "source_files": {}}
    groups = ["corrected_numa1_records_v1", "corrected_case02_records_v1",
              "selective_numa1_v1_records", "gpu_corrected_records_v1",
              "gpu_selective_v1_records"]
    for group in groups:
        for path in sorted((run_root / group).glob("*/record.json")):
            record = json.loads(path.read_text())
            row = {key: record.get(key) for key in (
                "status", "hostname", "cpu_affinity", "max_cpu_threads",
                "started_utc", "finished_utc", "wall_seconds", "returncode",
                "maximum_sampled_tree_rss_bytes", "maximum_sampled_tree_threads",
                "load_before", "load_after")}
            row.update(group=group, job_id=path.parent.name,
                       job_sha256=record.get("job_sha256"))
            time_file = path.parent / "time.txt"
            if time_file.exists():
                for line in time_file.read_text().splitlines():
                    if "Maximum resident set size (kbytes):" in line:
                        row["time_maximum_rss_kbytes"] = int(line.rsplit(":", 1)[1])
            guard = list(run_root.glob("*/*" + path.parent.name + ".guard.json"))
            if guard:
                metrics = json.loads(guard[0].read_text())
                row["gpu"] = {key: metrics[key] for key in (
                    "cli_api_seconds", "api_seconds_excluding_preprocessing",
                    "max_allocated_bytes", "max_reserved_bytes", "memory_budget_bytes")
                    if key in metrics}
            report["records"].append(row)
    for path in sorted((run_root / "comparisons").glob("*.json")):
        if path.name.startswith(("corrected_numa1_", "case02_", "selective_",
                                 "gpu_corrected_", "gpu_selective_", "ctab_corrected_")):
            report["comparisons"][path.name] = json.loads(path.read_text())
    for index in (1, 2, 3):
        item = {}
        for prefix in ("official", "fnit"):
            name = (f"official_case{index:02d}.json" if prefix == "official"
                    else f"fnit_case{index:02d}_corrected.json")
            path = run_root / "preprocess" / name
            if path.exists():
                item[prefix] = json.loads(path.read_text())
        report["preprocessing"][f"case{index:02d}"] = item
    for freeze in ("baseline_corrected", "candidate_corrected", "candidate_selective",
                   "baseline_final_frontend"):
        source = workspace / freeze / "src/fnit"
        if not source.exists():
            continue
        paths = list((source / "synthseg_parc").glob("*.py"))
        paths += [source / "_nib.py", source / "cli.py"]
        report["source_files"][freeze] = {
            str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}
    report["precision_acceptance"] = {
        "case01_cpu_soft_volumes": "not_assessed: no prospective numerical gate",
        "official_bitwise_equivalence": "not achieved on case01",
        "case02": "prospective acceptance_case02.json; inspect completed results separately"}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.run_root, args.workspace), indent=2, ensure_ascii=False))
