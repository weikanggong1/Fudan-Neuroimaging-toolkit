"""仅从本版真实报告生成数值汇总，不改验收值或报告身份。"""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.report_root / "reports/benchmark.json").read_text())
    if report["status"] != "complete" or report["scope"] != "five frozen real same-input CPU states; not whole recon-all":
        raise ValueError("report is not the completed frozen-state benchmark")
    columns = ["name", "vertices", "faces", "pending_branch_triggered", "changed_vertices",
               "different_normal_elements", "max_normal_error", "p99_normal_error",
               "first_candidate_API_seconds_including_topology",
               "repeated_candidate_API_seconds_including_topology",
               "Torch_CPU_normal_reference_seconds", "case_wall_seconds_including_read_hash_repeat_compare"]
    with (args.report_root / "stage_metrics.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in report["cases"])
    print(json.dumps({"states": len(report["cases"]), "numeric_values": "unchanged",
                      "pending_triggered": sum(row["pending_branch_triggered"] for row in report["cases"])}))


if __name__ == "__main__":
    main()
