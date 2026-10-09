"""核对两份完整白质报告的冻结输入、全部轨迹和输出SHA；不判单次提速。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--backend", default="torch", choices=("cpu", "torch"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [json.loads(path.read_text()) for path in
               (args.control_report, args.candidate_report)]
    rows = [report["python_runs"][args.backend] for report in reports]
    checks = {
        "same_frozen_input_sha256": reports[0]["input_sha256"] == reports[1]["input_sha256"],
        "inputs_unchanged": all(report["input_sha256"] == report["input_sha256_after"]
                                for report in reports),
        "both_complete": all(report["status"] == "complete" and row["status"] == "complete"
                             for report, row in zip(reports, rows)),
        "both_four_passes": all(row["stage"]["complete_four_passes"] and
                                len(row["stage"]["pass_ends"]) == 4 for row in rows),
        "both_final_zero_intersection": all(row["stage"]["cleanup"]["intersecting_faces_after"] == 0
                                            for row in rows),
        "same_all_coordinate_sha256": [x["coordinate_sha256"] for x in rows[0]["trace"]] ==
                                       [x["coordinate_sha256"] for x in rows[1]["trace"]],
        "same_all_trial_diagnostics": rows[0]["trace"] == rows[1]["trace"],
        "same_final_surface_file_sha256": rows[0]["output_sha256"] == rows[1]["output_sha256"],
        "same_final_MRI_file_sha256": rows[0]["output_volume_sha256"] == rows[1]["output_volume_sha256"],
    }
    result = {
        "scope": "complete_same_input_experimental_white_regression_not_speed_or_official_equivalence",
        "comparison_script_sha256": sha(Path(__file__)),
        "control_report_sha256": sha(args.control_report),
        "candidate_report_sha256": sha(args.candidate_report),
        "code_base_commits": [x["code_base_commit"] for x in reports],
        "actual_source_sha256": [x["source_sha256"] for x in reports],
        "checks": checks,
        "stage_API_seconds": [x["wall_seconds"] for x in rows],
        "steps": [len(x["trace"]) for x in rows],
        "performance_conclusion": "not_assessed; single full candidate integration run is not ABBA",
        "official_equivalence": "not_assessed",
        "whole_recon_all": "not_run",
        "strict_backend_regression": "passed" if all(checks.values()) else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
