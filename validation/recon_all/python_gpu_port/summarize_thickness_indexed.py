"""用两例双侧的既有八轮报告生成厚度摘要，不运行影像算法。

reports 为 performance_20261001，final-source 指向生产厚度源码，output
为新 JSON 路径。读取固定四份冻结配对、监控、可用的 Conda maps 对照。
固定检查优化门槛1e-6mm/相对0；另记录现有Conda门槛.005+.001rel。
写出实测源版本、最终源哈希、逐轮误差/耗时/采样范围；不推断整例提速。
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--final-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (("sub-01", "lh", "thickness_lh"), ("sub-01", "rh", "thickness_rh"),
             ("sub-02", "lh", "sub02_thickness_lh"), ("sub-02", "rh", "sub02_thickness_rh"))
    rows = []
    for subject, hemi, name in cases:
        path = args.reports / "diagnostics" / name / "report.json"
        report = json.loads(path.read_text())
        assert report["pass"] and len(report["runs"]) == 2
        assert report["thresholds"]["optimization"] == {"absolute_mm": 1e-6, "relative": 0}
        assert report["source_sha256"]["candidate"] == "fb3b665150a2c78c0a8a3a5813f29648e71b74efc7474f4b6c318489862e0136"
        monitor_path = args.reports / "diagnostics" / (name + "_monitor") / "monitor.json"
        monitor = json.loads(monitor_path.read_text())
        assert monitor["exit_code"] == 0
        runs = [{"repeat": run["repeat"], "order": run["order"],
                 "seconds_including_io": {k: v["seconds_including_io"]
                                           for k, v in run["implementations"].items()},
                 "comparison": run["optimization_comparison"]} for run in report["runs"]]
        assert all(run["comparison"]["pass"] for run in runs)
        dense = statistics.mean(run["seconds_including_io"]["dense"] for run in runs)
        indexed = statistics.mean(run["seconds_including_io"]["indexed"] for run in runs)
        row = {"subject": subject, "hemi": hemi, "vertices": report["vertices"],
               "faces": report["faces"], "report": str(path.relative_to(args.reports)),
               "report_sha256": sha256(path), "input_sha256": report["input_sha256"],
               "measured_code_commit": report["code_commit"], "source_sha256": report["source_sha256"],
               "host": report["host"], "device": report["device"], "threads": report["threads"],
               "precision": report["precision"], "cache_enabled": report["cuda_allocator_cache_enabled"],
               "runs": runs, "mean_seconds": {"dense": dense, "indexed": indexed},
               "observed_stage_reduction_percent": 100 * (1 - indexed / dense),
               "monitor": {"scope": "whole paired process, includes dense and indexed",
                           "report": str(monitor_path.relative_to(args.reports)),
                           "report_sha256": sha256(monitor_path),
                           **{k: monitor[k] for k in ("gpu_uuid", "peak_sampled_process_bytes", "samples",
                                                     "maximum_sampling_gap_seconds", "failed_app_queries",
                                                     "continuous_peak_verified", "command_wall_seconds")}}}
        conda_path = args.reports / "diagnostics" / f"{name}_vs_conda.json"
        if conda_path.exists():
            conda = json.loads(conda_path.read_text())
            assert conda["input_sha256"] == row["input_sha256"]
            assert conda["pair_source_sha256"] == row["source_sha256"]
            row["conda_source_build_comparison"] = {"report": str(conda_path.relative_to(args.reports)),
                    "report_sha256": sha256(conda_path), "pass": conda["pass"],
                    "threshold": conda["threshold"], "rows": conda["rows"]}
        else:
            row["conda_source_build_comparison"] = "not_assessed"
        rows.append(row)
    summary = {"scope": "frozen same-input surface thickness; 2 subjects x 2 hemispheres x 2 repeats",
               "script_sha256": sha256(Path(__file__)), "measured_candidate_sha256": rows[0]["source_sha256"]["candidate"],
               "final_default_source_sha256": sha256(args.final_source),
               "final_default_measurement_status": "not represented by these stage1 measurements",
               "optimization_threshold": {"absolute_mm": 1e-6, "relative": 0},
               "optimization_regression_pass": all(r["comparison"]["pass"] for row in rows for r in row["runs"]),
               "strict_old_new_equal": all(r["comparison"]["different_values"] == 0 for row in rows for r in row["runs"]),
               "new_official_validation": "not_run", "whole_pipeline_speedup": "not_measured_here",
               "overall_equivalence": "not_assessed", "cases": rows}
    with args.output.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "repeats": 8, "optimization_regression_pass": summary["optimization_regression_pass"]}))


if __name__ == "__main__":
    main()
