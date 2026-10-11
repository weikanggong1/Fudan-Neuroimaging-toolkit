"""从完成收据提取两例墙钟、逐阶段时间、程序审计及显存边界。

receipt_root：capture/export产生的JSON目录；summary_path、stage_csv为新文件。
报告仅使用真实原始T1运行的终态，不评价整体等效、不重算影像或补跑。
每行秒数为同期父stage墙钟，双侧/内部时间已包含，不重复相加。
输入缺失、状态仍running、输出已存在会失败；没有对应官方命令。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(*, receipt_root: Path, summary_path: Path, stage_csv: Path) -> dict:
    for path in (summary_path, stage_csv):
        if path.exists():
            raise FileExistsError(path)
    capture_path = receipt_root/"CAPTURE.json"
    captured = json.loads(capture_path.read_text())
    result = {"scope": "two completed raw T1 attempts; not completed reconstruction",
              "producer_code_version": None, "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
              "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "time_units": "seconds", "memory_budget_bytes": 20_000_000_000,
              "overall_metric_equivalence": "not_assessed", "clean_environment_isolation": "not_verified",
              "performance_speedup_of_successful_full_workflow": None, "cases": {}}
    rows = []
    for case in ("sub06", "sub07"):
        benchmark_path = receipt_root/f"{case}-candidate/benchmark.json"
        runtime_path = receipt_root/f"{case}-candidate/subject/fnit-native-free-run.json"
        benchmark = json.loads(benchmark_path.read_text())
        runtime = json.loads(runtime_path.read_text())
        if runtime["status"] not in ("complete", "failed"):
            raise RuntimeError(f"{case} has no final execution state")
        if benchmark["runtime_status"] != runtime["status"]:
            raise RuntimeError(f"{case} final status mismatch")
        if result["producer_code_version"] is None:
            result["producer_code_version"] = benchmark["code_version"]
        elif result["producer_code_version"] != benchmark["code_version"]:
            raise RuntimeError("paired producer versions differ")
        memory = benchmark["process_memory"]
        audit = benchmark["execution_audit"]
        data = {"benchmark_public_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
                "runtime_public_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
                "input_sha256": benchmark["input_sha256"],
                "harness_script_sha256": benchmark["script_sha256"],
                "native_free_source_sha256": benchmark["source_sha256"]["recon_all/native_free.py"],
                "status": runtime["status"], "exit_code": benchmark["exit_code"],
                "failed_stage": runtime.get("failed_stage"),
                "full_workflow_complete": captured["cases"][case]["full_workflow_complete"],
                "stage_count": len(benchmark["stages"]),
                "api_wall_seconds": benchmark["api_wall_seconds"],
                "cli_wall_seconds": benchmark["cli_wall_seconds"],
                "full_harness_seconds": benchmark["full_harness_seconds"],
                "target_gpu_uuid": memory["target_gpu_uuid"],
                "memory_ownership_status": memory["status"],
                "peak_tree_total_bytes": memory["peak_tree_total_bytes"],
                "peak_target_device_used_bytes_includes_other_users": memory["peak_target_device_used_bytes"],
                "memory_sampling_requested_seconds": memory["sampling_interval_seconds"],
                "memory_sampling_max_actual_seconds": memory["max_observed_interval_seconds"],
                "memory_failed_sample_count": len(memory["failed_samples"]),
                "memory_budget_passed": None,
                "original_program_sha256": benchmark["native_program_sha256"],
                "python_launch_audit_events": audit["events"],
                "original_program_attempts": audit["original_program_attempts"],
                "python_launch_audit_passed": audit["passed"],
                "output_completeness": benchmark["output_completeness"],
                "strict_reproduction": benchmark["strict_reproduction"],
                "optimization_regression": benchmark["optimization_regression"],
                "overall_metric_equivalence": benchmark["overall_metric_equivalence"]}
        groups = {}
        for group_path in sorted((runtime_path.parent/"scripts").glob("*.hemisphere-group.json")):
            group = json.loads(group_path.read_text())
            groups[group["operation"]] = {"status": group["status"],
                "group_wall_seconds": group.get("group_wall_seconds"),
                "workers": {hemi: {key: worker.get(key) for key in (
                    "status", "error", "failure_phase", "cuda_oom", "cuda_allocator")}
                    for hemi, worker in group.get("workers", {}).items()}}
        data["hemisphere_groups"] = groups
        result["cases"][case] = data
        for stage in benchmark["stages"]:
            rows.append({"case": case, "stage": stage["name"], "wall_seconds": stage["seconds"],
                         "outcome": "failed" if "error" in stage else "complete",
                         "error": stage.get("error", ""), "whole_execution_status": runtime["status"]})
    summary_path.write_text(json.dumps(result, indent=2)+"\n")
    with stage_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "stage", "wall_seconds", "outcome", "error", "whole_execution_status"),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--stage-csv", required=True, type=Path)
    args = parser.parse_args()
    summarize(receipt_root=args.receipt_root, summary_path=args.summary_path, stage_csv=args.stage_csv)
