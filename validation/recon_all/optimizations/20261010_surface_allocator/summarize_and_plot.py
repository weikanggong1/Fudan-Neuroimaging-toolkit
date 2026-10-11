"""只读汇总真实缓存实验的完整阶段计时与资源；不重跑算法或写影像。

输入为完整 summary JSON 与只读严格评价 JSON；输出目录必须不存在。
秒为墙钟，显存为字节；整卡采样包含其他进程，归属不明仍为 null。
生成数字摘要、CSV 和计时/负载图。没有原软件对应 CLI。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(*, summary_json: Path, evaluation_json: Path | None, output_dir: Path) -> dict:
    """汇总已完整完成各阶段；不把历史墙钟或阶段比率用于整例。"""
    value = json.loads(summary_json.read_text())
    if value["status"] != "complete":
        raise ValueError("summary producer is not complete")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not value.get("items") or any(row["returncode"] or row["child"]["status"] != "complete"
                                      for row in value["items"]):
        raise ValueError("all requested complete-stage items must have exit 0")
    output_dir.mkdir(parents=True)
    rows = []
    for item in value["items"]:
        child, sampler = item["child"], item["sampler"]
        row = {"index": item["index"], "mode": item["mode"], "policy": item["policy"],
               "cpu_threads": child["threads"], "accepted_steps": len(child["accepted_steps"]),
               "complete_api_seconds": child["complete_api_seconds"],
               "fresh_exec_wall_seconds": item["fresh_exec_wall_seconds"],
               "copy_and_validate_seconds": item["copy_and_validate_seconds"],
               "complete_item_wall_seconds": item["complete_item_wall_seconds"],
               "user_cpu_seconds": child["rusage"]["ru_utime"],
               "system_cpu_seconds": child["rusage"]["ru_stime"],
               "parent_environment_unchanged": item["parent_environment_unchanged"],
               "torch_peak_allocated_bytes": child["torch_memory"]["peak_allocated_bytes"],
               "torch_peak_reserved_bytes": child["torch_memory"]["peak_reserved_bytes"],
               "peak_tree_total_bytes": sampler["peak_tree_total_bytes"],
               "peak_target_device_used_bytes": sampler["peak_target_device_used_bytes"],
               "peak_target_compute_process_sum_bytes": sampler["peak_target_compute_process_sum_bytes"],
               "sampled_tree_ownership": sampler["status"],
               "target_interval_seconds": sampler["sampling_interval_seconds"],
               "maximum_observed_interval_seconds": sampler["max_observed_interval_seconds"],
               "sample_query_seconds_total": sum(sample.get("sample_query_seconds", 0)
                                                for sample in sampler["samples"]),
               "parent_side_queries_run_concurrently": True}
        row.update({"section_"+key+"_seconds": seconds
                    for key, seconds in child["stage"]["stage_seconds"].items()})
        details = child["stage"].get("collision_details")
        row["collision_profile_trial_count"] = len(details) if details is not None else None
        for key in ("candidate_build_seconds", "ordered_acceptance_seconds", "grid_candidates",
                    "sphere_candidates", "retained_candidates", "adaptive_chunk_reductions"):
            row["collision_total_"+key] = sum(trial.get(key, 0) for trial in details) if details is not None else None
        row["collision_any_candidate_truncation"] = any(trial.get("candidate_truncation", False)
                                                       for trial in details) if details is not None else None
        rows.append(row)
    cli = [row for row in rows if row["mode"] == "cli"]
    timing = {}
    for policy in ("enabled", "disabled"):
        selected = [row for row in cli if row["policy"] == policy]
        timing[policy] = ({"count": len(selected),
                          "median_complete_api_seconds": statistics.median(row["complete_api_seconds"] for row in selected),
                          "median_fresh_exec_seconds": statistics.median(row["fresh_exec_wall_seconds"] for row in selected)}
                         if selected else {"count": 0})
    if all(timing[policy]["count"] for policy in timing):
        timing["observed_api_speed_ratio_disabled_over_enabled"] = (
            timing["disabled"]["median_complete_api_seconds"] / timing["enabled"]["median_complete_api_seconds"])
        timing["observed_api_fraction_seconds_reduced"] = (
            1-timing["enabled"]["median_complete_api_seconds"] / timing["disabled"]["median_complete_api_seconds"])
    evaluation = json.loads(evaluation_json.read_text()) if evaluation_json is not None else None
    result = {"scope": value["scope"], "stage_name": value.get("stage_name", "white"),
              "producer_status": value["status"], "summary_sha256": digest(summary_json),
              "evaluator_report_sha256": digest(evaluation_json) if evaluation_json is not None else None,
              "summary_tool_sha256": digest(Path(__file__)), "rows": rows,
              "complete_stage_comparisons": value["comparisons"], "cli_timing_observations": timing,
              "initialized_parent_api": value["initialized_parent_api"],
              "frozen_baseline_diagnostic": evaluation,
              "unresolved_tree_memory_is_null": True,
              "target_device_sum_includes_other_processes": True,
              "no_cuda_allocation_api_trace_collected": True,
              "other_processes_retained": True,
              "whole_pipeline_speedup": None, "overall_metric_equivalence": "not_assessed",
              "interpretation": "fixed same-input stage observations with cold per-item Numba JIT and concurrent sampling; not a raw-T1 or exclusive-node benchmark"}
    (output_dir/"metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir/"metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    # 只画派生计时和设备总占用；MRI 与表面不进入图形或公共目录。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    labels = [f"{row['index']}: {row['mode']}\n{row['policy']}" for row in rows]
    colors = ["#3069a8" if row["policy"] == "enabled" else "#d28735" for row in rows]
    axes[0].bar(labels, [row["complete_api_seconds"] for row in rows], color=colors)
    for index, row in enumerate(rows):
        axes[0].text(index, row["complete_api_seconds"], f"{row['complete_api_seconds']:.2f}", ha="center", va="bottom")
    axes[0].set_ylabel("Complete API wall time (s)")
    axes[0].set_title("Frozen same-input complete surface stage; cold JIT + I/O included")
    for item in value["items"]:
        target_uuid = item["sampler"]["target_gpu_uuid"]
        points = []
        for sample in item["gpu_load_curve"]:
            for line in sample.get("csv", "").splitlines():
                columns = [part.strip() for part in line.split(",")]
                if len(columns) == 4 and columns[1] == target_uuid:
                    points.append((sample["monotonic"], float(columns[2])*1024**2/1e9))
        if points:
            axes[1].plot([time-points[0][0] for time, _ in points], [memory for _, memory in points],
                         label=f"{item['index']} {item['mode']} {item['policy']}")
    axes[1].axhline(20, ls="--", color="grey", lw=1, label="20 GB target")
    axes[1].set_xlabel("Seconds from first device-load sample in each item")
    axes[1].set_ylabel("Target device total (GB, decimal)")
    axes[1].set_title("Sampled device upper bound includes other processes; tree ownership unresolved")
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(output_dir/"timing_and_device_load.png", dpi=160)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(summary_json=args.summary_json, evaluation_json=args.evaluation_json,
              output_dir=args.output_dir)


if __name__ == "__main__":
    main()
