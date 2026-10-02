"""汇总 2026-09-30 两例整例报告；只读取已完成的运行、比较及显存采样。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


def main() -> None:
    """读取 --reports 下的固定报告，写入不存在的 --output-dir；失败时抛异常。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-tag", default="e036f57")
    parser.add_argument("--baseline-tag", default="279e09f")
    args = parser.parse_args()
    root = args.reports
    tag = args.candidate_tag
    run_tag = {"279e09f": "279"}.get(tag, tag)
    baseline_tag = {"279e09f": "279"}.get(args.baseline_tag, args.baseline_tag)
    consumed = {}

    def read(relative: str) -> dict:
        """relative 是相对 reports 的 JSON 路径；返回字典并记录文件 SHA-256。"""
        data = (root / relative).read_bytes()
        consumed[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    source = read(f"source_{tag}_manifest.json")
    initial_commit = "b8cd17bb441307d88dda78ba8f56e77695b595a8"
    baseline_commit = (initial_commit if args.baseline_tag == "b8cd" else
                       read(f"source_{args.baseline_tag}_manifest.json")["code_commit"])
    current = {"date": "2026-09-30", "profile": "single-t1-standard",
               "candidate_code_commit": source["code_commit"],
               "candidate_tag": tag, "baseline_tag": args.baseline_tag,
               "baseline_code_commit": baseline_commit,
               "initial_baseline_code_commit": initial_commit,
               "source_archive_sha256": source["source_archive_sha256"],
               "function_timing_scope": "after entry validation; includes stage loading, transfer and I/O",
               "whole_command_timing_scope": "includes imports, validation, CUDA initialization; GPU monitoring adds up to one query/sleep interval",
               "overall_metric_equivalence": "not_assessed; no confirmed whole-case gates",
               "shared_hardware": True, "subjects": {}}
    metrics = {"date": current["date"], "candidate_code_commit": source["code_commit"],
               "overall_metric_equivalence": current["overall_metric_equivalence"],
               "relative_error_policy": "exclude zero reference values; preserve them in per-region reports",
               "subjects": {}}
    fingerprints = read(f"runtime_fingerprints_{tag}.json")
    if fingerprints["code_commit"] != source["code_commit"]:
        raise ValueError("fingerprint and source versions differ")
    if fingerprints["mismatches"]:
        raise ValueError("input, resource or program fingerprints changed")
    for sid, host in (("01", "gpucw1"), ("02", "nodecw10")):
        run = read(f"sub{sid}/full_{run_tag}_run.json")
        baseline = read(f"sub{sid}/full_{baseline_tag}_run.json")
        initial = read(f"sub{sid}/full_b8cd_run.json")
        pair = f"sub{sid}/full_{tag}_pair"
        comparison = read(f"{pair}/summary.json")
        if run["status"] != "complete" or comparison["execution_status"] != "complete":
            raise ValueError(f"sub-{sid}: run or comparison incomplete")
        if comparison["candidate_code_commit"] != source["code_commit"]:
            raise ValueError("comparison and source versions differ")
        external_name = f"full_sub{sid}_{tag}" + ("_gpu_summary.txt" if sid == "01" else "_summary.txt")
        data = (root / external_name).read_bytes()
        consumed[external_name] = hashlib.sha256(data).hexdigest()
        external = {key: int(value) for key, value in
                    (line.split("=", 1) for line in data.decode().splitlines())}
        if external["exit_code"] != 0:
            raise ValueError(f"sub-{sid}: command failed")
        official = read(f"{pair}/region_vs_official.json")
        before = read(f"{pair}/region_vs_baseline.json")
        dice = read(f"{pair}/dice_vs_official.json")
        baseline_dice = read(f"{pair}/dice_vs_baseline.json")
        baseline_surfaces = read(f"{pair}/surface_vs_baseline.json")
        read(f"{pair}/surface_vs_official.json")
        strict = read(f"{pair}/strict_vs_baseline.json")
        row = {"host": host, "device": run["device"],
               "call_mode": "initialized CUDA Python API" if sid == "01" else "CLI",
               "precision": run["precision"], "gpu_memory_mode": run["gpu_memory_mode"],
               "input_t1_sha256": fingerprints["inputs"][f"sub{sid}_t1"]["sha256"],
               "run_status": run["status"], "stages": len(run["stages"]),
               "total_seconds": run["total_seconds"], "external": external,
               "baseline_total_seconds": baseline["total_seconds"],
               "observed_runtime_change_percent": 100 * (run["total_seconds"] / baseline["total_seconds"] - 1),
               "initial_baseline_total_seconds": initial["total_seconds"],
               "observed_runtime_change_vs_initial_percent": 100 * (run["total_seconds"] / initial["total_seconds"] - 1),
               "output_validation": run["output_validation"], "mesh_validation": run["mesh_validation"],
               "comparison": comparison["comparisons"],
               "additional_strict_failures_vs_baseline": [name for name, result in strict["files"].items() if not result["pass"]],
               "slowest_stages": sorted(run["stages"], key=lambda item: -item["seconds"])[:10],
               "thread_policy": {"torch_and_supported_native_threads": run["threads"],
                                 "global_numba_blas_openmp_cap": False},
               "hardware_report": f"hardware_{host}_{tag}.json"}
        read(row["hardware_report"])
        if sid == "01":
            row["parent_pytorch_peak_bytes"] = {"allocated": None, "reserved": None,
                                                "reason": "CUDA allocator cache explicitly disabled"}
            row["talairach_child_pytorch_peak_bytes"] = {
                "allocated": run.get("gpu_peak_allocated_bytes"),
                "reserved": run.get("gpu_peak_reserved_bytes")}
            sample_name = f"full_sub01_{tag}_gpu_samples.csv"
            consumed[sample_name] = hashlib.sha256((root / sample_name).read_bytes()).hexdigest()
            with (root / sample_name).open() as stream:
                samples = list(csv.DictReader(stream))
            times = [datetime.fromisoformat(sample["time_utc"].replace("Z", "+00:00")) for sample in samples]
            intervals = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
            peak = max(int(sample["total_bytes"]) for sample in samples)
            row["api_invocation"] = read(f"sub01/run_api_{run_tag}.json")
            row["gpu_sampling"] = {"physical_gpu": 1, "logical_device": "cuda:0",
                                   "gpu_uuid": row["api_invocation"].get("device_uuid"),
                                   "sample_count": len(samples), "peak_simultaneous_parent_children_bytes": peak,
                                   "peak_gb": peak / 1e9, "peak_gib": peak / 2**30,
                                   "median_interval_seconds": statistics.median(intervals),
                                   "maximum_interval_seconds": max(intervals),
                                   "intervals_over_30_seconds": sum(value > 30 for value in intervals),
                                   "largest_sample_gaps": [
                                       {"start_utc": samples[index]["time_utc"],
                                        "end_utc": samples[index + 1]["time_utc"],
                                        "seconds": intervals[index]}
                                       for index in sorted(range(len(intervals)),
                                                           key=lambda i: -intervals[i])[:5]],
                                   "memory_budget_bytes": 20_000_000_000,
                                   "budget_compliance": "unverified; periodic observations cannot bound continuous peak",
                                   "continuous_peak_verified": False}
        current["subjects"][f"sub-{sid}"] = row
        metrics["subjects"][f"sub-{sid}"] = {
            "aparc_68": {name: {key: value for key, value in result.items() if key != "per_region"}
                         for name, result in official["aparc_68"].items()},
            "aseg": {key: value for key, value in official["aseg"].items() if key != "per_region"},
            "wmparc": {key: value for key, value in official["wmparc"].items() if key != "per_region"},
            "global_brainvol_measures": official["global_brainvol_measures"],
            "dice": {name: {key: value for key, value in result.items() if key != "per_label"}
                     for name, result in dice["files"].items()},
            "baseline_aparc_maximum_absolute_change": {name: result["maximum_absolute_error"]
                                                       for name, result in before["aparc_68"].items()},
            "baseline_aseg_maximum_absolute_change": before["aseg"]["maximum_absolute_error"],
            "baseline_wmparc_maximum_absolute_change": before["wmparc"]["maximum_absolute_error"],
            "baseline_dice_minimum": {name: result["minimum_dice"]
                                      for name, result in baseline_dice["files"].items()},
            "baseline_surface_distances": {hemi: {name: result["indexed_vertex_distance"]
                                                  for name, result in stages.items()
                                                  if name in ("white", "pial")}
                                           for hemi, stages in baseline_surfaces["stages"].items()},
            "complete_reports": pair}
    current["consumed_reports_sha256"] = consumed
    metrics["consumed_reports_sha256"] = consumed
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, report in (("current_full_runs_20260930.json", current),
                         ("final_metric_consistency_20260930.json", metrics)):
        (args.output_dir / name).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
