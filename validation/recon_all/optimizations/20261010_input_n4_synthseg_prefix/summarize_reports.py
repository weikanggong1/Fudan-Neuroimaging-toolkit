"""从已校验公开JSON生成完整前段时间表和摘要；不改原报告或阈值。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(*, reports: Path, output: Path) -> dict:
    """reports为export_reports输出；output已有目录，两个衍生文件须不存在。"""
    manifest = json.loads((reports / "PUBLIC_EXPORT_MANIFEST.json").read_bytes())
    for name, row in {**manifest["reports"], **manifest["images"]}.items():
        if hashlib.sha256((reports / name).read_bytes()).hexdigest() != row["public_sha256"]:
            raise ValueError("published artifact changed: " + name)
    summary = {"scope": "two raw-T1 to filled continuous prefixes; no whole recon-all",
               "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "full_workflow_complete": False, "overall_metric_equivalence": "not_assessed",
               "stable_prefix_speedup_established": False, "default_input_execution": "serial",
               "official_rerun_this_experiment": False, "cases": {}}
    table = []
    names = {"sub07": {"serial": "1-serial", "parallel": "2-parallel"},
             "sub06": {"serial": "2-serial", "parallel": "1-parallel"}}
    for case, trials in names.items():
        comparison = json.loads((reports / "diagnostics/compare_plot_v1/comparisons" / (case + ".json")).read_bytes())
        result = {"source_manifest_sha256": comparison["source_manifest_sha256"],
                  "original_T1_sha256": comparison["original_T1_sha256"], "runs": {},
                  "all_14_voxel_arrays_exact": comparison["all_volume_values_exact"],
                  "geometry_dtype_exact": comparison["all_geometry_dtype_exact"],
                  "LTA_different_elements": comparison["talairach_lta"]["different_elements"],
                  "CSV_raw_bytes_equal": comparison["synthseg_volume_csv"]["raw_bytes_equal"],
                  "MGZ_files_bytes_equal": {name: row["files_bytes_equal"] for name, row in comparison["volumes"].items()},
                  "label_Dice": {name: row["labels"]["minimum_foreground_dice"]
                       for name, row in comparison["volumes"].items() if "labels" in row},
                  "WM_ge_5_foreground_Dice": {name: row["WM_foreground"]["WM_MIN_VAL_ge_5"]["dice"]
                       for name, row in comparison["volumes"].items() if "WM_foreground" in row}}
        for execution, name in trials.items():
            prefix = reports / case / name
            producer = json.loads((prefix / "benchmark.json").read_bytes())
            runtime = json.loads((prefix / "subject/fnit-native-free-run.json").read_bytes())
            process = json.loads((reports / case / (name + "-process.json")).read_bytes())
            if (producer["status"] != "complete_prefix" or process["exit_code"] != 0
                    or producer["full_workflow_complete"]):
                raise ValueError("incomplete prefix: " + case + " " + name)
            samples = process["samples"]
            result["runs"][execution] = {
                "API_seconds": producer["public_API_seconds_including_reads_writes"],
                "fresh_CLI_with_monitor_and_audit_seconds": process["fresh_CLI_seconds"],
                "CPU_affinity": producer["CPU_affinity"], "threads": producer["threads"],
                "target_GPU_UUID": samples["target_gpu_uuid"],
                "target_device_NVML_v1_sampled_peak_bytes_including_others": samples["peak_target_device_used_bytes"],
                "target_compute_process_sum_sampled_peak_bytes": samples["peak_target_compute_process_sum_bytes"],
                "simultaneous_parent_child_tree_peak_bytes": samples["peak_tree_total_bytes"],
                "sampling_interval_seconds": samples["sampling_interval_seconds"],
                "max_observed_interval_seconds": samples["max_observed_interval_seconds"],
                "failed_samples": samples["failed_samples"],
                "forbidden_program_attempts": process["execution_audit"]["blocked_events"],
                "continuous_GPU_peak_proven": False,
                "clean_OS_library_file_isolation_proven": False}
            for order, row in enumerate(runtime["stages"], 1):
                table.append({"case": case, "execution": execution, "stage_order": order,
                    **{key: row.get(key) for key in ("name", "seconds", "function_seconds",
                       "cuda_pre_sync_seconds", "cuda_post_sync_seconds", "cuda_synchronized",
                       "parent_cpu_seconds", "child_cpu_seconds", "torch_memory_stats_status")}})
        before = result["runs"]["serial"]["API_seconds"]
        after = result["runs"]["parallel"]["API_seconds"]
        result["observed_API_reduction_percent"] = 100 * (before - after) / before
        summary["cases"][case] = result
    output.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "stage_times.csv"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    with (output / "stage_times.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps({case: row["observed_API_reduction_percent"] for case, row in result["cases"].items()}))
