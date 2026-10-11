"""从已校验公开收据生成8次完整阶段时间和逐数组误差，不重算N4。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(*, reports: Path, output: Path) -> dict:
    manifest = json.loads((reports / "PUBLIC_EXPORT_MANIFEST.json").read_bytes())
    for name, entry in manifest["files"].items():
        if hashlib.sha256((reports / name).read_bytes()).hexdigest() != entry["published_sha256"]:
            raise ValueError("published receipt SHA mismatch: " + name)
    times, errors, cases = [], [], []
    for case in ("sub07", "sub06"):
        compare = json.loads((reports / (case + "_comparisons.json")).read_bytes())
        if compare["status"] != "complete" or compare["order"] != [1, 4, 4, 1]:
            raise ValueError("case not complete ABBA")
        for index, fit in enumerate((1, 4, 4, 1), 1):
            name = f"{case}-{index}-fit{fit}"
            record = json.loads((reports / (name + "-run.json")).read_bytes())
            process = json.loads((reports / (name + "-process.json")).read_bytes())
            if (manifest["files"][name + "-run.json"]["raw_sha256"] != compare["run_json_sha256"][index - 1]
                    or record["status"] != "complete" or process["exit_code"] != 0):
                raise ValueError("producer status/original SHA changed: " + name)
            parts = record["details"]["timing_parts_seconds"]
            times.append({"case": case, "ABBA_index": index, "fitting_threads": fit,
                "complete_API_seconds": record["complete_API_seconds_including_read_write_profile_hash"],
                "fitting_seconds": parts["fitting"], "imports_seconds": record["imports_seconds"],
                "full_child_process_seconds_including_diagnostics": process["full_process_wall_seconds"],
                "reconstruction_seconds": parts["full_bspline_reconstruction"],
                "exp_divide_download_seconds": parts["exp_divide_and_array_download"],
                "actual_fitting_work_units": record["details"]["actual_fitting_work_units"],
                "actual_fitting_filter_threads": record["details"]["actual_fitting_filter_threads"],
                "global_threads_restored": record["global_threads_restored"]})
        for array_name, contrasts in compare["comparisons"].items():
            for contrast, value in contrasts.items():
                errors.append({"case": case, "array": array_name, "contrast": contrast, **value})
        for index, value in enumerate(compare["native_comparisons_each_ABBA_run"], 1):
            for array_name, detail in value.items():
                errors.append({"case": case, "array": "native_" + array_name, "contrast": f"ABBA{index}_vs_existing_native", **detail})
        errors.append({"case": case, "array": "native_corrected_float", "contrast": "existing_native_repeat0_vs_repeat1",
                       **compare["native_repeat_float"]})
        repeat_exact = all(compare["comparisons"][array][contrast]["different_voxels"] == 0
            for array in ("log_bias_field", "corrected_float", "nu0")
            for contrast in ("A1_vs_A2", "B1_vs_B2"))
        cases.append({"case": case, "API_fit1_median_seconds": compare["median_A_API_seconds"],
            "API_fit4_median_seconds": compare["median_B_API_seconds"],
            "observed_stage_ratio": compare["observed_API_ratio_A_over_B"],
            "observed_stage_reduction_percent": 100.0 * (1.0 - compare["median_B_API_seconds"] / compare["median_A_API_seconds"]),
            "within_configuration_repeats_all_bit_exact": repeat_exact,
            "float_and_quantization_fit1_vs_fit4": compare["comparisons"],
            "native_reference_same_host": compare["native_reference_same_host"],
            "native_reference_repeat": compare["native_repeat_float"],
            "fit1_and_fit4_vs_existing_native": compare["native_comparisons_each_ABBA_run"],
            "geometry_exact": compare["all_geometry_exact"],
            "threads_restored": compare["all_thread_context_restored"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("stage_times.csv", times), ("stage_errors.csv", errors)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    result = {"status": "complete", "scope": "same-input full N4 file API ABBA; two public T1s",
        "cases": cases, "CPU_budget": 4, "CPU_affinity": [100, 101, 102, 103],
        "GPU_used": False, "GPU_memory_measurement": "not applicable; CPU stage only",
        "strict_reproduction": "fit1 versus fit4 and versus existing native differ; exact repeats within each configuration",
        "optimization_introduced_difference": "deterministic thread-dependent floating and quantization changes, reported without equivalence claim",
        "overall_metric_equivalence": "not_assessed", "whole_pipeline_speedup": "not_measured",
        "production_default": "fitting_threads=1 unchanged",
        "native_reference_scope": "same host existing two-repeat fixed-program outputs; no fresh native timing pair",
        "shared_host_limit": "CPU affinity fixed; other host tasks/load were recorded, observations are not dedicated-node throughput",
        "source_binding": json.loads((reports / "source_binding.json").read_bytes()),
        "public_export_manifest_sha256": hashlib.sha256((reports / "PUBLIC_EXPORT_MANIFEST.json").read_bytes()).hexdigest(),
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(**vars(args)), ensure_ascii=False))
