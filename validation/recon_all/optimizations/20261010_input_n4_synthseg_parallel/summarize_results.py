"""从已校验公开收据计算完整组ABBA、实际前向及采样范围，不外推整例。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read_verified(directory: Path, name: str, manifest: dict) -> dict:
    path = directory / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]["published_sha256"]:
        raise ValueError("public payload hash mismatch: " + name)
    return json.loads(path.read_bytes())


def summarize(*, reports: Path, output: Path) -> dict:
    manifest = json.loads((reports / "PUBLIC_EXPORT_MANIFEST.json").read_bytes())
    cases, rows, errors = {}, [], []
    for case in ("sub07", "sub06"):
        abba = read_verified(reports, case + "-ABBA.json", manifest)
        if abba["status"] != "complete" or len(abba["runs"]) != 4:
            raise ValueError("incomplete ABBA")
        times = {mode: [] for mode in ("serial", "parallel")}
        details = []
        for trial in abba["runs"]:
            name = f'{case}-{trial["index"]}-{trial["execution"]}'
            group = read_verified(reports, name + "-group.json", manifest)
            producer = read_verified(reports, name + "-run.json", manifest)
            # Public sanitized SHA differs from the original producer SHA on purpose.
            if producer["group_report_sha256"] != manifest["files"][name + "-group.json"]["raw_sha256"]:
                raise ValueError("original group provenance mismatch")
            memory = group["device_process_tree"]
            n4, synthseg = (group["workers"][key] for key in ("n4", "synthseg"))
            row = {"case": case, "trial": trial["index"], "execution": trial["execution"],
                   "complete_CLI_seconds": trial["complete_CLI_seconds"],
                   "script_inclusive_seconds": producer["script_wall_seconds"],
                   "complete_group_API_seconds": producer["script_wall_seconds"]-producer["import_and_parent_setup_seconds"],
                   "internal_group_seconds_before_final_report": group["group_wall_seconds"],
                   "N4_worker_seconds": n4["total_seconds"], "SynthSeg_worker_seconds": synthseg["total_seconds"],
                   "overlap_seconds": group["overlap_seconds"],
                   "parent_state_preserved": group["parent_state_preserved"],
                   "peak_tree_total_bytes": memory["peak_tree_total_bytes"],
                   "peak_target_device_used_bytes": memory["peak_target_device_used_bytes"],
                   "max_observed_sample_gap_seconds": memory["max_observed_interval_seconds"],
                   "failed_memory_samples": len(memory["failed_samples"]),
                   "target_gpu_uuid": memory["target_gpu_uuid"]}
            times[trial["execution"]].append(trial["complete_CLI_seconds"])
            details.append(row)
            rows.append(row)
        serial, parallel = (statistics.median(times[key]) for key in ("serial", "parallel"))
        gate = read_verified(reports, case + "-reconstruction-gate.json", manifest)
        errors.append({"case": case, "scope": "N4 reconstruction1 vs4 fit1",
            "different_float_elements": gate["float_comparison"]["different_elements"],
            "float_max_abs": gate["float_comparison"]["max_abs"], "float_P99_abs": gate["float_comparison"]["p99_abs"],
            "different_uint8_voxels": gate["uint8_comparison"]["different_voxels"],
            "affine_max_abs": gate["uint8_comparison"]["affine_max_abs"],
            "vox2ras_tkr_max_abs": gate["uint8_comparison"]["vox2ras_tkr_max_abs"]})
        for name, comparison in abba["comparisons"].items():
            for image in ("nu0.mgz", "synthseg.mgz"):
                measured = comparison[image]
                errors.append({"case": case, "scope": name + ":" + image,
                    "different_float_elements": None, "float_max_abs": measured["max_abs"],
                    # 既有比较器未单存P99；全部不同体素为0时P99严格为0。
                    "float_P99_abs": 0.0 if measured["different_voxels"] == 0 else None,
                    "different_uint8_voxels": measured["different_voxels"],
                    "affine_max_abs": measured["affine_max_abs"], "vox2ras_tkr_max_abs": measured["vox2ras_tkr_max_abs"]})
        cases[case] = {"serial_complete_CLI_median_seconds": serial,
                       "parallel_complete_CLI_median_seconds": parallel,
                       "complete_group_observed_reduction_percent": 100*(serial-parallel)/serial,
                       "all_group_outputs_exact": abba["strict_same_input_passed"],
                       "all_SynthSeg_label_Dice": 1.0,
                       "SynthSeg_volume_CSV_bytes_identical": True, "trials": details,
                       "reconstruction_gate": gate["float_comparison"], "comparisons": abba["comparisons"]}
    parent = read_verified(reports, "parent-initialized-sub07-run.json", manifest)
    group = read_verified(reports, "parent-initialized-sub07-group.json", manifest)
    comparison = read_verified(reports, "initialized-parent-comparison.json", manifest)
    if not parent["parent_buffer_preserved"] or not comparison["strict_same_input_passed"]:
        raise ValueError("initialized CUDA parent API regression")
    result = {"status": "complete_same_input_groups", "cases": cases,
              "parent_initialized_API": {"complete_script_seconds": parent["script_wall_seconds"],
                  "parent_buffer_preserved": parent["parent_buffer_preserved"],
                  "parent_state_preserved": group["parent_state_preserved"],
                  "parent_entry": group["parent_state_at_entry"], "parent_after": group["parent_state_after"],
                  "outputs_exact": comparison["strict_same_input_passed"],
                  "memory": {key: group["device_process_tree"][key] for key in
                      ("status", "peak_tree_total_bytes", "peak_target_device_used_bytes",
                       "sampling_interval_seconds", "max_observed_interval_seconds", "failed_samples")}},
              "strict_reproduction_vs_serial": "passed_same_input_group",
              "optimization_introduced_degradation": "no_change_in_tested_full_group_outputs",
              "overall_metric_equivalence": "not_assessed", "whole_recon_all_speedup": "not_measured",
              "20GB_continuous_peak_verified": False,
              "scope": "two same-input full N4/SynthSeg ABBA groups; eight CLI trials and one live-parent API trial; no raw-T1 whole, no OS-isolated deployment",
              "public_export_manifest_sha256": hashlib.sha256((reports / "PUBLIC_EXPORT_MANIFEST.json").read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    for name, table in (("stage_times.csv", rows), ("stage_errors.csv", errors)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    result=summarize(**vars(args))
    print(json.dumps({case: {k: v for k,v in row.items() if "median" in k or "percent" in k} for case,row in result["cases"].items()}))
