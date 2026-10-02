#!/usr/bin/env python3
"""Compare saved fixed-input FNIRT cold/warm/profile results without rerunning.

Uses the pipeline validator's image/header comparison and finite/hash checks.
Public output contains anonymous roles and scalar summaries, never input paths.
The sole QC exclusions are top-level execution and recursive elapsed_seconds.
No registration, profiler or CUDA operation is executed by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


IMAGE_SUFFIXES = {
    "warped_image": ".nii.gz",
    "coefficients": "_coeff.nii.gz",
    "pull_transform": "_pull.nii.gz",
    "full_pull_jacobian": "_jacobian.nii.gz",
    "nonlinear_jacobian": "_nonlinear_jacobian.nii.gz",
}
CONDITIONS = ("cold_0", "warm_1", "warm_2", "profiled")


def load_helper(path):
    spec = importlib.util.spec_from_file_location("fnirt_pipeline_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_qc(qc):
    if not isinstance(qc, dict) or "levels" not in qc:
        raise ValueError("FNIRT QC must contain its complete level/solver trace")

    def remove_timing(value):
        if isinstance(value, dict):
            return {key: remove_timing(item) for key, item in value.items()
                    if key != "elapsed_seconds"}
        if isinstance(value, list):
            return [remove_timing(item) for item in value]
        return value

    return remove_timing({key: value for key, value in qc.items() if key != "execution"})


def canonical_bytes(value):
    # JSON preserves -0.0: numeric == alone would lose signed-zero distinctions.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def difference_paths(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        changed = []
        for key in sorted(first.keys() | second.keys()):
            current = f"{prefix}/{key}"
            if key not in first or key not in second:
                changed.append(current)
            else:
                changed.extend(difference_paths(first[key], second[key], current))
        return changed
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return [f"{prefix}/length"]
        changed = []
        for index, (left, right) in enumerate(zip(first, second)):
            changed.extend(difference_paths(left, right, f"{prefix}/{index}"))
        return changed
    return [] if canonical_bytes(first) == canonical_bytes(second) else [prefix]


def record_summary(record, *, profiled):
    measurement = record["measurement"]
    return {
        "condition": record["condition"],
        "wall_seconds": record["wall_seconds"],
        "peak_cuda_allocated_bytes": record["peak_allocated_bytes"],
        "peak_cuda_reserved_bytes": record["peak_reserved_bytes"],
        "peak_cuda_allocated_gb_decimal": record["peak_allocated_bytes"] / 1e9,
        "peak_cuda_reserved_gb_decimal": record["peak_reserved_bytes"] / 1e9,
        "process_peak_cpu_rss_kb": record["process_peak_cpu_rss_kb"],
        "usable_for_formal_speed_summary": not profiled,
        "profile_instrumentation_changes_timing": bool(measurement["instrumentation_changes_timing"]),
        "phase_timing_scope": measurement["phase_timing_scope"],
        "phase_timings": measurement["phase_timings"],
        "operation_counts": measurement["operation_counts"],
        "cuda_tensor_api_calls": measurement["cuda_tensor_api_calls"],
        "actual_cuda_event_counts": measurement["actual_cuda_event_counts"],
        "profiled_cuda_device_time_us": measurement["profiled_cuda_device_time_us"],
        "profile_windows": measurement["profile_windows"],
        "profiler_scope": measurement["profiler_scope"],
        "whole_gpu_observation": record["whole_gpu_observation"],
    }


def summarize(baseline_root, candidate_root, helper, helper_path):
    import numpy as np
    import nibabel
    import torch

    roots = {"baseline": baseline_root, "candidate": candidate_root}
    reports = {role: json.loads((root / "report.safe.json").read_text())
               for role, root in roots.items()}
    variants = {}
    for role, report in reports.items():
        if {run["condition"] for run in report["runs"]} != set(CONDITIONS[:-1]):
            raise ValueError("expected exactly one cold and two warm runs")
        if report["profile"]["condition"] != "profiled":
            raise ValueError("profile result is missing")
        variants[role] = {
            "recorded_environment": report["environment"],
            "scientific_conditions": report["scientific_conditions"],
            "timing_scope": report["timing_scope"],
            "source_sha256": report["source_sha256"],
            "benchmark_tool_sha256": report["benchmark_tool_sha256"],
            "support_tool_sha256": report["support_tool_sha256"],
            "original_report_sha256": helper.sha256(roots[role] / "report.safe.json"),
            "formal_runs": [record_summary(run, profiled=False) for run in report["runs"]],
            "profile": record_summary(report["profile"], profiled=True),
        }
    conditions_equal = canonical_bytes(reports["baseline"]["scientific_conditions"]) == canonical_bytes(
        reports["candidate"]["scientific_conditions"]
    )
    checks = {}
    for condition in CONDITIONS:
        image_checks = {}
        for role, suffix in IMAGE_SUFFIXES.items():
            first, second = [root / f"{condition}{suffix}" for root in roots.values()]
            comparison = helper.compare_images(first, second)
            passed = all(value for key, value in comparison.items() if key.endswith("equal"))
            image_checks[role] = {
                "lossless_gate_passed": bool(passed),
                "comparison": comparison,
                "baseline": helper.image_check(first),
                "candidate": helper.image_check(second),
            }
        qc_paths = [root / f"{condition}.qc.private.json" for root in roots.values()]
        raw_qc = [json.loads(path.read_text()) for path in qc_paths]
        qc = [normalize_qc(value) for value in raw_qc]
        encoded = [canonical_bytes(value) for value in qc]
        raw_changes = difference_paths(*raw_qc)
        remaining_changes = difference_paths(*qc)
        qc_equal = encoded[0] == encoded[1]
        checks[condition] = {
            "all_scientific_outputs_equal": all(value["lossless_gate_passed"] for value in image_checks.values()),
            "images": image_checks,
            "complete_scientific_qc_equal": qc_equal,
            "qc": {
                "level_counts": [len(value["levels"]) for value in raw_qc],
                "baseline_file_sha256": helper.sha256(qc_paths[0]),
                "candidate_file_sha256": helper.sha256(qc_paths[1]),
                "baseline_scientific_qc_sha256": hashlib.sha256(encoded[0]).hexdigest(),
                "candidate_scientific_qc_sha256": hashlib.sha256(encoded[1]).hexdigest(),
                "raw_differing_field_paths": raw_changes,
                "scientific_differing_field_paths": remaining_changes,
            },
        }
    summary = {}
    for role, report in reports.items():
        warm = [run for run in report["runs"] if run["condition"].startswith("warm_")]
        summary[role] = {
            "cold_seconds": next(run["wall_seconds"] for run in report["runs"] if run["condition"] == "cold_0"),
            "warm_seconds": [run["wall_seconds"] for run in warm],
            "warm_median_seconds": statistics.median(run["wall_seconds"] for run in warm),
            "formal_peak_cuda_allocated_gb_decimal": max(run["peak_allocated_bytes"] for run in report["runs"]) / 1e9,
            "formal_peak_cuda_reserved_gb_decimal": max(run["peak_reserved_bytes"] for run in report["runs"]) / 1e9,
        }
    summary["cold_runtime_reduction_fraction"] = 1 - summary["candidate"]["cold_seconds"] / summary["baseline"]["cold_seconds"]
    summary["warm_runtime_reduction_fraction"] = 1 - summary["candidate"]["warm_median_seconds"] / summary["baseline"]["warm_median_seconds"]
    passed = conditions_equal and all(
        value["all_scientific_outputs_equal"] and value["complete_scientific_qc_equal"] for value in checks.values()
    )
    return {
        "schema_version": 1,
        "date": "2026-10-02",
        "function": "fixed_input_t1_fnirt",
        "baseline_commit": "747345223710e704f94ba850f58504fe2bac4f8e",
        "validation_only_no_registration_rerun": True,
        "baseline_and_candidate_scientific_conditions_equal": conditions_equal,
        "lossless_gate_passed": passed,
        "scope": {
            "same_input_and_initial_affine": True,
            "image_pairs": len(CONDITIONS) * len(IMAGE_SUFFIXES),
            "qc_pairs": len(CONDITIONS),
            "all_voxels_and_signed_zero_compared": True,
            "scientific_headers_affines_shapes_dtypes_extensions_compared": True,
            "qc_exclusions": {"top_level_keys": ["execution"], "recursive_exact_keys": ["elapsed_seconds"]},
            "all_other_qc_and_solver_fields_compared": True,
            "fsl_numerical_equivalence_claimed": False,
        },
        "timing_notes": {
            "formal_runs": "Cold/warm complete FNIRT function includes reads/decompression and CPU output conversion; excludes output persistence/offline comparison; light Python phase/count wrappers remain included; shared H100.",
            "profile": "Diagnostic only: CUDA profiler window fences/instrumentation change wall time; excluded from speed summary.",
            "phase_seconds": "Nested asynchronous CPU wall clocks are not additive or exclusive CUDA kernel times.",
            "memory": "Current process Torch CUDA allocated/reserved, decimal GB; whole GPU observations are separately reported.",
            "warm_sample_count_per_variant": 2,
        },
        "verification_environment": {"torch": torch.__version__, "numpy": np.__version__, "nibabel": nibabel.__version__},
        "comparison_tool_sha256": helper.sha256(Path(__file__)),
        "comparison_helper_sha256": helper.sha256(helper_path),
        "variants": variants,
        "paired_checks": checks,
        "formal_summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--comparison-helper", type=Path, required=True)
    parser.add_argument("--output", default="-", help="public JSON file; '-' prints to stdout")
    args = parser.parse_args()
    helper = load_helper(args.comparison_helper)
    report = summarize(args.baseline_results, args.candidate_results, helper, args.comparison_helper)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output == "-":
        print(encoded, end="")
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    return 0 if report["lossless_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
