"""汇总同源完整white索引/保留MHT两顺序配对；不修改验收或候选输出。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-first-report", type=Path, required=True)
    parser.add_argument("--candidate-first-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison",choices=("grid","retained"),default="grid")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = (args.control_first_report, args.candidate_first_report)
    reports = [json.loads(path.read_text()) for path in paths]
    if any(report.get("status") != "complete" for report in reports):
        raise ValueError("both complete four-pass pairs required")
    reference = reports[0]
    for report in reports:
        if report["input_sha256"] != reference["input_sha256"]:
            raise ValueError("stage inputs differ")
        if report["source_sha256"] != reference["source_sha256"]:
            raise ValueError("paired source versions differ")
        if (report["threads"], report["gpu"], report["device"], report["hostname"],
                report["cuda_allocator_environment"], report["tf32_matmul"], report["tf32_cudnn"]) != (
                reference["threads"], reference["gpu"], reference["device"], reference["hostname"],
                reference["cuda_allocator_environment"], reference["tf32_matmul"], reference["tf32_cudnn"]):
            raise ValueError("paired resources or precision differ")
    if reports[0]["python_order"] != ["cpu", "torch"] or reports[1]["python_order"] != ["torch", "cpu"]:
        raise ValueError("opposite control/candidate orders required")
    baseline_trace = reference["python_runs"]["cpu"]["trace"]
    rows = []
    for report in reports:
        for backend in report["python_order"]:
            run = report["python_runs"][backend]
            stage = run["stage"]
            if (not stage["complete_four_passes"] or len(stage["pass_ends"]) != 4
                    or stage["cleanup"]["intersecting_faces_after"] != 0):
                raise ValueError("four passes and final zero-intersection gate required")
            details = stage["collision_details"]
            rows.append({"order": len(rows), "role": "control" if backend == "cpu" else "candidate",
                "grid_cells_per_axis": stage["candidate_grid_cells_per_axis"],
                "api_wall_seconds": run["wall_seconds"], "cuda_setup_seconds": run["cuda_setup_seconds"],
                "trace_exact_to_first_control": run["trace"] == baseline_trace,
                "surface_sha256": run["output_sha256"], "MRI_sha256": run["output_volume_sha256"],
                "stage_seconds": stage["stage_seconds"], "prepare_components": stage["prepare_components"],
                "candidate_build_seconds": sum(row.get("candidate_build_seconds", 0) for row in details),
                "ordered_acceptance_seconds": sum(row.get("ordered_acceptance_seconds", 0) for row in details),
                "retained_MHT_tree_seconds": sum(row["total_collision_seconds"] for row in details
                    if row.get("effective_candidate_backend") == "tree_retained_mht"),
                "retained_MHT_compiled_seconds": sum(row["total_collision_seconds"] for row in details
                    if row.get("effective_candidate_backend") == "compiled_retained_mht"),
                "retained_bucket_checks": sum(row.get("retained_mht_checks",0) for row in details),
                "retained_mht_backend":stage.get("retained_mht_backend","tree"),
                "saved_real_retained_trial_count":len(run.get("retained_trial_checkpoints",[])),
                "peak_allocated_bytes": run["peak_allocated_bytes"], "peak_reserved_bytes": run["peak_reserved_bytes"],
                "steps": stage["steps"], "final_intersecting_faces": stage["cleanup"]["intersecting_faces_after"]})
    if args.comparison=="grid":
        if any(row["grid_cells_per_axis"] != (2 if row["role"] == "control" else 3) for row in rows):
            raise ValueError("actual grids do not match control2/candidate3")
    elif any(row["grid_cells_per_axis"]!=3 or row["retained_mht_backend"]!=(
            "tree" if row["role"]=="control" else "compiled") for row in rows):
        raise ValueError("retained comparison requires identical grid3 and actual tree/compiled policy")
    controls, candidates = ([row for row in rows if row["role"] == role] for role in ("control", "candidate"))
    control_wall = statistics.median(row["api_wall_seconds"] for row in controls)
    candidate_wall = statistics.median(row["api_wall_seconds"] for row in candidates)
    result = {"scope": "same_host_frozen_input_complete_white_preaparc_ABBA_not_final_white_or_recon_all",
        "script_sha256": sha(__file__), "source_report_sha256": [sha(path) for path in paths],
        "comparison":args.comparison,
        "input_sha256": reference["input_sha256"], "source_sha256": reference["source_sha256"],
        "gpu": reference["gpu"], "device": reference["device"], "threads": reference["threads"],
        "tf32_matmul": reference["tf32_matmul"], "tf32_cudnn": reference["tf32_cudnn"],
        "half_precision": False, "cuda_allocator_environment": reference["cuda_allocator_environment"],
        "allocator_policy_scope": "fresh benchmark process startup policy; no cache toggle or local cache scope in tested placement path; production cacheoff not evaluated",
        "timing_scope": "complete API including validation, MRI/surface loading, transfer, four passes, cleanup, trace IO and outputs; retained-trial checkpoint IO if enabled included; initial CUDA context and cold import separate; retained objmode JIT included",
        "rows": rows, "control_median_seconds": control_wall, "candidate_median_seconds": candidate_wall,
        "wall_reduction_percent_observation": 100*(1-candidate_wall/control_wall),
        "speed_ratio_observation": control_wall/candidate_wall,
        "all_step_trial_coordinates_exact": all(row["trace_exact_to_first_control"] for row in rows),
        "all_surface_files_exact": len({row["surface_sha256"] for row in rows}) == 1,
        "all_MRI_files_exact": len({row["MRI_sha256"] for row in rows}) == 1,
        "mesh_quality_final_zero_intersections": True,
        "strict_native_reproduction": "existing_same_input_native_geometry_difference; this comparison tests backend degradation only",
        "overall_metric_equivalence": "not_assessed", "whole_recon_all": "not_run_for_this_experimental_white"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
