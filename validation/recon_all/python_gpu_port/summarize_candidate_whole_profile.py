"""从已收回的整例 JSON 整理候选剖析；不运行重建、不读取影像或参考结果。

输入 --reports 为 performance_20261001 报告目录，--output 为新 JSON。
计时单位秒，显存单位字节；保留逐阶段、原网格检查和版本哈希。
此脚本没有独立 FreeSurfer 等价命令；官方全流程参考另在 benchmark 路径。
缺失报告、来源版本不符或输出已存在时失败，不猜测未测指标。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


COMMIT = "1b8c36d25a68e253a1e59b6d02114890afa467de"


def load_json(path: Path) -> dict:
    """读取既有 UTF-8 JSON 字典；缺失或格式错误向上传递，不改变原报告。"""
    data = path.read_bytes()
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dictionary report: {path}")
    return result


def binding(root: Path, relative: str) -> dict:
    """返回一个既有报告的相对路径、字节数与 SHA-256，不改变文件内容。"""
    path = root / relative
    data = path.read_bytes()
    return {"path": relative, "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def seconds_fields(value: dict) -> dict:
    """保留具名秒数，避免把迭代轨迹或嵌套峰值相加为整例时间。"""
    return {key: item for key, item in value.items()
            if "seconds" in key and isinstance(item, (int, float, dict))}


def summarize_case(root: Path, case: str, provenance: dict) -> dict:
    """汇总一例原始 T1 整例，输入仅既有报告，输出包含全部 66 阶段与限制。

    root 为报告目录，case 为 sub01/sub02，provenance 为已核验程序/资产清单。
    不为 CPU 例生成 GPU 峰值，不从跨硬件两例计算速度比。
    """
    run_path = f"whole/{case}/candidate_run.json"
    run = load_json(root / run_path)
    if run["status"] != "complete" or len(run["stages"]) != 66:
        raise ValueError(f"Candidate not a completed 66-stage run: {case}")
    device = run["device"]
    gpu = device.startswith("cuda")
    command_path = f"whole/{case}/candidate_monitor.json" if gpu else f"whole/{case}/candidate_command.json"
    command = load_json(root / command_path)
    if command["exit_code"] != 0:
        raise ValueError(f"Candidate command failed: {case}")
    api = load_json(root / f"whole/{case}/candidate_api.json") if gpu else None
    hardware_path = ("hardware_gpucw1_retry1_1b8c36d.json" if gpu
                     else "hardware_nodecw10_1b8c36d.json")
    hardware = load_json(root / hardware_path)
    stage_by_name = {row["name"]: row for row in run["stages"]}
    total = run["total_seconds"]
    stages = [{**row, "fraction_of_api_wall": row["seconds"] / total}
              for row in run["stages"]]
    sum_field = lambda field: sum(float(row.get(field, 0)) for row in stages)
    stage_total = sum_field("seconds")
    function_total = sum_field("function_seconds")
    pre = sum_field("cuda_pre_sync_seconds")
    post = sum_field("cuda_post_sync_seconds")
    substeps = {}
    folds = {}
    for hemi, surface in run["surfaces"].items():
        registration = run["sphere_registration"]["reports"][hemi]
        sulc = registration["sulc_pass"]
        smooth = registration["smoothwm_pass"]
        sphere = surface["standard_sphere_report"]
        # topology_python_seconds 包含 remesh 和 intersection，不能重复相加。
        initialization = {key: surface[key] for key in (
            "topology_python_seconds", "topology_native_seconds",
            "topology_remesh_seconds", "topology_intersection_seconds", "sphere_seconds")}
        initialization["white_preaparc"] = seconds_fields(surface["white_preaparc_report"])
        initialization["standard_sphere"] = seconds_fields(sphere)
        initialization["full_stage_seconds"] = stage_by_name[f"surface_{hemi}"]["seconds"]
        initialization["nested_times_overlap"] = True
        sulc_times = seconds_fields(sulc)
        sulc_times["integration_and_write_residual_seconds"] = (
            sulc["total_seconds_including_io"] - sulc["setup_seconds"])
        sulc_times["rigid_seconds_is_nested_in_setup"] = True
        substeps[hemi] = {
            "surface_initialization": initialization,
            "sphere_registration": {
                "full_stage_seconds": stage_by_name[f"register_{hemi}"]["seconds"],
                "full_function_including_io_seconds": registration["total_seconds_including_io"],
                "sulc": sulc_times, "smoothwm": seconds_fields(smooth),
                "overlap_device": registration["overlap_device"]},
            "final_white_pial_and_metrics": {
                "full_stage_seconds": stage_by_name[f"finish_surface_{hemi}"]["seconds"],
                "final_white_seconds": surface["final_white_report"]["seconds"],
                "pial_seconds": surface["pial_report"]["seconds"],
                "metric_seconds": surface["metric_seconds"],
                "other_finish_residual_seconds": stage_by_name[f"finish_surface_{hemi}"]["seconds"]
                    - surface["final_white_report"]["seconds"] - surface["pial_report"]["seconds"]
                    - sum(surface["metric_seconds"].values()),
                "residual_scope": "pial semantic copy, mid area, TH3 volume, checks, reads and wrapper overhead"}}
        folds[hemi] = {
            "sphere_cleanup_pre_step_negative_counts": sphere["negative_counts"],
            "sphere_reg_cleanup_pre_step_negative_counts": smooth["negative_counts"],
            "final_sphere_negative_face_count": None,
            "final_sphere_reg_negative_face_count": None,
            "status": "final_geometry_not_independently_checked",
            "reason": "Recorded cleanup lists are pre-step counts; the final returned mesh count is not recorded."}
    source_bindings = [binding(root, run_path), binding(root, command_path), binding(root, hardware_path)]
    if gpu:
        source_bindings.append(binding(root, f"whole/{case}/candidate_api.json"))
    precision = run["precision"]
    forwards = precision["SynthSeg_actual_forward"]["forwards"]
    memory_status = dict(Counter(row["torch_memory_stats_status"] for row in stages))
    return {
        "subject": case, "source_bindings": source_bindings,
        "invocation": "initialized CUDA Python API" if gpu else "CPU CLI",
        "host": hardware["host"], "cpu": hardware["cpu"], "device": device,
        "candidate_code_commit": COMMIT, "input": run["input"],
        "input_fingerprint": provenance["inputs"][case + "_t1"],
        "subject_directory": run["subject_dir"],
        "raw_t1_empty_directory_evidence": {
            "status": "supported_by_successful_standard_entry_and_driver_command",
            "reason": "The 1b production entry rejects nonempty subject directories; driver passes the raw T1, with no checkpoint/resume input.",
            "separate_pre_run_directory_snapshot_recorded": False},
        "execution_status": run["status"], "command_exit_code": command["exit_code"],
        "wall_time": {
            "command_wall_seconds": command["command_wall_seconds"],
            "command_scope": "whole command creation to exit; includes Python/CLI startup, imports, loading, transfers, file writes and final report output; excludes monitor join",
            "initialized_context_and_run_seconds": None if api is None else api["seconds_including_context_and_run"],
            "api_total_seconds": total,
            "api_scope": "public entry including thread setup/restore, validation, loading, transfers and output IO; excludes final public metadata write and pre-entry imports",
            "raw_timing_metadata": run["timing"],
            "thread_residual_scope": "1b field is public-wrapper residual; includes internal final report write/return, not pure thread-setting time",
            "sum_stage_seconds": stage_total,
            "sum_function_seconds": function_total,
            "sum_pre_sync_seconds": pre, "sum_post_sync_seconds": post,
            "sum_stage_measurement_overhead_seconds": stage_total - function_total - pre - post,
            "api_minus_stage_sum_seconds": total - stage_total,
            "sum_parent_cpu_seconds": sum_field("parent_cpu_seconds"),
            "sum_reaped_child_cpu_seconds": sum_field("child_cpu_seconds"),
            "cpu_scope": "process CPU seconds and RUSAGE_CHILDREN deltas; not GPU time, can exceed wall with parallel work"},
        "stages": stages,
        "top10_stage_bottlenecks": sorted(stages, key=lambda row: -row["seconds"])[:10],
        "hemisphere_substeps": substeps,
        "other_stage_substeps": {row["name"]: {key: row[key] for key in ("timings_seconds", "substep_seconds") if key in row}
                                  for row in stages if "timings_seconds" in row or "substep_seconds" in row},
        "output_integrity": run["output_validation"],
        "mesh_quality_actual_fields": run["mesh_validation"],
        "mesh_quality_scope": {
            "edge_manifold_check": "unclosed_edges counts orig edges with occurrence !=2, including boundary and overused edges; ordered faces checked on white/pial/sphere.reg",
            "white_pial_cross_intersections": "not_run", "vertex_link_manifold": "not_run",
            "connected_components": "not_run", "sphere_final_flip_checks": folds},
        "precision": precision,
        "precision_assessment": {
            "synthseg_actual_two_passes_fp32": all(row["input_dtype"] == "torch.float32"
                and row["output_dtype"] == "torch.float32" and row["model_dtypes"] == ["torch.float32"]
                and not any(v["enabled"] for v in row["autocast"].values()) for row in forwards),
            "synthseg_cudnn_tf32_disabled_on_cuda": all(not row["cudnn_tf32"] for row in forwards) if gpu else None,
            "cpu_tf32_flag_semantics": None if gpu else "CUDA backend TF32 flags are recorded but have no effect on these CPU forwards.",
            "other_fp32_exceptions": "Exception list is retained; this JSON only records SynthSeg actual forward flags directly. Do not treat the list as actual-forward evidence for every stage."},
        "thread_budget": run["thread_budget"],
        "observed_process_environment_snapshot": hardware.get("processes", []),
        "thread_scope_limitation": "Torch intraop and Numba masks are not a hard bound on total OS threads, interop, or all native/BLAS pools.",
        "surface_metrics_backend": run["surface_metrics"],
        "surface_stats_cache": run["surface_stats_cache"],
        "gpu_memory": {
            "scope": "same compute-apps query, requested physical GPU, parent and current descendants" if gpu else "CPU run; GPU not applicable",
            "monitor": command if gpu else None,
            "peak_sampled_process_bytes": command["peak_sampled_process_bytes"] if gpu else None,
            "sampled_peak_below_20000000000_bytes": command["peak_sampled_process_bytes"] < 20_000_000_000 if gpu else None,
            "continuous_peak_verified": False if gpu else None,
            "allocator_at_recon_entry": run["cuda_allocator"],
            "allocator_before_api_initialization": api["allocator_before_initialization"] if gpu else None,
            "torch_stage_memory_status_counts": memory_status,
            "top_level_torch_peak_allocated_bytes": run.get("gpu_peak_allocated_bytes"),
            "top_level_torch_peak_reserved_bytes": run.get("gpu_peak_reserved_bytes"),
            "top_level_torch_peak_scope": "maximum of available per-process peaks; may only be Talairach child here, not contemporaneous parent+child total",
            "shared_gpu_snapshot": hardware.get("gpu_snapshot"),
            "limitation": "Periodic sampled maxima are not continuous peaks; shared load is not controlled by candidate runtime."},
        "strict_reproduction": {"status": "not_run_in_this_run_report", "raw": run["numeric_validation"]},
        "optimization_introduced_degradation": {"status": "pending_separate_same_input_comparison"},
        "overall_metric_equivalence": "not_assessed", "speedup": None,
        "speedup_reason": "controlled corresponding baseline is still running; two candidate cases use different hosts/devices"}


def main() -> None:
    """读取具名报告参数并写一个全量剖析 JSON；输出存在时拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.reports.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    fingerprint = load_json(root / "whole/candidate_source_fingerprint.json")
    if fingerprint["candidate_code_commit"] != COMMIT or not all(fingerprint["git_snapshot_match"].values()):
        raise ValueError("Actual runtime sources do not match the declared 1b snapshot")
    provenance = load_json(root / "runtime_fingerprints_1b8c36d.json")
    if provenance["code_commit"] != COMMIT or provenance["mismatches"]:
        raise ValueError("Runtime asset/program provenance mismatch")
    manifest = load_json(root / "source_1b8c36d_manifest.json")
    summary = {
        "schema": "fnit.recon_all.candidate_whole_profile.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_code_commit": COMMIT,
        "deployment_archive_sha256": manifest["deployment_archive_sha256"],
        "collection_scope": "Existing completed whole-run JSON and source hashes only; no new reconstruction, no MRI copied",
        "source_fingerprint": fingerprint,
        "provenance_bindings": [binding(root, name) for name in (
            "whole/candidate_json_transfer_manifest.json", "whole/candidate_source_fingerprint.json",
            "source_1b8c36d_manifest.json", "runtime_fingerprints_1b8c36d.json")],
        "program_asset_provenance": {key: provenance[key] for key in (
            "build_manifest_sha256", "expected_manifest_sha256", "weights", "assets", "binaries", "reference_binaries", "inputs")},
        "cases": {case: summarize_case(root, case, provenance) for case in ("sub01", "sub02")},
        "speedup": None, "overall_metric_equivalence": "not_assessed",
        "comparability": "Sub01 CUDA API on gpucw1 and sub02 CPU CLI on nodecw10 are distinct candidate configurations; do not use their ratio as GPU speedup.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({case: {"command_seconds": value["wall_time"]["command_wall_seconds"],
        "output_present": value["output_integrity"]["present"], "stages": len(value["stages"]),
        "sum_sync_seconds": value["wall_time"]["sum_pre_sync_seconds"] + value["wall_time"]["sum_post_sync_seconds"]}
        for case, value in summary["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
