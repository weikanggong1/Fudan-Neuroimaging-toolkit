"""离线整理两次 sub02 CPU 整例的受控性能对照，不运行算法或读取真实影像。

--reports 是 performance_20261001，需有 whole/sub02 的候选、基线原 JSON；
--output 是新摘要 JSON，已存在则失败。输入/程序哈希、主机、设备、线程
不匹配时拒绝给速度比。旧 wrapper main 和候选外层 CLI 的墙钟范围不同，
只分别保存；共同校验后 pipeline 与阶段合计另列，不混作全命令提速。
本脚本没有独立 FreeSurfer 等价命令；严格比较和脑区指标由独立报告提供。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    """读取一个既有报告字典，缺失/无效 JSON 直接失败，不产生替代结果。"""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def file_binding(root: Path, name: str) -> dict:
    """绑定根目录下 JSON 的原始字节 SHA-256 与相对路径。"""
    value = (root / name).read_bytes()
    return {"path": name, "sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value)}


def elapsed_pair(baseline: float, candidate: float, *, scope: str) -> dict:
    """按明确共同范围返回秒数/速度比/耗时百分比，零或负耗时视为错误。"""
    if baseline <= 0 or candidate <= 0:
        raise ValueError("Elapsed times must be positive")
    return {"scope": scope, "baseline_seconds": baseline, "candidate_seconds": candidate,
            "candidate_minus_baseline_seconds": candidate - baseline,
            "baseline_divided_by_candidate": baseline / candidate,
            "time_reduction_fraction": (baseline - candidate) / baseline}


def native_fingerprints(run: dict) -> dict:
    """抽取真实调度报告的原生程序摘要，忽略算法输出/表面输入哈希。"""
    result = {}
    sections = ("n4_binary", "gca_registration", "topology_repair", "white_matter_chain",
                "white_preaparc", "surface_metrics", "extra_curvature", "defects_volume", "mni_nonlinear")
    for section in sections:
        for key, value in run[section].items():
            if "sha256" in key:
                result[f"{section}.{key}"] = value
    return result


def main() -> None:
    """核验固定 CPU 配置并写受控对照；不更改原报告或已有输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.reports.resolve()
    names = {"candidate": "whole/sub02/candidate_run.json",
             "candidate_command": "whole/sub02/candidate_command.json",
             "baseline": "whole/sub02/baseline_run.json",
             "baseline_control": "whole/sub02/baseline_control.json",
             "hardware": "hardware_nodecw10_1b8c36d.json",
             "runtime": "runtime_fingerprints_1b8c36d.json"}
    values = {key: read_json(root / name) for key, name in names.items()}
    candidate, baseline, control = (values[key] for key in ("candidate", "baseline", "baseline_control"))
    command = values["candidate_command"]
    checks = {
        "both_execution_complete": candidate["status"] == baseline["status"] == control["status"] == "complete",
        "same_host": command["host"] == control["hostname"] == values["hardware"]["host"],
        "both_cpu": candidate["device"] == baseline["device"] == control["device"] == "cpu",
        "same_raw_input_path": candidate["input"] == baseline["input"] == control["input"],
        "input_sha256": values["runtime"]["inputs"]["sub02_t1"]["sha256"] == control["input_sha256"],
        "same_requested_threads": candidate["threads"] == baseline["threads"] == control["threads"] == 4,
        "same_actual_torch_numba": control["torch_threads"] == control["numba_threads"]
            == candidate["thread_budget"]["torch"]["effective"]
            == candidate["thread_budget"]["numba"]["effective"] == 4,
        "same_torch_version": values["hardware"]["torch"] == control["torch_version"],
        "same_numba_version": values["hardware"]["numba"] == control["numba_version"],
        "same_native_program_fingerprints": native_fingerprints(candidate) == native_fingerprints(baseline),
        "both_138_present": candidate["output_validation"]["present"] == baseline["output_validation"]["present"] == 138,
    }
    failed = [key for key, equal in checks.items() if not equal]
    if failed:
        raise ValueError(f"Controlled CPU comparability checks failed: {failed}")
    before = {stage["name"]: stage for stage in baseline["stages"]}
    after = {stage["name"]: stage for stage in candidate["stages"]}
    if len(before) != 66 or before.keys() != after.keys():
        raise ValueError("Stage identity differs from the declared 66-stage profile")
    # 已核对旧 started 在资源/导入校验之后；候选 pipeline_started 在校验之后。
    common = elapsed_pair(baseline["total_seconds"], candidate["timing"]["pipeline_seconds"],
        scope="post-validation standard pipeline through output/mesh checks, before terminal internal report write; excludes initial imports/resource validation and public thread setup/restore")
    stage_pair = elapsed_pair(sum(row["seconds"] for row in before.values()),
        sum(row["seconds"] for row in after.values()),
        scope="sum of the same 66 stage rows; excludes between-stage report writes and initial validation; SynthSeg/brain_volume_stats per-stage IO boundaries differ")
    boundaries = {
        "SynthSeg": "candidate includes segmentation MGZ and CSV writes inside this stage; baseline writes them just after the stage",
        "brain_volume_stats": "candidate includes brainvol.stats and synthseg.tiv.dat writes; baseline writes them just after the stage"}
    stage_deltas = []
    for name in before:
        comparison = elapsed_pair(before[name]["seconds"], after[name]["seconds"], scope=name)
        comparison.update(name=name, stage_io_boundary_changed=name in boundaries,
            boundary_note=boundaries.get(name),
            candidate_parent_cpu_seconds=after[name].get("parent_cpu_seconds"),
            candidate_child_cpu_seconds=after[name].get("child_cpu_seconds"),
            baseline_cpu_seconds=None,
            baseline_cpu_reason="legacy stage wrapper only records wall seconds")
        stage_deltas.append(comparison)
    grouped = {}
    for group, prefix in (("surface_initialization", "surface_"), ("sphere_registration", "register_"),
                          ("final_surfaces", "finish_surface_"), ("annotation", "annot_"),
                          ("cortical_stats", "stats_")):
        grouped[group] = elapsed_pair(sum(row["seconds"] for name, row in before.items() if name.startswith(prefix)),
            sum(row["seconds"] for name, row in after.items() if name.startswith(prefix)), scope=group)
    report = {
        "schema": "fnit.recon_all.cpu_control_pair.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": "sub02", "host": control["hostname"], "cpu": values["hardware"]["cpu"],
        "baseline_calculation_commit": control["calculation_commit"],
        "candidate_calculation_commit": command["candidate_code_commit"],
        "source_report_bindings": {key: file_binding(root, name) for key, name in names.items()},
        "baseline_wrapper_sha256": control["wrapper_sha256"],
        "baseline_verified_source_sha256": control["verified_control_source_sha256"],
        "comparability_checks": checks,
        "thread_configuration": {"baseline_torch": control["torch_threads"],
            "baseline_numba": control["numba_threads"], "baseline_initial_NUMBA_NUM_THREADS": control["NUMBA_NUM_THREADS"],
            "candidate_budget": candidate["thread_budget"],
            "limitation": "Both active masks are 4; initial Numba pool capacity is 4 in baseline and 192 in candidate. OS thread totals and interop are not constrained to four."},
        "precision": {"baseline_control": control["precision_control"],
            "baseline_recorded_posterior_scopes": control["synthseg_posterior_scopes"],
            "candidate_actual": candidate["precision"],
            "cpu_interpretation": "CPU baseline posterior is unchanged and records no actual forward hooks; candidate two passes are observed float32/autocast-off. CUDA TF32 control flags do not change these CPU convolutions; do not claim direct actual-forward evidence for uninstrumented baseline."},
        "wall_clocks_with_different_scope": {
            "baseline_wrapper_main_seconds": control["seconds_including_first_sidecar_write"],
            "baseline_scope": control["wall_scope"],
            "candidate_external_command_seconds": command["command_wall_seconds"],
            "candidate_scope": "process creation through exit; includes interpreter/CLI startup, final JSON write and final stdout, unlike baseline wrapper-main clock",
            "candidate_public_api_seconds": candidate["total_seconds"],
            "baseline_native_api_seconds_excluding_prevalidation": baseline["total_seconds"],
            "candidate_public_api_scope": "includes validation/thread setup+restore; do not compare directly with legacy native_api_total_seconds",
            "exact_external_command_speedup": None},
        "common_postvalidation_pipeline": common, "same_stage_sum": stage_pair,
        "all_stage_deltas": stage_deltas, "grouped_stage_deltas": grouped,
        "candidate_top10_stages": sorted(stage_deltas, key=lambda row: -row["candidate_seconds"])[:10],
        "largest10_stage_increases": sorted(stage_deltas, key=lambda row: -row["candidate_minus_baseline_seconds"])[:10],
        "largest10_stage_savings": sorted(stage_deltas, key=lambda row: row["candidate_minus_baseline_seconds"])[:10],
        "output_integrity": {"baseline": baseline["output_validation"], "candidate": candidate["output_validation"]},
        "mesh_quality_reported": {"baseline": baseline["mesh_validation"], "candidate": candidate["mesh_validation"]},
        "whole_numeric_comparison": "separate paired_sub02_1b8c36d derived reports",
        "overall_metric_equivalence": "not_assessed", "gpu_memory": "not_applicable: CPU pair",
        "source_scope_note": "Common-scope ratios cover actual whole pipeline work including between-stage IO. Exact external command clock comparison is unavailable for baseline; stage deltas alone cannot establish isolated causal speedup.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"common_postvalidation_pipeline": common, "same_stage_sum": stage_pair}, indent=2))


if __name__ == "__main__":
    main()
