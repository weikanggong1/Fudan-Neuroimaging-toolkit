"""用已收回的真实报告更新性能索引，不运行重建或改写原始比较。

输入：--report-directory，已包含 stage_summary.json 及完整配对的目录；
--summary-repository-head，本次整理之前的完整 Git commit。
输出：同目录的 stage_summary.json；仅更新完成状态、摘要及 SHA-256 绑定。
坐标/单位：沿用原报告的 surface RAS/mm 和秒、字节，不转换影像或阈值。
失败：缺文件、旧绑定不符或未通过已声明阶段门槛时抛异常，不生成占位结果。
此脚本属于 FNIT benchmark 汇总，没有对应的独立 FreeSurfer 命令。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--summary-repository-head", required=True)
    args = parser.parse_args()
    report_directory = args.report_directory.resolve()

    def read(name: str) -> dict:
        return json.loads((report_directory / name).read_text(encoding="utf-8"))

    def binding(name: str) -> dict:
        data = (report_directory / name).read_bytes()
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    summary = read("stage_summary.json")
    # 最终线程日志明确更新过；其余已绑定证据必须保持原始字节。
    for name, expected in summary["report_sha256"].items():
        if name != "thread_runtime_logs.json":
            assert binding(name)["sha256"] == expected, (name, "source report changed")
    performance = read("gpu_control_pair_summary.json")
    precision = read("gpu_control_precision_summary.json")
    repeatability = read("curvature_repeat_rh_pial_1b/summary.json")
    changes = precision["controlled_optimization_changes"]
    assert all(performance["comparability_checks"].values())
    assert changes["strict_reproduction"]["passed"] == 138
    assert all(row["statistics"]["outliers"] == 0 for row in changes["nonzero_numeric_blocks"])
    assert changes["all_label_voxels_identical"]
    assert changes["all_reported_surface_geometry_identical"]
    assert repeatability["repeated_runs"]["reference"]["different_values"] == 0
    assert repeatability["repeated_runs"]["conda"]["different_values"] == 0
    assert repeatability["repeated_runs"]["pytorch"]["outliers"] == 0

    summary["updated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Real same-input stages, two complete raw-T1 candidate/control runs, "
        "independent official metrics and extended quality; GPU curvature repeatability."
    )
    summary["versions"]["summary_repository_head"] = args.summary_repository_head
    summary["versions"]["frozen_1b8c36d"]["measurement_scope"] = (
        "final-default LH thickness, posterior buffers, thread regressions and two complete whole runs"
    )
    summary["versions"]["post_1b_metadata_and_workers"]["real_gpu_worker_measurement_status"] = (
        "passed independent frozen sub-01 LH stage; not the calculation version of whole runs"
    )
    items = {item["item"]: item for item in summary["six_items"]}
    items[2]["status"] = "implemented_and_measured_frozen_1b_whole; later failure metadata local only"
    items[3]["status"] = "two_subject_stage_and_whole_reported_ROI_regression_passed"
    items[4]["status"] = "two_subject_stage_1b_lh_and_post_1b_worker_passed; whole_gates_unchanged"
    items[4]["post_1b_worker_candidate_status"] = "real stage passed; see workers_summary.json"
    items[5]["status"] = "CLI_API_stage_and_no_cache_whole_measured; cached_whole_not_tested"
    items[5]["whole_memory"] = performance["gpu_memory"]
    items[6]["finding"] = (
        "Torch/Numba scope fixes effective Python thread budget; native subprocess budgets "
        "require separate verification; final EM logs confirm OpenMP and FSRuntime 4, "
        "not all OS/library threads"
    )
    items[6]["runtime_logs"] = binding("thread_runtime_logs.json")

    status = summary["validation_status"]
    status["strict_reproduction"]["sub01_controlled_baseline"] = (
        "138/138 unchanged diagnostic gates; 20/174 numeric blocks nonzero, not byte-exact"
    )
    status["optimization_introduced_regression"] = (
        "No unacceptable changes in checked same-input gates; CPU parsed outputs exact; "
        "GPU label/region/ordered geometry exact, 20 map tails within existing gates. "
        "Current RH pial GPU repeats show comparable tails; no cause assigned to other maps. "
        "TF32 policy correction versus old effective TF32 separately changes 148 frozen labels."
    )
    status["whole_pipeline_speedup"] = performance["exact_outer_command_observation"]
    status["whole_pipeline_speedup_status"] = (
        "GPU matched complete-command observation 3.889% shorter; shared resources, one pair; "
        "CPU common postvalidation observation 1.495%, complete outer ratio unavailable"
    )
    status["gpu_memory_budget"] = {
        "sampled_parent_plus_children": performance["gpu_memory"],
        "continuous_peak_verified": False,
        "candidate_sampled_peak_higher_than_baseline": True,
        "default_cache_changed": False,
    }
    summary["whole_runs"]["sub01_gpu_controlled_baseline"].update(
        status="complete", performance=binding("gpu_control_pair_summary.json"),
        precision=binding("gpu_control_precision_summary.json"),
        command_wall_seconds=performance["exact_outer_command_observation"]["baseline_seconds"],
        notes="Original T1, empty output directory; same monitor clock, initialized API, threads and precision.",
    )
    summary["whole_runs"]["notes"] = (
        "All four actual whole runs completed; no checkpoint reruns substituted. "
        "GPU speed and precision use final controlled pair; earlier profile remains immutable."
    )
    names = [
        "gpu_control_pair_summary.json", "gpu_control_precision_summary.json",
        "gpu_control_cortex_label_identity.json", "thread_runtime_logs.json",
        "curvature_repeat_rh_pial_1b/summary.json",
        "curvature_repeat_rh_pial_1b/local_manifest.json",
    ]
    names.extend(f"whole/{subject}/paired/dice_vs_{target}_semantics_corrected.json"
                 for subject in ("sub01", "sub02") for target in ("baseline", "official"))
    for name in names:
        summary["report_sha256"][name] = binding(name)["sha256"]
    summary["independent_diagnostics"]["gpu_curvature_repeatability"] = binding(
        "curvature_repeat_rh_pial_1b/summary.json"
    )
    summary["next_priority_command"] = read("gpu_reuse_remaining_audit.json")["next_priority_command"]
    summary["next_priority_command"]["priority"] = "受控整例已完成；下一步剖析同输入 LH 注册的 averaging/force/line-search。"
    summary["finalization"] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {name: binding(name) for name in names},
        "no_image_calculation_or_threshold_change": True,
    }
    (report_directory / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "reports_bound": len(summary["report_sha256"]),
                      "gpu_time_reduction_fraction": status["whole_pipeline_speedup"]["time_reduction_fraction"]}))


if __name__ == "__main__":
    main()
