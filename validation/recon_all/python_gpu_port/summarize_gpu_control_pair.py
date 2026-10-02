"""离线整理 sub01 GPU 受控整例对照，不运行重建或读取原始影像。

输入 --reports 为 performance_20261001，需完整的 whole/sub01 原始运行、
控制器、监测 JSON 与 CSV；输出 --output 为新 JSON，禁止覆盖已有文件。
先核对原始 T1/原生程序 SHA、同一主机/物理 GPU、实际线程和 SynthSeg
前向 FP32 例外，再比较相同 run_monitored 包装器的全命令墙钟。完整墙钟
包含启动、模型加载、传输、读写和退出；不同内部 API 计时边界另列。

显存使用同一次 NVML compute-apps 查询中的父子进程合计；whole-GPU 查询
发生于另一时刻，不能相减冒充其他进程的同时占用。2 秒周期采样不能证明
连续峰值。本次各一个共享资源整例只支持本次观察，不证明稳定因果提速。
该报告不设整体指标等效门槛；数值/脑区结果由独立配对 JSON 提供。

示例（参数具名，路径为已下载的真实数据诊断产物）：
  python summarize_gpu_control_pair.py \
    --reports /path/to/performance_20261001 \
    --output /path/to/gpu_control_pair_summary.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from summarize_cpu_control_pair import elapsed_pair, file_binding, native_fingerprints, read_json


def sample_summary(path: Path, monitor: dict) -> dict:
    """核对一个最终 CSV 与 monitor，返回同查询显存与采样缺口统计。

    参数 path 为 NVML CSV 路径，monitor 为同次运行的最终 JSON。输出 dict
    显存单位 byte，同时给出 GB/GiB；查询失败保持缺失，不补零。记录行数、
    失败数或采样峰值与 JSON 不符时抛 ValueError，不报告虚假的资源达标。
    """
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != monitor["samples"]:
        raise ValueError("CSV count differs from final monitor JSON")
    failures = sum(row["apps_status"] != "ok" for row in rows)
    if failures != monitor["failed_app_queries"]:
        raise ValueError("CSV failed queries differ from final monitor JSON")
    measured = [row for row in rows if row["apps_status"] == "ok" and row["total_process_bytes"]]
    peak = max((int(row["total_process_bytes"]) for row in measured), default=None)
    if peak != monitor["peak_sampled_process_bytes"]:
        raise ValueError("CSV peak differs from final monitor JSON")
    peak_rows = [row for row in measured if int(row["total_process_bytes"]) == peak]
    gpu_rows = [row for row in rows if row["gpu_status"] == "ok" and row["gpu_total_mib"]]
    utilization = [float(row["gpu_utilization_percent"]) for row in gpu_rows if row["gpu_utilization_percent"]]
    return {
        "samples": len(rows), "failed_app_queries": failures,
        "failed_whole_gpu_queries": sum(row["gpu_status"] != "ok" for row in rows),
        "missing_whole_gpu_values_despite_successful_query": sum(row["gpu_status"] == "ok" and not row["gpu_total_mib"] for row in rows),
        "interval_requested_seconds": monitor["sampling_interval_requested_seconds"],
        "query_timeout_seconds": monitor["query_timeout_seconds"],
        "maximum_sampling_gap_seconds": monitor["maximum_sampling_gap_seconds"],
        "peak_sampled_parent_plus_children_bytes": peak,
        "peak_GB": peak / 1e9 if peak is not None else None,
        "peak_GiB": peak / 1024**3 if peak is not None else None,
        "peak_below_20_000_000_000_bytes": peak < 20_000_000_000 if peak is not None else None,
        "one_peak_same_query_record": peak_rows[0] if peak_rows else None,
        "continuous_peak_verified": monitor["continuous_peak_verified"],
        "monitor_thread_finished": monitor["monitor_thread_finished"],
        "whole_gpu_separate_query": {
            "peak_total_mib": max((float(row["gpu_total_mib"]) for row in gpu_rows), default=None),
            "median_total_mib": statistics.median(float(row["gpu_total_mib"]) for row in gpu_rows) if gpu_rows else None,
            "median_utilization_percent": statistics.median(utilization) if utilization else None,
            "scope": "whole device in separate later queries; includes other jobs; not simultaneous with compute-apps records",
        },
        "torch_allocator_stats": "unavailable with allocator cache disabled; never interpret reported zero as zero process memory",
    }


def fp32_forwards(candidate: dict, control: dict) -> tuple[bool, bool]:
    """核对候选与旧算法控制器的真实 SynthSeg 两次前向，返回两个布尔值。

    两者均要求 float32 输入/权重/输出，cuDNN TF32=False、matmul TF32=True、
    CPU/CUDA autocast 关闭。候选允许后验缓冲复用，控制器必须有 completed
    记录。只读取已有日志，不更改任何精度或把关闭 autocast 视为低精度输出。
    """
    after = candidate["precision"]["SynthSeg_actual_forward"]["forwards"]
    after_ok = len(after) == 2 and all(
        row["input_dtype"] == row["output_dtype"] == "torch.float32"
        and row["model_dtypes"] == ["torch.float32"]
        and row["matmul_tf32"] and not row["cudnn_tf32"]
        and not any(value["enabled"] for value in row["autocast"].values())
        for row in after
    )
    scopes = control["synthseg_posterior_scopes"]
    before = [row for scope in scopes for row in scope["forwards"]]
    before_ok = len(scopes) == 1 and scopes[0]["status"] == "complete" and len(before) == 2 and all(
        row["completed"] and row["input_dtype"] == row["output_dtype"] == "torch.float32"
        and row["parameter_dtypes"] == ["torch.float32"]
        and row["matmul_tf32"] and not row["cudnn_tf32"]
        and not row["autocast_cuda"] and not row["autocast_cpu"]
        for row in before
    )
    return after_ok, before_ok


def main() -> None:
    """读取最终受控报告，核验可比条件后写新 JSON；缺失/不一致直接失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="本次验证报告根目录")
    parser.add_argument("--output", type=Path, required=True, help="新建的 GPU 整例受控汇总 JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.reports.resolve()
    names = {
        "candidate": "whole/sub01/candidate_run.json",
        "candidate_api": "whole/sub01/candidate_api.json",
        "candidate_monitor": "whole/sub01/candidate_monitor.json",
        "baseline": "whole/sub01/baseline_run.json",
        "baseline_control": "whole/sub01/baseline_control.json",
        "baseline_monitor": "whole/sub01/baseline_monitor.json",
        "hardware": "hardware_gpucw1_retry1_1b8c36d.json",
        "runtime": "runtime_fingerprints_1b8c36d.json",
    }
    values = {key: read_json(root / name) for key, name in names.items()}
    candidate, baseline, control = (values[key] for key in ("candidate", "baseline", "baseline_control"))
    after_monitor, before_monitor, api = (values[key] for key in ("candidate_monitor", "baseline_monitor", "candidate_api"))
    candidate_fp32, baseline_fp32 = fp32_forwards(candidate, control)
    uuid = after_monitor["gpu_uuid"]
    checks = {
        "both_execution_complete": candidate["status"] == baseline["status"] == control["status"] == "complete",
        "both_command_exit_zero": after_monitor["exit_code"] == before_monitor["exit_code"] == 0,
        "same_host": control["hostname"] == values["hardware"]["host"] == "gpucw1",
        "same_gpu_uuid": before_monitor["gpu_uuid"] == control["CUDA_VISIBLE_DEVICES"] == api["CUDA_VISIBLE_DEVICES"] == uuid,
        "candidate_api_device_uuid_matches": api["device_uuid"].removeprefix("GPU-") == uuid.removeprefix("GPU-"),
        "same_requested_device": candidate["device"] == baseline["device"] == control["device"] == api["device"] == "cuda:0",
        "same_raw_input_path": candidate["input"] == baseline["input"] == control["input"],
        "input_sha256": values["runtime"]["inputs"]["sub01_t1"]["sha256"] == control["input_sha256"],
        "same_requested_threads": candidate["threads"] == baseline["threads"] == control["threads"] == api["threads"] == 4,
        "same_actual_torch_numba": control["torch_threads"] == control["numba_threads"]
            == candidate["thread_budget"]["torch"]["effective"] == candidate["thread_budget"]["numba"]["effective"] == 4,
        "both_numba_initial_capacity_4": control["NUMBA_NUM_THREADS"] == "4" and candidate["thread_budget"]["numba"]["initial_capacity"] == 4,
        "same_torch_version": values["hardware"]["torch"] == control["torch_version"],
        "same_numba_version": values["hardware"]["numba"] == control["numba_version"],
        "same_native_program_fingerprints": native_fingerprints(candidate) == native_fingerprints(baseline),
        "same_outer_monitor_source": after_monitor["script_sha256"] == before_monitor["script_sha256"],
        "same_sampling_interval_and_timeout": after_monitor["sampling_interval_requested_seconds"] == before_monitor["sampling_interval_requested_seconds"]
            and after_monitor["query_timeout_seconds"] == before_monitor["query_timeout_seconds"],
        "both_initialized_cuda_api": api["cuda_initialized_before_api"] and control["cuda_initialized_before_api"],
        "both_allocator_cache_disabled": api["allocator_before_initialization"]["effective"] == "disabled"
            and api["PYTORCH_NO_CUDA_MEMORY_CACHING"] == control["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1",
        "same_actual_fp32_synthseg_exception": candidate_fp32 and baseline_fp32,
        "both_138_present": candidate["output_validation"]["present"] == baseline["output_validation"]["present"] == 138,
    }
    failed = [key for key, equal in checks.items() if not equal]
    if failed:
        raise ValueError(f"Controlled GPU comparability checks failed: {failed}")
    before = {row["name"]: row for row in baseline["stages"]}
    after = {row["name"]: row for row in candidate["stages"]}
    if len(before) != 66 or before.keys() != after.keys():
        raise ValueError("Stage identity differs from the declared 66-stage profile")
    boundaries = {
        "SynthSeg": "candidate includes MGZ/CSV writes inside this stage; baseline writes just after the stage",
        "brain_volume_stats": "candidate includes brainvol.stats/tiv writes inside this stage; baseline writes just after the stage",
    }
    deltas = []
    for name in before:
        row = elapsed_pair(before[name]["seconds"], after[name]["seconds"], scope=name)
        row.update(name=name, stage_io_boundary_changed=name in boundaries, boundary_note=boundaries.get(name),
            candidate_parent_cpu_seconds=after[name].get("parent_cpu_seconds"),
            candidate_child_cpu_seconds=after[name].get("child_cpu_seconds"),
            candidate_cuda_sync_before_seconds=after[name].get("cuda_pre_sync_seconds"),
            candidate_cuda_sync_after_seconds=after[name].get("cuda_post_sync_seconds"),
            baseline_cpu_seconds=None, baseline_cpu_reason="legacy wrapper records wall seconds only; do not infer from mtimes")
        deltas.append(row)
    groups = {}
    for group, prefix in (("surface_initialization", "surface_"), ("sphere_registration", "register_"),
            ("final_surfaces", "finish_surface_"), ("annotation", "annot_"), ("cortical_stats", "stats_")):
        groups[group] = elapsed_pair(sum(row["seconds"] for name, row in before.items() if name.startswith(prefix)),
            sum(row["seconds"] for name, row in after.items() if name.startswith(prefix)), scope=group)
    csv_names = {"candidate_csv": "whole/sub01/candidate_gpu_samples.csv", "baseline_csv": "whole/sub01/baseline_gpu_samples.csv"}
    memory = {"candidate": sample_summary(root / csv_names["candidate_csv"], after_monitor),
        "baseline": sample_summary(root / csv_names["baseline_csv"], before_monitor)}
    report = {
        "schema": "fnit.recon_all.gpu_control_pair.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": "sub01", "host": control["hostname"], "cpu": values["hardware"]["cpu"], "gpu_uuid": uuid,
        "baseline_calculation_commit": control["calculation_commit"], "candidate_calculation_commit": values["hardware"]["code_commit"],
        "source_report_bindings": {key: file_binding(root, name) for key, name in (names | csv_names).items()},
        "baseline_wrapper_sha256": control["wrapper_sha256"], "baseline_verified_source_sha256": control["verified_control_source_sha256"],
        "comparability_checks": checks,
        "exact_outer_command_observation": elapsed_pair(before_monitor["command_wall_seconds"], after_monitor["command_wall_seconds"],
            scope="same run_monitored perf_counter start before process creation through child exit; includes interpreter/wrapper startup, validation, model loading, transfers, IO and final stdout; excludes monitor thread join/report write after exit"),
        "common_postvalidation_pipeline": elapsed_pair(baseline["total_seconds"], candidate["timing"]["pipeline_seconds"],
            scope="post-validation standard pipeline through output/mesh checks, before terminal internal report write; excludes initial imports/resource validation and public thread setup/restore"),
        "same_stage_sum": elapsed_pair(sum(row["seconds"] for row in before.values()), sum(row["seconds"] for row in after.values()),
            scope="sum of the same 66 stage rows; excludes between-stage report writes and initial validation; SynthSeg/brain_volume_stats IO boundaries differ"),
        "inner_clocks_with_different_scope": {
            "baseline_wrapper_main_seconds": control["seconds_including_first_sidecar_write"], "baseline_scope": control["wall_scope"],
            "candidate_initialized_api_context_seconds": api["seconds_including_context_and_run"],
            "candidate_public_api_seconds": candidate["total_seconds"], "baseline_native_api_seconds_excluding_prevalidation": baseline["total_seconds"],
            "candidate_public_api_scope": candidate["timing"]["total_scope"],
            "inner_api_speedup": None,
        },
        "all_stage_deltas": deltas, "grouped_stage_deltas": groups,
        "candidate_stage_sync_seconds": sum(row.get("cuda_pre_sync_seconds", 0.0) + row.get("cuda_post_sync_seconds", 0.0) for row in after.values()),
        "candidate_parent_cpu_seconds": sum(row.get("parent_cpu_seconds", 0.0) for row in after.values()),
        "candidate_reaped_children_cpu_seconds": sum(row.get("child_cpu_seconds", 0.0) for row in after.values()),
        "baseline_cpu_and_sync_totals": None,
        "candidate_top10_stages": sorted(deltas, key=lambda row: -row["candidate_seconds"])[:10],
        "largest10_stage_increases": sorted(deltas, key=lambda row: -row["candidate_minus_baseline_seconds"])[:10],
        "largest10_stage_savings": sorted(deltas, key=lambda row: row["candidate_minus_baseline_seconds"])[:10],
        "precision": {"baseline_actual": control["synthseg_posterior_scopes"], "candidate_actual": candidate["precision"],
            "default_matmul_and_cudnn_tf32": True, "synthseg_cudnn_fp32_exception": True, "automatic_half_precision": False},
        "thread_configuration": {"baseline_torch": control["torch_threads"], "baseline_numba": control["numba_threads"],
            "baseline_initial_NUMBA_NUM_THREADS": control["NUMBA_NUM_THREADS"], "candidate_budget": candidate["thread_budget"],
            "limitation": "active Torch/Numba masks four; interop/driver/monitor and total OS threads are not limited to four"},
        "gpu_memory": memory, "output_integrity": {"baseline": baseline["output_validation"], "candidate": candidate["output_validation"]},
        "mesh_quality_reported": {"baseline": baseline["mesh_validation"], "candidate": candidate["mesh_validation"]},
        "whole_numeric_comparison": "separate paired_sub01_1b8c36d derived reports", "overall_metric_equivalence": "not_assessed",
        "causal_limit": "one whole run each on shared CPU/GPU resources; unchanged native-stage wall differences cannot be attributed to a code optimization; exact clock scope does not establish stable causal speedup",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"exact_outer_command_observation": report["exact_outer_command_observation"], "gpu_memory": memory}, indent=2))


if __name__ == "__main__":
    main()
