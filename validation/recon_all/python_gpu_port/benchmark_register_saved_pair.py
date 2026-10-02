"""冻结 FNIT 输入与保存基线的完整球面配准回归，不读取官方被试结果。

参数：--subject 是已完成的冻结单 T1 FNIT 被试目录，必须包含
fnit-native-free-run.json 和 surf/H.sphere、H.smoothwm、H.sulc、
H.sphere.reg；--hemi 为 lh/rh；--atlas 是已声明的同半球 folding TIFF。
三角表面使用 surface RAS/mm，同索引比较要求有序面相同；sulc 是同序
(N,) 顶点特征，图谱单位继承配准参数化。--source 指实际冻结 src 目录，
--source-version 是其提交/归档标识，源码 SHA 另外计算。--device 默认
cuda:0，仅选择 averaging；overlap 固定 cpu；--threads 默认且只接受 4。
--output 必须不存在；--gpu-uuid 可额外核验明确设备，建议以 UUID 设置
CUDA_VISIBLE_DEVICES。无半精度选项，不修改已有输入或生产默认。

先核对保存报告的四个输入 SHA 和 sphere.reg SHA，再用独立进程执行一次
当前 API；复用 benchmark_register_pair.py 的 worker、表面/footer 比较和
保存轨迹提取规则。阶段时间含 API 读写、JIT、传输、初始化和目标 GPU
前后同步；进程时间还含 Python 导入。旧报告时间只作历史元数据，不计算
速度比，不声称同时对照或整例提速。CUDA allocator 是否关闭沿用调用环境；
关闭时 allocated/reserved 为 None，父子进程同时占用须另用外部监控。

输出：新目录下 report.json、candidate/api_report.json、完整候选
H.sphere.reg 和 candidate.log；记录版本、SHA、实际线程/设备/精度、
坐标最大/P99 mm 误差、有序面/footer、保存 dt/state/相交清理轨迹。
固定基线版本来自已有 1b 整例外部记录；旧运行 JSON 无嵌入提交号时明确
注明，不能把声明版本当成脚本独立核验的提交。严格复现仅覆盖该阶段。
结构/哈希/计算失败保存 failed JSON，退出 1；运行完成但严格比较失败退出
2，保留数值结果；严格阶段通过退出 0。整体指标等效始终 not_assessed。

具名参数示例（每个参数的中文注释在下一行对应说明）：
    python benchmark_register_saved_pair.py \\
      --subject /data/full_sub01_1b8c36d_retry1 \\
      --hemi rh \\
      --atlas /data/fnit_assets/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \\
      --source /data/frozen_gpu_candidate/src \\
      --source-version commit-plus-archive-sha256 \\
      --device cuda:0 \\
      --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \\
      --output /data/validation/sub01_rh_register_saved_new \\
      --threads 4
subject=冻结自产被试；hemi=对应半球；atlas=声明图谱；source=冻结计算源码；
source-version=真实版本标识；device=明确 GPU；gpu-uuid=目标核验；
output=新诊断目录；threads=固定 CPU/Numba/BLAS 线程预算。
对应官方 mris_register -curv 的 sulc+smoothwm 阶段；这里只运行 FNIT API。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from benchmark_register_pair import (
    THREADS, compare_geometry, inputs, sha256, source_manifest, trajectory, write_json,
)


BASELINE_VERSION = "1b8c36d25a68e253a1e59b6d02114890afa467de"


def validate_baseline(args) -> tuple[dict, dict, dict]:
    """读冻结运行 JSON；四份输入及保存输出必须与阶段原记录逐文件同 SHA。

    输入为具名 argparse 参数；输出为全例报告、该半球 API 报告、哈希核对
    字典。只读文件，不计算影像；缺失字段、未完成基线或哈希改变抛异常。
    """
    run = json.loads(args.baseline_report.read_text())
    if run.get("status") != "complete":
        raise ValueError("保存 FNIT 基线不是 complete，不能作为冻结阶段精度基线")
    saved = run["sphere_registration"]["reports"][args.hemi]
    for name in ("sulc_pass", "smoothwm_pass"):
        if not isinstance(saved[name].get("updates"), list) or not saved[name]["updates"]:
            raise ValueError(f"保存基线缺少完整 {name} 更新轨迹")
    expected = saved["input_sha256"]
    if set(expected) != {"sphere", "smoothwm", "sulc", "atlas"}:
        raise ValueError("保存基线必须包含四个输入 SHA")
    current = {name: sha256(path) for name, path in inputs(args).items()}
    checks = {name: {"recorded_sha256": expected[name], "actual_sha256": value,
                     "equal": value == expected[name]} for name, value in current.items()}
    reference_hash = sha256(args.saved_surface)
    checks["saved_sphere_reg"] = {
        "recorded_sha256": saved["output_sha256"], "actual_sha256": reference_hash,
        "equal": reference_hash == saved["output_sha256"]}
    if not all(row["equal"] for row in checks.values()):
        raise ValueError(f"冻结输入或 sphere.reg 与保存阶段哈希不符：{checks}")
    return run, saved, checks


def main() -> int:
    """新目录中执行一次独立候选；保留失败或精确比较未通过的报告。"""
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, choices=(THREADS,), default=THREADS)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("subject", "atlas", "source", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.baseline_report = args.subject / "fnit-native-free-run.json"
    args.saved_surface = args.subject / "surf" / f"{args.hemi}.sphere.reg"
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    worker_script = script.with_name("benchmark_register_pair.py")
    report = {
        "status": "running", "scope": "saved-FNIT-same-input-register-stage-regression;not-whole",
        "started_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
        "subject": str(args.subject), "hemi": args.hemi, "source": str(args.source),
        "candidate_source_version": args.source_version,
        "baseline_source_version_declared": BASELINE_VERSION,
        "baseline_version_scope": "external frozen 1b whole-run provenance; not inferred from subject name",
        "baseline_report": str(args.baseline_report), "saved_surface": str(args.saved_surface),
        "script_sha256": sha256(script), "worker_script_sha256": None,
        "threads": THREADS, "averaging_device_requested": args.device, "overlap_device": "cpu",
        "gpu_uuid_requested": args.gpu_uuid, "new_numerical_tolerance": None,
        "strict_reproduction_scope": "four fixed input hashes, ordered coordinates/faces/footer and all reported selected-step trajectories",
        "whole_speedup": "not_assessed", "overall_metric_equivalence": "not_assessed",
        "historical_timing_is_simultaneous_control": False,
    }
    write_json(args.output / "report.json", report)
    try:
        if args.device != "cpu" and (not args.device.startswith("cuda:")
                or not args.device.removeprefix("cuda:").isdigit()):
            raise ValueError("device 必须为 cpu 或显式 cuda:N")
        report["worker_script_sha256"] = sha256(worker_script)
        report["baseline_report_sha256"] = sha256(args.baseline_report)
        run, saved, checks = validate_baseline(args)
        report["baseline_hash_checks"] = checks
        report["baseline_embedded_code_version"] = next(
            (run[key] for key in ("candidate_code_commit", "code_commit", "source_version")
             if key in run), None)
        report["baseline_embedded_code_version_available"] = report["baseline_embedded_code_version"] is not None
        report["baseline_version_independently_verified_by_this_script"] = False
        report["baseline_historical_api_seconds_including_io"] = saved["total_seconds_including_io"]
        report["baseline_historical_timing_note"] = "saved whole-run metadata only; not a concurrent CPU control; no speed ratio"
        report["source_sha256_before"] = source_manifest(args.source)
        if not report["source_sha256_before"]:
            raise ValueError("source 必须是包含 fnit/recon_all 的 src 目录")
        baseline_trajectory = trajectory(saved)
        write_json(args.output / "baseline_stage_reference.json", {
            "baseline_report": str(args.baseline_report),
            "baseline_report_sha256": report["baseline_report_sha256"],
            "baseline_version_declared": BASELINE_VERSION,
            "input_sha256": saved["input_sha256"], "output_sha256": saved["output_sha256"],
            "trajectory": baseline_trajectory})
        cache = args.output / "numba_cache_candidate"
        triton_cache = args.output / "triton_cache_candidate"
        cache.mkdir()
        triton_cache.mkdir()
        environment = dict(os.environ, PYTHONPATH=str(args.source),
            NUMBA_NUM_THREADS=str(THREADS), OMP_NUM_THREADS=str(THREADS),
            MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
            NUMEXPR_NUM_THREADS=str(THREADS), NUMBA_CACHE_DIR=str(cache),
            TRITON_CACHE_DIR=str(triton_cache))
        # 外层几何比较在计算子进程退出后才导入 NumPy，同样保持线程预算。
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(THREADS)
        child_output = args.output / "candidate"
        command = [sys.executable, str(worker_script), "--worker",
            "--subject", str(args.subject), "--hemi", args.hemi,
            "--atlas", str(args.atlas), "--candidate-source", str(args.source),
            "--candidate-version", args.source_version, "--averaging-device", args.device,
            "--output", str(child_output), "--threads", str(THREADS)]
        if args.gpu_uuid is not None:
            command.extend(["--gpu-uuid", args.gpu_uuid])
        report["command"] = command
        write_json(args.output / "report.json", report)
        tick = time.perf_counter()
        with (args.output / "candidate.log").open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        report["candidate_process_wall_seconds_including_imports"] = time.perf_counter() - tick
        report["candidate_returncode"] = result.returncode
        child_path = child_output / "report.json"
        child = json.loads(child_path.read_text()) if child_path.exists() else {"status": "failed"}
        report["candidate_stage_report"] = child
        report["candidate_stage_report_sha256"] = sha256(child_path) if child_path.exists() else None
        if result.returncode or child.get("status") != "complete":
            raise RuntimeError(f"完整候选配准未完成：{child.get('exception', child_path)}")
        candidate_api = json.loads((child_output / "api_report.json").read_text())
        if candidate_api.get("averaging_device") != args.device or candidate_api.get("overlap_device") != "cpu":
            raise ValueError("实际 API 设备与请求不符")
        report["candidate_input_hashes_match_saved_baseline"] = (
            child["input_sha256"] == candidate_api["input_sha256"] == saved["input_sha256"])
        if not report["candidate_input_hashes_match_saved_baseline"]:
            raise ValueError("候选实际输入 SHA 与冻结基线不同")
        report["candidate_source_hashes_match_requested_snapshot"] = (
            child["source_sha256"] == report["source_sha256_before"])
        if not report["candidate_source_hashes_match_requested_snapshot"]:
            raise ValueError("候选实际源码 SHA 与请求快照不同")
        report["source_sha256_after"] = source_manifest(args.source)
        report["source_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
        report["baseline_report_unchanged"] = sha256(args.baseline_report) == report["baseline_report_sha256"]
        _, _, after_checks = validate_baseline(args)
        report["input_and_reference_hashes_unchanged"] = after_checks == checks
        if not all(report[key] for key in ("source_unchanged", "baseline_report_unchanged", "input_and_reference_hashes_unchanged")):
            raise ValueError("验证期间计算源码、冻结输入或保存参考被改写")
        comparison = compare_geometry(args.saved_surface, Path(child["output"]))
        comparison["reported_trajectory_equal"] = baseline_trajectory == trajectory(candidate_api)
        comparison["sulc_seed_sha256_equal"] = saved["sulc_seed_sha256"] == candidate_api["sulc_seed_sha256"]
        report["comparison"] = comparison
        report["strict_reproduction"] = (comparison["strict_reproduction"]
            and comparison["reported_trajectory_equal"] and comparison["sulc_seed_sha256_equal"])
        report["optimization_degradation"] = (
            "not_detected_under_exact_stage_comparison" if report["strict_reproduction"] else "not_assessed")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, exception=str(exc))
        traceback.print_exc()
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    report["wrapper_wall_seconds_until_final_report_write"] = time.perf_counter() - started
    report["wrapper_wall_scope"] = "argument validation, provenance IO, candidate imports/init/full-stage IO/JIT/sync, geometry and trajectory comparison, intermediate report writes; excludes this final JSON write"
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "status": report["status"],
                      "strict_reproduction": report.get("strict_reproduction")}, ensure_ascii=False))
    return 1 if report["status"] != "complete" else (0 if report["strict_reproduction"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
