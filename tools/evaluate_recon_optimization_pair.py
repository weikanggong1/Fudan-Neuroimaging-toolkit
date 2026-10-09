"""绑定两次完成的原始T1整例，复用既有比较器评价性能优化是否改变输出。

只读控制/候选收据、自产结果和冻结源码；不运行生产或修改算法门槛。
完整计时来自各自benchmark，比较耗时单列。整体官方等效仍未判定。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

from evaluate_recon_torch_run import sha, validate_candidate


def main() -> int:
    """具名路径指定两整例/实际源码、比较器、LUT与新报告目录。

    全部路径必填；case是两次共享的公开被试标识。仅接受四线程，
    同主机/CPU/亲和性/目标GPU环境、同原始T1及完整138项输出。
    输出严格文件、Dice、逐脑区/no-th3、双向表面距离及绑定JSON；
    mm、mm²、mm³与原比较器相同。非法配对或修改过的源码抛异常，
    比较错误保留失败收据。通过文件诊断不自动判定整体官方等效。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-benchmark", "candidate-benchmark", "control-source-root",
                 "candidate-source-root", "driver", "scripts-dir", "label-table", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("existing comparison and paired thread budget require four threads")
    if args.output.exists():
        raise FileExistsError(args.output)
    control_raw = json.loads(args.control_benchmark.read_text())
    manifest = {"cases": {args.case: {"input_sha256": control_raw["input_sha256"]}}}
    control, control_runtime = validate_candidate(benchmark_path=args.control_benchmark,
        source_root=args.control_source_root, case=args.case, reference_manifest=manifest)
    candidate, candidate_runtime = validate_candidate(benchmark_path=args.candidate_benchmark,
        source_root=args.candidate_source_root, case=args.case, reference_manifest=manifest)
    for key in ("host", "cpu", "cpu_affinity", "threads", "device"):
        if key not in control or control[key] != candidate.get(key):
            raise ValueError("pair differs in hardware/budget field: " + key)
    if control["threads"] != args.threads:
        raise ValueError("comparison and whole-case thread budgets differ")
    if control.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != candidate.get("environment", {}).get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("pair GPU visibility differs")
    for report in (control, candidate):
        if report.get("cli_wall_seconds", 0) <= 0:
            raise ValueError("complete CLI wall time missing")
    spec = importlib.util.spec_from_file_location("_existing_optimization_comparison", args.driver)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    driver.configure_runtime()
    args.output.mkdir(parents=True)
    save = lambda name, data: driver.write(args.output / name, data)
    control_subject = args.control_benchmark.parent / "subject"
    candidate_subject = args.candidate_benchmark.parent / "subject"
    state = {"scope": "two completed raw T1 empty-directory FNIT runs; no official performance ratio",
        "status": "running", "case": args.case, "input_sha256": control["input_sha256"],
        "code_versions": {"control": control["code_version"], "candidate": candidate["code_version"]},
        "source_sha256": {"control": control["source_sha256"], "candidate": candidate["source_sha256"]},
        "benchmark_sha256": {"control": sha(args.control_benchmark), "candidate": sha(args.candidate_benchmark)},
        "run_sha256": {"control": sha(control_subject / "fnit-native-free-run.json"),
                       "candidate": sha(candidate_subject / "fnit-native-free-run.json")},
        "comparator_sha256": {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name("evaluate_recon_torch_run.py"),
                                                        args.driver, *args.scripts_dir.glob("*.py"))},
        "hardware": {key: control[key] for key in ("host", "cpu", "cpu_affinity", "threads", "device")},
        "cuda_visible_devices": control.get("environment", {}).get("CUDA_VISIBLE_DEVICES"),
        "cli_wall_seconds": {"control": control["cli_wall_seconds"], "candidate": candidate["cli_wall_seconds"]},
        "full_harness_seconds": {"control": control["full_harness_seconds"], "candidate": candidate["full_harness_seconds"]},
        "speedup": control["cli_wall_seconds"] / candidate["cli_wall_seconds"],
        "wall_reduction_percent": 100 * (1 - candidate["cli_wall_seconds"] / control["cli_wall_seconds"]),
        "output_validation": {"control": control_runtime["output_validation"], "candidate": candidate_runtime["output_validation"]},
        "production_mesh_validation": {"control": control_runtime["mesh_validation"], "candidate": candidate_runtime["mesh_validation"]},
        "strict_reproduction": "not_assessed", "optimization_regression": "not_assessed",
        "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds",
        "performance_limit": "one AB/BA pair per case; shared load and cold startup/JIT included; no repeated whole-case ABBA"}
    save("pair.json", state)
    cache, commands, started = driver.ExactDistanceCache(), [], time.perf_counter()
    try:
        no_th3 = {str(p): driver.no_th3_subject(p) for p in (control_subject, candidate_subject)}
        save("no_th3_inputs.json", no_th3)
        config = {"python": __import__("sys").executable, "scripts_dir": str(args.scripts_dir), "label_table": str(args.label_table)}
        strict = driver.compare_pair(control_subject, candidate_subject, "candidate_vs_control", candidate["code_version"],
                                     config, args.output, commands, no_th3, cache)
        state["strict_reproduction"] = strict
        state["optimization_regression"] = "measured_file_region_dice_surface_reports; review_values_and_existing_operator_tolerances"
        save("distance_cache.json", cache.report())
        state["status"] = "complete"
        return 0
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        cache.clear()
        state["comparison_wall_seconds"] = time.perf_counter() - started
        state["timing_scope"] = "post-run numerical diagnosis; excluded from production wall time"
        save("pair.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
