"""复用现有整例比较器，评价一次已完成的原始T1运行；不运行生产。

候选benchmark、冻结代码、既有官方参考清单、比较脚本目录和新输出目录
均须显式指定。严格138诊断、几何对应、Dice、逐脑区与no-th3统计、
双向点到三角面、质量覆盖和脑图分别保存；整体等效不设新阈值。
官方参考输入绑定必须与候选原始T1 SHA一致。对照跨环境/历史时间单列。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_candidate(*, benchmark_path: Path, source_root: Path,
                       case: str, reference_manifest: dict) -> tuple[dict, dict]:
    """绑定实际完成收据/原始输入/模块哈希，返回benchmark与运行报告。

    benchmark_path为整例脚本JSON；source_root为实际冻结仓库根目录；case
    必须在reference_manifest.cases中。所有参数必填；哈希为SHA-256。
    只读校验，不创建输出或修改输入。失败/不完整/源码变化/输入错配抛
    ValueError，文件和JSON错误传播。内部验证步骤没有独立官方CLI。
    """
    report = json.loads(benchmark_path.read_text())
    runtime = json.loads((benchmark_path.parent / "subject/fnit-native-free-run.json").read_text())
    if report.get("execution") != "complete" or report.get("exit_code") != 0:
        raise ValueError("candidate raw T1 execution did not complete successfully")
    if runtime.get("status") != "complete":
        raise ValueError("candidate output/production mesh validation failed")
    complete = runtime.get("output_validation", {})
    if complete.get("expected") != 138 or complete.get("present") != 138 or complete.get("missing"):
        raise ValueError("candidate fixed 138 output manifest is incomplete")
    if report.get("input_sha256") != reference_manifest["cases"][case]["input_sha256"]:
        raise ValueError("candidate and official original T1 hashes differ")
    if sha(Path(runtime["input"])) != report["input_sha256"]:
        raise ValueError("runtime original T1 differs from benchmark binding")
    sources = report.get("source_sha256", {})
    if not sources or "recon_all/native_free.py" not in sources:
        raise ValueError("generator source binding is missing")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid generator relative path")
        if sha(source_root / "src/fnit" / path) != expected:
            raise ValueError("generator source changed: " + relative)
    return report, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--label-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads != 4:
        raise ValueError("existing frozen comparison uses four threads")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.reference_root / "TRANSFER_MANIFEST.private.json"
    manifest = json.loads(manifest_path.read_text())
    benchmark, runtime = validate_candidate(benchmark_path=args.benchmark,
        source_root=args.source_root, case=args.case, reference_manifest=manifest)
    reference = args.reference_root / "subjects" / args.case
    subject = args.benchmark.parent / "subject"
    spec = importlib.util.spec_from_file_location("_existing_whole_comparison", args.driver)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.configure_runtime()
    args.output.mkdir(parents=True)
    save = lambda name, data: driver.write(args.output / name, data)
    started = time.perf_counter()
    state = {"scope": "one complete raw T1 candidate versus archived official; not optimized/control pairing",
        "status": "running", "case": args.case, "execution_complete": True,
        "output_validation": runtime["output_validation"], "production_mesh_validation": runtime["mesh_validation"],
        "benchmark_sha256": sha(args.benchmark),
        "run_sha256": sha(subject / "fnit-native-free-run.json"),
        "tested_code_version": benchmark["code_version"],
        "generator_source_sha256": benchmark["source_sha256"],
        "reference_manifest_sha256": sha(manifest_path),
        "reference": manifest["cases"][args.case],
        "comparator_sha256": {str(p): sha(p) for p in [Path(__file__), args.driver, *args.scripts_dir.glob("*.py")]},
        "cli_wall_seconds": benchmark["cli_wall_seconds"],
        "full_harness_seconds": benchmark["full_harness_seconds"],
        "process_memory": {k:v for k,v in benchmark.get("process_memory",{}).items() if k!="samples"},
        "strict_reproduction": "not_assessed", "optimization_regression": "not_assessed_waiting_same_case_control",
        "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds"}
    save("evaluation.json", state)
    commands, cache = [], driver.ExactDistanceCache()
    try:
        no_th3 = {str(p): driver.no_th3_subject(p) for p in (reference, subject)}
        save("no_th3_inputs.json", no_th3)
        config = {"python": sys.executable, "scripts_dir": str(args.scripts_dir), "label_table": str(args.label_table)}
        strict = driver.compare_pair(reference, subject, "candidate_vs_official", benchmark["code_version"],
            config, args.output, commands, no_th3, cache)
        state["strict_reproduction"] = strict
        save("distance_cache.json", cache.report())
        cache.clear()
        for role, path in (("candidate", subject), ("official", reference)):
            driver.execute([sys.executable, str(args.scripts_dir / "benchmark_surface_quality_extended.py"),
                "--subject", str(path), "--output", str(args.output / ("quality_"+role)),
                "--code-version", benchmark["code_version"] if role=="candidate" else manifest["cases"][args.case]["official_version"],
                "--source-kind", "fnit" if role=="candidate" else "official", "--threads", "4",
                "--cross-timeout-seconds", "180", "--max-bbox-pairs", "20000000"], args.output / "commands.log", commands)
        driver.execute([sys.executable, str(args.scripts_dir / "plot_recon_all_comparison.py"),
            "--reference", str(reference), "--candidate", str(subject),
            "--region-report", str(args.output / "region_candidate_vs_official.json"),
            "--dice-report", str(args.output / "dice_candidate_vs_official.json"),
            "--output-dir", str(args.output / "figures"), "--code-commit", benchmark["code_version"]],
            args.output / "commands.log", commands)
        state.update(status="complete", quality={role: driver.read(args.output/("quality_"+role)/"report.json")["status"]
                    for role in ("candidate","official")})
        return 0
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        cache.clear()
        state["comparison_wall_seconds"] = time.perf_counter() - started
        state["timing_scope"] = "post-run diagnostics and figures; excluded from recon-all runtime"
        save("evaluation.json", state)


if __name__ == "__main__":
    sys.exit(main())
