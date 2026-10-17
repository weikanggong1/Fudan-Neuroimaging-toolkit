"""只读适配已完成的mixed-native-surfaces FNIT控制/候选整例，不运行生产。

复用当前evaluate_recon_torch_run.validate_candidate及已有driver.compare_pair；
控制组始终是FNIT control，绝不创建官方参考manifest或改标签伪装官方。
启动前要求本轮四份benchmark和给定外层EXIT都完成；全部事后计算串行、
固定4个CPU线程并隐藏CUDA。原门槛不变，严格复现、优化差异与整体等效分列。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def regular(path):
    path = Path(os.path.abspath(path))
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("required plain file contains a link or is missing: " + str(path))
    return path


def read(path):
    return json.loads(regular(path).read_text())


def bound(path, expected):
    path = regular(path)
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or sha(path) != expected:
        raise ValueError("SHA binding differs: " + str(path))
    return path


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def all_finished(*, benchmarks, completion_exits):
    """仅校验四份整例metadata和外层EXIT；失败时不开始事后数值计算。"""
    paths = [regular(path) for path in benchmarks]
    if len(paths) != 4 or len(set(paths)) != 4 or not completion_exits:
        raise ValueError("four distinct completed benchmarks and outer EXIT required")
    binding = {"benchmarks": {}, "completion_exits": {}}
    for path in paths:
        record = read(path)
        runtime_path = regular(path.parent / "subject/fnit-native-free-run.json")
        runtime = read(runtime_path)
        outputs = runtime.get("output_validation", {})
        if (record.get("execution") != "complete" or record.get("exit_code") != 0 or
                record.get("runtime_status") != "complete" or runtime.get("status") != "complete" or
                outputs.get("status") != "passed" or outputs.get("expected") != 138 or
                outputs.get("present") != 138 or outputs.get("missing") or
                runtime.get("mesh_validation", {}).get("status") != "passed"):
            raise ValueError("all four raw-T1 runs must finish before evaluation: " + str(path))
        binding["benchmarks"][str(path)] = {"sha256": sha(path), "runtime_sha256": sha(runtime_path),
                                           "code_version": record.get("code_version"),
                                           "input_sha256": record.get("input_sha256")}
    for filename in completion_exits:
        path = regular(filename)
        if path.read_text().strip() != "0":
            raise ValueError("outer whole-case launcher is not EXIT0: " + str(path))
        binding["completion_exits"][str(path)] = sha(path)
    return binding


def same_budget(control, candidate):
    """同输入、资源、CPU/GPU可见性、线程与已安装程序必须相同。"""
    required = ("input_sha256", "host", "cpu", "cpu_affinity", "threads", "device",
                "environment", "resource_sha256", "native_program_sha256")
    for name in required:
        if name not in control or control[name] != candidate.get(name):
            raise ValueError("control/candidate resource or hardware differs: " + name)
    if control["threads"] != 4:
        raise ValueError("both whole cases must use the four-thread budget")
    if not control["resource_sha256"] or not control["native_program_sha256"]:
        raise ValueError("missing resource/program SHA bindings")
    if control.get("n4_runtime_dependencies") != candidate.get("n4_runtime_dependencies"):
        raise ValueError("N4 runtime dependency binding differs")
    return {name: control[name] for name in required}


def helpers_manifest(path, expected, driver, scripts_dir):
    """绑定既有helper清单全部文件；仅这些声明的driver/scripts可被调用。"""
    manifest_path = bound(path, expected)
    value = read(manifest_path)
    records = value.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("helpers manifest requires the existing nonempty files list")
    root = manifest_path.parent
    verified = {}
    for row in records:
        payload = regular(row["path"])
        if not payload.is_relative_to(root) or str(payload) in verified:
            raise ValueError("helper path escapes manifest root or is duplicated")
        bound(payload, row["sha256"])
        verified[str(payload)] = row["sha256"]
    required = [regular(driver)]
    required += [regular(Path(scripts_dir) / (name + ".py")) for name in (
        "compare_complete_subject", "compare_surface_chain", "compare_region_stats",
        "compare_parcellation_dice", "benchmark_surface_quality_extended")]
    if any(str(path) not in verified for path in required):
        raise ValueError("driver or mandatory script is not covered by helpers manifest")
    return {"manifest_sha256": expected, "git_commit": value.get("git_commit"),
            "verified_files": len(verified), "files": verified}


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(args):
    """具名路径均为冻结文件；输出新目录，失败保留pair.json和既有结果。"""
    if args.output.exists():
        raise FileExistsError(args.output)
    # 在任何Torch/Numba导入前设定；无GPU初始化，不与整例争用设备。
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
    completion = all_finished(benchmarks=[args.control_benchmark, args.candidate_benchmark,
                                        *args.other_benchmark], completion_exits=args.completion_exit)
    evaluator = bound(args.evaluator, args.evaluator_sha256)
    checks = bound(evaluator.with_name("recon_evaluation_checks.py"), args.checks_sha256)
    label_table = bound(args.label_table.resolve(strict=True), args.label_table_sha256)
    helper_binding = helpers_manifest(args.helpers_manifest, args.helpers_manifest_sha256,
                                      args.driver, args.scripts_dir)
    sys.path.insert(0, str(evaluator.parent))
    sys.path.insert(0, str(args.candidate_source_root / "src"))
    validation = module_from_path("_native_pair_validation", evaluator)
    native = validation.validate_native_program_manifest(
        path=args.expected_native_programs_manifest,
        expected_sha256=args.expected_native_programs_manifest_sha256)
    raw_control = read(args.control_benchmark)
    # 此参数只是既有validator的同原始T1字段绑定，不是官方参考清单或视图。
    input_binding = {"kind": "fnit_control_candidate_input_binding",
                     "cases": {args.case: {"input_sha256": raw_control["input_sha256"]}}}
    reports, runtimes = {}, {}
    for role, benchmark, source in (("control", args.control_benchmark, args.control_source_root),
                                    ("candidate", args.candidate_benchmark, args.candidate_source_root)):
        reports[role], runtimes[role] = validation.validate_candidate(
            benchmark_path=benchmark, source_root=source, case=args.case,
            reference_manifest=input_binding, candidate_profile="mixed-native-surfaces",
            expected_native_programs=native)
    budget = same_budget(reports["control"], reports["candidate"])
    driver = module_from_path("_frozen_native_pair_driver", args.driver)
    driver.configure_runtime()
    import numba
    import torch
    from recon_evaluation_checks import all_partition_stats
    if torch.get_num_threads() != 4 or numba.get_num_threads() != 4 or torch.cuda.is_initialized():
        raise RuntimeError("posthoc must run with four CPU threads and no initialized CUDA")
    args.output.mkdir(parents=True, mode=0o700)
    state = {"scope": "same-environment FNIT candidate versus FNIT control; neither side is official",
             "status": "running", "case": args.case, "candidate_profile": "mixed-native-surfaces",
             "completion_gate": completion, "budget_binding": budget,
             "helper_binding": helper_binding,
             "evaluator_sha256": sha(evaluator), "checks_sha256": sha(checks),
             "label_table_sha256": sha(label_table),
             "adapter_sha256": sha(__file__), "native_manifest_sha256": args.expected_native_programs_manifest_sha256,
             "code_versions": {role: row["code_version"] for role, row in reports.items()},
             "cli_wall_seconds": {role: row["cli_wall_seconds"] for role, row in reports.items()},
             "full_harness_seconds": {role: row["full_harness_seconds"] for role, row in reports.items()},
             "output_validation": {role: row["output_validation"] for role, row in runtimes.items()},
             "production_mesh_validation": {role: row["mesh_validation"] for role, row in runtimes.items()},
             "output_byte_bindings": {role: row["candidate_output_binding"] for role, row in reports.items()},
             "strict_reproduction": "not_assessed", "optimization_regression": "not_assessed",
             "overall_metric_equivalence": "not_assessed_no_confirmed_prospective_thresholds",
             "official_reference_read": False, "threads": 4, "cuda_visible_during_evaluation": "",
             "actual_torch_threads": torch.get_num_threads(), "actual_numba_threads": numba.get_num_threads(),
             "cuda_initialized_before": torch.cuda.is_initialized()}
    save(args.output / "pair.json", state)
    started = time.perf_counter()
    cache, commands = driver.ExactDistanceCache(), []
    subjects = {"control": args.control_benchmark.parent / "subject",
                "candidate": args.candidate_benchmark.parent / "subject"}
    try:
        derived = {str(path): driver.no_th3_subject(path) for path in subjects.values()}
        save(args.output / "no_th3_inputs.json", derived)
        config = {"python": sys.executable, "scripts_dir": str(args.scripts_dir), "label_table": str(label_table)}
        strict = driver.compare_pair(subjects["control"], subjects["candidate"], "candidate_vs_control",
                                     reports["candidate"]["code_version"], config, args.output,
                                     commands, derived, cache)
        state["strict_reproduction"] = strict
        region_tool = driver.tool("compare_region_stats", args.scripts_dir)
        save(args.output / "all_regions_candidate_vs_control.json", all_partition_stats(
            reference=subjects["control"], candidate=subjects["candidate"], region_tool=region_tool))
        save(args.output / "distance_cache.json", cache.report())
        cache.clear()
        for role, subject in subjects.items():
            driver.execute([sys.executable, str(args.scripts_dir / "benchmark_surface_quality_extended.py"),
                            "--subject", str(subject), "--output", str(args.output / ("quality_" + role)),
                            "--code-version", reports[role]["code_version"], "--source-kind", "fnit",
                            "--threads", "4", "--cross-timeout-seconds", "180", "--max-bbox-pairs", "20000000"],
                           args.output / "commands.log", commands)
        state.update(status="complete", optimization_regression="measured_not_automatically_accepted; review_existing_operator_tolerances_and_local_quality",
                     quality={role: read(args.output / ("quality_" + role) / "report.json")["status"]
                              for role in subjects})
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
    finally:
        cache.clear()
        state["comparison_wall_seconds"] = time.perf_counter() - started
        state["timing_scope"] = "post-run diagnostics; excluded from production runtime; preflight/imports excluded from this comparison timer"
        state["commands"] = commands
        state["cuda_initialized_after"] = torch.cuda.is_initialized()
        save(args.output / "pair.json", state)
    return 0 if state["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-benchmark", "candidate-benchmark", "control-source-root", "candidate-source-root",
                 "evaluator", "driver", "scripts-dir", "label-table", "helpers-manifest",
                 "expected-native-programs-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("evaluator-sha256", "checks-sha256", "label-table-sha256", "helpers-manifest-sha256",
                 "expected-native-programs-manifest-sha256", "case"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--other-benchmark", type=Path, action="append", required=True,
                        help="另两份完整整例benchmark；必须重复两次，保证四次整例都完成后评估")
    parser.add_argument("--completion-exit", type=Path, action="append", required=True,
                        help="已核对的整例外层launcher.EXIT；每份须已存在且内容为0")
    args = parser.parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
