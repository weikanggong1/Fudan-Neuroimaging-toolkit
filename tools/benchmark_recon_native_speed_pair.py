"""同节点、同设备的两例原始 T1 旧新整例反序配对监督器。

仅调用两份冻结源码各自的 run_recon_native_end_to_end_cases.py，不实现
重建、不读官方参考、不重试或补跑。病例清单恰好两例，顺序为首例 A→B、
次例 B→A；每次都使用新的空目录。A 是基线，B 仅额外选择已有的 torch
球面法向。两臂各自新建 Numba/Triton 缓存，并在各自两例之间复用。
外层负责 GPU 租约、CPU 总预算及精度环境；本工具仅改变 PYTHONPATH 和
两种缓存目录，记录实际边界。许可证由环境继承，不记录路径或内容。
输入、输出结构、全部参数和完整例子见 RECON_NATIVE_SPEED_PAIR.md。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


SUPERVISOR = "tools/run_recon_native_end_to_end_cases.py"
BENCHMARK = "tools/benchmark_recon_torch_end_to_end.py"
FORBIDDEN_ARGUMENTS = frozenset({
    "--t1", "--output-root", "--weights-dir", "--assets-dir", "--native-bin-dir",
    "--code-version", "--backend", "--topology-backend", "--topology-finish-backend",
    "--surface-placement-backend", "--device", "--threads", "--n4-backend",
    "--n4-execution", "--sphere-normals-backend",
})
RECORDED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OMP_MAX_ACTIVE_LEVELS", "OMP_DYNAMIC", "MKL_DYNAMIC",
    "PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_DEVICE_ORDER", "FNIT_RECON_FORBID_ORIGINAL_STAGES",
)


def sha256_file(path: Path) -> str:
    """读取显式文件并返回 SHA-256；读取失败直接抛 OSError。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    """以同目录临时文件原子写出 JSON，不删除已有科学产物。"""
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def source_manifest(source: Path) -> dict[str, str]:
    """记录 src 下计算源码和两个实际被调用工具的 SHA，不读取 Git 凭据。"""
    paths = {source / SUPERVISOR, source / BENCHMARK}
    extensions = {".py", ".cpp", ".c", ".h", ".cu", ".cuh"}
    paths.update(path for path in (source / "src").rglob("*")
                 if path.is_file() and path.suffix in extensions and "__pycache__" not in path.parts)
    return {str(path.relative_to(source)): sha256_file(path) for path in sorted(paths)}


def _load_list(path: Path, name: str) -> list:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return value


def run_pair(*, cases_json: Path, output_root: Path,
             baseline_source: Path, baseline_version: str,
             candidate_source: Path, candidate_version: str,
             weights_dir: Path, assets_dir: Path, native_bin_dir: Path,
             dependency_pythonpath_json: Path, common_arguments_json: Path,
             python_interpreter: Path,
             n4_backend: str = "simpleitk", n4_execution: str = "in-process") -> dict:
    """运行两例 A/B 与 B/A，返回私有完整报告；算法失败保留并继续规定顺序。

    所有 source 为包含 src/ 和 tools/ 的冻结源码根；version 是实际提交
    标识。cases_json 为 [{case,t1,sha256},...]，恰好两例不同原始影像。
    dependency_pythonpath_json 是依赖目录列表，common_arguments_json
    是两臂完全相同的附加 benchmark argv 字符串列表。source 和依赖目录
    组成各自子进程 PYTHONPATH，不继承可能污染的旧 PYTHONPATH。
    weights/assets/native 为已授权资源目录，解释器为显式可执行文件。
    N4 默认 simpleitk/in-process，值限制沿用现有监督器。设备 cuda:0、
    threads=4 由其固定；其他线程环境与 GPU UUID 由外层设置并记录。
    输出根必须不存在。输出 speed_pair.private.json、四个 one-case JSON、
    外层日志和四套现有监督器产物。墙钟为秒，影像空间与质量由原流程
    检查。参数或哈希错误在创建目录前抛异常；子进程失败不重试，最终状态
    非 complete。该编排无独立官方命令，对应完整 recon-all -i T1 -all。
    """
    started = time.perf_counter()
    output_root = Path(output_root).absolute()
    if os.path.lexists(output_root):
        raise FileExistsError(f"output-root must not exist: {output_root}")
    sources = {"baseline": Path(baseline_source).resolve(),
               "candidate": Path(candidate_source).resolve()}
    versions = {"baseline": baseline_version, "candidate": candidate_version}
    if not all(isinstance(value, str) and value.strip() for value in versions.values()):
        raise ValueError("both source versions must be explicit nonempty identifiers")
    for source in sources.values():
        if not (source / "src").is_dir():
            raise NotADirectoryError(source / "src")
        for relative in (SUPERVISOR, BENCHMARK):
            if not (source / relative).is_file():
                raise FileNotFoundError(source / relative)
    resources = {"weights_dir": Path(weights_dir).resolve(),
                 "assets_dir": Path(assets_dir).resolve(),
                 "native_bin_dir": Path(native_bin_dir).resolve()}
    for resource in resources.values():
        if not resource.is_dir():
            raise NotADirectoryError(resource)
    interpreter = Path(python_interpreter).absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("python-interpreter must be an existing executable file")
    dependency_values = _load_list(dependency_pythonpath_json, "dependency-pythonpath-json")
    if not all(isinstance(value, str) and value for value in dependency_values):
        raise ValueError("dependency PYTHONPATH entries must be nonempty path strings")
    dependencies = [Path(value).resolve() for value in dependency_values]
    for dependency in dependencies:
        if not dependency.is_dir():
            raise NotADirectoryError(dependency)
        if (dependency / "fnit").exists() or (dependency / "fnit.py").exists():
            raise ValueError("dependency PYTHONPATH must not shadow the declared FNIT source")
    common = _load_list(common_arguments_json, "common-arguments-json")
    if not all(isinstance(value, str) and "\x00" not in value for value in common):
        raise ValueError("common arguments must be argv strings without NUL")
    if any(value.split("=", 1)[0] in FORBIDDEN_ARGUMENTS for value in common):
        raise ValueError("common arguments cannot override frozen profile, budget or normals arm")
    if n4_backend not in {"native", "simpleitk", "torch"}:
        raise ValueError("invalid N4 backend")
    if n4_execution not in {"in-process", "isolated"} or (n4_execution == "isolated" and n4_backend != "torch"):
        raise ValueError("isolated N4 requires torch")
    cases = _load_list(cases_json, "cases-json")
    if len(cases) != 2:
        raise ValueError("exactly two real raw-T1 cases are required")
    for case in cases:
        if not isinstance(case, dict) or not {"case", "t1", "sha256"}.issubset(case):
            raise ValueError("each case needs case, t1 and sha256")
        identifier = case["case"]
        if (not isinstance(identifier, str) or not identifier or identifier in {".", ".."}
                or Path(identifier).name != identifier):
            raise ValueError("case must be a safe directory name")
        if not isinstance(case["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", case["sha256"]):
            raise ValueError("raw T1 SHA-256 must be 64 lowercase hexadecimal digits")
        if sha256_file(Path(case["t1"])) != case["sha256"]:
            raise ValueError(f"raw T1 hash mismatch: {identifier}")
    if len({case["case"] for case in cases}) != 2 or len({case["sha256"] for case in cases}) != 2:
        raise ValueError("two case names and raw T1 inputs must be distinct")
    manifests = {arm: source_manifest(source) for arm, source in sources.items()}
    preflight_seconds = time.perf_counter() - started
    output_root.mkdir(parents=True)
    (output_root / "plans").mkdir()
    caches = {}
    for arm in sources:
        caches[arm] = {}
        for name in ("numba", "triton"):
            directory = output_root / "cache" / arm / name
            directory.mkdir(parents=True)
            caches[arm][name] = str(directory)
    report_path = output_root / "speed_pair.private.json"
    report = {
        "schema_version": 1, "scope": "two-raw-T1-four-empty-directory-paired-whole-runs",
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": {"system": platform.system(),
                                               "release": platform.release(),
                                               "machine": platform.machine()},
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "script_sha256": sha256_file(Path(__file__)),
        "cases_json": str(Path(cases_json).resolve()), "cases_json_sha256": sha256_file(cases_json),
        "cases": cases, "sources": {arm: {"path": str(source), "version": versions[arm],
                                            "sha256": manifests[arm]} for arm, source in sources.items()},
        "resource_directories": {key: str(value) for key, value in resources.items()},
        "python_interpreter": str(interpreter), "python_interpreter_sha256": sha256_file(interpreter),
        "dependency_pythonpath_json_sha256": sha256_file(dependency_pythonpath_json),
        "dependency_pythonpath": [str(path) for path in dependencies],
        "common_arguments_json_sha256": sha256_file(common_arguments_json),
        "common_benchmark_arguments": common,
        "candidate_only_benchmark_arguments": ["--sphere-normals-backend", "torch"],
        "n4_backend": n4_backend, "n4_execution": n4_execution,
        "inherited_environment": {name: os.environ.get(name) for name in RECORDED_ENVIRONMENT},
        "environment_scope": "all other environment inherited; license path/content intentionally omitted; external lease owns GPU and thread budget",
        "fresh_caches": caches, "cache_scope": "empty per arm at pair start; reused within that arm across both subjects",
        "preflight_wall_seconds": preflight_seconds,
        "outer_run_wall_scope": "immediately before one-case supervisor Popen through supervisor exit; includes its input validation, benchmark, reads/writes and final report",
        "cli_wall_scope": "copied unchanged from each actual benchmark.json; never replaced by historical timing",
        "whole_metric_equivalence": "not_assessed", "error_assessment": "not_assessed", "runs": [],
    }
    save_json(report_path, report)
    for case_index, case in enumerate(cases):
        order = ("baseline", "candidate") if case_index == 0 else ("candidate", "baseline")
        for arm in order:
            run_id = f"{case_index}_{case['case']}_{arm}"
            plan = output_root / "plans" / (run_id + ".json")
            save_json(plan, [case])
            run_root = output_root / "runs" / run_id
            source = sources[arm]
            extras = common + (["--sphere-normals-backend", "torch"] if arm == "candidate" else [])
            command = [str(interpreter), "-u", str(source / SUPERVISOR),
                       "--cases-json", str(plan), "--output-root", str(run_root),
                       "--benchmark-script", str(source / BENCHMARK),
                       "--weights-dir", str(resources["weights_dir"]),
                       "--assets-dir", str(resources["assets_dir"]),
                       "--native-bin-dir", str(resources["native_bin_dir"]),
                       "--code-version", versions[arm], "--python-interpreter", str(interpreter),
                       "--n4-backend", n4_backend, "--n4-execution", n4_execution, "--", *extras]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in dependencies),
                                                 str(source / "src"), str(source / "tools")])
            env["NUMBA_CACHE_DIR"] = caches[arm]["numba"]
            env["TRITON_CACHE_DIR"] = caches[arm]["triton"]
            row = {"run_id": run_id, "case": case["case"], "case_index": case_index,
                   "arm": arm, "code_version": versions[arm], "input_sha256": case["sha256"],
                   "command": command, "output_root": str(run_root), "exit_code": None,
                   "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "environment_override": {name: env[name] for name in
                                              ("PYTHONPATH", "NUMBA_CACHE_DIR", "TRITON_CACHE_DIR")},
                   "supervisor_report": str(run_root / "cases.private.json"),
                   "benchmark_path": str(run_root / case["case"] / "benchmark.json")}
            report["runs"].append(row)
            save_json(report_path, report)
            log_path = output_root / (run_id + ".outer.log")
            tick = time.perf_counter()
            try:
                with log_path.open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                    row["exit_code"] = process.wait()
                row["status"] = "complete" if row["exit_code"] == 0 else "failed"
            except OSError as exc:
                row.update(status="failed", launch_error=f"{type(exc).__name__}: {exc}")
            row["outer_supervisor_wall_seconds"] = time.perf_counter() - tick
            row["ended_utc"] = datetime.now(timezone.utc).isoformat()
            row["outer_log"] = str(log_path)
            supervisor_path, benchmark_path = Path(row["supervisor_report"]), Path(row["benchmark_path"])
            try:
                if not supervisor_path.is_file() or not benchmark_path.is_file():
                    raise FileNotFoundError("supervisor or benchmark report missing")
                supervisor = json.loads(supervisor_path.read_text())
                if (supervisor.get("status") != "complete" or len(supervisor.get("cases", [])) != 1
                        or supervisor["cases"][0].get("case") != case["case"]
                        or supervisor["cases"][0].get("input_sha256") != case["sha256"]):
                    raise ValueError("one-case supervisor did not complete the declared raw input")
                row["supervisor_report_sha256"] = sha256_file(supervisor_path)
                row["supervisor_case"] = supervisor["cases"][0]
                row["supervisor_case_outer_wall_seconds"] = supervisor["cases"][0].get("outer_case_wall_seconds")
                row["benchmark_sha256"] = sha256_file(benchmark_path)
                benchmark = json.loads(benchmark_path.read_text())
                row["benchmark_summary"] = {name: benchmark[name] for name in
                    ("execution", "runtime_status", "failed_stage", "output_completeness", "grid_quality",
                     "cli_wall_seconds", "full_harness_seconds", "process_memory") if name in benchmark}
                row["benchmark_cli_wall_seconds"] = benchmark.get("cli_wall_seconds")
                row["benchmark_full_harness_seconds"] = benchmark.get("full_harness_seconds")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status="failed", report_error=f"{type(exc).__name__}: {exc}")
            save_json(report_path, report)
    try:
        report["integrity_after_runs"] = {
            "source_sha256_unchanged": {arm: source_manifest(source) == manifests[arm]
                                         for arm, source in sources.items()},
            "raw_inputs_unchanged": {case["case"]: sha256_file(Path(case["t1"])) == case["sha256"]
                                      for case in cases},
        }
        intact = all(all(values.values()) for values in report["integrity_after_runs"].values())
    except OSError as exc:
        report["integrity_after_runs"] = {"error": f"{type(exc).__name__}: {exc}"}
        intact = False
    report["status"] = ("complete" if intact and all(row["status"] == "complete" for row in report["runs"])
                        else "completed_with_failures")
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    report["total_pair_wall_seconds"] = time.perf_counter() - started
    report["total_wall_scope"] = "run_pair input/source validation, four supervisor runs, interim report writes and integrity recheck; excludes final report publication and interpreter import"
    save_json(report_path, report)
    return report


def main() -> int:
    """具名参数读取现有路径；失败非零退出，报告不覆盖历史运行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases-json", "output-root", "baseline-source", "candidate-source",
                 "weights-dir", "assets-dir", "native-bin-dir", "dependency-pythonpath-json",
                 "common-arguments-json", "python-interpreter"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--n4-backend", choices=("native", "simpleitk", "torch"), default="simpleitk")
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process")
    arguments = vars(parser.parse_args())
    report = run_pair(**arguments)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
