"""从原始 T1 顺序运行 Conda 自编译阶段配方，并记录完整整例墙钟。

病例清单为 JSON 列表，每项含 case（目录名）、t1（原始 T1 路径）和
sha256（输入文件 SHA-256）。输出根必须不存在；所有输入先验哈希通过后才
创建目录。每例独立启动 benchmark，保存退出码、完整外层墙钟和原始报告。
许可证只从进程环境 FS_LICENSE 继承，不写入命令或本脚本报告。

本脚本只调度 FNIT 的可选 Conda 自编译阶段配方，不读取官方参考结果，
也不替代 benchmark 内的 138 项输出及网格质量门。异常会保留现场；某例
失败后仍运行下一例。GPU 资源租约由外层 run_recon_single_gpu_job.py 管理。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


FIXED_OPTIONS = frozenset({
    "--t1", "--output-root", "--weights-dir", "--assets-dir", "--native-bin-dir",
    "--code-version", "--backend", "--topology-backend",
    "--topology-finish-backend", "--surface-placement-backend",
    "--device", "--threads", "--n4-backend", "--n4-execution",
})


def sha256_file(path: Path) -> str:
    """计算输入或工具文件的 SHA-256；不可读文件直接抛 OSError。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    """以同目录临时文件原子发布私有机器报告。"""
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_cases(*, cases: list[dict], output_root: Path, benchmark_script: Path,
              weights_dir: Path, assets_dir: Path, native_bin_dir: Path,
              code_version: str, python_interpreter: str = sys.executable,
              n4_backend: str = "simpleitk", n4_execution: str = "in-process",
              benchmark_arguments: tuple[str, ...] = ()) -> dict:
    """运行两例或更多原始 T1 空目录整例，返回逐例退出及质量摘要。

    所有路径均为调用者显式提供的本机文件；输出根必须不存在。
    原始 T1 含影像及几何，格式由 FNIT benchmark 接口接受；本函数只读取
    字节计算哈希。输出为 output_root/case/benchmark.json、subject/ 和
    case.outer.log，外加 cases.private.json。墙钟单位为秒，逐例范围从
    启动 benchmark 解释器前至其最终文件写出并退出，包含加载和读写。
    设备固定为当前 CUDA_VISIBLE_DEVICES 映射下的 cuda:0，总线程固定 4。
    附加参数仅用于未冻结的算法选项，不能覆盖输入、后端、设备或预算。
    若资源或输入不符，在创建输出前抛异常；算法失败保留原日志并继续下一例。
    对应原软件入口为 recon-all -i T1 -s SUBJECT -all；本脚本本身无官方 CLI。
    """
    started = time.perf_counter()
    output_root = Path(output_root)
    benchmark_script = Path(benchmark_script)
    weights_dir, assets_dir = Path(weights_dir), Path(assets_dir)
    native_bin_dir = Path(native_bin_dir)
    if output_root.exists():
        raise FileExistsError(f"output-root must not exist: {output_root}")
    if not cases:
        raise ValueError("cases must contain at least one raw T1")
    if not benchmark_script.is_file():
        raise FileNotFoundError(benchmark_script)
    for resource in (weights_dir, assets_dir, native_bin_dir):
        if not resource.is_dir():
            raise NotADirectoryError(resource)
    license_path = os.environ.get("FS_LICENSE")
    if not license_path or not Path(license_path).is_file():
        raise RuntimeError("native stages require an existing private FS_LICENSE file")
    if os.environ.get("FNIT_RECON_FORBID_ORIGINAL_STAGES", "").lower() not in ("", "0", "false"):
        raise RuntimeError("remove Python-only original-stage audit from native benchmark environment")
    if n4_backend not in {"native", "simpleitk", "torch"}:
        raise ValueError("invalid N4 backend")
    if n4_execution not in {"in-process", "isolated"} or (n4_execution == "isolated" and n4_backend != "torch"):
        raise ValueError("isolated N4 requires torch")
    if any(argument.split("=", 1)[0] in FIXED_OPTIONS for argument in benchmark_arguments):
        raise ValueError("benchmark arguments cannot override frozen input/profile/budget")
    identifiers = []
    for case in cases:
        identifier = case["case"]
        if (not isinstance(identifier, str) or not identifier
                or Path(identifier).name != identifier or identifier in {".", ".."}):
            raise ValueError("case must be a unique directory name")
        identifiers.append(identifier)
        if sha256_file(Path(case["t1"])) != case["sha256"]:
            raise ValueError(f"raw T1 hash mismatch: {identifier}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case names must be unique")

    output_root.mkdir(parents=True)
    report = {
        "schema_version": 1, "code_version": code_version,
        "recipe": {"backend": "native", "topology_backend": "native",
                   "topology_finish_backend": "conda-cpp",
                   "surface_placement_backend": "native",
                   "n4_backend": n4_backend, "n4_execution": n4_execution},
        "benchmark_script_sha256": sha256_file(benchmark_script),
        "supervisor_sha256": sha256_file(Path(__file__)),
        "cases_manifest": cases, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "outer_case_wall_scope": "Popen before benchmark interpreter startup through final report writes and process exit",
        "supervisor_wall_scope": "run_cases including raw input validation, all cases and prior report writes; excludes interpreter imports and final publication",
        "status": "running", "cases": [], "overall_metric_equivalence": "not_assessed",
    }
    report_path = output_root / "cases.private.json"
    save_json(report_path, report)
    for case in cases:
        command = [python_interpreter, "-X", "faulthandler", "-u", str(benchmark_script),
                   "--t1", str(case["t1"]), "--output-root", str(output_root / case["case"]),
                   "--weights-dir", str(weights_dir), "--assets-dir", str(assets_dir),
                   "--native-bin-dir", str(native_bin_dir), "--backend", "native",
                   "--topology-backend", "native", "--topology-finish-backend", "conda-cpp",
                   "--surface-placement-backend", "native", "--n4-backend", n4_backend,
                   "--n4-execution", n4_execution, "--device", "cuda:0", "--threads", "4",
                   "--code-version", code_version, *benchmark_arguments]
        with (output_root / (case["case"] + ".outer.log")).open("w") as stream:
            tick = time.perf_counter()
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            exit_code = process.wait()
            outer_wall = time.perf_counter() - tick
        row = {"case": case["case"], "input_sha256": case["sha256"],
               "command": command, "exit_code": exit_code, "outer_case_wall_seconds": outer_wall,
               "execution": "complete" if exit_code == 0 else "failed",
               "output_completeness": "not_assessed", "grid_quality": "not_assessed"}
        benchmark_path = output_root / case["case"] / "benchmark.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())
            row["benchmark_sha256"] = sha256_file(benchmark_path)
            for key in ("execution", "runtime_status", "failed_stage", "runtime_error",
                        "quality_failure", "output_completeness", "grid_quality", "process_memory",
                        "cli_wall_seconds", "full_harness_seconds"):
                if key in benchmark:
                    row[key] = benchmark[key]
        runtime_path = output_root / case["case"] / "subject" / "fnit-native-free-run.json"
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
            row["runtime_report_sha256"] = sha256_file(runtime_path)
            for key in ("failed_stage", "quality_failure"):
                if key in runtime:
                    row[key] = runtime[key]
            if "error" in runtime:
                row["runtime_error"] = runtime["error"]
        report["cases"].append(row)
        save_json(report_path, report)
    report["status"] = "complete" if all(row["exit_code"] == 0 for row in report["cases"]) else "completed_with_failures"
    report["supervisor_wall_seconds"] = time.perf_counter() - started
    save_json(report_path, report)
    return report


def main() -> int:
    """解析具名路径和固定原生配方；输入错误不创建输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--python-interpreter", default=sys.executable)
    parser.add_argument("--n4-backend", choices=("native", "simpleitk", "torch"), default="simpleitk")
    parser.add_argument("--n4-execution", choices=("in-process", "isolated"), default="in-process")
    parser.add_argument("benchmark_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extras = args.benchmark_arguments
    if extras[:1] == ["--"]:
        extras = extras[1:]
    result = run_cases(cases=json.loads(args.cases_json.read_text()),
                       output_root=args.output_root, benchmark_script=args.benchmark_script,
                       weights_dir=args.weights_dir, assets_dir=args.assets_dir,
                       native_bin_dir=args.native_bin_dir, code_version=args.code_version,
                       python_interpreter=args.python_interpreter, n4_backend=args.n4_backend,
                       n4_execution=args.n4_execution, benchmark_arguments=tuple(extras))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
