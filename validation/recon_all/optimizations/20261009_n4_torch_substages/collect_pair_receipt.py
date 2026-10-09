"""绑定两例 N4 配对的实际程序、编译输入、加载库、硬件及独立测试层。

仅读取明确传入的私有任务路径；不读取权重、许可证或登录配置。
硬件记录不构成干净环境隔离证明；已有报告的源码哈希不被改写。
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="冻结 v1/control/ 与诊断构建目录")
    parser.add_argument("--environment", type=Path, required=True, help="实际 Conda prefix")
    parser.add_argument("--native", type=Path, required=True, help="当前 Conda 源码构建 N4")
    parser.add_argument("--diagnostic-source", type=Path, required=True, help="独立 final-capture 生成源码")
    parser.add_argument("--diagnostic-build", type=Path, required=True, help="同源码编译目录")
    parser.add_argument("--data", type=Path, required=True, help="两例公开数据 manifest 目录")
    parser.add_argument("--run", type=Path, required=True, help="完整配对输出目录")
    parser.add_argument("--file-run", type=Path, help="可选已完成文件 API/CLI 输出目录")
    parser.add_argument("--code-base", required=True, help="冻结完整 FNIT 提交；算子另由逐文件 SHA 绑定")
    parser.add_argument("--output", type=Path, required=True, help="新 JSON；拒绝覆盖")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pair = json.loads((args.run / "summary.json").read_text())
    paths = [Path(__file__), args.native, args.diagnostic_build / "fnit_n4_profile",
        args.diagnostic_build / "CMakeCache.txt", args.diagnostic_build / "build.ninja",
        args.diagnostic_source / "source_manifest.json", args.diagnostic_source / "n4_diagnostic.cpp",
        args.diagnostic_source / "include/fnit_n4_profile.h", args.diagnostic_source / "CMakeLists.txt",
        args.data / "manifest.json", args.run / "summary.json",
        args.environment / "bin/x86_64-conda-linux-gnu-g++", args.environment / "bin/cmake",
        args.environment / "bin/python", args.workspace / "control/unit_v1.xml",
        args.workspace / "control/unit_v1.log", args.workspace / "control/test_tools_install.log",
        args.workspace / "division_variant/division_variant_manifest.json"]
    paths.extend(sorted((args.workspace / "control").glob("*.py")))
    paths.extend(args.environment / "include/ITK-5.4" / name for name in (
        "itkVersion.h", "itkVersionConfig.h", "itkN4BiasFieldCorrectionImageFilter.h",
        "itkN4BiasFieldCorrectionImageFilter.hxx", "itkBSplineScatteredDataPointSetToImageFilter.hxx",
        "itkBSplineControlPointImageFilter.hxx", "itkBSplineKernelFunction.h"))
    libraries = {}
    for program in (args.native, args.diagnostic_build / "fnit_n4_profile", args.environment / "bin/python"):
        text = subprocess.run(["ldd", str(program)], capture_output=True, text=True, check=True).stdout
        resolved = set()
        for line in text.splitlines():
            for field in line.split():
                if field.startswith("/") and Path(field).is_file():
                    resolved.add(Path(field))
        libraries[str(program)] = {"ldd": text, "resolved_library_sha256": {str(path): sha(path) for path in sorted(resolved)}}
    test_tools = {}
    for name in ("pytest", "pluggy", "packaging", "iniconfig"):
        try:
            distribution = importlib.metadata.distribution(name)
            test_tools[name] = {"version": distribution.version, "metadata_sha256": hashlib.sha256(
                (distribution.read_text("METADATA") or "").encode()).hexdigest()}
        except importlib.metadata.PackageNotFoundError:
            test_tools[name] = {"unavailable": True}
    cpu = json.loads(subprocess.run(["lscpu", "-J"], capture_output=True, text=True, check=True).stdout)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    receipt = {"collected_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "cpu": cpu, "gpu_inventory": gpu, "code_base": args.code_base,
        "actual_algorithm_identity": "per-candidate report loaded-module SHA256, not inferred from collection commit",
        "files": {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths if path.is_file()},
        "unavailable_explicit_files": [str(path) for path in paths if not path.is_file()],
        "libraries": libraries, "test_tools": test_tools, "test_tool_scope": "independent overlay; frozen runtime not modified",
        "collected_affinity": sorted(os.sched_getaffinity(0)), "benchmark_affinity": pair["cpu_affinity"],
        "benchmark_threads": pair["threads"], "native_fitting_threads": pair["native_fitting_threads"],
        "native_reconstruction_threads": pair["native_reconstruction_threads"], "pair_status": pair["status"],
        "native_repeats": pair["cases"], "production_default_changed": False,
        "whole_n4_equivalence": "not established", "whole_recon_all_acceleration": "not measured",
        "isolation_validation": "not tested; imports/ldd/SHA checks alone do not prove clean deployment",
        "official_scope": "fresh FNIT ITK5.4.7 native; old FreeSurfer ITK4.8 record remains separate",
        "weights_assets": "N4 has no external model weights or atlas"}
    if args.file_run:
        path = args.file_run / "summary.json"
        receipt["file_interface_receipt"] = {"path": str(path), "sha256": sha(path),
            "status": json.loads(path.read_text()).get("status")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
