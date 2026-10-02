"""复核本次已声明输入、资源及程序，记录计算主机和环境版本。

--runtime-root 为已有 Conda/weights/assets 根目录；--expected-manifest 为
已声明的 JSON；--build-manifest 为14程序独立源码构建记录或带programs列表的清单；
--native-bin-dir 显式指定独立源码构建程序目录，默认环境bin；--code-commit
标记实际计算提交；--output 必须不存在。--reference-bin-dir 仅在独立
benchmark 中核验官方程序，不执行程序或读官方影像；未提供则不核验。
--subject 可记录本用户对应运行进程的 /proc 线程和白名单环境。
--hardware-only 不读取影像/资产，仅写硬件快照（不等同资源核验）。
返回 JSON 含大小、SHA-256、与声明比较、CPU/库版本及可见 GPU 查询；
缺失文件、哈希不符或已存在输出抛异常。无独立官方等价命令。
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import nibabel
import numba
import numpy
import scipy
import torch


def fingerprint(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return {"size_bytes": path.stat().st_size, "sha256": result.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--native-bin-dir", type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-bin-dir", type=Path)
    parser.add_argument("--subject", type=Path)
    parser.add_argument("--hardware-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    report = {"code_commit": args.code_commit, "host": platform.node(), "cpu": cpu,
              "utc": datetime.now(timezone.utc).isoformat(), "hardware_only": args.hardware_only,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__,
              "scipy": scipy.__version__, "nibabel": nibabel.__version__, "numba": numba.__version__,
              "numba_initial_capacity_for_probe": numba.config.NUMBA_NUM_THREADS,
              "script_sha256": fingerprint(Path(__file__))["sha256"],
              "mismatches": [], "processes": []}
    if args.subject:
        keys = ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTORCH_NO_CUDA_MEMORY_CACHING")
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                if str(args.subject) not in command or not command.startswith(str(args.runtime_root / "fnit_main_env/bin/python")):
                    continue
                status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
                environ = dict(row.split("=", 1) for row in (proc / "environ").read_text().split("\0") if "=" in row)
                report["processes"].append({"pid": int(proc.name), "command": command,
                    "os_threads": status["Threads"].strip(),
                    "thread_environment": {k: environ.get(k) for k in keys}})
            except (OSError, UnicodeError):
                continue
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        report["gpu_snapshot"] = gpu.stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        report["gpu_snapshot_status"] = type(error).__name__
    if not args.hardware_only:
        expected = json.loads(args.expected_manifest.read_text())
        build = json.loads(args.build_manifest.read_text())
        report["expected_manifest_sha256"] = fingerprint(args.expected_manifest)["sha256"]
        report["build_manifest_sha256"] = fingerprint(args.build_manifest)["sha256"]
        folders = {"weights": args.runtime_root / "weights", "assets": args.runtime_root / "assets",
                   "binaries": args.native_bin_dir or args.runtime_root / "fnit_main_env/bin"}
        if args.reference_bin_dir:
            folders["reference_binaries"] = args.reference_bin_dir
        for group, folder in folders.items():
            if group == "binaries":
                program_hashes = build.get("installed_program_sha256")
                if program_hashes is None:
                    program_hashes = {row["name"]: row["sha256"] for row in build["programs"]}
                entries = {name: {"sha256": value} for name, value in program_hashes.items()}
            else:
                entries = expected[group]
            report[group] = {}
            for name, previous in entries.items():
                actual = fingerprint(folder / name)
                report[group][name] = actual
                if any(actual[k] != value for k, value in previous.items()):
                    report["mismatches"].append(group + "/" + name)
        report["inputs"] = {}
        for name, previous in expected["inputs"].items():
            sid = name.split("_")[0].replace("sub", "sub-")
            path = args.runtime_root.parents[1] / "examples/data" / (sid + "_T1w.nii.gz")
            actual = fingerprint(path)
            report["inputs"][name] = actual
            if actual != previous:
                report["mismatches"].append(name)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if report["mismatches"]:
        raise ValueError(report["mismatches"])


if __name__ == "__main__":
    main()

