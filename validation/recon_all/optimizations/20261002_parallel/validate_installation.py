"""在既有主页 Conda 环境构建 wheel、安装到新目录并检查 recon-all CLI/API。

--config JSON 包含 code_root、code_commit、source_archive、archive_sha256 和 output；
可选 modules 指定需要导入的 FNIT 模块。Python 必须来自待测 Conda 环境。
输出 report.json、wheel、installed/ 和逐命令日志，包含返回码、耗时及 SHA-256。
使用该环境的 C/C++ 编译器，不下载依赖、不分配 GPU 张量、不修改已有安装。
失败保留报告并非零退出；output 必须不存在。属于 FNIT 安装验证，没有对应的
FreeSurfer CLI。不代表全新 Conda 创建、原生程序重建或物理隔离整例验收。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    source = Path(config["code_root"]).resolve()
    report = {
        "code_commit": config["code_commit"], "status": "failed",
        "host": platform.node(), "python": sys.version,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "config_sha256": sha256(args.config),
        "steps": {}, "isolation_status": "not_verified",
        "scope": "existing homepage Conda; wheel build and target install only",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    started = time.perf_counter()
    try:
        archive = Path(config["source_archive"])
        observed_hash = sha256(archive)
        report["source_archive_sha256"] = observed_hash
        if observed_hash != config["archive_sha256"]:
            raise ValueError("source archive SHA-256 mismatch")
        if not (source / "pyproject.toml").is_file():
            raise FileNotFoundError("candidate pyproject.toml unavailable")
        bin_dir = Path(sys.executable).resolve().parent
        compilers = {"CC": bin_dir / "x86_64-conda-linux-gnu-cc",
                     "CXX": bin_dir / "x86_64-conda-linux-gnu-c++"}
        for compiler in compilers.values():
            if not compiler.is_file() or not os.access(compiler, os.X_OK):
                raise FileNotFoundError("declared Conda compiler unavailable: " + str(compiler))
        report["compiler_paths"] = {key: str(value) for key, value in compilers.items()}
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CONDA_PREFIX=str(bin_dir.parent))
        env.update({key: str(value) for key, value in compilers.items()})
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL", "MAX_JOBS"):
            env[key] = "4"
        # 不沿用其他源码工作树的 PYTHONPATH 或可能带模块别名的 Numba 缓存。
        env.pop("PYTHONPATH", None)
        env["NUMBA_CACHE_DIR"] = str(output / "numba_cache")

        def run(name: str, command: list[str], *, pythonpath: str | None = None):
            command_env = dict(env)
            if pythonpath is not None:
                command_env["PYTHONPATH"] = pythonpath
            tick = time.perf_counter()
            log_path = output / (name + ".log")
            with log_path.open("w") as log:
                result = subprocess.run(command, cwd=source, env=command_env,
                                        stdout=log, stderr=subprocess.STDOUT)
            report["steps"][name] = {
                "command": command, "exit_code": result.returncode,
                "seconds": time.perf_counter() - tick, "log_sha256": sha256(log_path),
            }
            save()
            if result.returncode:
                raise RuntimeError(name + " returned " + str(result.returncode))

        run("wheel", [sys.executable, "-m", "pip", "wheel", "--no-deps",
                      "--no-build-isolation", str(source), "-w", str(output / "wheels")])
        wheels = list((output / "wheels").glob("fudan_neuroimaging_toolkit-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected exactly one FNIT wheel")
        report["wheel_sha256"] = sha256(wheels[0])
        run("install", [sys.executable, "-m", "pip", "install", "--no-deps",
                        "--target", str(output / "installed"), str(wheels[0])])
        target = str(output / "installed")
        run("recon_cli", [sys.executable, "-m", "fnit.recon_all.native_free", "--help"],
            pythonpath=target)
        modules = config.get("modules", ["fnit", "fnit.recon_all.native_free",
                              "fnit.recon_all.normalization._normalization_cuda"])
        if not all(isinstance(name, str) and name.startswith("fnit") for name in modules):
            raise ValueError("modules must contain FNIT import names")
        program = (
            "import importlib,json,pathlib; "
            "names=json.loads(" + repr(json.dumps(modules)) + "); "
            "result={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) "
            "for n in names}; print(json.dumps(result))"
        )
        run("api_import", [sys.executable, "-c", program], pythonpath=target)
        imported = json.loads((output / "api_import.log").read_text().splitlines()[-1])
        for module_path in imported.values():
            if not Path(module_path).is_relative_to(output.resolve() / "installed"):
                raise RuntimeError("API imported outside target installation")
        report["imported_modules"] = imported
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
