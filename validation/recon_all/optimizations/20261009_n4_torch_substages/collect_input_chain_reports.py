"""收集本轮已完成的公开输入链/N4 worker收据，不打包影像或许可证。

--fnit-root为私有统一目录，--output必须不存在。固定文件白名单包括
JSON/log/JUnit XML和误差PNG；输入影像、MGH/LTA、权重均不复制。
请在原benchmark环境shell内运行；runtime_receipt仅是运行后现场核验，
不能作为物理隔离验收。公开副本另用publish_reports.py去目录/主机信息。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(arguments: list[str]) -> dict:
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    return {"arguments": arguments, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def collect(*, fnit_root: Path, output: Path) -> dict:
    """只复制显式文件白名单；缺失、符号链接或未完成报告直接报错。"""
    root = fnit_root.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    raw = root / "runs/input_n4_raw_t1_a100_two_cases_20261009_v1"
    worker = root / "runs/n4_cached_exec_same_orig_a100_two_cases_20261009_v2"
    figures = root / "runs/input_n4_raw_t1_a100_two_cases_20261009_v1_figures"
    workspace = root / "workspaces/recon_input_n4_torch_20261009"
    if json.loads((raw / "summary.json").read_text())["status"] != "complete_raw_input_to_nu_pair":
        raise ValueError("raw input pair is incomplete")
    if json.loads((worker / "summary.json").read_text())["status"] != "complete_cached_exec_two_modes":
        raise ValueError("cached exec pair is incomplete")
    files: dict[str, Path] = {
        "raw_pair/summary.json": raw / "summary.json",
        "cached_worker/summary.json": worker / "summary.json",
        "figures/figure_receipt.json": figures / "figure_receipt.json",
        "figures/raw_chain_nu_error.png": figures / "raw_chain_nu_error.png",
        "source/raw_v1_patch_manifest.json": workspace / "v1/SOURCE_PATCH_MANIFEST.json",
        "source/worker_v2_patch_manifest.json": workspace / "v2/SOURCE_PATCH_MANIFEST.json",
        "source/raw_input_config.json": workspace / "fnit-input-n4-config-v1.json",
    }
    for name in ("unit_v1.log", "unit_v1.xml", "unit_v2.log", "unit_v2.xml",
                 "full_chain_v1.log", "worker_pair_v2.log", "plots_v2.log"):
        files["logs/" + name] = workspace / name
    for case in ("ds000114_sub-06", "ds000114_sub-07"):
        for backend in ("native", "torch"):
            relative = f"{case}/{backend}/subject/scripts/talairach-child-gpu.json"
            files["raw_pair/" + relative] = raw / relative
        for mode in ("cold_cli", "initialized_api"):
            relative = f"{case}/{mode}/child.json"
            files["cached_worker/" + relative] = worker / relative
        relative = f"{case}/cold_cli/cli.log"
        files["cached_worker/" + relative] = worker / relative
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise ValueError("whitelisted report must be an existing regular file")
    output.mkdir(parents=True)
    receipt = {"scope": "report-only exact copies, no MRI/weights/licences",
               "collector_sha256": sha(Path(__file__)), "files": {}}
    for relative, source in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        receipt["files"][relative] = {"source": str(source), "sha256": sha(source),
                                     "bytes": source.stat().st_size}
    native = root / "assets/recon_resources_20261009_v1/native/bin/fnit_n4_itk"
    libraries = command(["ldd", str(native)])
    paths = re.findall(r"(?:=>\s+)?(/\S+)\s+\(", libraries["stdout"])
    library_sha = {path: sha(Path(path)) for path in sorted(set(paths)) if Path(path).is_file()}
    import numpy
    import nibabel
    import torch
    cpuinfo = Path("/proc/cpuinfo").read_text()
    models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)))
    runtime = {
        "scope": "post-run verification in the same declared runtime shell, not a new benchmark",
        "physical_clean_environment_isolation": "not verified",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(), "python_executable": sys.executable,
        "python_version": platform.python_version(), "cpu_models": models,
        "logical_cpu_count": os.cpu_count(), "collector_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numpy_version": numpy.__version__, "nibabel_version": nibabel.__version__,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "collector_cuda_initialized": torch.cuda.is_initialized(),
        "native_binary": str(native), "native_sha256": sha(native),
        "dynamic_libraries": libraries, "library_sha256": library_sha,
        "gpu_query": command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
        "selected_runtime_environment": {name: os.environ.get(name) for name in
            ("FNIT_RECON_ENV", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTORCH_NO_CUDA_MEMORY_CACHING", "LD_LIBRARY_PATH")},
        "notes": ["collector does not run CUDA work or load models",
                  "actual benchmark precision, threads and source hashes remain in original reports",
                  "library hashes observed after the completed run; receipt does not prove physical isolation"]}
    (output / "runtime_receipt.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (output / "report_collection_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"files": len(files), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnit-root", type=Path, required=True, help="已授权的FNIT统一私有目录")
    parser.add_argument("--output", type=Path, required=True, help="新报告目录，不包含影像")
    arguments = parser.parse_args()
    print(json.dumps(collect(fnit_root=arguments.fnit_root, output=arguments.output)))


if __name__ == "__main__":
    main()
