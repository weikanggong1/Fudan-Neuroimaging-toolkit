"""为同一冻结源码准备「仅拓扑 GA 使用 Conda C++」的整例命令。

输入：已完成的原始 T1 整例 benchmark.json、原租约 job JSON、该整例冻结源码、同一 Conda
环境及其独立编译的 mris_fix_topology_fnit、私有 FreeSurfer 许可证路径、
全新预检目录和全新整例目录。T1、权重、资产沿用 benchmark.json 的路径；
体积为原始 T1 网格，后续表面仍是 surface RAS 毫米坐标。

输出：权限 700 的预检目录，包括 provenance.json、只含拓扑程序软链的
native-bin、launch.sh。脚本只做哈希及配置校验，不运行整例，也不读取
官方结果；launch.sh 须经 GPU 租约调度后执行。预检失败在创建目录前抛错。
完整计时、显存、138 输出与网格质量由原始 T1 benchmark 程序写出。

Python 调用示例（全部具名参数；此处不会启动 GPU 作业）：

    prepare_hybrid(
        baseline_benchmark=Path("/private/old/benchmark.json"),  # 已完成的 Python GA 整例
        baseline_job=Path("/private/old/job.private.json"),      # 原租约环境，包括依赖和 JIT 缓存
        source_root=Path("/private/source_cbaffa990c7f"),       # 同一冻结源码
        conda_prefix=Path("/private/fnit_main_env"),            # 独立构建程序的 Conda 环境
        fs_license=Path("/private/license.txt"),               # 私有许可证文件，不复制内容
        preflight_root=Path("/private/hybrid-preflight"),       # 必须尚不存在
        run_output_root=Path("/private/hybrid-whole"),          # 必须尚不存在
    )

CLI 用法见 --help。原软件对应 mris_fix_topology -ga -seed 1234；FNIT 实际
执行 mris_fix_topology_fnit，使用固定 FreeSurfer 8.2 源码和 Conda 构建。
其余整例步骤沿用冻结 FNIT 配方，不能称纯 Python 整例。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess


UPSTREAM_COMMIT = "d932c45b7941662ea380a05efef580568b98d41a"
BASELINE_FLAGS = {
    "--backend": "python",
    "--native-optimizations": "torch",
    "--n4-backend": "simpleitk",
    "--wm-backend": "torch-optimized",
    "--wm-edit-backend": "python",
    "--defects-backend": "torch",
    "--inflate-backend": "torch",
    "--surface-placement-backend": "python",
    "--topology-backend": "python-ga",
    "--topology-distance-backend": "gpu_defect_signed_distance",
    "--topology-state-backend": "packed",
    "--hemisphere-execution": "persistent",
}
HYBRID_CHANGES = {
    "--backend": "native",
    "--topology-backend": "native",
    "--topology-distance-backend": "serial",
    "--topology-state-backend": "legacy",
    "--hemisphere-execution": "groups",
}
BOOLEAN_FLAGS = {"--profile-stages", "--gca-reuse-atlas-context"}
VALUE_FLAGS = {
    "--weights-dir", "--assets-dir", "--backend", "--device", "--threads",
    "--hemisphere-workers", "--hemisphere-execution", "--native-optimizations",
    "--defects-backend", "--sphere-normals-backend", "--wm-edit-backend",
    "--wm-backend", "--gca-inverse-backend", "--gca-candidate-chunk",
    "--gca-execution", "--fill-backend", "--n4-backend", "--n4-execution",
    "--input-execution", "--normalization-controls-backend",
    "--normalization-initial-bias-backend", "--inflate-backend",
    "--sphere-finish-backend", "--surface-placement-backend",
    "--topology-backend", "--topology-distance-backend", "--topology-state-backend",
    "--annotation-gibbs-backend", "--remesh-scalar-storage", "--mni-execution",
    "--wm-execution", "--native-bin-dir",
}
RECORDED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "PYTORCH_NO_CUDA_MEMORY_CACHING",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _child_command(report: dict) -> tuple[Path, Path, dict[str, str | bool], list[str]]:
    command = report["cli_command"]
    if command[1:6] != ["-X", "faulthandler", "-u", "-m", "fnit.recon_all.native_free"]:
        raise ValueError("baseline is not the expected FNIT raw-T1 CLI child")
    options: dict[str, str | bool] = {}
    order: list[str] = []
    offset = 8
    while offset < len(command):
        flag = command[offset]
        if flag in options or flag not in VALUE_FLAGS | BOOLEAN_FLAGS:
            raise ValueError(f"duplicate or unknown baseline CLI option: {flag}")
        order.append(flag)
        if flag in BOOLEAN_FLAGS:
            options[flag] = True
            offset += 1
        else:
            if offset + 1 >= len(command) or command[offset + 1].startswith("--"):
                raise ValueError(f"missing baseline CLI value: {flag}")
            options[flag] = command[offset + 1]
            offset += 2
    return Path(command[6]), Path(command[7]), options, order


def _verify_frozen_source(source: Path, baseline: dict) -> dict:
    manifest_path = source / "SOURCE_MANIFEST.private.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != baseline["code_version"]:
        raise ValueError("frozen source commit differs from the completed baseline")
    fnit_root = source / "src/fnit"
    expected = baseline["source_sha256"]
    observed = {str(path.relative_to(fnit_root)) for path in fnit_root.rglob("*.py")}
    if observed != set(expected):
        raise ValueError("frozen FNIT Python file set differs from the completed baseline")
    for relative, checksum in expected.items():
        if _sha256(fnit_root / relative) != checksum:
            raise ValueError(f"frozen source SHA differs: {relative}")
    benchmark = source / "tools/benchmark_recon_torch_end_to_end.py"
    if _sha256(benchmark) != baseline["script_sha256"]:
        raise ValueError("frozen benchmark script differs from the completed baseline")
    return {"commit": manifest["commit"], "manifest_sha256": _sha256(manifest_path),
            "python_file_count": len(expected), "benchmark_sha256": baseline["script_sha256"]}


def _verify_resources(t1: Path, weights: Path, assets: Path, baseline: dict) -> dict:
    if _sha256(t1) != baseline["input_sha256"]:
        raise ValueError("raw T1 differs from the completed baseline")
    counts = {}
    for kind, root in (("weights", weights), ("assets", assets)):
        expected = baseline["resource_sha256"][kind]
        observed = {str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and "license" not in path.name.lower()}
        if observed != set(expected):
            raise ValueError(f"{kind} file set differs from the completed baseline")
        for relative, checksum in expected.items():
            if _sha256(root / relative) != checksum:
                raise ValueError(f"{kind} SHA differs: {relative}")
        counts[kind] = len(expected)
    return {"input_sha256": baseline["input_sha256"], "resource_counts": counts,
            "resource_hashes_match_completed_baseline": True}


def _manifest_sha_for_name(path: Path, name: str) -> str:
    rows = [row.split() for row in path.read_text().splitlines() if row.strip()]
    matched = [parts[0] for parts in rows if len(parts) == 2
               and Path(parts[1]).name == name]
    if len(matched) != 1:
        raise ValueError(f"expected exactly one {name} in {path}")
    return matched[0]


def _verify_conda_build(prefix: Path) -> dict:
    binary = prefix / "bin/mris_fix_topology_fnit"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(binary)
    binary_sha = _sha256(binary)
    build = prefix / "share/fnit/recon_all_native_full"
    provenance = dict(line.split("=", 1) for line in
                      (build / "build-provenance.txt").read_text().splitlines() if "=" in line)
    if provenance.get("SOURCE_COMMIT") != UPSTREAM_COMMIT:
        raise ValueError("topology binary does not have pinned FreeSurfer source provenance")
    if Path(provenance.get("CONDA_PREFIX", "")).resolve() != prefix:
        raise ValueError("topology build was recorded for a different Conda prefix")
    if _manifest_sha_for_name(build / "installed-bin.sha256", binary.name) != binary_sha:
        raise ValueError("topology binary differs from the Conda installed-program manifest")
    built = build / "bin/mris_fix_topology_fnit"
    built_sha = _sha256(built)
    if _manifest_sha_for_name(build / "bin.sha256", binary.name) != built_sha:
        raise ValueError("topology build artifact differs from the Conda build manifest")
    if b"FNIT_CENTERED_SUBSTITUTION_USED" not in binary.read_bytes():
        raise ValueError("topology binary lacks the required FNIT centered-sphere patch")
    # setup_recon_all_native_conda.sh applies an RPATH change when installing.
    # Therefore built_sha and binary_sha may differ; both must match their own manifest.
    return {"binary": str(binary), "binary_sha256": binary_sha,
            "build_binary_sha256": built_sha,
            "upstream_commit": provenance["SOURCE_COMMIT"],
            "source_validation": provenance.get("SOURCE_VALIDATION"),
            "build_provenance_sha256": _sha256(build / "build-provenance.txt"),
            "installed_manifest_sha256": _sha256(build / "installed-bin.sha256"),
            "build_manifest_sha256": _sha256(build / "bin.sha256"),
            "conda_prefix": str(prefix)}


def prepare_hybrid(*, baseline_benchmark: Path, baseline_job: Path, source_root: Path,
                   conda_prefix: Path, fs_license: Path, preflight_root: Path,
                   run_output_root: Path,
                   target_physical_gpu_index: int | None = None,
                   target_gpu_uuid: str | None = None) -> dict:
    """核对同输入/程序来源，生成不执行的混合整例命令；返回私有预检字典。

    七个必需参数均为路径。GPU索引和完整UUID必须同时指定或同时省略；
    指定时只用 nvidia-smi 核对设备身份，不建立 CUDA context。
    预检目录和整例目录必须尚不存在；只创建前者。
许可证不复制到预检目录，报告仅记录 SHA；旧租约的 expected_host
必须与实际计算主机一致。失败抛错，不启动任务。
    """
    baseline_path = Path(baseline_benchmark).resolve()
    baseline_job_path = Path(baseline_job).resolve()
    source = Path(source_root).resolve()
    prefix = Path(conda_prefix).resolve()
    license_path = Path(fs_license).resolve()
    preflight = Path(preflight_root).resolve()
    output = Path(run_output_root).resolve()
    if preflight.exists() or output.exists():
        raise FileExistsError("preflight_root and run_output_root must not exist")
    if not license_path.is_file() or license_path.stat().st_size == 0:
        raise FileNotFoundError("FS_LICENSE must be a readable nonempty private file")
    baseline = json.loads(baseline_path.read_text())
    old_job = json.loads(baseline_job_path.read_text())
    if (baseline.get("execution") != "complete" or baseline.get("exit_code") != 0
            or baseline.get("output_completeness", {}).get("status") != "passed"
            or baseline.get("grid_quality", {}).get("status") != "passed"):
        raise ValueError("baseline was not a complete 138-output, mesh-passed whole T1 run")
    t1, old_subject, flags, order = _child_command(baseline)
    if old_subject != baseline_path.parent / "subject":
        raise ValueError("baseline CLI subject does not match its benchmark directory")
    for flag, expected in BASELINE_FLAGS.items():
        if flags.get(flag) != expected:
            raise ValueError(f"baseline option {flag} differs from the frozen Python GA recipe")
    if flags.get("--profile-stages") is not True or flags.get("--hemisphere-workers") != "2":
        raise ValueError("baseline must have two profiled hemisphere workers")
    if "--native-bin-dir" in flags:
        raise ValueError("Python baseline unexpectedly declared a native program directory")
    weights, assets = Path(str(flags["--weights-dir"])), Path(str(flags["--assets-dir"]))
    source_provenance = _verify_frozen_source(source, baseline)
    resource_provenance = _verify_resources(t1, weights, assets, baseline)
    expected_host = old_job.get("expected_host")
    actual_host = socket.gethostname().split(".", 1)[0]
    if (not isinstance(expected_host, str) or not expected_host
            or expected_host.split(".", 1)[0] != actual_host
            or old_job.get("expected_gpu_uuid") != baseline["environment"]["CUDA_VISIBLE_DEVICES"]
            or str(source / "tools/benchmark_recon_torch_end_to_end.py") not in old_job.get("argv", [])
            or source_provenance["commit"] not in old_job.get("argv", [])):
        raise ValueError("baseline job does not bind the same host, GPU, source and benchmark")
    runtime_environment = {key: value for key, value in old_job["env"].items()
                           if key not in {"FNIT_RECON_AUDIT_SOURCE_DIR",
                                          "FNIT_RECON_FORBID_ORIGINAL_STAGES"}}
    if not any(Path(part).resolve() == source / "src" for part in
               runtime_environment["PYTHONPATH"].split(os.pathsep)):
        raise ValueError("baseline runtime PYTHONPATH does not bind the frozen source")
    for key in ("TRITON_PTXAS_PATH", "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR"):
        if key not in runtime_environment:
            raise ValueError(f"baseline runtime job lacks {key}")
    if Path(baseline["cli_command"][0]).resolve() != (prefix / "bin/python").resolve():
        raise ValueError("hybrid interpreter differs from the completed baseline Conda environment")
    binary_provenance = _verify_conda_build(prefix)
    options = {**flags, **HYBRID_CHANGES}
    command = [str(prefix / "bin/python"), str(source / "tools/benchmark_recon_torch_end_to_end.py"),
               "--t1", str(t1), "--output-root", str(output),
               "--weights-dir", str(weights), "--assets-dir", str(assets),
               "--native-bin-dir", str(preflight / "native-bin")]
    for flag in order:
        if flag in {"--weights-dir", "--assets-dir", "--profile-stages"}:
            continue
        command.append(flag)
        if flag not in BOOLEAN_FLAGS:
            command.append(str(options[flag]))
    command += ["--code-version", source_provenance["commit"]]
    environment = dict(baseline["environment"])
    if set(environment) != set(RECORDED_ENVIRONMENT):
        raise ValueError("baseline resource/thread/CUDA environment record is incomplete")
    if not environment["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("baseline target GPU was not explicit")
    baseline_gpu_uuid = environment["CUDA_VISIBLE_DEVICES"]
    if (target_physical_gpu_index is None) != (target_gpu_uuid is None):
        raise ValueError("target physical GPU index and UUID must be supplied together")
    if target_gpu_uuid is not None:
        if (type(target_physical_gpu_index) is not int or target_physical_gpu_index < 0
                or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                                target_gpu_uuid) is None):
            raise ValueError("target GPU identity must be a nonnegative index and complete UUID")
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15).stdout
        identities = dict(row.strip().split(", ", 1) for row in inventory.splitlines())
        if identities.get(str(target_physical_gpu_index)) != target_gpu_uuid:
            raise ValueError("target physical GPU index/UUID does not match current host")
        environment["CUDA_VISIBLE_DEVICES"] = target_gpu_uuid
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.append("unset FNIT_RECON_AUDIT_SOURCE_DIR FNIT_RECON_FORBID_ORIGINAL_STAGES")
    for key, value in sorted(runtime_environment.items()):
        lines.append(f"export {key}={shlex.quote(value)}")
    for key in RECORDED_ENVIRONMENT:
        value = environment[key]
        lines.append(f"export {key}={shlex.quote(value)}" if value is not None
                     else f"unset {key}")
    lines += [f"export FS_LICENSE={shlex.quote(str(license_path))}",
              f"cd {shlex.quote(str(source))}",
              "exec " + shlex.join(command)]
    result = {
        "status": "prepared_not_run", "scope": "raw-T1 whole; only original topology program permitted",
        "baseline_benchmark": str(baseline_path), "baseline_benchmark_sha256": _sha256(baseline_path),
        "baseline_job": str(baseline_job_path), "baseline_job_sha256": _sha256(baseline_job_path),
        "runtime_environment_keys": sorted(runtime_environment),
        "jit_cache_policy": "reuse the completed baseline job cache locations; compiler/import state may be warm",
        "source": source_provenance, "resources": resource_provenance,
        "native_program": binary_provenance, "fs_license_sha256": _sha256(license_path),
        "baseline_backend": "python", "hybrid_backend": "native",
        "baseline_topology": "python-ga", "hybrid_topology": "native",
        "forced_configuration_differences": {flag: {"baseline": flags[flag], "hybrid": value}
                                             for flag, value in HYBRID_CHANGES.items()},
        "gpu_selection": {"baseline_uuid": baseline_gpu_uuid,
                          "target_physical_index": target_physical_gpu_index,
                          "target_uuid": environment["CUDA_VISIBLE_DEVICES"],
                          "changed_from_baseline": environment["CUDA_VISIBLE_DEVICES"] != baseline_gpu_uuid,
                          "requires_idle_gpu_before_launch": True},
        "hemisphere_lifecycle_note": "persistent requires backend=python; mixed backend uses groups, so whole-run timings include this scheduler difference",
        "audit_note": "Python-only process-launch audit is absent in mixed benchmark; inspect actual process tree after execution",
        "native_bin_contents": ["mris_fix_topology_fnit symlink to verified Conda install"],
        "benchmark_command": command,
        "launch_script": str(preflight / "launch.sh"),
        "run_output_root": str(output),
        "execution_status": "not_run", "138_outputs": "not_assessed",
        "grid_quality": "not_assessed", "strict_reproduction": "not_assessed",
        "overall_metric_equivalence": "not_assessed", "wall_seconds": None,
    }
    # 所有校验通过后才创建预检目录；整例目录保持不存在。
    preflight.mkdir(mode=0o700)
    native_bin = preflight / "native-bin"
    native_bin.mkdir(mode=0o700)
    (native_bin / "mris_fix_topology_fnit").symlink_to(prefix / "bin/mris_fix_topology_fnit")
    launch = preflight / "launch.sh"
    launch.write_text("\n".join(lines) + "\n")
    launch.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    (preflight / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-benchmark", type=Path, required=True)
    parser.add_argument("--baseline-job", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--conda-prefix", type=Path, required=True)
    parser.add_argument("--fs-license", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--run-output-root", type=Path, required=True)
    parser.add_argument("--target-physical-gpu-index", type=int)
    parser.add_argument("--target-gpu-uuid")
    arguments = parser.parse_args()
    result = prepare_hybrid(
        baseline_benchmark=arguments.baseline_benchmark,
        baseline_job=arguments.baseline_job,
        source_root=arguments.source_root, conda_prefix=arguments.conda_prefix,
        fs_license=arguments.fs_license, preflight_root=arguments.preflight_root,
        run_output_root=arguments.run_output_root,
        target_physical_gpu_index=arguments.target_physical_gpu_index,
        target_gpu_uuid=arguments.target_gpu_uuid)
    print(json.dumps({key: result[key] for key in
                      ("status", "native_program", "forced_configuration_differences",
                       "launch_script", "run_output_root")}, indent=2))


if __name__ == "__main__":
    main()
