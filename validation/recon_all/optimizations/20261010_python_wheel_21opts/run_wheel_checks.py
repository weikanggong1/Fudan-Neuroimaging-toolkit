"""当前21项Python配方的wheel/独立target检查；不运行MRI或改共享环境。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import io
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tarfile


MODULES = (
    "fnit.recon_all.python_profile", "fnit.recon_all.n4_simpleitk",
    "fnit.recon_all.input_n4_chain", "fnit.recon_all.native_free",
    "fnit.recon_all.place_white_preaparc_python", "fnit.recon_all.place_final_white_python",
    "fnit.recon_all.place_pial_python", "fnit.recon_all.topology_volume_python",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_installation(*, output: Path) -> dict:
    """检查新进程的独立target，记录模块SHA/依赖/CLI；不初始化CUDA。"""
    target = output / "installed"
    binding = json.loads((output / "SOURCE_BINDING.private.json").read_text())
    dist = importlib.metadata.distribution("fudan-neuroimaging-toolkit")
    assert Path(dist.locate_file("")).resolve() == target.resolve()
    import torch
    cuda_initialized_before = torch.cuda.is_initialized()
    assert cuda_initialized_before is False
    from packaging.requirements import Requirement
    requirements = [Requirement(value) for value in dist.requires or ()]
    simpleitk = [req for req in requirements if req.name.lower() == "simpleitk"]
    assert len(simpleitk) == 1 and str(simpleitk[0].specifier) == "==2.5.6"
    entry = next(item for item in dist.entry_points
                 if item.group == "console_scripts" and item.name == "fnit-recon-all")
    assert entry.value == "fnit.recon_all.native_free:main"
    modules = {}
    for name in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        relative = path.relative_to(target.resolve())
        digest = sha256(path)
        assert digest == binding["source_files"]["src/" + str(relative)]["sha256"]
        modules[name] = {"file": str(path), "sha256": digest}
    import SimpleITK
    import torch
    assert SimpleITK.Version_VersionString() == "2.5.6"
    profile = importlib.import_module("fnit.recon_all.python_profile").python_profile_options(
        device="cuda:1", threads=4)
    expected_profile = {"native_optimizations": "torch", "n4_backend": "simpleitk",
        "n4_execution": "in-process", "wm_backend": "torch-optimized",
        "gca_execution": "isolated", "gca_inverse_backend": "torch", "gca_candidate_chunk": 1024,
        "normalization_controls_backend": "torch", "normalization_initial_bias_backend": "torch",
        "sphere_finish_backend": "torch", "mni_execution": "parallel-late", "wm_execution": "isolated",
        "wm_edit_backend": "python", "defects_backend": "torch", "inflate_backend": "torch",
        "surface_placement_backend": "python", "topology_backend": "python",
        "annotation_gibbs_backend": "numba", "remesh_scalar_storage": "python",
        "fill_backend": "numba", "hemisphere_workers": 2}
    assert profile == expected_profile and len(profile) == 21
    assert torch.cuda.is_initialized() is False
    assert entry.load() is importlib.import_module("fnit.recon_all.native_free").main
    completed = subprocess.run([sys.executable, str(target / "bin/fnit-recon-all"), "--help"],
                               capture_output=True, text=True)
    (output / "cli-help.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0
    assert all(flag in completed.stdout for flag in
               ("--topology-backend", "--surface-placement-backend", "simpleitk",
                "--gca-execution", "--gca-inverse-backend", "--gca-candidate-chunk",
                "--normalization-controls-backend", "--normalization-initial-bias-backend",
                "--sphere-finish-backend", "--mni-execution"))
    return {"status": "complete", "scope": "isolated target metadata and imports only; not MRI or clean Conda",
            "source_binding_sha256": sha256(output / "SOURCE_BINDING.private.json"),
            "requires_dist": dist.requires, "entrypoint": entry.value, "imports": modules,
            "simpleitk_version": SimpleITK.Version_VersionString(),
            "torch_version": torch.__version__, "cuda_initialized": torch.cuda.is_initialized(),
            "snapshot_profile": profile, "profile_option_count": len(profile),
            "cuda_initialized_before_imports": cuda_initialized_before,
            "no_CUDA_init_after_imports_profile_help": not torch.cuda.is_initialized(),
            "check_script_sha256": sha256(Path(__file__)),
            "cli_help_sha256": sha256(output / "cli-help.log"),
            "cli_help_exit_code": completed.returncode}


def run_wheel_checks(*, repo: Path, output: Path, dependency_path: Path | None = None,
                     git_only: bool = False) -> dict:
    """repo为实际冻结工作树；output须不存在；dependency_path为已有额外Python库。

    当前解释器须已有pip/setuptools/wheel和项目声明依赖。复制源码至独立目录，
    记录Git版本及全部源文件SHA，执行无依赖构建/安装和新进程导入。不会下载
    影像或依赖，不改共享环境；失败保留日志/退出码并抛异常。输出全部为私有
    JSON/日志及wheel，含本机路径，发布前应脱敏。MSM自有扩展可由setup编译。
    """
    repo, output = repo.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    first_head = git("rev-parse", "HEAD")
    snapshot_start = time.perf_counter()
    archive = subprocess.check_output(["git", "archive", "--format=tar", first_head,
        "src", "pyproject.toml", "setup.py", "README.md", "THIRD_PARTY_NOTICES.md", "licenses"], cwd=repo)
    if git_only:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("非法归档路径")
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    else:
        shutil.copytree(repo / "src", snapshot / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbc", "*.nbi", "*.so"))
        for name in ("pyproject.toml", "setup.py", "README.md", "THIRD_PARTY_NOTICES.md", "licenses"):
            source = repo / name
            if source.is_dir():
                shutil.copytree(source, snapshot / name)
            else:
                shutil.copy2(source, snapshot / name)
    files = {str(path.relative_to(snapshot)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        original = {member.name: hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                    for member in tar.getmembers() if member.isfile()}
    differences = {name: {"snapshot_sha256": record["sha256"], "git_blob_sha256": original.get(name)}
                   for name, record in files.items() if original.get(name) != record["sha256"]}
    binding = {"git_head_at_snapshot_start": first_head, "git_head_at_snapshot_end": git("rev-parse", "HEAD"),
               "repo": str(repo), "source_file_count": len(files), "source_files": files,
               "snapshot_wall_seconds": time.perf_counter() - snapshot_start,
               "scope": "actual worktree snapshot, each source SHA bound; uncommitted overlays explicitly checked",
               "git_archive_sha256": hashlib.sha256(archive).hexdigest(),
               "git_only_snapshot": git_only,
               "snapshot_diff_from_git_head": differences,
               "all_snapshot_files_match_actual_git_blobs": not differences,
               "build_recon_FS_native_stage": False}
    (output / "SOURCE_BINDING.private.json").write_text(json.dumps(binding, indent=2))
    def execute(command, name):
        tick = time.perf_counter()
        with (output / (name + ".log")).open("w") as stream:
            process = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
        receipt = {"command": command, "exit_code": process.returncode,
                   "wall_seconds": time.perf_counter() - tick}
        (output / (name.upper() + ".private.json")).write_text(json.dumps(receipt, indent=2))
        process.check_returncode()
    execute([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir",
             "--wheel-dir", str(output / "wheel"), str(snapshot)], "build")
    wheels = list((output / "wheel").glob("*.whl"))
    assert len(wheels) == 1
    execute([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--no-cache-dir",
             "--target", str(output / "installed"), str(wheels[0])], "install")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "installed") + (os.pathsep + str(dependency_path) if dependency_path else "")
    environment["NUMBA_CACHE_DIR"] = str(output / "numba-cache")
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "4"
    checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--internal-check", str(output)],
                             cwd=output, env=environment, capture_output=True, text=True)
    (output / "check.log").write_text(checked.stdout + checked.stderr)
    checked.check_returncode()
    report = json.loads((output / "CHECK.private.json").read_text())
    report["wheel"] = {"name": wheels[0].name, "size_bytes": wheels[0].stat().st_size, "sha256": sha256(wheels[0])}
    (output / "CHECK.private.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dependency-path", type=Path)
    parser.add_argument("--git-only", action="store_true", help="只构建捕获HEAD的Git归档，不复制未提交工作树")
    parser.add_argument("--internal-check", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.internal_check:
        receipt = check_installation(output=options.internal_check)
        (options.internal_check / "CHECK.private.json").write_text(json.dumps(receipt, indent=2))
    else:
        if options.repo is None or options.output is None:
            parser.error("--repo and --output are required")
        receipt = run_wheel_checks(repo=options.repo, output=options.output,
                                  dependency_path=options.dependency_path, git_only=options.git_only)
    print(json.dumps(receipt, indent=2))
