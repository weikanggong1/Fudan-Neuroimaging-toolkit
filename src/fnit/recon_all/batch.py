"""Multi-subject scheduler for the Conda recon-all pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import subprocess
import sys


def run_recon_all_python_batch(
    jobs: list[dict], weights_dir: str | Path, assets_dir: str | Path,
    *, devices: tuple[str, ...] = ("cuda:0",), threads: int = 4,
    native_bin_dir: str | Path | None = None,
    profile_stages: bool = False, cuda_allocator_cache: str = "auto",
    hemisphere_workers: int = 1,
    native_optimizations: str = "auto",
    wm_backend: str = "native",
    backend: str = "native",
) -> list[dict]:
    """每设备独立子进程执行单 T1，按 jobs 顺序返回完整报告列表。

    jobs 每项含 t1 原始影像和 subject_dir 空目录；weights_dir/assets_dir
    为已校验资源，devices 为互不重复的逻辑 CPU/CUDA 设备，threads 为
    每个被试线程预算（默认4），native_bin_dir 默认当前 Conda bin。
    profile_stages=False 不增加阶段同步；cuda_allocator_cache=auto
    延续低显存默认，也可在子进程初始化前选择 enabled/disabled。
    输出体积为 conform 网格，表面为 surface RAS/mm；输出/错误语义同
    单例入口。输入非法抛 ValueError/FileNotFoundError，任务失败汇总为
    RuntimeError。每设备仅执行一个被试；hemisphere_workers 默认1，设2时被试内部
    独立进程并行双侧，threads 在双侧之间分配（总预算不翻倍）。
    backend=python-gpu使用严格纯Python/CUDA profile；只要仍有未完成阶段就
    在创建被试目录前失败，不回退到原生程序。native_optimizations=auto按已验证能力选择完整GCA缓存及white快速程序；
    original用于原生阶段配对控制，原样传递给每个被试CLI。
    """
    from .hemisphere_parallel import validate_hemisphere_workers
    validate_hemisphere_workers(hemisphere_workers, threads)
    if not devices or len(set(devices)) != len(devices) or any(
        device != "cpu" and re.fullmatch(r"cuda:\d+", device) is None for device in devices
    ):
        raise ValueError("devices must be distinct CPU/CUDA device names")
    if threads < 1:
        raise ValueError("threads must be positive")
    if cuda_allocator_cache not in {"auto", "enabled", "disabled"}:
        raise ValueError("cuda_allocator_cache must be auto, enabled, or disabled")
    if native_optimizations not in {"auto", "original", "torch"}:
        raise ValueError("native_optimizations must be auto, original, or torch")
    if wm_backend not in {"native", "torch"}:
        raise ValueError("wm_backend must be native or torch")
    if wm_backend == "torch" and any(device == "cpu" for device in devices):
        raise ValueError("wm_backend='torch' requires CUDA devices")
    if backend not in {"native", "python-gpu"}:
        raise ValueError("backend must be native or python-gpu")
    weights, assets = Path(weights_dir).resolve(), Path(assets_dir).resolve()
    if not weights.is_dir() or not assets.is_dir():
        raise FileNotFoundError("weights_dir and assets_dir must exist")
    prepared = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"t1", "subject_dir"}:
            raise ValueError(f"job {index} needs t1 and subject_dir")
        t1, subject = Path(job["t1"]).resolve(), Path(job["subject_dir"]).resolve()
        if not t1.is_file():
            raise FileNotFoundError(t1)
        if subject.exists() and (not subject.is_dir() or any(subject.iterdir())):
            raise ValueError(f"job {index} subject_dir must be empty")
        if any(subject == previous or subject.is_relative_to(previous)
               or previous.is_relative_to(subject) for _, previous in prepared):
            raise ValueError(f"job {index} subject_dir overlaps another job")
        prepared.append((t1, subject))

    def run_device(device: str, assignments: list[tuple[int, Path, Path]]):
        results = []
        for index, t1, subject in assignments:
            command = [sys.executable, "-m", "fnit.recon_all.native_free",
                       str(t1), str(subject), "--weights-dir", str(weights),
                       "--assets-dir", str(assets), "--device", device,
                       "--threads", str(threads), "--cuda-allocator-cache", cuda_allocator_cache]
            command += ["--backend", backend]
            if wm_backend != "native":
                command += ["--wm-backend", wm_backend]
            if hemisphere_workers != 1:
                command += ["--hemisphere-workers", str(hemisphere_workers)]
            if native_optimizations != "auto":
                command += ["--native-optimizations", native_optimizations]
            if profile_stages:
                command.append("--profile-stages")
            if native_bin_dir is not None:
                command += ["--native-bin-dir", str(Path(native_bin_dir).resolve())]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                results.append((index, None, completed.stderr.strip()))
            else:
                results.append((index, json.loads(
                    (subject / "fnit-native-free-run.json").read_text()), None))
        return results

    assignments = [[] for _ in devices]
    for index, (t1, subject) in enumerate(prepared):
        assignments[index % len(devices)].append((index, t1, subject))
    reports, errors = [None] * len(prepared), []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run_device, device, assigned)
                   for device, assigned in zip(devices, assignments) if assigned]
        for future in futures:
            for index, report, error in future.result():
                reports[index] = report
                if error:
                    errors.append(f"job {index}: {error}")
    if errors:
        raise RuntimeError("batch reconstruction failed: " + "; ".join(errors))
    return reports
