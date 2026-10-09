"""完整WM阶段的新exec入口：复用既有Torch算子，只在子进程启用缓存。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .mri_segment import segment_white_matter_mgz
from .profiling import StageProfiler, autocast_state, configure_cuda_allocator
from .thread_budget import native_thread_environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(*, output_path: Path, report_path: Path, device: str,
              histogram_batch_size: int, planar_batch_size: int) -> None:
    target = torch.device(device)
    if target.type != "cuda" or target.index is None:
        raise ValueError("WM worker requires an explicit logical CUDA index")
    for name, value in (("histogram_batch_size", histogram_batch_size),
                        ("planar_batch_size", planar_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("WM worker requires new output and report paths")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("WM image and report paths must differ")


def run_isolated_segmentation(*, source_path: Path, output_path: Path,
                              report_path: Path, device: str, threads: int = 4,
                              histogram_batch_size: int = 2048,
                              planar_batch_size: int = 256,
                              profile_stages: bool = False,
                              code_version: str = "FNIT-source-hashes") -> dict:
    """父API用新exec运行完整缓存WM，父CUDA/TF32/分配器环境不修改。

    source_path为三维uint8自产强度MGZ；output_path为原XYZ网格/毫米affine
    的uint8 WM，report_path为新JSON。device必须明确如cuda:1，保留父
    CUDA_VISIBLE_DEVICES映射。threads默认4，仅约束子线程环境；直方图
    批量2048、平面批量256只分块全部候选。profile_stages默认False，
    True复用公共StageProfiler同步剖析。code_version记录冻结源码标识。
    返回worker报告及isolated_cli_wall_seconds，包含校验/导入/exec/
    初始化/哈希/读取/传输/全部计算/压缩写出/子退出，不是原始T1整例。
    子失败抛CalledProcessError，不原生回退；参数/路径失败传播。支持已
    初始化CUDA的父API，不fork计算，不读取参考，不调用上游CLI。
    """
    started = time.perf_counter()
    _validate(output_path=output_path, report_path=report_path, device=device,
              histogram_batch_size=histogram_batch_size, planar_batch_size=planar_batch_size)
    environment, _ = native_thread_environment(threads=threads)
    environment.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    # 独立模块目录用于冻结源码覆盖；普通安装中也是当前包目录。传递为
    # argv而非shell代码，不复制模块、不改变父进程包解析或环境。
    bootstrap = ("import sys,runpy,fnit.recon_all; "
                 "fnit.recon_all.__path__.insert(0,sys.argv[1]); "
                 "sys.argv=sys.argv[2:]; "
                 "runpy.run_module('fnit.recon_all.wm_torch_worker',run_name='__main__')")
    command = [sys.executable, "-c", bootstrap, str(Path(__file__).parent),
               "fnit.recon_all.wm_torch_worker", "--source", str(source_path),
               "--output", str(output_path), "--report", str(report_path),
               "--device", device, "--threads", str(threads),
               "--histogram-batch-size", str(histogram_batch_size),
               "--planar-batch-size", str(planar_batch_size),
               "--code-version", code_version]
    if profile_stages:
        command.append("--profile-stages")
    subprocess.run(command, env=environment, check=True)
    result = json.loads(report_path.read_text())
    result["isolated_cli_wall_seconds"] = time.perf_counter() - started
    result["isolation"] = "fresh exec; worker cache enabled; parent policy preserved"
    temporary = report_path.with_suffix(".pending")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(report_path)
    return result


def run_worker(*, source_path: Path, output_path: Path, report_path: Path,
               device: str = "cuda:0", threads: int = 4,
               histogram_batch_size: int = 2048, planar_batch_size: int = 256,
               profile_stages: bool = False, code_version: str) -> dict:
    """新进程用完整WM已有GPU直方图+缓存平面，返回/写出JSON报告。

    输入输出/批量含义同run_isolated_segmentation；device默认cuda:0，
    threads默认4。仅在CUDA初始化前复用configure_cuda_allocator(enabled)，
    TF32开启，无FP16/BF16。有序CPU规则保留。report含源码/强度/输出SHA、
    worker PID、实际精度/线程/缓存策略及allocated/reserved字节；后两者
    不代表进程树占用。内部墙钟含全部校验/初始化/读写，不含模块导入。
    复用StageProfiler且默认不额外同步。输入/路径/已初始化CUDA失败抛
    异常；调用原生mri_segment的等价配方见专页，不执行原生命令。
    """
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        raise ValueError("WM worker must be a fresh exec before CUDA initialization")
    _validate(output_path=output_path, report_path=report_path, device=device,
              histogram_batch_size=histogram_batch_size, planar_batch_size=planar_batch_size)
    native_thread_environment(threads=threads)
    input_sha = _sha256(source_path)
    allocator = configure_cuda_allocator(device=device, policy="enabled")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(device)
    profiler = StageProfiler(device=device, synchronize=profile_stages, allocator=allocator)
    api = profiler.run("complete_wm_cached", segment_white_matter_mgz,
        source_path=source_path, output_path=output_path, device=device,
        histogram_backend="torch", histogram_batch_size=histogram_batch_size,
        planar_backend="cached", planar_batch_size=planar_batch_size)
    from . import mri_segment, mri_segment_histogram_torch, mri_segment_planar_torch, profiling
    report = {
        "scope": "complete FNIT WM in isolated cached worker; not raw T1 whole recon",
        "code_version": code_version, "worker_pid": os.getpid(),
        "input_sha256": input_sha, "output_sha256": _sha256(output_path),
        "source_sha256": {"worker": _sha256(Path(__file__)), **{
            name: _sha256(Path(module.__file__)) for name, module in
            (("segmentation", mri_segment),
             ("histogram", mri_segment_histogram_torch), ("planar", mri_segment_planar_torch),
             ("profiling", profiling))}},
        "cuda_allocator": allocator, "logical_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads": {"torch": torch.get_num_threads(), "requested": threads,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "environment": {key: os.environ.get(key) for key in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}},
        "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                      "autocast": autocast_state("cuda"), "half_requested": False},
        "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(device),
        "api": api, "stage_profile": profiler.last_row,
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
        "process_tree_GPU_memory": "caller benchmark uses shared ProcessTreeDeviceSampler",
        "worker_api_wall_seconds_including_validation_init_read_transfer_write": time.perf_counter() - started,
        "overall_metric_equivalence": "not_assessed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".pending")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--histogram-batch-size", type=int, default=2048)
    parser.add_argument("--planar-batch-size", type=int, default=256)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    run_worker(source_path=args.source, output_path=args.output, report_path=args.report,
               device=args.device, threads=args.threads,
               histogram_batch_size=args.histogram_batch_size, planar_batch_size=args.planar_batch_size,
               profile_stages=args.profile_stages, code_version=args.code_version)


if __name__ == "__main__":
    main()
