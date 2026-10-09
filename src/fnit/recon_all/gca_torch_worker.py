"""独立GCA Torch阶段：新exec进程内启用缓存，退出时释放全部CUDA资源。

仅计算nu/brainmask到voxel LTA；不调用原生程序，不更改父进程allocator。
用于验证阶段隔离是否改善完整搜索，未据此切换生产默认或宣称官方等价。
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

import torch

from .mri_em_register_python import register_t1
from .mri_em_register_score_gpu import GCASearchScorer
from .profiling import configure_cuda_allocator, autocast_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_isolated_registration(*, nu_path: Path, mask_path: Path, atlas_path: Path,
                              output_path: Path, report_path: Path,
                              device: str, threads: int = 4,
                              candidate_chunk: int = 1024,
                              inverse_backend: str = "torch",
                              code_version: str = "FNIT-source-hashes") -> dict:
    """从父API启动新exec注册，子进程缓存开启，父allocator和精度不变。

    文件/空间/参数同run_worker；必须指定逻辑CUDA设备，保留父进程的
    CUDA_VISIBLE_DEVICES映射。父CUDA已初始化也可调用；不fork CUDA计算。
    threads控制子进程Torch/OpenMP/BLAS/Numba总线程上限；candidate_chunk
    默认1024、inverse_backend默认torch只改变已有评分分块/同公式求逆。
    输出LTA及子JSON，返回JSON附isolated_cli_wall_seconds（包含子导入、
    哈希、校验、加载、计算、写出和退出）。子进程失败抛CalledProcessError，
    不回退原生、不使用参考。文件已存在或参数无效抛异常。无新增依赖。
    """
    tick = time.perf_counter()
    if torch.device(device).type != "cuda":
        raise ValueError("isolated registration requires an explicit CUDA device")
    if isinstance(candidate_chunk, bool) or not isinstance(candidate_chunk, int) or candidate_chunk < 1:
        raise ValueError("candidate_chunk must be a positive integer")
    if inverse_backend not in {"cpu", "torch"}:
        raise ValueError("inverse_backend must be cpu or torch")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("isolated registration requires new output/report paths")
    from .thread_budget import native_thread_environment
    environment, _ = native_thread_environment(threads=threads)
    environment.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    command = [sys.executable, "-m", "fnit.recon_all.gca_torch_worker",
               "--nu", str(nu_path), "--mask", str(mask_path),
               "--atlas", str(atlas_path), "--output", str(output_path),
               "--report", str(report_path), "--device", device,
               "--threads", str(threads), "--candidate-chunk", str(candidate_chunk),
               "--inverse-backend", inverse_backend, "--code-version", code_version]
    subprocess.run(command, env=environment, check=True)
    result = json.loads(report_path.read_text())
    result["isolated_cli_wall_seconds"] = time.perf_counter() - tick
    result["isolation"] = "fresh exec; child cache enabled; parent allocator preserved"
    pending = report_path.with_suffix(".pending")
    pending.write_text(json.dumps(result, indent=2))
    pending.replace(report_path)
    return result


def run_worker(*, nu_path: Path, mask_path: Path, atlas_path: Path,
               output_path: Path, report_path: Path, device: str = "cuda:0",
               threads: int = 4, candidate_chunk: int = 1024,
               inverse_backend: str = "torch", code_version: str) -> dict:
    """运行单次完整FNIT注册并写LTA/JSON，输入输出均为文件路径。

    nu为3D强度图，mask为同shape/affine掩膜，atlas为固定GCA；输入空间为
    conform voxel，输出LTA是源voxel到图谱voxel的4×4矩阵与双方几何。
    device默认cuda:0且必须CUDA，threads默认4，candidate_chunk默认1024
    只改变完整候选分块；inverse_backend默认torch，可cpu保留原求逆。
    report_path须不存在；output_path须不存在。code_version绑定冻结源码。
    精度为FP32及原有FP64例外，默认TF32，无半精度。返回注册报告字典，
    附输入/源码SHA256、实际精度/线程、缓存设置、GPU allocated/reserved
    及评分时间/候选数；allocated/reserved不是整个进程占用。
    内部计时含读入/传输/全部搜索/EM/输出，不含进程导入，完整CLI另计。
    输入/参数、CUDA或注册失败抛异常，不原生回退；原命令mri_em_register。
    """
    if torch.cuda.is_initialized():
        raise ValueError("worker must be a fresh exec process before CUDA initialization")
    if torch.device(device).type != "cuda" or isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("worker requires a CUDA device and positive thread budget")
    if isinstance(candidate_chunk, bool) or not isinstance(candidate_chunk, int) or candidate_chunk < 1:
        raise ValueError("candidate_chunk must be a positive integer")
    if inverse_backend not in {"cpu", "torch"}:
        raise ValueError("inverse_backend must be cpu or torch")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("worker output and report must be new paths")
    started = time.perf_counter()
    inputs_sha256 = {name: _sha256(path) for name, path in
                    (("nu", nu_path), ("mask", mask_path), ("atlas", atlas_path))}
    allocator = configure_cuda_allocator(device=device, policy="enabled")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    scoring = {"calls": 0, "candidates": 0, "wall_seconds": 0.0,
               "scope": "existing score_many including its final CPU copy; no added CUDA synchronization"}
    original = GCASearchScorer.score_many

    def timed_score(scorer, matrices):
        tick = time.perf_counter()
        try:
            return original(scorer, matrices)
        finally:
            scoring["calls"] += 1
            scoring["candidates"] += len(matrices)
            scoring["wall_seconds"] += time.perf_counter() - tick

    GCASearchScorer.score_many = timed_score
    try:
        report = register_t1(
            nu_path=nu_path, atlas_path=atlas_path, mask_path=mask_path,
            output_path=output_path, device=device, search_backend="torch",
            candidate_chunk=candidate_chunk, sample_chunk=8192,
            reduce_on_device=True, inverse_backend=inverse_backend)
        torch.cuda.synchronize(device)
    finally:
        GCASearchScorer.score_many = original
    report.update(
        scope="same-input complete FNIT GCA in isolated cached worker; not raw-T1 end-to-end",
        code_version=code_version, worker_pid=os.getpid(), cuda_allocator=allocator,
        inputs_sha256=inputs_sha256,
        threads={"torch": torch.get_num_threads(),
                 "cpu_affinity": sorted(os.sched_getaffinity(0)),
                 "environment": {key: os.environ.get(key) for key in
                                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}},
        torch_version=torch.__version__, logical_device=device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                       for name, module in (("worker", __import__(__name__, fromlist=["_"])),
                                            ("registration", __import__(register_t1.__module__, fromlist=["_"])),
                                            ("scorer", __import__(GCASearchScorer.__module__, fromlist=["_"])))},
        precision={"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                   "autocast": autocast_state("cuda"), "half_requested": False},
        gpu_name=torch.cuda.get_device_name(device),
        allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
        reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
        scoring_profile=scoring,
        worker_api_wall_seconds_including_validation_init_read_transfer_write=time.perf_counter()-started,
        overall_metric_equivalence="not_assessed")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".pending")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nu", "mask", "atlas", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=1024)
    parser.add_argument("--inverse-backend", choices=("cpu", "torch"), default="torch")
    parser.add_argument("--code-version", required=True)
    args = parser.parse_args()
    run_worker(nu_path=args.nu, mask_path=args.mask, atlas_path=args.atlas,
               output_path=args.output, report_path=args.report, device=args.device,
               threads=args.threads, candidate_chunk=args.candidate_chunk,
               inverse_backend=args.inverse_backend, code_version=args.code_version)


if __name__ == "__main__":
    main()
