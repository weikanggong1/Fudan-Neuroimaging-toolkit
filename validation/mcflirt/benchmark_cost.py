#!/usr/bin/env python3
r"""真实数据的 MCFLIRT 单次代价函数验证与计时。

输入：四维 BOLD、同网格三维参考、冻结 FNIT 源码，以及逐帧 MAT_####
目录或 N×4×4 .npy。固定使用 BOLD 第 0 帧，参考降采样至 8 mm、4 mm；
在所选帧的既有变换上反复求值，不重新估计运动。
输出：--output 指定的 JSON 和标准输出，只有匿名标量、逐 bit 比较、
耗时、SHA-256、GPU 负载和显存；不写影像或记录输入文件路径。

比较冻结 CPU、冻结 CUDA 张量准备、候选 CUDA 融合准备。两个 CUDA
版本共用冻结的 torch.compile 归约器，并复用相同 moving、centre 和参考。
先预热，再按分块交替计时；准备阶段逐次同步，完整代价的 float 返回值
负责同步。此结果衡量固定变换的代价函数，不能代替完整运动校正耗时。
共享 GPU 上的计时只描述当次负载；输入读取和编译时间不计入测量。
参考头中的体素尺寸须与比较条件一致，包括 NIfTI pixdim 的实际值。

示例（变量由操作者填入；短时序应相应修改 --frames）：
python validation/mcflirt/benchmark_cost.py \
  --bold "$raw_bold_nifti" --reference "$reference_nifti" \
  --frozen-source "$frozen_fnit_source" --baseline-matrices "$frozen_motion_matrices" \
  --frames 0 100 489 --repetitions 100 300 --rounds 2 \
  --frozen-revision "$frozen_git_revision" --candidate-revision "$candidate_git_revision" \
  --output "$cost_report_json"
其中 raw_bold_nifti 是真实四维 BOLD；reference_nifti 是同网格三维参考；
frozen_fnit_source 是冻结源码根目录；frozen_motion_matrices 是完整既有矩阵；
cost_report_json 是本次标量报告路径；两个 git_revision 记录源码版本。

对应原软件命令：mcflirt -in "$raw_bold_nifti" -reffile "$reference_nifti" -mats -plots
原命令须另行运行；本脚本不调用原软件，也不据此给出原 FSL 加速比。
源码：https://git.fmrib.ox.ac.uk/fsl/mcflirt/-/blob/2111.0/mcflirt.cc
参考：Jenkinson et al. NeuroImage 17:825-841 (2002), doi:10.1006/nimg.2002.1132。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch


DEFAULT_CANDIDATE_SOURCE = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def source_root(path):
    path = Path(path).resolve()
    if (path / "src" / "fnit").is_dir():
        return path / "src"
    if (path / "fnit").is_dir():
        return path
    raise ValueError("source_layout_unrecognized")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_load():
    """Read GPU utilization and memory only; exclude processes and identifiers."""
    fields = ["index", "utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "power.draw", "clocks.sm"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=5)
        if result.returncode:
            return {"available": False}
        rows = []
        for line in result.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(fields):
                continue
            row = {}
            for key, value in zip(fields, values):
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = None
            rows.append(row)
        return {"available": True, "gpus": rows}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}


def float32_bits(value):
    return int(np.float32(value).view(np.uint32))


def scalar_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max())}


class CaptureReducer:
    """Capture prepared tensor inputs without reduction, allocation, or CUDA reads."""

    def __init__(self):
        self.prepared = None

    def __call__(self, reference, values, weights):
        self.prepared = (reference, values, weights)
        return 0.0


@contextmanager
def preparation_only(cost):
    original = cost.reducer
    capture = CaptureReducer()
    cost.reducer = capture
    try:
        yield capture
    finally:
        cost.reducer = original


def capture_once(cost, matrix):
    with preparation_only(cost) as capture:
        cost(matrix)
        synchronize(cost.device)
        return capture.prepared


def compare_tensors(left, right):
    """Only scalar differences leave the device; compare actual float32 bits."""
    result = {}
    for name, a, b in zip(("reference", "moving", "weights"), left, right):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("prepared_tensor_contract_changed")
        bit_differences = torch.count_nonzero(a.view(torch.int32) != b.view(torch.int32))
        result[name] = {
            "bit_exact_equal": int(bit_differences) == 0,
            "unequal_bits": int(bit_differences),
            "max_abs": float((a - b).abs().max()),
            "finite_left": bool(torch.isfinite(a).all()),
            "finite_right": bool(torch.isfinite(b).all()),
        }
    return result


def timed_block(cost, matrix, calls, *, per_call_sync):
    synchronize(cost.device)
    started = time.perf_counter()
    last = None
    for _ in range(calls):
        last = cost(matrix)
        if per_call_sync:
            synchronize(cost.device)
    synchronize(cost.device)
    return time.perf_counter() - started, last


def alternating_timings(costs, matrix, calls, chunk, rounds, phase):
    """AB/BA CUDA chunks in one process, with separate CPU comparator blocks."""
    modes = ("frozen_cpu", "frozen_eager_cuda", "candidate_fused_cuda")
    elapsed = {mode: [] for mode in modes}
    last_values = {}
    for round_number in range(rounds):
        seconds = {mode: 0.0 for mode in modes}
        remaining = calls
        chunk_number = 0
        while remaining:
            count = min(chunk, remaining)
            cuda_order = ["frozen_eager_cuda", "candidate_fused_cuda"]
            if (round_number + chunk_number) % 2:
                cuda_order.reverse()
            # Place the CPU chunk on alternate sides, avoiding a persistent
            # placement advantage when the concurrent job's load changes.
            order = (["frozen_cpu"] + cuda_order if (round_number + chunk_number) % 2
                     else cuda_order + ["frozen_cpu"])
            for mode in order:
                value, last = timed_block(
                    costs[mode], matrix, count,
                    per_call_sync=(phase == "preparation"))
                seconds[mode] += value
                last_values[mode] = float(last)
            remaining -= count
            chunk_number += 1
        for mode in modes:
            elapsed[mode].append(seconds[mode])
    result = {}
    for mode in modes:
        seconds = elapsed[mode]
        result[mode] = {
            "calls_per_round": calls,
            "rounds": rounds,
            "elapsed_seconds": seconds,
            "microseconds_per_call": scalar_statistics(
                [value * 1e6 / calls for value in seconds]),
        }
        if phase == "full_cost":
            result[mode]["last_cost"] = last_values[mode]
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bold", type=Path, required=True,
                        help="真实四维 BOLD NIfTI；固定使用第 0 帧作为 moving")
    parser.add_argument("--reference", type=Path, required=True,
                        help="同网格三维参考 NIfTI；保留实际头信息和体素尺寸")
    parser.add_argument("--frozen-source", type=Path, required=True,
                        help="冻结 FNIT 源码目录，含 src/fnit 或 fnit")
    parser.add_argument("--candidate-source", type=Path, default=DEFAULT_CANDIDATE_SOURCE,
                        help="候选 FNIT 源码目录；默认本脚本所在仓库根目录")
    parser.add_argument("--baseline-matrices", type=Path, required=True,
                        help="完整既有 MAT_#### 目录，或逐帧 N×4×4 .npy")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 100, 489],
                        help="抽取既有矩阵的帧号；默认 0 100 489，不改变 moving 第 0 帧")
    parser.add_argument("--repetitions", nargs="+", type=int, default=[100, 300],
                        help="每个固定矩阵每轮调用次数；默认分别测 100、300 次")
    parser.add_argument("--rounds", type=int, default=2,
                        help="每种调用次数测几轮；默认 2")
    parser.add_argument("--chunk", type=int, default=25,
                        help="CPU/CUDA 交替测量的每块调用次数；默认 25")
    parser.add_argument("--warmup-calls", type=int, default=5,
                        help="每个矩阵、每个版本的预热次数；默认 5")
    parser.add_argument("--threads", type=int, default=8,
                        help="PyTorch CPU 线程数；默认 8")
    parser.add_argument("--device", default="cuda:0",
                        help="候选和冻结 CUDA 使用的逻辑设备；默认 cuda:0")
    parser.add_argument("--frozen-revision", default=None,
                        help="可选：冻结源码 Git 版本；文件另以 SHA-256 记录")
    parser.add_argument("--candidate-revision", default=None,
                        help="可选：候选源码 Git 版本；文件另以 SHA-256 记录")
    parser.add_argument("--output", type=Path, required=True,
                        help="本次匿名标量 JSON 报告的输出路径")
    parser.add_argument("--skip-preparation-timing", action="store_true",
                        help="仅跳过准备阶段计时；仍比较准备张量和完整代价")
    args = parser.parse_args()
    if any(value < 1 for value in (
            args.rounds, args.chunk, args.warmup_calls, args.threads, *args.repetitions)):
        parser.error("次数、分块大小和线程数必须为正整数")
    if len(set(args.frames)) != len(args.frames) or any(value < 0 for value in args.frames):
        parser.error("帧号必须非负且不重复")
    return args


def main():
    args = parse_arguments()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("cuda_required")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    frozen_src = source_root(args.frozen_source)
    candidate_src = source_root(args.candidate_source)
    sys.path.insert(0, str(candidate_src))
    current = importlib.import_module("fnit.mcflirt.core")
    # The relative FLIRT import intentionally resolves to the same current
    # mature helper module for both cost classes; both source hashes are saved.
    frozen_file = frozen_src / "fnit" / "mcflirt" / "core.py"
    spec = importlib.util.spec_from_file_location("fnit.mcflirt._frozen", frozen_file)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frozen
    spec.loader.exec_module(frozen)
    flirt_core = importlib.import_module("fnit.flirt.core")
    sampler_module = importlib.import_module("fnit.mcflirt._cost_cuda")
    flirt_cuda = importlib.import_module("fnit.flirt._batched_cuda")
    imported_source = Path(current.__file__).resolve()
    if imported_source != (candidate_src / "fnit" / "mcflirt" / "core.py").resolve():
        raise RuntimeError("candidate_import_not_from_requested_snapshot")

    raw_path = args.bold
    reference_path = args.reference
    raw = nib.load(str(raw_path))
    reference = nib.load(str(reference_path))
    if raw.ndim != 4 or reference.ndim != 3 or raw.shape[:3] != reference.shape:
        raise ValueError("real_bold_and_reference_grid_required")
    if not np.allclose(raw.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("real_bold_and_reference_affine_mismatch")
    if any(frame >= raw.shape[3] for frame in args.frames):
        raise ValueError("requested_fitted_matrix_frame_unavailable")
    moving_array = np.asarray(raw.dataobj[..., 0], dtype=np.float32)
    reference_array = np.asarray(reference.dataobj, dtype=np.float32)
    if not np.isfinite(moving_array).all() or not np.isfinite(reference_array).all():
        raise ValueError("input_arrays_must_be_finite")
    moving_cpu = torch.from_numpy(flirt_core._flip_to_radiological(moving_array, raw.affine))
    reference_cpu = torch.from_numpy(
        flirt_core._flip_to_radiological(reference_array, reference.affine))
    moving_gpu = moving_cpu.to(device)
    reference_gpu = reference_cpu.to(device)
    moving_sizes = tuple(float(value) for value in raw.header.get_zooms()[:3])
    reference_sizes = tuple(float(value) for value in reference.header.get_zooms()[:3])

    if args.baseline_matrices.is_dir():
        matrix_files = [args.baseline_matrices / f"MAT_{frame:04d}" for frame in args.frames]
        matrices = [np.loadtxt(path, dtype=np.float64) for path in matrix_files]
        matrix_hash = aggregate_hash(matrix_files)
    else:
        all_matrices = np.load(args.baseline_matrices, allow_pickle=False)
        if all_matrices.shape != (raw.shape[3], 4, 4):
            raise ValueError("baseline_matrix_array_shape_mismatch")
        matrices = [all_matrices[frame] for frame in args.frames]
        matrix_hash = sha256(args.baseline_matrices)
    if any(matrix.shape != (4, 4) or not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("fitted_matrix_must_be_finite_4_by_4")

    hashes = {
        "raw_bold": sha256(raw_path),
        "reference": sha256(reference_path),
        "selected_baseline_matrices": matrix_hash,
        "frozen_mcflirt_core": sha256(frozen_file),
        "candidate_mcflirt_core": sha256(current.__file__),
        "candidate_mcflirt_cost_cuda": sha256(sampler_module.__file__),
        "shared_current_flirt_core": sha256(flirt_core.__file__),
        "shared_current_flirt_batched_cuda": sha256(flirt_cuda.__file__),
        "frozen_flirt_core": sha256(frozen_src / "fnit" / "flirt" / "core.py"),
        "benchmark_script": sha256(__file__),
    }
    props = torch.cuda.get_device_properties(device)
    report = {
        "status": "complete",
        "comparison": "Frozen FNIT CPU / frozen FNIT eager CUDA preparation / candidate fused CUDA; no native FSL execution",
        "frozen_revision": args.frozen_revision,
        "candidate_revision": args.candidate_revision,
        "hashes": hashes,
        "hardware": {"cuda_device_name": props.name,
                     "cuda_total_memory_bytes": int(props.total_memory),
                     "cuda_multiprocessors": int(props.multi_processor_count),
                     "cuda_compute_capability_major": int(props.major),
                     "cuda_compute_capability_minor": int(props.minor),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "cpu_threads": args.threads,
                     "torch_version": str(torch.__version__)},
        "real_input_shape": list(raw.shape),
        "moving_frame": 0,
        "sampled_fitted_matrix_frames": args.frames,
        "scope": "Cost-only, fixed previously fitted matrices; full motion optimization and I/O excluded",
        "shared_gpu_load": True,
        "timing_control": "Within-process alternating CUDA AB/BA chunks, CPU chunks alternate placement; all kernels/reducer warmed",
        "preparation_timing_control": "Reducer replaced with allocation-free capture; each CUDA preparation call synchronized",
        "reducer_control": "Both CUDA classes use the same frozen torch.compile reducer object; CPU uses frozen eager reducer",
        "reference_control": "8 mm/4 mm reference made once on CUDA with current isotropic helper; CPU receives byte-identical copy",
        "gpu_load_before": gpu_load(),
        "cases": [],
    }
    torch.cuda.reset_peak_memory_stats(device)

    for scale in (8.0, 4.0):
        target_gpu = current._isotropic_reference(reference_gpu, reference_sizes, scale)
        target_cpu = target_gpu.cpu()
        baseline_cpu = frozen.FSLMotionNormCorr(
            target_cpu, moving_cpu, (scale,) * 3, moving_sizes)
        baseline_gpu = frozen.FSLMotionNormCorr(
            target_gpu, moving_gpu, (scale,) * 3, moving_sizes)
        candidate_gpu = current.FSLMotionNormCorr(
            target_gpu, moving_gpu, (scale,) * 3, moving_sizes,
            centre=baseline_gpu.centre)
        if candidate_gpu.sampler is None:
            raise RuntimeError("fused_sampler_not_active")
        candidate_gpu.reducer = baseline_gpu.reducer
        baseline_cpu.centre = baseline_gpu.centre
        costs = {"frozen_cpu": baseline_cpu, "frozen_eager_cuda": baseline_gpu,
                 "candidate_fused_cuda": candidate_gpu}

        # Warm every selected affine's constexpr direction branch and shape,
        # plus the shared compiled reducer. Compilation is outside timed blocks.
        warm_started = time.perf_counter()
        for matrix in matrices:
            for _ in range(args.warmup_calls):
                for cost in costs.values():
                    cost(matrix)
        synchronize(device)
        warm_seconds = time.perf_counter() - warm_started

        for frame, matrix in zip(args.frames, matrices):
            baseline_prepared = capture_once(baseline_gpu, matrix)
            candidate_prepared = capture_once(candidate_gpu, matrix)
            preparation_comparison = compare_tensors(baseline_prepared, candidate_prepared)
            fixed_costs = {name: float(cost(matrix)) for name, cost in costs.items()}
            if not all(np.isfinite(value) for value in fixed_costs.values()):
                raise RuntimeError("nonfinite_fixed_cost")
            cuda_exact = (float32_bits(fixed_costs["frozen_eager_cuda"]) ==
                          float32_bits(fixed_costs["candidate_fused_cuda"]))
            entry = {
                "scale_mm": scale,
                "fitted_matrix_frame": frame,
                "reference_xyz_shape": list(target_gpu.shape),
                "reference_voxels": int(target_gpu.numel()),
                "warmup_seconds_for_scale": warm_seconds,
                "prepared_cuda": preparation_comparison,
                "fixed_cost": fixed_costs,
                "cost_cuda_bit_exact_equal": cuda_exact,
                "cost_cpu_vs_frozen_cuda_abs": abs(
                    fixed_costs["frozen_cpu"] - fixed_costs["frozen_eager_cuda"]),
                "full_cost_timings": [],
                "preparation_timings": [],
                "gpu_load_before_case": gpu_load(),
            }
            for repetitions in args.repetitions:
                timing = alternating_timings(
                    costs, matrix, repetitions, args.chunk, args.rounds, "full_cost")
                timing["calls_per_round"] = repetitions
                entry["full_cost_timings"].append(timing)
                if not args.skip_preparation_timing:
                    with preparation_only(baseline_cpu), preparation_only(baseline_gpu), preparation_only(candidate_gpu):
                        prep_timing = alternating_timings(
                            costs, matrix, repetitions, args.chunk, args.rounds, "preparation")
                    prep_timing["calls_per_round"] = repetitions
                    entry["preparation_timings"].append(prep_timing)
            entry["gpu_load_after_case"] = gpu_load()
            report["cases"].append(entry)
        del baseline_cpu, baseline_gpu, candidate_gpu, costs
        synchronize(device)

    report["all_prepared_cuda_bit_exact_equal"] = all(
        tensor["bit_exact_equal"] for entry in report["cases"]
        for tensor in entry["prepared_cuda"].values())
    report["all_cost_cuda_bit_exact_equal"] = all(
        entry["cost_cuda_bit_exact_equal"] for entry in report["cases"])
    report["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    report["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    report["gpu_load_after"] = gpu_load()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, separators=(",", ":"), ensure_ascii=False))
    if not (report["all_prepared_cuda_bit_exact_equal"] and report["all_cost_cuda_bit_exact_equal"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

