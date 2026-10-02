#!/usr/bin/env python3
"""用真实 BOLD 和既有矩阵分解 MCFLIRT 单次 NCC 求值的耗时。

本脚本不重新估计运动，不调用 FSL。默认取第 0 帧 moving，在 8/4 mm
参考上使用第 0、100、489 帧的冻结矩阵。影像读取、参考生成、GPU 传输、
首次编译和预热均不计入组件耗时。公开 JSON 只含标量、文件哈希和设备
负载；不记录路径、逐体素数组或逐帧参数。

每个组件的计时独立进行，不能相加当作完整 cost 的耗时。同步的 wall
计时包含 Python、内核提交和实际计算；CUDA event 区间用于说明 GPU
工作量，仍会受共享卡负载影响。完整 MCFLIRT 必须另行运行
benchmark_optimization.py，确认矩阵、参数和校正图保持不变。

例如：python validation/mcflirt/profile_components.py \\
  --bold "$raw_bold_nifti" --reference "$saved_feat_reference_nifti" \\
  --matrices "$frozen_motion_matrices" --source "$frozen_fnit_source" \\
  --source-revision "$source_git_revision" --output "$profile_json"
raw_bold_nifti 是真实四维 BOLD；saved_feat_reference_nifti 是实际估计时
使用的参考文件，须保留原 NIfTI pixdim；frozen_motion_matrices 是逐帧
MAT_#### 目录或 N×4×4 NumPy 文件；frozen_fnit_source 是待测源码根目录；
profile_json 是匿名报告输出路径。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_root(path):
    path = Path(path).resolve()
    for source in (path / "src", path):
        if (source / "fnit" / "mcflirt" / "core.py").is_file():
            return source
    raise ValueError("source_layout_unrecognized")


def gpu_load():
    fields = ("index", "utilization.gpu", "utilization.memory", "memory.used",
              "memory.total", "power.draw", "clocks.sm")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    if result.returncode:
        return {"available": False}
    rows = []
    for line in result.stdout.splitlines():
        values = line.split(",")
        if len(values) == len(fields):
            row = {}
            for field, value in zip(fields, values):
                try:
                    row[field] = float(value.strip())
                except ValueError:
                    row[field] = None
            rows.append(row)
    return {"available": True, "gpus": rows}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max())}


def timed(function, calls, device, *, per_call_sync, cuda_events):
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True) if cuda_events else None
    end_event = torch.cuda.Event(enable_timing=True) if cuda_events else None
    if start_event is not None:
        start_event.record()
    started = time.perf_counter()
    result = None
    for _ in range(calls):
        result = function()
        if per_call_sync:
            torch.cuda.synchronize(device)
    if end_event is not None:
        end_event.record()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "wall_microseconds_per_call": seconds * 1e6 / calls,
        "cuda_event_microseconds_per_call": (start_event.elapsed_time(end_event) * 1000 / calls
                                              if start_event is not None else None),
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bold", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 100, 489])
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-calls", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.calls, args.rounds, args.warmup_calls, args.threads) < 1:
        parser.error("调用次数、轮数、预热次数和线程数须为正整数")
    if len(set(args.frames)) != len(args.frames) or any(frame < 0 for frame in args.frames):
        parser.error("帧号须非负且不重复")

    source = source_root(args.source)
    sys.path.insert(0, str(source))
    core = importlib.import_module("fnit.mcflirt.core")
    flirt = importlib.import_module("fnit.flirt.core")
    sampler_module = importlib.import_module("fnit.mcflirt._cost_cuda")
    if Path(core.__file__).resolve() != source / "fnit/mcflirt/core.py":
        raise RuntimeError("import_not_from_requested_source")
    paths = {"fnit.mcflirt.core": Path(core.__file__),
             "fnit.mcflirt._cost_cuda": Path(sampler_module.__file__),
             "fnit.flirt.core": Path(flirt.__file__),
             "fnit.flirt._batched_cuda": source / "fnit/flirt/_batched_cuda.py",
             "profile_driver": Path(__file__)}
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("cuda_required")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    raw = nib.load(str(args.bold))
    target = nib.load(str(args.reference))
    if raw.ndim != 4 or target.ndim != 3 or raw.shape[:3] != target.shape:
        raise ValueError("bold_and_reference_grid_required")
    if not np.allclose(raw.affine, target.affine, atol=1e-4, rtol=0):
        raise ValueError("bold_and_reference_affine_mismatch")
    if any(frame >= raw.shape[3] for frame in args.frames):
        raise ValueError("matrix_frame_outside_real_series")
    moving = torch.as_tensor(flirt._flip_to_radiological(
        np.asarray(raw.dataobj[..., 0], np.float32), raw.affine), device=device)
    reference = torch.as_tensor(flirt._flip_to_radiological(
        np.asarray(target.dataobj, np.float32), target.affine), device=device)
    moving_sizes = tuple(map(float, raw.header.get_zooms()[:3]))
    reference_sizes = tuple(map(float, target.header.get_zooms()[:3]))
    matrices_hash = hashlib.sha256()
    if args.matrices.is_dir():
        matrices = []
        for frame in args.frames:
            path = args.matrices / f"MAT_{frame:04d}"
            matrices_hash.update(path.read_bytes())
            matrices.append(np.loadtxt(path, dtype=np.float64))
    else:
        array = np.load(args.matrices, allow_pickle=False)
        if array.shape != (raw.shape[3], 4, 4):
            raise ValueError("matrix_array_shape_mismatch")
        matrices = [array[frame] for frame in args.frames]
        matrices_hash.update(args.matrices.read_bytes())
    if any(matrix.shape != (4, 4) or not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("finite_matrix_required")

    report = {
        "source_revision": args.source_revision, "source_sha256": hashes_before,
        "input_sha256": {"bold": sha256(args.bold), "reference": sha256(args.reference),
                         "selected_matrices": matrices_hash.hexdigest()},
        "input_shape": list(raw.shape), "moving_frame": 0,
        "matrix_frames": args.frames, "calls_per_round": args.calls, "rounds": args.rounds,
        "warmup_calls": args.warmup_calls, "cpu_threads": args.threads,
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__),
        "reference_pixdim_float32_bits": np.asarray(reference_sizes, np.float32).view(np.uint32).tolist(),
        "scope": "Fixed real-image NCC evaluations only; no motion estimation, final sampling or I/O timing.",
        "timing_note": "Components run independently in rotated order after warmup. Their timings are not additive. Serial wall timing includes host work and synchronization; event intervals can include idle gaps from host submission and shared-GPU scheduling.",
        "privacy": "Only anonymous scalar aggregates, source/input hashes and GPU utilization are public.",
        "gpu_load_before": gpu_load(), "cases": [],
    }
    for scale in (8.0, 4.0):
        reduced = core._isotropic_reference(reference, reference_sizes, scale)
        cost = core.FSLMotionNormCorr(reduced, moving, (scale,) * 3, moving_sizes)
        if cost.sampler is None:
            raise RuntimeError("fused_cuda_sampler_required")
        for frame, matrix in zip(args.frames, matrices):
            def coefficients():
                return flirt._fsl_pull_coefficients(matrix, moving_sizes, (scale,) * 3,
                                                     device="cpu").numpy()

            fixed_coefficients = coefficients()
            rigid_parameters = flirt.fsl_parameters_from_affine(matrix, cost.centre)

            def rigid_matrix():
                return flirt.fsl_affine_from_parameters(
                    torch.as_tensor(rigid_parameters, dtype=torch.float64), cost.centre, 6).numpy()

            prepared = cost.sampler.prepare(fixed_coefficients)
            for _ in range(args.warmup_calls):
                cost(matrix)
            torch.cuda.synchronize(device)
            fixed_cost = cost(matrix)
            fixed_bits = int(np.float32(fixed_cost).view(np.uint32))
            functions = {
                "cpu_rigid_matrix": (rigid_matrix, False, False),
                "cpu_pull_coefficients": (coefficients, False, False),
                "prepare_sync": (lambda: cost.sampler.prepare(fixed_coefficients), True, True),
                "reducer_to_float_sync": (lambda: float(cost.reducer(*prepared)), False, True),
                "prepared_cost_sync": (lambda: float(cost.reducer(*cost.sampler.prepare(
                    fixed_coefficients))), False, True),
                "full_cost_sync": (lambda: cost(matrix), False, True),
                "prepare_queued": (lambda: cost.sampler.prepare(fixed_coefficients), False, True),
                "reducer_queued": (lambda: cost.reducer(*prepared), False, True),
            }
            measurements = {name: [] for name in functions}
            names = list(functions)
            for round_number in range(args.rounds):
                order = names[round_number:] + names[:round_number]
                for name in order:
                    function, per_call_sync, events = functions[name]
                    measurement, last = timed(function, args.calls, device,
                                               per_call_sync=per_call_sync, cuda_events=events)
                    if name in ("reducer_to_float_sync", "prepared_cost_sync", "full_cost_sync"):
                        if int(np.float32(last).view(np.uint32)) != fixed_bits:
                            raise RuntimeError("profile_changed_fixed_cost_bits")
                    measurements[name].append(measurement)
            entry = {"scale_mm": scale, "matrix_frame": frame, "reference_shape_xyz": list(reduced.shape),
                     "fixed_cost_float32_bits": fixed_bits, "components": {}}
            for name, measurements_for_component in measurements.items():
                entry["components"][name] = {"round_measurements": measurements_for_component,
                    "wall_microseconds_per_call": stats([row["wall_microseconds_per_call"]
                                                          for row in measurements_for_component])}
                gpu_values = [row["cuda_event_microseconds_per_call"] for row in measurements_for_component
                              if row["cuda_event_microseconds_per_call"] is not None]
                if gpu_values:
                    entry["components"][name]["cuda_event_microseconds_per_call"] = stats(gpu_values)
            report["cases"].append(entry)
        del cost
    report["gpu_load_after"] = gpu_load()
    report["runtime_sources_unchanged"] = hashes_before == {name: sha256(path) for name, path in paths.items()}
    report["valid_run"] = report["runtime_sources_unchanged"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"valid_run": report["valid_run"], "cases": len(report["cases"]),
                      "output_sha256": sha256(args.output)}, separators=(",", ":")))
    if not report["valid_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
