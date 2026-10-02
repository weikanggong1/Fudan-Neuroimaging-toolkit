#!/usr/bin/env python3
"""用真实 BOLD 和既有矩阵分解 MCFLIRT 单次 NCC 求值的耗时。

本脚本不重新估计运动，不调用 FSL。默认取第 0 帧 moving，在 8/4 mm
参考上使用第 0、100、489 帧的冻结矩阵。影像读取、参考生成、GPU 传输、
首次编译和预热均不计入组件耗时。公开 JSON 只含标量、文件哈希和设备
负载；不记录路径、逐体素数组或逐帧参数。

对照包括原共享 CPU composer 与缓存 composer、原共享 pull coefficients
与缓存 coefficients、原 compiled NCC reducer 与其 CUDA graph replay，
以及使用既有矩阵的完整 NCC 求值。两份 CUDA cost 使用同一个 compiled
reducer 实例和未改动的采样算法；逐 bit 检查采样张量和 NCC。CPU composer
重复固定参数，旋转缓存可持续命中；这不是 Brent 搜索中的实际命中率，
也不能用该组件的加速比推算完整运动估计。每个 scale 只建一份 workspace。

每个组件的计时独立进行，不能相加当作完整 cost 的耗时。同步的 wall
计时包含 Python、内核提交和实际计算；CUDA event 区间用于说明 GPU
工作量，仍会受共享卡负载影响。完整 MCFLIRT 必须另行运行
benchmark_optimization.py，确认矩阵、参数和校正图保持不变。

例如：python validation/mcflirt/profile_components.py \\
  --bold "$raw_bold_nifti" --reference "$saved_feat_reference_nifti" \\
  --matrices "$frozen_motion_matrices" --source "$fnit_source_snapshot" \\
  --source-revision "$source_git_revision" --output "$profile_json"
raw_bold_nifti 是真实四维 BOLD；saved_feat_reference_nifti 是实际估计时
使用的参考文件，须保留原 NIfTI pixdim；frozen_motion_matrices 是逐帧
MAT_#### 目录或 N×4×4 NumPy 文件；fnit_source_snapshot 是包含当前优化模块的
待测源码快照根目录；
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
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True) if cuda_events else None
        end_event = torch.cuda.Event(enable_timing=True) if cuda_events else None
        if start_event is not None:
            start_event.record(torch.cuda.current_stream(device))
        started = time.perf_counter()
        result = None
        for _ in range(calls):
            result = function()
            if per_call_sync:
                torch.cuda.synchronize(device)
        if end_event is not None:
            end_event.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        return {
            "wall_microseconds_per_call": seconds * 1e6 / calls,
            "cuda_event_microseconds_per_call": (start_event.elapsed_time(end_event) * 1000 / calls
                                                  if start_event is not None else None),
        }, result


def float32_bits(value):
    return int(np.float32(float(value)).view(np.uint32))


def array_bits_equal(left, right, dtype):
    left = np.asarray(left, dtype=dtype)
    right = np.asarray(right, dtype=dtype)
    unsigned_dtype = np.uint64 if np.dtype(dtype).itemsize == 8 else np.uint32
    return left.shape == right.shape and np.array_equal(
        left.view(unsigned_dtype), right.view(unsigned_dtype))


def prepared_bits_equal(left, right):
    return all(a.shape == b.shape and a.dtype == b.dtype == torch.float32 and
               bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
               for a, b in zip(left, right))


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

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("cuda_required")
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.synchronize(device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    source = source_root(args.source)
    sys.path.insert(0, str(source))
    core = importlib.import_module("fnit.mcflirt.core")
    flirt = importlib.import_module("fnit.flirt.core")
    sampler_module = importlib.import_module("fnit.mcflirt._cost_cuda")
    executor_module = importlib.import_module("fnit.mcflirt._cuda_executor")
    rigid_module = importlib.import_module("fnit.mcflirt._rigid")
    shared_sampler_module = importlib.import_module("fnit.flirt._batched_cuda")
    modules = {"fnit.mcflirt.core": core, "fnit.mcflirt._cost_cuda": sampler_module,
               "fnit.mcflirt._cuda_executor": executor_module,
               "fnit.mcflirt._rigid": rigid_module, "fnit.flirt.core": flirt,
               "fnit.flirt._batched_cuda": shared_sampler_module}
    for name, module in modules.items():
        expected_path = source / (name.replace(".", "/") + ".py")
        if Path(module.__file__).resolve() != expected_path:
            raise RuntimeError("import_not_from_requested_source")
    paths = {"fnit." + ".".join(path.relative_to(source / "fnit").with_suffix("").parts): path
             for path in sorted((source / "fnit/mcflirt").rglob("*.py"))}
    paths.update({name: Path(module.__file__).resolve() for name, module in modules.items()})
    paths["profile_driver"] = Path(__file__).resolve()
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    raw = nib.load(str(args.bold))
    target = nib.load(str(args.reference))
    if raw.ndim != 4 or target.ndim != 3 or raw.shape[:3] != target.shape:
        raise ValueError("bold_and_reference_grid_required")
    if not np.allclose(raw.affine, target.affine, atol=1e-4, rtol=0):
        raise ValueError("bold_and_reference_affine_mismatch")
    if any(frame >= raw.shape[3] for frame in args.frames):
        raise ValueError("matrix_frame_outside_real_series")
    moving_array = np.asarray(raw.dataobj[..., 0], np.float32)
    reference_array = np.asarray(target.dataobj, np.float32)
    if not np.isfinite(moving_array).all() or not np.isfinite(reference_array).all():
        raise ValueError("finite_bold_and_reference_required")
    moving = torch.as_tensor(flirt._flip_to_radiological(moving_array, raw.affine),
                             device=device)
    reference = torch.as_tensor(flirt._flip_to_radiological(reference_array, target.affine),
                                device=device)
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
        "scope": "Fixed real-image NCC and fixed CPU parameter evaluations only; no motion estimation, final sampling or I/O timing.",
        "comparison": "Shared CPU rigid composer and coefficients versus cached CPU helpers; original compiled NCC reduction versus replay of that exact same compiled reducer, with one workspace per scale.",
        "timing_note": "Components run independently in rotated and reversed order after warmup. Their timings are not additive and do not estimate the API wall time. Serial wall timing includes host work and synchronization; event intervals can include idle gaps from host submission and shared-GPU scheduling.",
        "fixed_parameter_note": "The CPU composer repeatedly receives the same fixed parameters, so its rotation cache can stay fully warm. This does not measure Brent's actual rotation-cache hit rate or imply a whole-optimizer speedup.",
        "original_cost_definition": "Unchanged fused sampler, shared _fsl_pull_coefficients and original compiled _normcorr_reduce.",
        "workspace_cost_definition": "The same sampling algorithm, cached FSLPullCoefficients and CudaMotionCostExecutor graph replay of the very same original compiled reducer object.",
        "privacy": "Only anonymous scalar aggregates, source/input hashes and GPU utilization are public.",
        "gpu_load_before": gpu_load(), "scales": [], "cases": [],
    }
    for scale in (8.0, 4.0):
        reduced = core._isotropic_reference(reference, reference_sizes, scale)
        original_cost = core.FSLMotionNormCorr(reduced, moving, (scale,) * 3, moving_sizes)
        if original_cost.sampler is None:
            raise RuntimeError("fused_cuda_sampler_required")
        original_reducer = original_cost.reducer

        def shared_coefficients(matrix):
            return flirt._fsl_pull_coefficients(matrix, moving_sizes, (scale,) * 3,
                                                device="cpu").numpy()

        # The current standalone class already uses cached coefficients.
        # Restore this callback explicitly so the original-cost comparator
        # represents the shared pre-optimization CPU preparation.
        original_cost.pull_coefficients = shared_coefficients
        workspace_started = time.perf_counter()
        workspace = executor_module.CudaMotionCostExecutor(
            original_cost.reference, original_cost.moving, moving_sizes, original_reducer)
        workspace_cost = core.FSLMotionNormCorr(
            reduced, moving, (scale,) * 3, moving_sizes,
            centre=original_cost.centre, workspace=workspace)
        torch.cuda.synchronize(device)
        workspace_creation_seconds = time.perf_counter() - workspace_started
        cached_coefficients = rigid_module.FSLPullCoefficients(moving_sizes, (scale,) * 3)
        composer = rigid_module.RigidAffineComposer(original_cost.centre)
        graph_initialization_seconds = None
        for frame, matrix in zip(args.frames, matrices):
            def coefficients_original():
                return shared_coefficients(matrix)

            def coefficients_cached():
                return cached_coefficients(matrix)

            fixed_coefficients = coefficients_original()
            coefficients_equal = array_bits_equal(fixed_coefficients, coefficients_cached(),
                                                   np.float32)
            if not coefficients_equal:
                raise RuntimeError("cached_coefficients_changed_bits")
            rigid_parameters = flirt.fsl_parameters_from_affine(matrix, original_cost.centre)

            def rigid_matrix_original():
                return flirt.fsl_affine_from_parameters(
                    torch.as_tensor(rigid_parameters, dtype=torch.float64),
                    original_cost.centre, 6).numpy()

            def rigid_matrix_cached():
                return composer(rigid_parameters).numpy()

            rigid_equal = array_bits_equal(rigid_matrix_original(), rigid_matrix_cached(),
                                           np.float64)
            if not rigid_equal:
                raise RuntimeError("cached_composer_changed_bits")
            original_prepared = original_cost.sampler.prepare(fixed_coefficients)
            workspace_prepared = workspace.sampler.prepare(fixed_coefficients)
            prepared_equal = prepared_bits_equal(original_prepared, workspace_prepared)
            if not prepared_equal:
                raise RuntimeError("workspace_changed_sampling_bits")
            # First warm the original reducer, then measure graph setup
            # separately. It is excluded from the repeated component times;
            # the full-run benchmark includes construction and first capture.
            for _ in range(args.warmup_calls):
                original_cost(matrix)
            if graph_initialization_seconds is None:
                torch.cuda.synchronize(device)
                capture_started = time.perf_counter()
                workspace_cost(matrix)
                torch.cuda.synchronize(device)
                graph_initialization_seconds = time.perf_counter() - capture_started
            for _ in range(args.warmup_calls):
                workspace_cost(matrix)
            torch.cuda.synchronize(device)
            fixed_bits = float32_bits(original_cost(matrix))
            workspace_bits = float32_bits(workspace_cost(matrix))
            if fixed_bits != workspace_bits:
                raise RuntimeError("workspace_changed_fixed_cost_bits")
            functions = {
                "cpu_rigid_matrix_original": (rigid_matrix_original, False, False),
                "cpu_rigid_matrix_cached": (rigid_matrix_cached, False, False),
                "cpu_pull_coefficients_original": (coefficients_original, False, False),
                "cpu_pull_coefficients_cached": (coefficients_cached, False, False),
                "prepare_sync": (lambda: original_cost.sampler.prepare(fixed_coefficients), True, True),
                "reducer_original_to_float_sync": (lambda: float(original_reducer(*original_prepared)), False, True),
                "reducer_graph_to_float_sync": (lambda: float(workspace.reduce(*workspace_prepared)), False, True),
                "prepared_cost_original_sync": (lambda: float(original_reducer(*original_cost.sampler.prepare(
                    fixed_coefficients))), False, True),
                "prepared_cost_graph_sync": (lambda: float(workspace.reduce(*workspace.sampler.prepare(
                    fixed_coefficients))), False, True),
                "full_cost_original_sync": (lambda: original_cost(matrix), False, True),
                "full_cost_graph_sync": (lambda: workspace_cost(matrix), False, True),
                "prepare_queued": (lambda: original_cost.sampler.prepare(fixed_coefficients), False, True),
                "reducer_original_queued": (lambda: original_reducer(*original_prepared), False, True),
                "reducer_graph_queued": (lambda: workspace.reduce(*workspace_prepared), False, True),
            }
            measurements = {name: [] for name in functions}
            names = list(functions)
            for round_number in range(args.rounds):
                shift = round_number % len(names)
                order = names[shift:] + names[:shift]
                if round_number % 2:
                    order.reverse()
                for name in order:
                    function, per_call_sync, events = functions[name]
                    measurement, last = timed(function, args.calls, device,
                                               per_call_sync=per_call_sync, cuda_events=events)
                    if name.startswith(("reducer_", "prepared_cost_", "full_cost_")):
                        if float32_bits(last) != fixed_bits:
                            raise RuntimeError("profile_changed_fixed_cost_bits")
                    if name.startswith("cpu_rigid_matrix_"):
                        if not array_bits_equal(last, rigid_matrix_original(), np.float64):
                            raise RuntimeError("profile_changed_rigid_composer_bits")
                    if name.startswith("cpu_pull_coefficients_"):
                        if not array_bits_equal(last, fixed_coefficients, np.float32):
                            raise RuntimeError("profile_changed_pull_coefficients_bits")
                    measurements[name].append(measurement)
            entry = {"scale_mm": scale, "matrix_frame": frame, "reference_shape_xyz": list(reduced.shape),
                     "fixed_cost_float32_bits": fixed_bits,
                     "graph_cost_float32_bits": workspace_bits,
                     "cost_bits_equal": fixed_bits == workspace_bits,
                     "prepared_bits_equal": prepared_equal,
                     "rigid_matrix_float64_bits_equal": rigid_equal,
                     "pull_coefficients_float32_bits_equal": coefficients_equal,
                     "same_compiled_reducer_object": workspace.compiled_reducer is original_reducer,
                     "components": {}}
            for name, measurements_for_component in measurements.items():
                entry["components"][name] = {"round_measurements": measurements_for_component,
                    "wall_microseconds_per_call": stats([row["wall_microseconds_per_call"]
                                                          for row in measurements_for_component])}
                gpu_values = [row["cuda_event_microseconds_per_call"] for row in measurements_for_component
                              if row["cuda_event_microseconds_per_call"] is not None]
                if gpu_values:
                    entry["components"][name]["cuda_event_microseconds_per_call"] = stats(gpu_values)
            report["cases"].append(entry)
        report["scales"].append({"scale_mm": scale,
                                  "workspace_creation_seconds": workspace_creation_seconds,
                                  "first_graph_cost_including_capture_seconds": graph_initialization_seconds,
                                  "workspace_instances": 1,
                                  "graph_captured": workspace.graph is not None})
        del original_cost, workspace_cost, workspace
    report["gpu_load_after"] = gpu_load()
    report["runtime_sources_unchanged"] = hashes_before == {name: sha256(path) for name, path in paths.items()}
    report["valid_run"] = report["runtime_sources_unchanged"] and all(
        all(case[key] for key in ("cost_bits_equal", "prepared_bits_equal",
                                 "rigid_matrix_float64_bits_equal",
                                 "pull_coefficients_float32_bits_equal",
                                 "same_compiled_reducer_object")) for case in report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"valid_run": report["valid_run"], "cases": len(report["cases"]),
                      "output_sha256": sha256(args.output)}, separators=(",", ":")))
    if not report["valid_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
