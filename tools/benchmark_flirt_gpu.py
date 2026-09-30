#!/usr/bin/env python3
"""Source-bound real-data FLIRT timing, profiling and FSL paired validation.

No official executable is called. Supply already verified official matrix and
trilinear output from the same images and command. Cold means the first complete
registration in this process; warm repeats reuse the CUDA context, not a fitted
matrix or a changed search. Profiling overhead is excluded from the unprofiled
cold/warm rows by running profiling as a separate complete registration.

Example (private absolute image paths are intentionally supplied by the caller)::

    python tools/benchmark_flirt_gpu.py --moving B0 --reference T1 \
      --fsl-matrix FSL_MAT --fsl-moved FSL_MOVED --dof 6 --cost normmi \
      --device cuda:0 --output-dir PRIVATE_OUTPUT --warm-repeats 1 \
      --profile-costs 8 --source-root REPOSITORY

``--profile-costs`` profiles the first N real cost calls per resolution/cost type.
Those kernel/memcpy counts are explicitly sampled, never reported as full-run
counts. ``--profile-full`` captures all calls, with potentially large host-memory
and runtime overhead. Full-run scalar/CPU-transfer counts are also obtained from
instrumented PyTorch API calls; these are API counts, not CUPTI runtime counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import inspect
import os
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_record(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "minimum": float(values.min()),
            "maximum": float(values.max())}


class GPUMonitor:
    """Observe the physical GPU, including other processes; not process utilization."""
    def __init__(self, torch, device, interval=1.0):
        self.interval = interval
        self.samples = []
        self.finished = threading.Event()
        self.thread = None
        self.uuid = None
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            self.uuid = getattr(properties, "uuid", None)
            # CUDA_VISIBLE_DEVICES can remap logical indices. Resolve UUID when
            # the runtime exposes it; otherwise explicitly retain this limitation.
            self.index = device.index if device.index is not None else torch.cuda.current_device()
        else:
            self.index = None

    def __enter__(self):
        if self.index is None:
            return self
        def sample():
            while not self.finished.is_set():
                try:
                    command = ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
                    rows = [line.split(",") for line in result.stdout.splitlines()]
                    row = next((row for row in rows if self.uuid and row[1].strip().removeprefix("GPU-") == str(self.uuid).removeprefix("GPU-")), None)
                    if row is None:
                        row = next(row for row in rows if int(row[0]) == self.index)
                    self.samples.append({"elapsed_seconds": time.perf_counter() - self.started,
                                         "gpu_utilization_percent": float(row[2]),
                                         "whole_gpu_used_mib": float(row[3]),
                                         "whole_gpu_total_mib": float(row[4])})
                except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
                    pass
                self.finished.wait(self.interval)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.finished.set()
        if self.thread:
            self.thread.join(timeout=6)

    def report(self):
        return {"scope": "whole GPU; includes unrelated processes",
                "sample_interval_seconds": self.interval,
                "samples": len(self.samples),
                "gpu_utilization_percent": percentile_record([row["gpu_utilization_percent"] for row in self.samples]),
                "whole_gpu_used_mib": percentile_record([row["whole_gpu_used_mib"] for row in self.samples]),
                "uuid_mapping_available": self.uuid is not None}


class StageTimings:
    """Lightweight method timers; no profiler, scalar hooks or extra CUDA fences."""
    def __init__(self, core, dof):
        self.core = core
        self.engine = core._RigidNMIEngine if dof == 6 else core._DefaultFLIRTEngine
        self.stages = defaultdict(lambda: {"calls": 0, "wall_seconds": 0.0,
                                          "cost_evaluations": 0})

    def __enter__(self):
        self.patches = ExitStack()
        names = {"_prepare_reference_pyramid": "reference_pyramid",
                 "set_scale": "moving_level_preparation",
                 "angular_candidates": "angular_search", "_optimize": "local_optimization"}
        for method, label in names.items():
            original = getattr(self.engine, method)
            def record(instance, *args, _original=original, _method=method, _label=label, **kwargs):
                scale = args[0] if _method == "set_scale" and args else getattr(instance, "requested_scale", None)
                name = _label if _method in ("angular_candidates", "_prepare_reference_pyramid") else f"{_label}/{scale:g}mm"
                before = instance.cost_evaluations
                started = time.perf_counter()
                try:
                    return _original(instance, *args, **kwargs)
                finally:
                    value = self.stages[name]
                    value["calls"] += 1
                    value["wall_seconds"] += time.perf_counter() - started
                    value["cost_evaluations"] += instance.cost_evaluations - before
            self.patches.enter_context(patch.object(self.engine, method, record))
        original = self.core._resample_output
        def resample(*args, _original=original, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                value = self.stages["final_resampling"]
                value["calls"] += 1
                value["wall_seconds"] += time.perf_counter() - started
        self.patches.enter_context(patch.object(self.core, "_resample_output", resample))
        return self

    def __exit__(self, *exception):
        self.patches.__exit__(*exception)

    def report(self):
        return {"stages": dict(self.stages),
                "scope": "CPU wall clocks around actual methods; no profiler or extra CUDA synchronization; final resampling dispatch can finish asynchronously before the following result copy"}


class Instrumentation:
    """External instrumentation; registration source and scientific behavior unchanged."""
    def __init__(self, core, torch, output, profile_costs=0, profile_full=False):
        self.core, self.torch, self.output = core, torch, output
        self.profile_costs, self.profile_full = profile_costs, profile_full
        self.stack = []
        self.phases = defaultdict(lambda: {"calls": 0, "wall_seconds": 0.0, "exclusive_wall_seconds": 0.0})
        self.evaluations = Counter()
        self.api = Counter()
        self.sample_count = Counter()
        self.batch_calls = Counter()
        self.sample_events = Counter()
        self.sample_cuda_us = Counter()
        self.search = []
        self.profiler = None
        self.profile_windows = 0

    @contextmanager
    def phase(self, name):
        frame = [name, time.perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[1]
            self.stack.pop()
            record = self.phases[name]
            record["calls"] += 1
            record["wall_seconds"] += elapsed
            record["exclusive_wall_seconds"] += elapsed - frame[2]
            if self.stack:
                self.stack[-1][2] += elapsed

    def _collect_profiler(self, profiler, scope):
        # Individual device events are kernel launches/memcpys. CPU API events
        # count actual recorded runtime synchronization calls in this scope.
        for event in profiler.events():
            name = event.name
            kind = str(event.device_type)
            if "CUDA" in kind:
                key = "device_memcpy" if "Memcpy" in name or "memcpy" in name else "device_memset" if "Memset" in name else "kernel"
                self.sample_events[f"{scope}:{key}"] += 1
                self.sample_cuda_us[scope] += float(getattr(event, "device_time_total", 0) or 0)
                if "DtoH" in name or "Device to Host" in name:
                    self.sample_events[f"{scope}:D2H"] += 1
                if "HtoD" in name or "Host to Device" in name:
                    self.sample_events[f"{scope}:H2D"] += 1
            if name in ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize",
                        "cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel",
                        "cudaLaunchCooperativeKernel", "cudaGraphLaunch", "cudaMemcpyAsync", "cudaMemcpy"):
                self.sample_events[f"{scope}:{name}"] += 1

    def __enter__(self):
        self.patches = ExitStack()
        self.started = time.perf_counter()
        torch, core = self.torch, self.core
        for name in ("item", "__float__", "__bool__", "cpu"):
            original = getattr(torch.Tensor, name)
            def wrapper(tensor, *args, _original=original, _name=name, **kwargs):
                if tensor.device.type == "cuda":
                    self.api[_name] += 1
                return _original(tensor, *args, **kwargs)
            self.patches.enter_context(patch.object(torch.Tensor, name, wrapper))
        synchronize = torch.cuda.synchronize
        def sync(*args, **kwargs):
            self.api["explicit_cuda_synchronize"] += 1
            return synchronize(*args, **kwargs)
        self.patches.enter_context(patch.object(torch.cuda, "synchronize", sync))

        # Record nested stages separately; exclusive times prevent double counting.
        engine = core._DefaultFLIRTEngine
        for method in ("_prepare_reference_pyramid", "set_scale", "angular_candidates", "_optimize_search_subset", "optimize_matrix"):
            original = getattr(engine, method)
            def stage(instance, *args, _original=original, _name=method, **kwargs):
                scale = getattr(instance, "requested_scale", None)
                name = f"{_name}/{scale:g}mm" if scale is not None else _name
                with self.phase(name):
                    result = _original(instance, *args, **kwargs)
                if _name == "angular_candidates":
                    self.search.append({"optimized_candidates": len(result[0]),
                                        "preoptimized_candidates": len(result[1]),
                                        "optimized_costs": [float(item[0]) for item in result[0]],
                                        "matrix_sha256": hashlib.sha256(np.asarray([item[1] for item in result[0]], dtype=np.float64).tobytes()).hexdigest()})
                return result
            self.patches.enter_context(patch.object(engine, method, stage))
        original = core._resample_output
        def resample(*args, _original=original, **kwargs):
            with self.phase("final_resampling"):
                return _original(*args, **kwargs)
        self.patches.enter_context(patch.object(core, "_resample_output", resample))

        for cls in (core.FSLCorrelationRatio, core.FSLNormalizedMutualInformation):
            original = cls.__call__
            def evaluate(instance, *args, _original=original, **kwargs):
                scope = f"{type(instance).__name__}/{instance.smooth_size:g}mm"
                self.evaluations[scope] += 1
                if self.sample_count[scope] < self.profile_costs and not self.profile_full:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as profiler:
                        result = _original(instance, *args, **kwargs)
                    self.sample_count[scope] += 1
                    self._collect_profiler(profiler, scope)
                    return result
                result = _original(instance, *args, **kwargs)
                if self.profiler is not None:
                    self.profiler.step()
                return result
            self.patches.enter_context(patch.object(cls, "__call__", evaluate))
        batch_module = Path(core.__file__).with_name("batched.py")
        if batch_module.exists():
            from fnit.flirt.batched import BatchedAffineCost
            original_batch = BatchedAffineCost.__call__
            def evaluate_batch(instance, matrices, *args, **kwargs):
                scope = f"batched_{type(instance.cost).__name__}/{instance.cost.smooth_size:g}mm"
                count = int(matrices.shape[0]) if getattr(matrices, "ndim", 3) == 3 else 1
                self.evaluations[scope] += count
                self.batch_calls[scope] += 1
                if self.sample_count[scope] < self.profile_costs and not self.profile_full:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) as profiler:
                        result = original_batch(instance, matrices, *args, **kwargs)
                    self.sample_count[scope] += 1
                    self._collect_profiler(profiler, scope)
                    return result
                result = original_batch(instance, matrices, *args, **kwargs)
                if self.profiler is not None:
                    self.profiler.step()
                return result
            self.patches.enter_context(patch.object(BatchedAffineCost, "__call__", evaluate_batch))
        if self.profile_full:
            def collect_window(profiler):
                self.profile_windows += 1
                self._collect_profiler(profiler, "complete_registration")
                progress = {**self.report(), "elapsed_seconds": time.perf_counter() - self.started,
                            "complete": False, "benchmark_tool_sha256": sha256(__file__)}
                temporary = self.output / "profile.progress.private.json.tmp"
                temporary.write_text(json.dumps(progress, indent=2) + "\n")
                temporary.replace(self.output / "profile.progress.private.json")
                print(json.dumps({"profile_progress": {"windows": self.profile_windows,
                                  "cost_evaluations": sum(self.evaluations.values()),
                                  "elapsed_seconds": progress["elapsed_seconds"]}}), flush=True)
            # Drain event windows instead of retaining millions of events. Each
            # step is one real cost call; schedule/optimizer behavior is unchanged.
            self.profiler = self.patches.enter_context(torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=128, repeat=0),
                on_trace_ready=collect_window, record_shapes=False, profile_memory=False))
        return self

    def __exit__(self, *exception):
        self.patches.__exit__(*exception)

    def report(self):
        return {"stages": dict(self.phases), "cost_evaluations_by_type_and_scale": dict(self.evaluations), "batch_cost_calls_by_type_and_scale": dict(self.batch_calls),
                "cuda_tensor_api_calls": dict(self.api),
                "cuda_tensor_api_scope": "whole profiled registration; scalar conversions and cpu() API calls, not an exact runtime synchronization count",
                "profiler_scope": "complete registration; raw runtime counts include profiler window boundaries" if self.profile_full else "first N scalar/batched cost calls per resolution/cost type",
                "profiled_cost_calls": dict(self.sample_count), "full_profile_event_windows": self.profile_windows, "profiler_event_counts": dict(self.sample_events),
                "profiled_cuda_device_time_us": dict(self.sample_cuda_us),
                "cuda_device_timeline_available": any(key.endswith(":kernel") for key in self.sample_events),
                "kernel_launch_runtime_api_count": sum(value for key, value in self.sample_events.items() if key.rsplit(":", 1)[-1] in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cudaLaunchCooperativeKernel")),
                "angular_search": self.search}


def paired_metrics(result, moving, fixed, fsl_matrix, fsl_moved, comparison_mask, core):
    matrix = np.loadtxt(fsl_matrix)
    official_world = core.flirt_to_world_affine(matrix, moving.affine, fixed.affine, moving.shape, fixed.shape)
    axes = [np.linspace(0, size - 1, 13) for size in moving.shape]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    points = nib.affines.apply_affine(moving.affine, points)
    distances = np.linalg.norm(nib.affines.apply_affine(result.moving_to_fixed_world, points) - nib.affines.apply_affine(official_world, points), axis=1)
    official = nib.load(str(fsl_moved))
    first = np.asarray(official.dataobj, dtype=np.float32)
    second = np.asarray(result.moved.dataobj, dtype=np.float32)
    if official.shape != fixed.shape or first.shape != second.shape or not np.allclose(official.affine, fixed.affine, atol=1e-5, rtol=0):
        raise ValueError("Official output must be on the same reference grid")
    if comparison_mask:
        mask_image = nib.load(str(comparison_mask))
        if mask_image.shape != fixed.shape or not np.allclose(mask_image.affine, fixed.affine, atol=1e-5, rtol=0):
            raise ValueError("Comparison mask must match the reference grid")
        mask = np.asarray(mask_image.dataobj) > 0
        mask_definition = "provided mask; unchanged across paths"
    else:
        mask = first != 0
        mask_definition = "official output foreground; unchanged across paths"
    x, y = first[mask].astype(np.float64), second[mask].astype(np.float64)
    difference = y - x
    intersection = (first != 0) & (second != 0)
    return {"world_grid_displacement_mm": {**percentile_record(distances), "rms": float(np.sqrt(np.mean(distances * distances)))},
            "warped": {"comparison_mask": mask_definition, "voxels": int(mask.sum()),
                       "pearson_r": float(np.corrcoef(x, y)[0, 1]), "mae": float(np.abs(difference).mean()),
                       "rmse": float(np.sqrt(np.mean(difference * difference))),
                       "support_dice": float(2 * intersection.sum() / ((first != 0).sum() + (second != 0).sum()))},
            "grid_header": {"shape_equal": tuple(result.moved.shape) == tuple(official.shape),
                            "affine_max_abs": float(np.abs(result.moved.affine - official.affine).max()),
                            "dtype_equal": str(result.moved.header.get_data_dtype()) == str(official.header.get_data_dtype()),
                            "qform_code_equal": int(result.moved.header["qform_code"]) == int(official.header["qform_code"]),
                            "sform_code_equal": int(result.moved.header["sform_code"]) == int(official.header["sform_code"])},
            "cost": float(result.qc["cost_value"]), "cost_evaluations": int(result.qc["cost_evaluations"])}


def worker(args):
    source = Path(args.source_root).resolve()
    sys.path.insert(0, str(source / "src"))
    import torch
    from fnit.flirt import core
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.mem_get_info(device)  # establish the selected context before timing
    moving = nib.load(args.moving)
    fixed = nib.load(args.reference)
    # Materialize image data before the timed call, making I/O scope explicit.
    moving.get_fdata(dtype=np.float32)
    fixed.get_fdata(dtype=np.float32)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameters = {"device": str(device), "dof": args.dof, "cost": args.cost}
    if args.execution != "default":
        if "execution" not in inspect.signature(core.TorchFLIRT).parameters:
            raise ValueError("This source snapshot has no execution selection API")
        parameters["execution"] = args.execution
    if args.candidate_batch_size is not None:
        parameters["candidate_batch_size"] = args.candidate_batch_size
    tracker = core.TorchFLIRT(**parameters)
    inputs = {"moving": sha256(args.moving), "reference": sha256(args.reference),
              "fsl_matrix": sha256(args.fsl_matrix), "fsl_moved": sha256(args.fsl_moved)}
    if args.mask:
        inputs["comparison_mask"] = sha256(args.mask)
    if args.init:
        inputs["initial_matrix"] = sha256(args.init)
    report = {"schema_version": 1, "benchmark_tool_sha256": sha256(__file__), "source_sha256": {str(path.relative_to(source)): sha256(path) for path in [*(source / "src/fnit/flirt").glob("*.py"), source / "src/fnit/_nib.py"]},
              "input_sha256": inputs, "device": str(device), "dof": args.dof, "cost": args.cost,
              "execution": args.execution, "threads": args.threads, "runs": [],
              "environment": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                              "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "gpu_name": torch.cuda.get_device_properties(device).name if device.type == "cuda" else None,
                              "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                              "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
                              "physical_gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", None)) if device.type == "cuda" else None},
              "timing_scope": "already materialized input images; full schedule, optimization, resampling, device-to-host result; excludes input/output file I/O",
              "cold_definition": "first registration in fresh process; CUDA context and constructor setup precede timing",
              "warm_definition": "repeat full registration in same process; no fitted-matrix reuse",
              "official_command": args.official_command, "official_wall_seconds": args.official_wall_seconds}
    result = None
    for index in range(0 if args.profile_only else 1 + args.warm_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with GPUMonitor(torch, device) as monitor:
            started = time.perf_counter()
            with StageTimings(core, args.dof) as stage_timers:
                result = tracker(moving, fixed, init=args.init)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        record = {"condition": "cold" if index == 0 else "warm", "wall_seconds": elapsed,
                  "external_stage_timings": stage_timers.report(),
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
                  "gpu_observation": monitor.report(), "qc": result.qc,
                  "paired": paired_metrics(result, moving, fixed, args.fsl_matrix, args.fsl_moved, args.mask, core)}
        report["runs"].append(record)
        result.save(output / f"{record['condition']}_{index}.nii.gz", output / f"{record['condition']}_{index}.mat")
        (output / "report.private.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"condition": record["condition"], "wall_seconds": elapsed, "paired": record["paired"]}), flush=True)
    if args.profile_costs or args.profile_full:
        if device.type != "cuda":
            raise ValueError("CUDA profiler requires a CUDA device")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with GPUMonitor(torch, device) as monitor:
            started = time.perf_counter()
            with Instrumentation(core, torch, output, args.profile_costs, args.profile_full) as instrumentation:
                profiled = tracker(moving, fixed, init=args.init)
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        report["profile"] = {**instrumentation.report(), "instrumented_wall_seconds": elapsed,
                             "instrumentation_changes_timing": True, "gpu_observation": monitor.report(),
                             "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                             "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                             "paired": paired_metrics(profiled, moving, fixed, args.fsl_matrix, args.fsl_moved, args.mask, core),
                             "profiled_vs_last_unprofiled_matrix_max_abs": float(np.abs(profiled.matrix - result.matrix).max()) if result is not None else None}
        # Persist the independent profiled result for exact output regression
        # checks; file I/O occurs after profiling and its timed scope.
        profiled.save(output / "profiled.nii.gz", output / "profiled.mat")
        (output / "report.private.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"profile": report["profile"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--fsl-matrix", required=True)
    parser.add_argument("--fsl-moved", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--init")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dof", type=int, choices=(6, 12), required=True)
    parser.add_argument("--cost", choices=("normmi", "corratio"), required=True)
    parser.add_argument("--execution", choices=("default", "reference", "batched"), default="default")
    parser.add_argument("--candidate-batch-size", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--profile-costs", type=int, default=0)
    parser.add_argument("--profile-full", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="Separate instrumented run without an extra cold/warm repetition")
    parser.add_argument("--official-command")
    parser.add_argument("--official-wall-seconds", type=float)
    args = parser.parse_args()
    if args.profile_only and not (args.profile_full or args.profile_costs):
        parser.error("--profile-only requires --profile-full or --profile-costs")
    if args.warm_repeats < 0 or args.profile_costs < 0:
        parser.error("repeat and sampled profile counts must be nonnegative")
    worker(args)


if __name__ == "__main__":
    main()
