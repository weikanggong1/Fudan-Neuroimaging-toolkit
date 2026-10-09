"""现有归一化的完整文件 API 分配器剖析；不改变控制点或偏置算法。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import threading
import time

import nibabel as nib
import numpy as np
import torch


def sha(path):
    """分块读取文件并返回 SHA-256；不把影像内容写入报告。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def timed_cpu_helpers(modules, records, gpu_context_class=None):
    """仅计时已有 CPU 辅助函数，保留对象、参数、更新顺序并在退出恢复。

    records 追加名字、调用次数、CPU 秒与墙钟；嵌套值不可直接相加。
    不插入 CUDA 同步，不访问或修改控制图。异常保持传播。
    """
    patches = []
    names = {
        "normalize_3d_controls": ("_homogeneous", "_neighbor_sum", "tissue_peaks", "_remove_outliers_ordered"),
        "normalize_gentle_source": ("_remove_outliers_ordered",),
        "aseg_pipeline": ("prepare_aseg_source", "medial_ridge", "filter_aseg_ridge", "apply_initial_aseg_bias"),
    }
    for module in modules:
        for name in names.get(module.__name__.rsplit(".", 1)[-1], ()):
            original = getattr(module, name)
            key = module.__name__.rsplit(".", 1)[-1] + "." + name
            row = records.setdefault(key, {"calls": 0, "seconds": 0., "cpu_seconds": 0.})

            def wrapped(*args, _function=original, _row=row, **kwargs):
                tick, cpu = time.perf_counter(), time.process_time()
                try:
                    return _function(*args, **kwargs)
                finally:
                    _row["calls"] += 1
                    _row["seconds"] += time.perf_counter() - tick
                    _row["cpu_seconds"] += time.process_time() - cpu

            setattr(module, name, wrapped)
            patches.append((module, name, original))
    if gpu_context_class is not None:
        original = gpu_context_class.__call__
        row = records.setdefault("NeighborSumTorch.complete_transfer_kernel_download", {
            "calls": 0, "seconds": 0., "cpu_seconds": 0.})

        def neighbor_wrapped(*args, **kwargs):
            tick, cpu = time.perf_counter(), time.process_time()
            try:
                return original(*args, **kwargs)
            finally:
                row["calls"] += 1
                row["seconds"] += time.perf_counter() - tick
                row["cpu_seconds"] += time.process_time() - cpu

        gpu_context_class.__call__ = neighbor_wrapped
        patches.append((gpu_context_class, "__call__", original))
    try:
        yield
    finally:
        for module, name, original in reversed(patches):
            setattr(module, name, original)


def main():
    """固定同网格输入→原算法 uint8 MGZ＋JSON；默认两轮，失败传播。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("t1", "aseg"), required=True,
                        help="第一轮 T1 或带 aseg 的第二轮归一化")
    parser.add_argument("--cache", choices=("enabled", "disabled"), required=True,
                        help="启动前已选择的分配器；不能在 CUDA 初始化后切换")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="冻结 FNIT 源码包的 src 目录")
    parser.add_argument("--candidate-overlay", type=Path,
                        help="可选只读两文件覆盖目录；只加载normalize_aseg_source/aseg_pipeline")
    parser.add_argument("--mri-dir", type=Path, required=True,
                        help="自产检查点目录；不作为原始 T1 整例验收")
    parser.add_argument("--reference", type=Path, required=True,
                        help="相同输入的旧 FNIT 输出，仅在完成后比较")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="新目录，保存 T1/brain.mgz 与 report.json")
    parser.add_argument("--device", default="cuda:1", help="明确 CUDA 逻辑设备")
    parser.add_argument("--threads", type=int, default=4, help="总 CPU 线程预算")
    parser.add_argument("--code-commit", required=True, help="冻结源码 commit；实际另绑定文件 SHA")
    parser.add_argument("--controls-neighbor-backend", choices=("cpu", "torch"), default="cpu",
                        help="默认现有 CPU 邻域；torch 仅用于已冻结候选显式 GPU 邻域")
    parser.add_argument("--initial-bias-backend", choices=("cpu", "torch"), default="cpu",
                        help="aseg 初始偏场；torch 复用既有GPU传播和严格平滑")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.stage != "aseg" and args.initial_bias_backend != "cpu":
        raise ValueError("initial bias backend only applies to aseg stage")
    disabled = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") is not None
    if disabled != (args.cache == "disabled"):
        raise ValueError("allocator environment must be selected before Python startup")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or torch.cuda.is_initialized():
        raise ValueError("explicit CUDA device and fresh uninitialized process required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # 源码路径只能读取；不改正在运行的冻结包。
    import sys
    sys.path.insert(0, str(args.source_dir.resolve()))
    if args.candidate_overlay is not None:
        # 诊断只装载明确的两个候选模块，避免复制既有整个workspace。
        namespace = importlib.import_module("fnit.recon_all.normalization")
        for name in ("normalize_aseg_source", "aseg_pipeline"):
            full_name = "fnit.recon_all.normalization." + name
            file = args.candidate_overlay / (name + ".py")
            if not file.is_file():
                raise FileNotFoundError(file)
            spec = importlib.util.spec_from_file_location(full_name, file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            setattr(namespace, name, module)
    modules = [importlib.import_module("fnit.recon_all.normalization." + name)
               for name in ("pipeline", "aseg_pipeline", "normalize_3d_controls", "normalize_gentle_source")]
    from fnit.recon_all.profiling import (configure_cuda_allocator, StageProfiler,
                                        ProcessTreeDeviceSampler)
    allocator = configure_cuda_allocator(device=str(device), policy=args.cache)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    initialization_tick = time.perf_counter()
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_tick
    filenames = (("nu.mgz", "transforms/talairach.xfm") if args.stage == "t1" else
                 ("norm.mgz", "aseg.presurf.mgz", "brainmask.mgz"))
    input_paths = {name: args.mri_dir / name for name in filenames}
    input_hashes = {name: sha(path) for name, path in input_paths.items()}
    sampler = ProcessTreeDeviceSampler(device=str(device), parent_pid=os.getpid(), interval=.5)
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            sampler.sample_if_due(force=True)
            stop.wait(.5)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    profiler = StageProfiler(device=str(device), synchronize=True, allocator=allocator)
    output = args.output_dir / ("T1.mgz" if args.stage == "t1" else "brain.mgz")
    helpers = {}
    control_traces = []
    active_pipeline = modules[0] if args.stage == "t1" else modules[1]
    original_controls = active_pipeline.controls_3d

    def trace_controls(*positional, **keywords):
        source = positional[0] if positional else keywords["source"]
        input_digest = hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()
        control, details = original_controls(*positional, **keywords)
        control_traces.append({"round": len(control_traces) + 1,
            "input_data_sha256": input_digest, "input_shape": list(source.shape), "input_dtype": str(source.dtype),
            "control_data_sha256": hashlib.sha256(np.ascontiguousarray(control).tobytes()).hexdigest(),
            "control_dtype": str(control.dtype), "details": details})
        return control, details

    active_pipeline.controls_3d = trace_controls
    gpu_context_class = (importlib.import_module(
        "fnit.recon_all.normalization.normalize_neighbor_sum_torch").NeighborSumTorch
        if args.controls_neighbor_backend == "torch" else None)
    extra = ({} if args.controls_neighbor_backend == "cpu" else
             {"controls_neighbor_backend": args.controls_neighbor_backend})
    if args.initial_bias_backend != "cpu":
        extra["initial_bias_backend"] = args.initial_bias_backend
    initial_trace = {}
    original_initial_bias = modules[1].apply_initial_aseg_bias

    def trace_initial_bias(source, controls, **keywords):
        initial_trace["source_data_sha256"] = hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()
        initial_trace["controls_data_sha256"] = hashlib.sha256(np.ascontiguousarray(controls).tobytes()).hexdigest()
        initial = original_initial_bias(source, controls, **keywords)
        initial_trace["output_data_sha256"] = hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()
        initial_trace["dtype"] = str(initial.dtype)
        return initial

    if args.stage == "aseg":
        modules[1].apply_initial_aseg_bias = trace_initial_bias
    try:
        with timed_cpu_helpers(modules, helpers, gpu_context_class=gpu_context_class):
            if args.stage == "t1":
                api = profiler.run("normalize_t1", modules[0].normalize_t1,
                    input_file=input_paths["nu.mgz"], xfm_file=input_paths["transforms/talairach.xfm"],
                    output_file=output, device=str(device), three_d_iterations=2, diagnostic_dir=None, **extra)
            else:
                api = profiler.run("normalize_t1_aseg", modules[1].normalize_t1_aseg,
                    norm_file=input_paths["norm.mgz"], aseg_file=input_paths["aseg.presurf.mgz"],
                    brainmask_file=input_paths["brainmask.mgz"], output_file=output,
                    device=str(device), three_d_iterations=2, **extra)
    finally:
        active_pipeline.controls_3d = original_controls
        modules[1].apply_initial_aseg_bias = original_initial_bias
        stop.set()
        monitor.join(timeout=10)
    sampler.sample_if_due(force=True)
    unchanged = {name: sha(path) == input_hashes[name] for name, path in input_paths.items()}
    if not all(unchanged.values()):
        raise RuntimeError("input checkpoint changed during benchmark")
    reference, result = [nib.load(str(path)) for path in (args.reference, output)]
    lhs, rhs = np.asarray(reference.dataobj), np.asarray(result.dataobj)
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise ValueError("reference/output shape or dtype differs")
    difference = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    loaded_modules = {Path(module.__file__).name: sha(module.__file__) for name, module in sys.modules.items()
                      if name.startswith("fnit.recon_all.normalization") and getattr(module, "__file__", None)}
    report = {
        "scope": "same-input full existing normalization API allocator ABBA component; not raw T1 end-to-end",
        "stage": args.stage, "cache": args.cache, "code_commit": args.code_commit,
        "controls_neighbor_backend": args.controls_neighbor_backend,
        "initial_bias_backend": args.initial_bias_backend,
        "source_sha256": loaded_modules, "script_sha256": sha(__file__),
        "profiling_sha256": sha(sys.modules["fnit.recon_all.profiling"].__file__),
        "input_sha256": input_hashes, "input_unchanged": unchanged,
        "reference_kind": "same frozen self-produced 803 chain stage output; comparison only",
        "reference_sha256": sha(args.reference), "output_sha256": sha(output),
        "host": platform.node(), "cpu_model": next((line.split(":", 1)[1].strip() for line in
            Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unavailable"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
        "thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "nibabel": nib.__version__, "torch_cuda": torch.version.cuda,
        "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_tf32": torch.backends.cudnn.allow_tf32, "autocast": torch.is_autocast_enabled(),
                      "fp16_bf16_used": False},
        "allocator": allocator, "initialization_seconds": initialization_seconds,
        "full_API": profiler.last_row, "api": api, "cpu_helpers": helpers,
        "three_d_control_traces": control_traces,
        "initial_bias_trace": initial_trace,
        "timing_includes": "first API validation, file loading, H2D/D2H, control search, first JIT/cache load, all two iterations and compressed write; target CUDA pre/post sync",
        "timing_excludes": "Python import, CUDA initialization, SHA collection and comparison; shell records cold process separately",
        "helper_timing_scope": "CPU helpers or GPU initial bias/full neighbor upload/kernel/synchronous download; nested values not summed; no additional synchronization; initial and round input/control/output data SHA tracing included in paired API",
        "shape": list(rhs.shape), "dtype": str(rhs.dtype), "different_voxels": int(np.count_nonzero(difference)),
        "max_abs": float(difference.max()), "p99_abs": float(np.percentile(difference, 99)),
        "affine_equal": bool(np.array_equal(result.affine, reference.affine)),
        "mgh_header_equal": result.header.binaryblock == reference.header.binaryblock,
        "byte_equal": sha(args.reference) == sha(output), "process_tree_GPU_memory": sampler.report(),
        "strict_stage_replication": "pass" if not np.any(difference) else "fail",
        "overall_metric_equivalence": "not_assessed", "whole_recon_speedup": "not_measured",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"full_API_seconds": report["full_API"]["seconds"],
                      "different_voxels": report["different_voxels"], "cpu_helpers": helpers}), flush=True)
    if np.any(difference) or not report["affine_equal"] or not report["mgh_header_equal"]:
        raise RuntimeError("strict unchanged-output regression failed; report retained")


if __name__ == "__main__":
    main()
