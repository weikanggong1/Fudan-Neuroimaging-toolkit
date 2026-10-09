"""真实 NIfTI 输入上逐轨严格比较两个 tracking 源码，并交错计时。

两个 tracking 源码复用 --fod-module 指定的同一 FOD/SH 实现。
默认两轮配对的计时顺序为 baseline、candidate、candidate、baseline。
每版首次全量调用单列，不以小规模预热证明全量编译已完成。
输入读取、H2D、tracking、D2H、摘要及写盘分别计时；不生成模拟影像。
示例（输入路径须先由 FNIT 服务器索引及 SHA-256 核对）：

    python tools/benchmark_connectome_tracking_exact.py \\
      --baseline-tracking /path/to/frozen/tracking.py \\
      --candidate-tracking /path/to/candidate/tracking.py \\
      --fod-module /path/to/fixed/fod.py \\
      --fod /path/to/real/fod.nii.gz \\
      --five-tissue /path/to/real/5tt.nii.gz \\
      --gmwmi /path/to/real/gmwmi.nii.gz \\
      --n-seeds 1000 --batch-size 1000 --seed 0 --device cuda:0 \\
      --repeats 2 --output /path/to/run/tracking_exact.json --profile

--compile-arc 对两版一同生效；比较未编译实现时省略此参数。
--cold 单列每版首次调用，不能消除共享 FOD、CUDA 和编译缓存的顺序影响。
--save-tck 保存每次实际输出；TCK 转换不计入 tracking 或 D2H。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types

import nibabel as nib
import numpy as np
import torch

@dataclass
class Snapshot:
    """仅持有 CPU 输出；paths 为连续 packed_points 的视图。"""

    packed_points: torch.Tensor
    paths: tuple[torch.Tensor, ...]
    point_counts: torch.Tensor
    endpoints: torch.Tensor
    lengths_mm: torch.Tensor
    accepted_seeds: torch.Tensor
    mean_fa: torch.Tensor | None
    seeds_attempted: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Python source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sources(baseline_path: Path, candidate_path: Path, fod_path: Path):
    # 独立空 package 只加载这三个文件，不触发 FNIT __init__、vendor 或资产。
    package_name = "_fnit_tracking_exact"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    fod = load_source(package_name + ".fod", fod_path)
    package.fod = fod
    modules = {
        "baseline": load_source(package_name + ".baseline", baseline_path),
        "candidate": load_source(package_name + ".candidate", candidate_path),
    }
    for variant, module in modules.items():
        setattr(package, variant, module)
        if not callable(getattr(module, "probabilistic_tractography", None)):
            raise ValueError(f"{variant} tracking source lacks probabilistic_tractography")
        for function_name in ("tracking_sh_precomputed", "real_sh"):
            if getattr(module, function_name, None) is not getattr(fod, function_name, None):
                raise ValueError(f"{variant} did not reuse fixed FOD function {function_name}")
    baseline, candidate = modules["baseline"], modules["candidate"]
    for function_name in ("tracking_sh_precomputed", "real_sh"):
        if getattr(baseline, function_name, None) is not getattr(candidate, function_name, None):
            raise ValueError(f"baseline did not reuse current FOD function {function_name}")
    return modules, fod


def load_real_inputs(paths: dict[str, Path]):
    images = {name: nib.load(str(path)) for name, path in paths.items()}
    tensors = {
        name: torch.from_numpy(np.asarray(image.dataobj, dtype=np.float32).copy())
        for name, image in images.items()
    }
    affines = {
        name: torch.as_tensor(np.array(image.affine, copy=True), dtype=torch.float64)
        for name, image in images.items()
    }
    if tensors["fod"].ndim != 4 or tensors["fod"].shape[-1] != 45:
        raise ValueError("real FOD must be float32 [X,Y,Z,45] for lmax=8")
    if tensors["five_tissue"].ndim != 4 or tensors["five_tissue"].shape[-1] != 5:
        raise ValueError("real 5TT must be float32 [A,B,C,5]")
    if tensors["gmwmi"].shape != tensors["five_tissue"].shape[:3]:
        raise ValueError("GMWMI and 5TT must share one grid")
    if not torch.equal(affines["gmwmi"], affines["five_tissue"]):
        raise ValueError("GMWMI and 5TT must have exactly the same NIfTI affine")
    for name, affine in affines.items():
        if not bool(torch.isfinite(affine).all()):
            raise ValueError(f"{name} affine contains nonfinite values")
    spacing = tuple(float(value) for value in images["five_tissue"].header.get_zooms()[:3])
    return tensors, affines, spacing


def cpu_snapshot(tracks, device: torch.device) -> tuple[Snapshot, float, float]:
    # 一次打包路径后搬到 CPU，避免逐条路径分别触发 CUDA 同步。
    counts = tuple(path.shape[0] for path in tracks.paths)
    started = time.perf_counter()
    packed = (torch.cat(tracks.paths, dim=0) if tracks.paths else
              tracks.endpoints.new_empty((0, 3)))
    sync(device)
    pack_seconds = time.perf_counter() - started
    started = time.perf_counter()
    packed_cpu = packed.detach().to(device="cpu", copy=True).contiguous()
    result = Snapshot(
        packed_points=packed_cpu,
        paths=tuple(packed_cpu.split(counts)) if counts else (),
        point_counts=torch.tensor(counts, dtype=torch.int64),
        endpoints=tracks.endpoints.detach().to(device="cpu", copy=True).contiguous(),
        lengths_mm=tracks.lengths_mm.detach().to(device="cpu", copy=True).contiguous(),
        accepted_seeds=tracks.accepted_seeds.detach().to(device="cpu", copy=True).contiguous(),
        mean_fa=(None if tracks.mean_fa is None else
                 tracks.mean_fa.detach().to(device="cpu", copy=True).contiguous()),
        seeds_attempted=int(tracks.seeds_attempted),
    )
    sync(device)
    return result, pack_seconds, time.perf_counter() - started


def update_tensor_digest(digest, name: str, value: torch.Tensor) -> None:
    array = value.detach().contiguous().numpy()
    # 将 dtype、shape 和小端字节一起编码，区分同字节不同结构的张量。
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
    digest.update(header.encode("utf-8") + b"\0")
    if array.size:
        digest.update(memoryview(array).cast("B"))


def tensor_sha256(name: str, value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    update_tensor_digest(digest, name, value)
    return digest.hexdigest()


def output_summary(snapshot: Snapshot) -> dict:
    fields = ("point_counts", "packed_points", "accepted_seeds", "lengths_mm", "endpoints")
    field_hashes = {name: tensor_sha256(name, getattr(snapshot, name)) for name in fields}
    if snapshot.mean_fa is not None:
        field_hashes["mean_fa"] = tensor_sha256("mean_fa", snapshot.mean_fa)
    overall = hashlib.sha256()
    overall.update(json.dumps([snapshot.seeds_attempted, field_hashes],
                              sort_keys=True, separators=(",", ":")).encode("utf-8"))
    per_streamline = []
    for index, path in enumerate(snapshot.paths):
        digest = hashlib.sha256()
        update_tensor_digest(digest, "path", path)
        update_tensor_digest(digest, "seed", snapshot.accepted_seeds[index])
        update_tensor_digest(digest, "length_mm", snapshot.lengths_mm[index:index + 1])
        update_tensor_digest(digest, "endpoints", snapshot.endpoints[index])
        if snapshot.mean_fa is not None:
            update_tensor_digest(digest, "mean_fa", snapshot.mean_fa[index:index + 1])
        per_streamline.append({"point_count": len(path), "sha256": digest.hexdigest()})
    return {
        "accepted_streamlines": len(snapshot.paths),
        "total_path_points": len(snapshot.packed_points),
        "seeds_attempted": snapshot.seeds_attempted,
        "field_sha256": field_hashes,
        "output_sha256": overall.hexdigest(),
        "streamlines": per_streamline,
        "digest_format": "v1: named dtype/shape headers plus little-endian tensor bytes",
    }


def strict_compare(reference: Snapshot, actual: Snapshot) -> dict:
    path_count_equal = len(reference.paths) == len(actual.paths)
    mismatch_indices = []
    paths_equal = path_count_equal
    # 不使用误差阈值或只比较摘要：每一条实际路径都执行 torch.equal。
    for index, (expected, observed) in enumerate(zip(reference.paths, actual.paths)):
        equal = torch.equal(expected, observed)
        paths_equal &= equal
        if not equal and len(mismatch_indices) < 10:
            mismatch_indices.append(index)
    fields = {
        name: torch.equal(getattr(reference, name), getattr(actual, name))
        for name in ("point_counts", "accepted_seeds", "lengths_mm", "endpoints")
    }
    fields["paths"] = bool(paths_equal)
    fields["seeds_attempted"] = reference.seeds_attempted == actual.seeds_attempted
    fields["mean_fa"] = (
        reference.mean_fa is None and actual.mean_fa is None if reference.mean_fa is None
        or actual.mean_fa is None else torch.equal(reference.mean_fa, actual.mean_fa)
    )
    return {
        "assessed": True, "all_equal": all(fields.values()), "torch_equal": fields,
        "path_count_equal": path_count_equal,
        "first_mismatched_path_indices": mismatch_indices,
    }


def memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None,
                "peak_allocated_gb": None, "peak_reserved_gb": None}
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
            "peak_allocated_gb": allocated / 1e9, "peak_reserved_gb": reserved / 1e9}


def profile_top_events(profiler) -> list[dict]:
    events = list(profiler.key_averages())
    events.sort(key=lambda event: getattr(event, "self_device_time_total", 0.), reverse=True)
    return [{
        "name": event.key, "count": event.count,
        "cpu_seconds_inclusive": event.cpu_time_total / 1e6,
        "cpu_seconds_self": event.self_cpu_time_total / 1e6,
        "device_seconds_inclusive": getattr(event, "device_time_total", 0.) / 1e6,
        "device_seconds_self": getattr(event, "self_device_time_total", 0.) / 1e6,
    } for event in events[:60]]


def run_once(module, variant: str, phase: str, ordinal: int, inputs: dict,
             options: dict, device: torch.device, trace_path: Path | None = None):
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profiler = None
    if trace_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, record_shapes=False,
                                          profile_memory=True, with_stack=False)
    context = profiler if profiler is not None else nullcontext()
    with context:
        span = torch.profiler.record_function(f"tracking/{variant}") if profiler else nullcontext()
        with span, torch.inference_mode():
            sync(device)
            started = time.perf_counter()
            tracks = module.probabilistic_tractography(**inputs, **options)
            sync(device)
            tracking_seconds = time.perf_counter() - started
    tracking_memory = memory_stats(device)
    snapshot, pack_seconds, d2h_seconds = cpu_snapshot(tracks, device)
    memory_after_pack = memory_stats(device)
    del tracks
    record = {
        "variant": variant, "phase": phase, "ordinal": ordinal,
        "n_seeds": options["n_seeds"], "tracking_seconds": tracking_seconds,
        "d2h_seconds": d2h_seconds, "path_pack_seconds": pack_seconds,
        "tracking_memory": tracking_memory,
        "tracking_and_d2h_memory": memory_after_pack,
    }
    if profiler is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        profiler.export_chrome_trace(str(trace_path))
        record["profile_export_seconds"] = time.perf_counter() - started
        record["profile_trace"] = str(trace_path)
        record["profile_top_events"] = profile_top_events(profiler)
    return record, snapshot


def current_commit(source: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(source.parent), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("baseline-tracking", "fod", "five-tissue", "gmwmi", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-tracking", type=Path,
                        help="候选 tracking.py；默认当前仓库 src/fnit/connectome/tracking.py")
    parser.add_argument("--fod-module", type=Path,
                        help="两版共享的 fod.py；默认候选 tracking.py 的同目录 fod.py")
    parser.add_argument("--baseline-commit", help="可选：调用者已核对的基线 commit 标签")
    parser.add_argument("--candidate-commit", help="可选：调用者已核对的候选 commit 标签")
    parser.add_argument("--n-seeds", type=int, default=1000, help="真实 GMWMI 播种次数")
    parser.add_argument("--batch-size", type=int, default=8192, help="两版相同的并行播种批量")
    parser.add_argument("--seed", type=int, default=0, help="两版相同的 PyTorch 随机种子")
    parser.add_argument("--device", default="cuda:0", help="显式 CPU 或 CUDA 设备")
    parser.add_argument("--repeats", type=int, default=2,
                        help="热配对次数，交替 AB/BA；默认两次得到 ABBA")
    parser.add_argument("--warmup", type=int, default=1, help="每版额外预热调用次数")
    parser.add_argument("--warmup-seeds", type=int, default=128,
                        help="小规模预热播种数；0禁用预热；不计入全量计时")
    parser.add_argument("--cold", action="store_true", help="在预热前单列每版首次调用")
    parser.add_argument("--compile-arc", action="store_true", help="两版统一编译 arc 概率核")
    parser.add_argument("--memory-budget-gb", type=float, default=20.,
                        help="十进制 GB；Torch allocator 使用预算的90%%，其余留给上下文")
    parser.add_argument("--save-tck", type=Path, help="可选：逐次 TCK 输出目录")
    parser.add_argument("--profile", action="store_true", help="热计时后单列候选 profiler")
    parser.add_argument("--profile-seeds", "--profile-n-seeds", dest="profile_seeds",
                        type=int, default=128, help="候选 profiler 播种次数上限")
    parser.add_argument("--profile-dir", type=Path, help="默认 JSON 同目录下的 <stem>_profile")
    args = parser.parse_args()
    for name in ("n_seeds", "batch_size", "repeats", "profile_seeds"):
        if getattr(args, name) < 1:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if args.warmup < 0 or args.warmup_seeds < 0 or not 0 < args.memory_budget_gb <= 20:
        parser.error("warmup counts must be nonnegative and --memory-budget-gb must be in (0,20]")
    if args.candidate_tracking is None:
        args.candidate_tracking = Path(__file__).resolve().parents[1] / "src/fnit/connectome/tracking.py"
    if args.fod_module is None:
        args.fod_module = args.candidate_tracking.parent / "fod.py"
    for name in ("baseline_tracking", "candidate_tracking", "fod_module", "fod", "five_tissue", "gmwmi"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} is not an existing file: {path}")
        setattr(args, name, path)
    args.output = args.output.resolve()
    return args


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this benchmark supports CPU and CUDA devices")
    if args.compile_arc and device.type != "cuda":
        raise ValueError("--compile-arc requires CUDA")
    modules, fixed_fod = load_sources(args.baseline_tracking, args.candidate_tracking, args.fod_module)
    paths = {"fod": args.fod, "five_tissue": args.five_tissue, "gmwmi": args.gmwmi}
    started = time.perf_counter()
    input_hashes = {name: {"name": path.name, "size_bytes": path.stat().st_size,
                           "sha256": file_sha256(path)} for name, path in paths.items()}
    input_hashing_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tensors, affines, spacing = load_real_inputs(paths)
    io_seconds = time.perf_counter() - started
    allocator_budget = None
    device_metadata = {"device": str(device)}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = True
        properties = torch.cuda.get_device_properties(device)
        allocator_budget = min(args.memory_budget_gb * 1e9 * .9, properties.total_memory * .9)
        torch.cuda.set_per_process_memory_fraction(allocator_budget / properties.total_memory, device)
        device_metadata.update({"name": properties.name, "total_memory_bytes": properties.total_memory,
                                "uuid": str(getattr(properties, "uuid", "unavailable"))})
        sync(device)
    device_initialization_seconds = time.perf_counter() - started
    started = time.perf_counter()
    device_tensors = {name: value.to(device) for name, value in tensors.items()}
    device_affines = {name: value.to(device) for name, value in affines.items()}
    sync(device)
    h2d_seconds = time.perf_counter() - started
    inputs = {"wm_sh": device_tensors["fod"], "fod_affine": device_affines["fod"],
              "five_tissue": device_tensors["five_tissue"],
              "five_tissue_affine": device_affines["five_tissue"],
              "gmwmi": device_tensors["gmwmi"]}
    options = {"n_seeds": args.n_seeds, "lmax": 8, "five_tissue_spacing_mm": spacing,
               "seed": args.seed, "batch_size": args.batch_size, "arc_proposals": 16,
               "max_length_mm": 250., "min_length_mm": None, "step_mm": None,
               "max_angle_degrees": 45., "cutoff": .1, "power": .5,
               "compile_arc": args.compile_arc}
    source_paths = {"baseline_tracking": args.baseline_tracking,
                    "candidate_tracking": args.candidate_tracking,
                    "fixed_fod": Path(fixed_fod.__file__).resolve(),
                    "benchmark": Path(__file__).resolve()}
    report = {
        "schema_version": 1, "status": "running",
        "input_kind": "existing real NIfTI; no generated inputs",
        "input_files": input_hashes,
        "input_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "input_affines": {name: value.tolist() for name, value in affines.items()},
        "source_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
        "baseline_commit": args.baseline_commit,
        "candidate_commit": args.candidate_commit or current_commit(source_paths["candidate_tracking"]),
        "fixed_fod_function_identity_equal": True,
        "device": device_metadata, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "python_version": platform.python_version(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "autocast": False, "input_dtype": "float32", "affine_dtype": "float64",
        "options": options,
        "memory_budget_bytes": int(args.memory_budget_gb * 1e9),
        "torch_allocator_budget_bytes": int(allocator_budget) if allocator_budget is not None else None,
        "memory_note": "Torch allocated/reserved exclude other processes and CUDA context allocations.",
        "shared_input_timing": {"input_hashing_seconds": input_hashing_seconds,
                                "io_seconds": io_seconds, "h2d_seconds": h2d_seconds,
                                "device_initialization_seconds": device_initialization_seconds},
        "timing_note": "Tracking excludes I/O, H2D, D2H, hashing, comparison and TCK writes. Each variant first full call is separate unless a prior full cold/warmup call was made. Profile timings are excluded from hot summaries.",
        "cold_note": "Optional cold entries are each implementation's first call in this process; CUDA/FOD/Inductor caches are shared.",
        "warmup_seeds": args.warmup_seeds, "profile_seeds": args.profile_seeds,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    oracle = None
    full_calls = {variant: 0 for variant in modules}

    def save_report():
        temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)

    def execute(variant: str, phase: str, ordinal: int, run_options: dict,
                trace_path: Path | None = None, reference: Snapshot | None = None):
        record, output = run_once(modules[variant], variant, phase, ordinal,
                                  inputs, run_options, device, trace_path)
        started = time.perf_counter()
        record["output"] = output_summary(output)
        record["summary_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        record["strict_comparison"] = (strict_compare(reference, output) if reference is not None
                                       else {"assessed": False, "all_equal": None,
                                             "reason": "no same-scale reference output"})
        record["comparison_seconds"] = time.perf_counter() - started
        if args.save_tck is not None:
            args.save_tck.mkdir(parents=True, exist_ok=True)
            tck_path = args.save_tck / f"{phase}_{ordinal:03d}_{variant}.tck"
            started = time.perf_counter()
            nib.streamlines.save(nib.streamlines.Tractogram(
                [path.numpy() for path in output.paths], affine_to_rasmm=np.eye(4)), str(tck_path))
            record["tck_write_seconds"] = time.perf_counter() - started
            record["tck_path"] = str(tck_path.resolve())
            record["tck_sha256"] = file_sha256(tck_path)
        report["runs"].append(record)
        if run_options["n_seeds"] == args.n_seeds and phase != "profile":
            full_calls[variant] += 1
        save_report()
        print(json.dumps({"phase": phase, "variant": variant,
                          "tracking_seconds": record["tracking_seconds"],
                          "accepted_streamlines": len(output.paths),
                          "strict_equal": record["strict_comparison"]["all_equal"]}), flush=True)
        return output

    save_report()
    try:
        if args.cold:
            for variant in ("baseline", "candidate"):
                output = execute(variant, "cold", 0, options, reference=oracle)
                if oracle is None:
                    oracle = output
                else:
                    del output
        if args.warmup_seeds:
            warmup_reference = None
            warmup_options = dict(options, n_seeds=min(args.warmup_seeds, args.n_seeds))
            for ordinal in range(args.warmup):
                for variant in ("baseline", "candidate"):
                    output = execute(variant, "warmup", ordinal, warmup_options,
                                     reference=warmup_reference)
                    if warmup_reference is None:
                        warmup_reference = output
                    else:
                        del output
            del warmup_reference
        for repeat in range(args.repeats):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for variant in order:
                phase = "hot" if full_calls[variant] else "first_full_call"
                output = execute(variant, phase, repeat, options, reference=oracle)
                if oracle is None:
                    oracle = output
                else:
                    del output
        if args.profile:
            profile_dir = args.profile_dir or args.output.parent / (args.output.stem + "_profile")
            profile_options = dict(options, n_seeds=min(args.profile_seeds, args.n_seeds))
            profile_reference = oracle if profile_options["n_seeds"] == args.n_seeds else None
            output = execute("candidate", "profile", 0, profile_options,
                             trace_path=profile_dir / "candidate.chrome.json", reference=profile_reference)
            del output
        report["hot_order"] = [run["variant"] for run in report["runs"] if run["phase"] == "hot"]
        report["hot_summary"] = {}
        for variant in modules:
            hot = [run for run in report["runs"] if run["phase"] == "hot" and run["variant"] == variant]
            values = [run["tracking_seconds"] for run in hot]
            report["hot_summary"][variant] = {
                "tracking_seconds": values, "sample_count": len(values),
                "median_seconds": statistics.median(values) if values else None,
                "mean_seconds": statistics.mean(values) if values else None,
                "min_seconds": min(values) if values else None,
                "max_seconds": max(values) if values else None,
            }
        baseline_median = report["hot_summary"]["baseline"]["median_seconds"]
        candidate_median = report["hot_summary"]["candidate"]["median_seconds"]
        report["candidate_speedup_from_hot_medians"] = (
            baseline_median / candidate_median
            if baseline_median is not None and candidate_median is not None else None
        )
        assessed = [run for run in report["runs"] if run["strict_comparison"]["assessed"]]
        report["strict_comparison_count"] = len(assessed)
        report["all_strict_equal"] = (
            all(run["strict_comparison"]["all_equal"] for run in assessed) if assessed else None
        )
        report["status"] = ("no_comparison" if not assessed else
                            "passed" if report["all_strict_equal"] else "strict_equality_failed")
        save_report()
        print(json.dumps({"report": str(args.output), "status": report["status"],
                          "speedup": report["candidate_speedup_from_hot_medians"]}), flush=True)
        return 0 if report["all_strict_equal"] else 2
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save_report()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
