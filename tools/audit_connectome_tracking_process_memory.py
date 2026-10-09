"""用本机实际基线报告审计一次完整 tracking 调用的单进程显存。

共享或配对 FOD 均可；全部 tracking 参数取自 SHA 绑定的参考报告。
不使用历史 GPU 的轨迹数或摘要作为当前设备 oracle，也不计为速度样本。
复用显式 SHA 绑定的旧审计模块 OwnProcessMonitor，绝不执行其 main。
只有本进程所选物理 GPU 的采样可进入门槛；采样不证明连续上界。
源文件、真实 NIfTI 和参考报告须先核对；报告不保存路径、PID 或 UUID。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


OUTPUT_FIELDS = ("accepted_streamlines", "total_path_points", "seeds_attempted",
                 "field_sha256", "output_sha256", "streamlines", "digest_format")
REQUIRED_OPTIONS = {"n_seeds", "lmax", "seed", "batch_size", "arc_proposals",
                    "max_length_mm", "min_length_mm", "step_mm", "max_angle_degrees",
                    "cutoff", "power", "compile_arc", "five_tissue_spacing_mm"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_argument(value):
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark-module", "candidate-tracking", "candidate-fod-module",
                 "reference-report", "monitor-module"):
        parser.add_argument("--" + name, type=Path, required=True,
                            help="本轮冻结的源码或本机基线报告")
        parser.add_argument("--" + name + "-sha256", type=sha256_argument, required=True,
                            help="上述文件的预先核验 SHA-256")
    for name in ("fod", "five-tissue", "gmwmi", "output"):
        parser.add_argument("--" + name, type=Path, required=True,
                            help="真实 NIfTI 输入；output 为无路径和标识符的 JSON 报告")
    parser.add_argument("--device", default="cuda:0", help="显式 CUDA 设备；不支持 CPU 显存验收")
    parser.add_argument("--memory-budget-gb", type=positive_float, default=20.,
                        help="十进制 GB，上限20；allocator 为该预算90%%，进程采样须严格小于预算")
    parser.add_argument("--sample-interval-seconds", type=positive_float, default=.5)
    parser.add_argument("--query-timeout-seconds", type=positive_float, default=3.)
    parser.add_argument("--max-sample-gap-seconds", type=positive_float, default=2.,
                        help="事先声明的最大采样起点间隔，含查询耗时")
    args = parser.parse_args(argv)
    if args.memory_budget_gb > 20:
        parser.error("--memory-budget-gb must be at most 20")
    if args.max_sample_gap_seconds < args.sample_interval_seconds:
        parser.error("--max-sample-gap-seconds must be at least the sample interval")
    bound_names = ("benchmark_module", "candidate_tracking", "candidate_fod_module", "reference_report",
                   "monitor_module", "fod", "five_tissue", "gmwmi")
    if args.output.expanduser().resolve() in {getattr(args, name).expanduser().resolve() for name in bound_names}:
        parser.error("--output cannot overwrite an input, source or reference report")
    return args


def load_module(name, path, expected_sha, functions):
    if file_sha256(path) != expected_sha:
        raise ValueError("bound module SHA mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("bound module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not all(callable(getattr(module, function, None)) for function in functions):
        raise ValueError("bound module helper unavailable")
    return module


def compact_output(output):
    if not all(key in output for key in OUTPUT_FIELDS):
        raise ValueError("reference needs full per-streamline byte digests")
    result = {key: output[key] for key in OUTPUT_FIELDS}
    count = result["accepted_streamlines"]
    if not isinstance(count, int) or count < 0 or len(result["streamlines"]) != count:
        raise ValueError("reference streamline digest count is inconsistent")
    if not {"point_counts", "packed_points", "accepted_seeds", "lengths_mm", "endpoints"}.issubset(result["field_sha256"]):
        raise ValueError("reference raw-byte field digests are incomplete")
    for digest in (result["output_sha256"], *result["field_sha256"].values(),
                   *(row["sha256"] for row in result["streamlines"])):
        sha256_argument(digest)
    return result


def validate_reference(reference, sources, input_files):
    """取当前设备同配置 baseline oracle；允许 baseline/candidate FOD SHA 不同。"""
    if reference.get("status") != "passed" or reference.get("all_strict_equal") is not True:
        raise ValueError("reference paired benchmark did not pass strict equality")
    recorded = reference.get("source_sha256", {})
    baseline_fod = recorded.get("baseline_fod", recorded.get("fixed_fod"))
    candidate_fod = recorded.get("candidate_fod", recorded.get("fixed_fod"))
    for key, expected in (("benchmark", sources["benchmark_module"]),
                          ("candidate_tracking", sources["candidate_tracking"])):
        if recorded.get(key) != expected:
            raise ValueError("reference source SHA mismatch")
    if candidate_fod != sources["candidate_fod_module"] or baseline_fod is None:
        raise ValueError("reference FOD SHA mismatch")
    sha256_argument(baseline_fod)
    sha256_argument(recorded.get("baseline_tracking", ""))
    mode = reference.get("fod_mode", "shared")
    if mode not in ("shared", "paired"):
        raise ValueError("reference FOD mode is unavailable")
    if mode == "shared" and (baseline_fod != candidate_fod or
                             reference.get("fixed_fod_function_identity_equal") is not True):
        raise ValueError("reference shared FOD binding is inconsistent")
    if mode == "paired":
        identities = reference.get("fod_function_identity", {})
        for variant in ("baseline", "candidate"):
            if not all(identities.get(variant, {}).get(name) is True
                       for name in ("real_sh", "tracking_sh_precomputed")):
                raise ValueError("reference paired FOD binding is unavailable")
    for name, actual in input_files.items():
        expected = reference.get("input_files", {}).get(name, {})
        if any(expected.get(key) != actual[key] for key in ("sha256", "size_bytes")):
            raise ValueError("reference real input SHA or size mismatch")
    options = reference.get("options", {})
    if set(options) != REQUIRED_OPTIONS or not isinstance(options["n_seeds"], int) or options["n_seeds"] < 1:
        raise ValueError("reference tracking options are incomplete")
    baselines = [run for run in reference.get("runs", [])
                 if run.get("variant") == "baseline" and run.get("phase") != "profile"
                 and run.get("n_seeds") == options["n_seeds"]]
    if not baselines:
        raise ValueError("reference lacks full-scale baseline output")
    oracle = compact_output(baselines[0].get("output", {}))
    if oracle["seeds_attempted"] != options["n_seeds"]:
        raise ValueError("reference seeds attempted mismatch")
    if any(compact_output(run.get("output", {})) != oracle for run in baselines[1:]):
        raise ValueError("reference baseline outputs differ")
    return dict(options), oracle, {"fod_mode": mode, "baseline_fod_sha256": baseline_fod,
                                  "candidate_fod_sha256": candidate_fod,
                                  "baseline_tracking_sha256": recorded.get("baseline_tracking"),
                                  "full_baseline_output_count": len(baselines)}


def validate_runtime(reference, runtime, device):
    """纯 metadata 检查；当前物理 GPU 和运行时必须与参考一致。"""
    for name, actual in runtime.items():
        if reference.get(name) != actual:
            raise ValueError("reference runtime differs")
    recorded = reference.get("device", {})
    normalize_uuid = lambda value: "GPU-" + str(value).removeprefix("GPU-")
    if (recorded.get("name") != device["name"] or
            recorded.get("total_memory_bytes") != device["total_memory_bytes"] or
            not device.get("uuid") or
            normalize_uuid(recorded.get("uuid")) != normalize_uuid(device["uuid"])):
        raise ValueError("reference baseline did not use this physical GPU")


def compare_output(oracle, actual):
    actual = compact_output(actual)
    fields = {key: actual[key] == oracle[key] for key in OUTPUT_FIELDS if key != "streamlines"}
    expected_paths, actual_paths = oracle["streamlines"], actual["streamlines"]
    paths_equal = expected_paths == actual_paths
    return {"all_equal": all(fields.values()) and paths_equal,
            "fields_equal": fields, "all_streamline_byte_digests_equal": paths_equal,
            "first_mismatched_streamline_indices": [index for index, (expected, observed)
                in enumerate(zip(expected_paths, actual_paths)) if expected != observed][:10],
            "method": "Same digest format, named dtype/shape/raw-byte field hashes and every streamline digest; no numeric tolerance."}


def memory_gate(peaks, process, *, allocator_cap, process_budget, execution_ok, output_ok, snapshot_ok):
    allocated = [row["peak_allocated_bytes"] for row in peaks.values() if row.get("peak_allocated_bytes") is not None]
    reserved = [row["peak_reserved_bytes"] for row in peaks.values() if row.get("peak_reserved_bytes") is not None]
    allocator_ok = bool(allocated and reserved and max(allocated) <= allocator_cap and max(reserved) <= allocator_cap)
    peak = process.get("whole_audit_selected_process_sampled_peak_bytes")
    process_ok = peak is not None and 0 < peak < process_budget
    coverage_ok = process.get("coverage_complete_under_declared_gap_limit") is True
    return {"torch_peak_allocated_bytes": max(allocated) if allocated else None,
            "torch_peak_reserved_bytes": max(reserved) if reserved else None,
            "sampled_own_process_peak_bytes": peak, "torch_allocator_within_cap": allocator_ok,
            "sampled_own_process_below_budget": process_ok,
            "sampling_coverage_complete": coverage_ok, "baseline_output_bytes_digest_matches": output_ok,
            "snapshot_d2h_completed": snapshot_ok, "execution_completed": execution_ok,
            "passed_sampled_full_process_memory_gate": bool(execution_ok and output_ok and snapshot_ok
                                                           and allocator_ok and process_ok and coverage_ok),
            "continuous_memory_upper_bound_proven": False}


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=output.name + ".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(output)


def main(argv=None):
    args = parse_args(argv)
    sys.dont_write_bytecode = True
    started = time.perf_counter()
    process_budget = int(args.memory_budget_gb * 1e9)
    allocator_cap = int(process_budget * .9)
    output = args.output.expanduser().resolve()
    report = {"schema_version": 1, "status": "running", "audit_only_not_abba": True,
              "speedup_assessed": False, "warmup_calls": 0, "tracking_calls": 0,
              "baseline_tracking_executed": False, "identifiers_and_paths_in_report": False,
              "process_memory_budget_bytes": process_budget, "torch_allocator_cap_bytes": allocator_cap,
              "source_sha256": {}, "input_files": {}, "torch_allocator_peaks": {}, "timing_seconds": {},
              "timing_note": "First full candidate call with sampling overhead; excluded from benchmark speedup samples."}
    monitor = benchmark = device = None
    cuda_initialized = False
    scope_start = scope_end = None
    error = None
    stage = "source_validation"
    try:
        for name in ("benchmark_module", "candidate_tracking", "candidate_fod_module", "reference_report", "monitor_module"):
            path = getattr(args, name).expanduser().resolve()
            if not path.is_file() or file_sha256(path) != getattr(args, name + "_sha256"):
                raise ValueError("bound source or reference file SHA mismatch")
            setattr(args, name, path)
            report["source_sha256"][name] = file_sha256(path)
        report["source_sha256"]["audit_script"] = file_sha256(Path(__file__))
        reference = json.loads(args.reference_report.read_text())
        paths = {name: getattr(args, name).expanduser().resolve() for name in ("fod", "five_tissue", "gmwmi")}
        phase_started = time.perf_counter()
        report["input_files"] = {name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                                 for name, path in paths.items()}
        report["timing_seconds"]["input_hashing"] = time.perf_counter() - phase_started
        options, oracle, provenance = validate_reference(reference, report["source_sha256"], report["input_files"])
        report["options"], report["reference_provenance"] = options, provenance
        report["expected_output_sha256"] = oracle["output_sha256"]
        benchmark = load_module("_tracking_memory_bound_benchmark", args.benchmark_module,
                                args.benchmark_module_sha256,
                                ("load_sources", "load_real_inputs", "run_once", "sync", "memory_stats", "output_summary"))
        monitor_module = load_module("_tracking_memory_bound_monitor", args.monitor_module,
                                    args.monitor_module_sha256, ("OwnProcessMonitor",))
        import torch
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("memory gate requires an explicit CUDA device")
        # Load only candidate sources; the second tracking object is never executed.
        modules, candidate_fod = benchmark.load_sources(args.candidate_tracking, args.candidate_tracking,
                                                       args.candidate_fod_module)
        report["candidate_fod_function_identity"] = {
            name: callable(getattr(candidate_fod, name, None)) and
                  getattr(modules["candidate"], name, None) is getattr(candidate_fod, name, None)
            for name in ("real_sh", "tracking_sh_precomputed")}
        if not all(report["candidate_fod_function_identity"].values()):
            raise ValueError("candidate FOD function binding differs")
        stage = "real_input_loading"
        phase_started = time.perf_counter()
        tensors, affines, spacing = benchmark.load_real_inputs(paths)
        report["timing_seconds"]["real_input_io"] = time.perf_counter() - phase_started
        if list(spacing) != options["five_tissue_spacing_mm"]:
            raise ValueError("reference 5TT spacing differs")
        report["input_shapes"] = {name: list(value.shape) for name, value in tensors.items()}
        if report["input_shapes"] != reference.get("input_shapes"):
            raise ValueError("reference input shapes differ")
        report["runtime"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                             "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                             "cpu_interop_threads": torch.get_num_interop_threads()}
        save_report(report, output)
        monitor = monitor_module.OwnProcessMonitor(args.sample_interval_seconds, args.query_timeout_seconds, started)
        monitor.start()
        stage = "cuda_initialization"
        torch.cuda.set_device(device)
        torch.cuda.init()
        cuda_initialized = True
        properties = torch.cuda.get_device_properties(device)
        identity = getattr(properties, "uuid", None)
        if identity is None or properties.total_memory < allocator_cap:
            raise ValueError("physical GPU identity or allocator capacity unavailable")
        normalize_uuid = lambda value: "GPU-" + str(value).removeprefix("GPU-")
        validate_runtime(reference, {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                                     "python_version": platform.python_version(), "cpu_threads": torch.get_num_threads(),
                                     "cpu_interop_threads": torch.get_num_interop_threads()},
                         {"name": properties.name, "total_memory_bytes": properties.total_memory, "uuid": identity})
        monitor.set_physical_device(normalize_uuid(identity))
        torch.backends.cuda.matmul.allow_tf32 = reference["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = reference["tf32_cudnn"]
        if reference.get("autocast") is not False or reference.get("input_dtype") != "float32" or reference.get("affine_dtype") != "float64":
            raise ValueError("reference precision strategy differs")
        torch.cuda.set_per_process_memory_fraction(allocator_cap / properties.total_memory, device)
        benchmark.sync(device)
        report["device"] = {"logical_device": str(device), "name": properties.name,
                            "total_memory_bytes": properties.total_memory, "same_physical_gpu_as_reference": True}
        report["precision"] = {name: reference[name] for name in ("tf32_matmul", "tf32_cudnn", "autocast", "input_dtype", "affine_dtype")}
        stage = "input_h2d"
        monitor.set_phase(stage)
        monitor.sample(marker="before_input_h2d")
        scope_start = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats(device)
        device_tensors = {name: value.to(device) for name, value in tensors.items()}
        device_affines = {name: value.to(device) for name, value in affines.items()}
        benchmark.sync(device)
        report["torch_allocator_peaks"]["input_h2d"] = benchmark.memory_stats(device)
        monitor.sample(marker="after_input_h2d")
        inputs = {"wm_sh": device_tensors["fod"], "fod_affine": device_affines["fod"],
                  "five_tissue": device_tensors["five_tissue"], "five_tissue_affine": device_affines["five_tissue"],
                  "gmwmi": device_tensors["gmwmi"]}
        stage = "tracking_and_snapshot_d2h"
        monitor.set_phase(stage)
        monitor.sample(marker="before_full_tracking")
        report["tracking_calls"] = 1
        record, snapshot = benchmark.run_once(modules["candidate"], "candidate", "full_memory_audit", 0,
                                               inputs, options, device, trace_path=None)
        scope_end = time.perf_counter() - started
        monitor.sample(marker="after_snapshot_d2h")
        report["torch_allocator_peaks"].update(tracking=record["tracking_memory"],
                                               tracking_and_snapshot_d2h=record["tracking_and_d2h_memory"])
        report["timing_seconds"].update({key: record[key] for key in ("tracking_seconds", "path_pack_seconds", "d2h_seconds")})
        report["snapshot_d2h_completed"] = True
        stage = "output_hash_validation"
        monitor.set_phase(stage)
        summary = benchmark.output_summary(snapshot)
        report["output_comparison"] = compare_output(oracle, summary)
        report["output"] = {key: summary[key] for key in OUTPUT_FIELDS if key != "streamlines"}
        monitor.sample(marker="after_output_hash")
    except BaseException as caught:
        error = caught
        report["execution_error"] = {"stage": stage, "type": type(caught).__name__,
                                     "message": "Details omitted to exclude filesystem paths and identifiers."}
    finally:
        if monitor is not None:
            if scope_start is not None and scope_end is None:
                scope_end = time.perf_counter() - started
            if error is not None:
                monitor.sample(marker="failure_final_sample")
            report["process_memory_monitor"] = monitor.summary(scope_start, scope_end,
                                                                 args.max_sample_gap_seconds, monitor.stop())
        if cuda_initialized:
            try:
                report["torch_allocator_peaks"]["at_audit_end"] = benchmark.memory_stats(device)
            except BaseException as caught:
                report["allocator_stats_error_type"] = type(caught).__name__
        try:
            report["bound_source_files_unchanged"] = all(
                file_sha256(getattr(args, name)) == getattr(args, name + "_sha256")
                for name in ("benchmark_module", "candidate_tracking", "candidate_fod_module", "reference_report", "monitor_module"))
        except (OSError, ValueError):
            report["bound_source_files_unchanged"] = False
        report["gate"] = memory_gate(report["torch_allocator_peaks"], report.get("process_memory_monitor", {}),
                                      allocator_cap=allocator_cap, process_budget=process_budget,
                                      execution_ok=error is None and report["bound_source_files_unchanged"],
                                      output_ok=report.get("output_comparison", {}).get("all_equal") is True,
                                      snapshot_ok=report.get("snapshot_d2h_completed") is True)
        passed = report["gate"]["passed_sampled_full_process_memory_gate"]
        report["status"] = "passed_sampled_memory_gate" if passed else "execution_failed" if error else "memory_gate_not_passed"
        report["timing_seconds"]["audit_wall_including_monitoring"] = time.perf_counter() - started
        save_report(report, output)
        print(json.dumps({"status": report["status"], "gate": report["gate"]}), flush=True)
    return 0 if passed else 3 if error is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
