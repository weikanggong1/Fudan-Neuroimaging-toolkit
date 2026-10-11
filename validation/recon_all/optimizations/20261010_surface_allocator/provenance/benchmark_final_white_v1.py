"""完整自产 final-white 的 fresh-child CUDA 缓存对照；不修改生产或父策略。

输入为冻结 FNIT src、自产 subject、独立输出目录和显式 CUDA 设备。
复用已有 worker_cuda_cache_environment 与完整 place_final_white；所有子进程
使用同源、同输入、TF32、float32 和固定线程。支持 CLI 父进程与已经初始化
CUDA 的父 API 对照。输出私有 JSON/JSONL、独立表面和预处理 MRI；公共发布
须使用现有收据 exporter 脱敏，禁止发布原始影像/许可证/服务器地址。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256")
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def required_inputs(hemi: str) -> list[str]:
    return [f"surf/{hemi}.white.preaparc", f"surf/autodet.gw.stats.{hemi}.dat",
            "mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz",
            f"label/{hemi}.cortex.label", f"label/{hemi}.aparc.annot"]


def proc_io() -> dict:
    try:
        return {key: int(value) for key, value in
                (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}
    except OSError:
        return {"status": "unavailable"}


def cgroup_snapshot() -> dict:
    paths = ["/proc/self/cgroup", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat",
             "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
             "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "/sys/fs/cgroup/cpu/cpu.stat",
             "/sys/fs/cgroup/cpuset/cpuset.cpus"]
    result = {}
    for name in paths:
        path = Path(name)
        try:
            result[name] = path.read_text()
        except OSError as error:
            result[name] = {"status": "unavailable", "error": type(error).__name__}
    return result


def gpu_load() -> dict:
    try:
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True, timeout=4)
        return {"monotonic": time.monotonic(), "csv": value}
    except Exception as error:
        return {"monotonic": time.monotonic(), "error": repr(error)}


def selected_environment() -> dict:
    """只记录策略相关变量，不将令牌、许可证或其他环境写入收据。"""
    return {name: os.environ.get(name) for name in
            ("PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}


def process_snapshot(pid: int) -> dict:
    result = {"monotonic": time.monotonic(), "pid": pid}
    for name in ("io", "status", "wchan"):
        try:
            result[name] = Path(f"/proc/{pid}/{name}").read_text()
        except OSError as error:
            result[name] = {"status": "unavailable", "error": type(error).__name__}
    return result


def child(request: dict) -> int:
    started = time.perf_counter()
    sys.path.insert(0, request["source_root"])
    import numpy as np
    import nibabel as nib
    import torch
    from numba import set_num_threads
    from fnit.recon_all.profiling import configure_cuda_allocator, autocast_state
    from fnit.recon_all.place_final_white_python import place_final_white
    from fnit.recon_all.surface_placement_python import placement_options, _check_ordered_mesh
    result = {"status": "running", "pid": os.getpid(), "source_root": request["source_root"],
              "import_seconds": time.perf_counter() - started, "request": request,
              "input_sha256": {}, "driver_sha256": digest(Path(__file__))}
    destination = Path(request["report_path"])
    io_before = proc_io()
    try:
        threads = request["threads"]
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        set_num_threads(threads)
        allocator = configure_cuda_allocator(device=request["device"], policy=request["policy"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        subject = Path(request["subject"])
        for name, expected in request["input_sha256"].items():
            actual = digest(subject / name)
            if actual != expected:
                raise RuntimeError("input SHA changed: " + name)
            result["input_sha256"][name] = actual
        options = placement_options(device=request["device"])
        result.update(allocator=allocator, options=options, precision={
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "autocast": autocast_state("cuda"), "algorithm_coordinate_dtype": "float32"},
            cpu_affinity=sorted(os.sched_getaffinity(0)), threads=threads,
            versions={name: importlib.metadata.version(name) for name in
                      ("torch", "numpy", "scipy", "numba", "nibabel")})
        bootstrap = time.perf_counter()
        torch.cuda.set_device(request["device"])
        torch.cuda.synchronize(request["device"])
        result["cuda_bootstrap_seconds"] = time.perf_counter() - bootstrap
        result["gpu_name"] = torch.cuda.get_device_name(request["device"])
        result["gpu_uuid"] = str(torch.cuda.get_device_properties(request["device"]).uuid)
        if allocator["torch_stats_known_valid"]:
            torch.cuda.reset_peak_memory_stats(request["device"])
        accepted = []
        trace_overhead = 0.0
        trace_path = destination.parent / "accepted-steps.private.jsonl"
        trace_stream = trace_path.open("w")
        def trace(step, pass_index, coordinates, diagnostics):
            nonlocal trace_overhead
            tick = time.perf_counter()
            row = {"step": step, "pass_index": pass_index, "diagnostics": diagnostics,
                   "coordinates_sha256": hashlib.sha256(np.ascontiguousarray(coordinates).tobytes()).hexdigest(),
                   "coordinates_dtype": str(coordinates.dtype), "coordinates_shape": list(coordinates.shape)}
            accepted.append(row)
            trace_stream.write(json.dumps(row, allow_nan=False) + "\n")
            trace_stream.flush()
            trace_overhead += time.perf_counter() - tick
        api_tick = time.perf_counter()
        output = subject / "surf" / f"{request['hemi']}.white"
        output_volume = subject / "mri/mrisps.white.mgz"
        stage = place_final_white(subject_dir=subject, hemi=request["hemi"], output=output,
            output_volume=output_volume, max_steps=400, trace_callback=trace,
            collision_profile=True, **options)
        trace_stream.close()
        vertices, faces = _check_ordered_mesh(subject / f"surf/{request['hemi']}.white.preaparc", output)
        # 输出写入与目标 GPU 同步均在完整 API 墙钟内，统计/最终哈希另计。
        write_json(subject / "scripts" / f"{request['hemi']}.white.python.json", stage)
        sync_tick = time.perf_counter()
        torch.cuda.synchronize(request["device"])
        result["cuda_post_sync_seconds"] = time.perf_counter() - sync_tick
        result["complete_api_seconds"] = time.perf_counter() - api_tick
        xyz, triangle, metadata = nib.freesurfer.read_geometry(str(output), read_metadata=True)
        result.update(status="complete", stage=stage, accepted_steps=accepted,
            trace_sink_seconds=trace_overhead, vertices=vertices, faces=faces,
            output_sha256={"white": digest(output), "mrisps.white": digest(output_volume)},
            coordinate_sha256=hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest(),
            ordered_faces_sha256=hashlib.sha256(np.ascontiguousarray(triangle).tobytes()).hexdigest(),
            finite_coordinates=bool(np.isfinite(xyz).all()),
            surface_metadata={k: v.tolist() if hasattr(v, "tolist") else v for k, v in metadata.items()})
        result["torch_memory"] = ({"status": "available", "allocated_bytes": int(torch.cuda.memory_allocated(request["device"])),
            "reserved_bytes": int(torch.cuda.memory_reserved(request["device"])),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(request["device"])),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(request["device"]))}
            if allocator["torch_stats_known_valid"] else {"status": "unavailable_allocator_disabled",
                "allocated_bytes": None, "reserved_bytes": None,
                "peak_allocated_bytes": None, "peak_reserved_bytes": None})
    except Exception as error:
        result.update(status="failed", error=repr(error))
    result["child_main_seconds"] = time.perf_counter() - started
    result["io_before"], result["io_after"] = io_before, proc_io()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["rusage"] = {key: getattr(usage, key) for key in
                         ("ru_utime", "ru_stime", "ru_maxrss", "ru_minflt", "ru_majflt", "ru_inblock", "ru_oublock")}
    write_json(destination, result)
    return 0 if result["status"] == "complete" else 1


def driver(args) -> int:
    start = args.main_started_at
    source_root = args.source_root.resolve()
    subject = args.subject.resolve()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    policies = args.order.split(",")
    if any(policy not in {"enabled", "disabled"} for policy in policies):
        raise ValueError("order must contain enabled/disabled only")
    if not (source_root / "fnit/recon_all/surface_placement_python.py").is_file():
        raise FileNotFoundError("source-root must be frozen src")
    inputs = {name: digest(subject / name) for name in required_inputs(args.hemi)}
    source_hashes = {str(path.relative_to(source_root)): digest(path)
                     for path in sorted(source_root.rglob("*.py"))}
    output_root.mkdir(parents=True)
    proof = {"status": "prepared", "scope": "frozen same-input complete final-white; not raw T1 or whole recon-all",
        "source_root": str(source_root), "subject": str(subject), "input_sha256": inputs,
        "source_py_sha256": source_hashes, "source_file_count": len(source_hashes),
        "driver_sha256": digest(Path(__file__)), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads, "device": args.device, "gpu_load_at_prepare": gpu_load(),
        "cgroup_at_prepare": cgroup_snapshot(), "precision": "TF32 allowed, existing CPU regularization/sampling; no half precision",
        "overall_metric_equivalence": "not_assessed", "whole_pipeline_speedup": None}
    write_json(output_root / "manifest.private.json", proof)
    if args.prepare_only:
        return 0
    sys.path.insert(0, str(source_root))
    import torch
    import numpy as np
    import nibabel as nib
    from fnit.recon_all.hemisphere_parallel import worker_cuda_cache_environment
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler, configure_cuda_allocator
    parent_environment = dict(os.environ)
    def run_one(index, policy, mode):
        tick = time.perf_counter()
        run_dir = output_root / f"{index:02d}-{mode}-{policy}"
        run_dir.mkdir()
        own_subject = run_dir / "subject"
        for name, expected in inputs.items():
            target = own_subject / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(subject / name, target)
            if digest(target) != expected:
                raise RuntimeError("copied input SHA changed")
        copy_seconds = time.perf_counter() - tick
        env, selection = worker_cuda_cache_environment(os.environ, cuda_allocator_cache=policy)
        env.update(PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(run_dir / "numba-cache"),
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads), NUMBA_NUM_THREADS=str(args.threads))
        request = {"source_root": str(source_root), "subject": str(own_subject), "hemi": args.hemi,
            "device": args.device, "threads": args.threads, "policy": policy,
            "input_sha256": inputs, "report_path": str(run_dir / "child.private.json")}
        request_path = run_dir / "request.private.json"
        write_json(request_path, request)
        command = [sys.executable, str(Path(__file__).resolve()), "--child-request", str(request_path)]
        log_stream = (run_dir / "child.log").open("w")
        process_start = time.perf_counter()
        proc = subprocess.Popen(command, env=env, stdout=log_stream, stderr=subprocess.STDOUT)
        sampler = ProcessTreeDeviceSampler(device=args.device, parent_pid=os.getpid(), interval=args.interval)
        sampler.add_worker(proc.pid)
        loads, process_samples = [], []
        next_load = 0.0
        while proc.poll() is None:
            sampler.sample_if_due()
            if time.monotonic() >= next_load:
                loads.append(gpu_load())
                process_samples.append(process_snapshot(proc.pid))
                next_load = time.monotonic() + 5.0
            time.sleep(.1)
        exec_wall = time.perf_counter() - process_start
        sampler.sample_if_due(force=True)
        log_stream.close()
        report = json.loads(Path(request["report_path"]).read_text()) if Path(request["report_path"]).exists() else {"status": "missing"}
        row = {"index": index, "mode": mode, "policy": policy, "returncode": proc.returncode,
            "copy_and_validate_seconds": copy_seconds, "fresh_exec_wall_seconds": exec_wall,
            "complete_item_wall_seconds": time.perf_counter() - tick,
            "child_selection": selection, "child": report, "sampler": sampler.report(),
            "gpu_load_curve": loads, "process_samples": process_samples, "cgroup_after": cgroup_snapshot(),
            "parent_environment_unchanged": dict(os.environ) == parent_environment}
        write_json(run_dir / "benchmark.private.json", row)
        return row
    results = []
    for index, policy in enumerate(policies, 1):
        results.append(run_one(index, policy, "cli"))
        if results[-1]["returncode"]:
            break
    parent_api = None
    if all(row["returncode"] == 0 for row in results) and args.api_policy:
        parent_policy = configure_cuda_allocator(device=args.device, policy="disabled")
        live = torch.full((8 * 1024 * 1024,), 1.0, device=args.device, dtype=torch.float32)
        torch.cuda.synchronize(args.device)
        full_parent_environment = dict(os.environ)
        before = {"environment": selected_environment(), "value_sum": float(live.sum().item()),
                  "live_data_ptr": live.data_ptr(), "allocator": parent_policy}
        parent_environment = dict(os.environ)
        results.append(run_one(len(results) + 1, args.api_policy, "initialized-api"))
        torch.cuda.synchronize(args.device)
        after = {"environment": selected_environment(), "value_sum": float(live.sum().item()), "live_data_ptr": live.data_ptr()}
        parent_api = {"parent_initialized": True, "live_tensor_bytes": live.numel() * live.element_size(),
                      "before": before, "after": after, "live_tensor_and_environment_unchanged":
                      full_parent_environment == dict(os.environ) and before["value_sum"] == after["value_sum"]
                      and before["live_data_ptr"] == after["live_data_ptr"],
                      "torch_memory_stats_status": "unavailable_allocator_disabled"}
    complete = all(row["returncode"] == 0 and row["child"]["status"] == "complete" for row in results)
    comparisons = []
    if complete:
        first = results[0]
        before, before_faces = nib.freesurfer.read_geometry(str(Path(first["child"]["request"]["subject"]) / f"surf/{args.hemi}.white"))
        for row in results[1:]:
            after, after_faces = nib.freesurfer.read_geometry(str(Path(row["child"]["request"]["subject"]) / f"surf/{args.hemi}.white"))
            ordered = np.array_equal(before_faces, after_faces) and before.shape == after.shape
            error = np.linalg.norm(after - before, axis=1) if ordered else None
            comparisons.append({"first_index": first["index"], "candidate_index": row["index"],
                "same_ordered_mesh": ordered, "different_coordinate_elements": int(np.count_nonzero(after != before)) if ordered else None,
                "coordinate_distance_mean_mm": float(error.mean()) if ordered else None,
                "coordinate_distance_p99_mm": float(np.quantile(error, .99)) if ordered else None,
                "coordinate_distance_max_mm": float(error.max()) if ordered else None,
                "white_and_volume_bytes_equal": first["child"]["output_sha256"] == row["child"]["output_sha256"],
                "complete_accepted_trajectory_equal": first["child"]["accepted_steps"] == row["child"]["accepted_steps"],
                "per_step_accept_reject_equal": first["child"]["stage"]["per_step"] == row["child"]["stage"]["per_step"]})
    summary = {**proof, "status": "complete" if complete else "failed", "items": results,
        "comparisons": comparisons, "initialized_parent_api": parent_api,
        "full_driver_wall_seconds": time.perf_counter() - start,
        "timing_scope": "main starts after initial stdlib imports; includes parameter parsing/validation/hash/copy, each fresh Python startup/import/JIT, API transfers/read/write, sampler and final diagnostics; parent interpreter initial import excluded",
        "memory_limit_bytes": 20_000_000_000, "memory_limit_verdict": "unassessed_when_process_ownership_unresolved",
        "rejected_trial_coordinate_capture": "not captured; accepted-step coordinates and every scalar trial decision recorded",
        "sharing": "other processes retained; full load curves recorded; stage observations do not establish whole-pipeline speedup"}
    write_json(output_root / "summary.private.json", summary)
    return 0 if complete else 1


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-request", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--subject", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hemi", choices=("lh", "rh"), default="lh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("--order", default="enabled,disabled,disabled,enabled")
    parser.add_argument("--api-policy", choices=("enabled", "disabled"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.main_started_at = started
    if args.child_request:
        return child(json.loads(args.child_request.read_text()))
    if not all((args.source_root, args.subject, args.output_dir)):
        parser.error("--source-root, --subject and --output-dir are required")
    if args.threads < 1 or args.interval <= 0 or not args.device.startswith("cuda:"):
        parser.error("positive threads/interval and explicit cuda:N required")
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
