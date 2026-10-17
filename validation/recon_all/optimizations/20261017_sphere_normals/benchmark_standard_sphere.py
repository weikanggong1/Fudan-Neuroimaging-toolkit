"""两例双侧自产标准球面：Numba/CUDA法向同输入ABBA及独立完整轨迹。

只调用已有run_standard_sphere，不修改生产算法。两个后端都使用CUDA有序
平均和完整dense CUDA收尾，仅normals_device不同；每次独立解释器，4线程。
timing无observer；trace另跑完整阶段，不将其墙钟混入ABBA。输出目录须不存在。
完整阶段墙钟含函数内部读写/JIT/搬运，并在其前后同步目标CUDA；另记包含
解释器启动、manifest校验及质量检查的父进程墙钟。显存不可归属时保持未知。
保存后质量检查在CUDA不可见的新解释器中调用同一inspect_surface；原来的
FP32/FP64和结构判据不变。子检查失败立即失败，不自动重试或补写旧报告。

freeze子命令为冻结源码生成完整src/fnit/*.py SHA清单；prepare子命令只
从两份成功生产benchmark.json的同级subject/surf读取双侧inflated/smoothwm。
benchmark、源码和输入清单须由协调者核对；不接受官方检查点/atlas路径。
所有表面坐标为surface RAS/mm，有序面为整数(F,3)，不重跑inflation。
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


BASELINE = "5ca07f999375d02281ab60589cb2840e7b488790"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    """保留JSON中非有限诊断的含义，不把未知/NaN写成零。"""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def regular(path):
    """拒绝任意路径组件上的符号链接，避免参考视图或动态替换被当冻结输入。"""
    path = Path(os.path.abspath(path))
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("frozen input/source must not traverse symlinks: " + str(path))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_bound(path, expected):
    path = regular(path)
    if sha(path) != expected:
        raise ValueError("manifest SHA changed: " + str(path))
    return json.loads(path.read_text())


def freeze(args):
    """source为仓库根，commit为实际冻结40位提交；输出完整Python源码清单。"""
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.commit) != 40 or any(c not in "0123456789abcdef" for c in args.commit):
        raise ValueError("commit must be the actual full lowercase Git commit")
    root = args.source.absolute()
    paths = sorted((root / "src/fnit").rglob("*.py"))
    if not paths:
        raise ValueError("source must contain src/fnit")
    files = {str(regular(p).relative_to(root)): sha(p) for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         text=True, capture_output=True, check=False)
    actual_git_head = git.stdout.strip() if git.returncode == 0 else None
    if actual_git_head is not None and actual_git_head != args.commit:
        raise ValueError("declared commit differs from actual Git HEAD")
    save(args.output, {"schema": "fnit-sphere-source-v1", "commit": args.commit,
                       "actual_git_head": actual_git_head,
                       "git_provenance": "HEAD_checked; source SHA controls actual content" if actual_git_head else "archive_commit_declared_by_coordinator; actual content bound by complete SHA manifest",
                       "files": files, "scope": "all Python files under src/fnit"})
    print(json.dumps({"manifest": str(args.output), "sha256": sha(args.output),
                      "files": len(files)}))


def verify_source(root, manifest, expected):
    value = read_bound(manifest, expected)
    if value.get("schema") != "fnit-sphere-source-v1" or not value.get("files"):
        raise ValueError("unsupported or empty source manifest")
    root = Path(os.path.abspath(root))
    actual = {str(p.relative_to(root)) for p in (root / "src/fnit").rglob("*.py")}
    if actual != set(value["files"]):
        raise ValueError("source manifest coverage changed")
    for name, expected_sha in value["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("src", "fnit"):
            raise ValueError("invalid source manifest member")
        if sha(regular(root / relative)) != expected_sha:
            raise ValueError("source SHA changed: " + name)
    return value


def prepare(args):
    """恰好两份成功生产benchmark→四份同输入清单；完整报告及输入只读。"""
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.benchmark) != 2:
        raise ValueError("exactly two completed production benchmarks required")
    cases, names = [], set()
    for filename in args.benchmark:
        path = regular(filename)
        record = json.loads(path.read_text())
        if record.get("execution") != "complete" or record.get("exit_code") != 0:
            raise ValueError("producer must have successful complete benchmark: " + str(path))
        subject = path.parent / "subject"
        runtime_path = regular(subject / "fnit-native-free-run.json")
        runtime = json.loads(runtime_path.read_text())
        if runtime.get("status") != "complete":
            raise ValueError("producer runtime is incomplete")
        case = path.parent.name
        if case in names:
            raise ValueError("case directory names must be distinct")
        names.add(case)
        for hemi in ("lh", "rh"):
            inputs = {name: regular(subject / "surf" / f"{hemi}.{name}")
                      for name in ("inflated", "smoothwm")}
            cases.append({"case": case, "hemisphere": hemi,
                          "benchmark": str(path), "benchmark_sha256": sha(path),
                          "producer_code_version": record.get("code_version"),
                          "raw_t1_sha256": record.get("input_sha256"),
                          "runtime": str(runtime_path), "runtime_sha256": sha(runtime_path),
                          "inputs": {name: {"path": str(p), "sha256": sha(p)}
                                     for name, p in inputs.items()}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, {"schema": "fnit-sphere-production-inputs-v1", "cases": cases,
                       "scope": "two successful FNIT production subjects, own inflated/smoothwm only"})
    print(json.dumps({"manifest": str(args.output), "sha256": sha(args.output),
                      "hemisphere_inputs": len(cases)}))


def verify_inputs(path, expected):
    value = read_bound(path, expected)
    cases = value.get("cases", [])
    if value.get("schema") != "fnit-sphere-production-inputs-v1" or len(cases) != 4:
        raise ValueError("exactly two production subjects and four hemispheres required")
    identities = {(r["case"], r["hemisphere"]) for r in cases}
    names = {r["case"] for r in cases}
    if len(names) != 2 or identities != {(c, h) for c in names for h in ("lh", "rh")}:
        raise ValueError("duplicated or incomplete hemisphere inputs")
    for row in cases:
        if Path(row["case"]).name != row["case"] or row["case"] in {".", ".."}:
            raise ValueError("unsafe case basename")
        benchmark = read_bound(row["benchmark"], row["benchmark_sha256"])
        runtime = read_bound(row["runtime"], row["runtime_sha256"])
        if benchmark.get("execution") != "complete" or benchmark.get("exit_code") != 0 or runtime.get("status") != "complete":
            raise ValueError("producer completion changed")
        expected_subject = Path(row["benchmark"]).parent / "subject"
        if Path(row["runtime"]) != expected_subject / "fnit-native-free-run.json":
            raise ValueError("runtime does not belong to the declared producer")
        for name in ("inflated", "smoothwm"):
            binding = row["inputs"][name]
            required = expected_subject / "surf" / f"{row['hemisphere']}.{name}"
            if Path(binding["path"]) != required or sha(regular(required)) != binding["sha256"]:
                raise ValueError("production input path/SHA changed")
    return value


def array_sha(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def observe_complete_sphere(operation):
    """只读sys trace覆盖每次真实更新及全部dense收尾；不替换任何计算函数。"""
    from fnit.recon_all.mris_register_overlap import remove_overlap_sphere
    run = operation.run_standard_sphere
    finish = inspect.unwrap(remove_overlap_sphere)
    search_code = operation.first_epoch_line_search.__code__

    def line_number(function, marker):
        lines, begin = inspect.getsourcelines(function)
        found = [begin + i for i, text in enumerate(lines) if text.lstrip().startswith(marker)]
        if len(found) != 1:
            raise ValueError("unable to locate readonly trace marker: " + marker)
        return found[0]

    update_line = line_number(run, "updates.append(")
    finish_line = line_number(finish, "old_count = count")
    updates, cleanups, search_inputs = [], [], []

    def observer(frame, event, arg):
        if event == "call" and frame.f_code is search_code:
            values = frame.f_locals
            search_inputs.append({"coordinates_before_fp32_sha256": array_sha(values["vertices"]),
                                  "gradient_before_search_fp32_sha256": array_sha(values["gradient"])})
            return None
        if event != "line":
            return observer if frame.f_code in (run.__code__, finish.__code__) else None
        values = frame.f_locals
        if frame.f_code is run.__code__ and frame.f_lineno == update_line:
            if not updates or updates[-1]["index"] != values["index"]:
                row = {key: plain(values[key]) for key in (
                    "index", "stage", "weight", "averages", "steps_at_scale",
                    "prior_scale", "next_scale", "ending_sse", "search")}
                row.update(coordinates_fp32_sha256=array_sha(values["xyz"]),
                           gradient_fp32_sha256=array_sha(values["gradient"]))
                if len(search_inputs) != len(updates) + 1:
                    raise RuntimeError("missing or duplicated real line-search input trace")
                row.update(search_inputs[-1])
                updates.append(row)
        elif frame.f_code is finish.__code__ and frame.f_lineno == finish_line:
            row = {key: plain(values[key]) for key in (
                "iteration", "count", "dt", "max_neighbors", "min_negative",
                "min_iteration", "last_expand", "same")}
            row.update(negative_mask_sha256=array_sha(values["negative"]),
                       marked_sha256=array_sha(values["marked"]),
                       marked_displacement_sha256=array_sha(values["displacement"][values["marked"]]),
                       coordinates_fp32_sha256=array_sha(values["xyz"]))
            cleanups.append(row)
        return observer if frame.f_code in (run.__code__, finish.__code__) else None

    return observer, updates, cleanups


def inspect_surface(path, inflated):
    """独立检查保存后的有序面、几何头、闭合流形结构及FP32/FP64翻折。"""
    import nibabel.freesurfer.io as fsio
    import numpy as np
    import torch
    from fnit.recon_all.native_free import _native_final_mesh_structure
    from fnit.recon_all.mris_register_nonlinear import face_area_normals
    xyz, faces, metadata = fsio.read_geometry(str(path), read_metadata=True)
    ix, original_faces, original_metadata = fsio.read_geometry(str(inflated), read_metadata=True)
    xyz32 = np.asarray(xyz, np.float32)
    area32, _ = face_area_normals(torch.from_numpy(xyz32), torch.from_numpy(faces.astype(np.int64)), signed_sphere=True)
    corners = xyz.astype(np.float64)[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    signed = np.sum(corners.sum(axis=1) * cross, axis=1)
    area = np.linalg.norm(cross, axis=1) * 0.5
    structure = _native_final_mesh_structure(xyz, faces)
    metadata_equal = {key: bool(np.array_equal(metadata.get(key), original_metadata.get(key)))
                      for key in set(metadata) | set(original_metadata)}
    quality = {"vertices": len(xyz), "faces": len(faces),
               "ordered_faces_equal_input": bool(np.array_equal(faces, original_faces) and len(xyz) == len(ix)),
               "volume_geometry_equal": metadata_equal, "structure": structure,
               "fp32_negative_faces": int((area32 < 0).sum()),
               "fp32_saved_surface_predicate_backend": "existing face_area_normals on CPU float32; actual CUDA stop predicate also retained in API overlap diagnostics",
               "fp64_negative_faces": int(np.count_nonzero(signed < 0)),
               "fp64_negative_area_mm2": float(area[signed < 0].sum()),
               "coordinates_fp32_sha256": array_sha(xyz32), "faces_sha256": array_sha(faces),
               "sha256": sha(path), "all_finite": bool(np.isfinite(xyz).all()),
               "scope": "saved surface: closed connected genus0/manifold/orientation/zero-area and two radial predicates; no general triangle collision test"}
    quality["passed"] = bool(quality["ordered_faces_equal_input"] and all(metadata_equal.values())
                              and structure["status"] == "passed" and quality["all_finite"]
                              and quality["fp32_negative_faces"] == quality["fp64_negative_faces"] == 0)
    return quality


def quality_worker(args):
    """绑定的保存球面→独立CPU原判据JSON；无GPU、重建、重试或门槛修改。

    args.payload及payload_sha256指向本次GPU worker的冻结JSON及SHA；
    surface_sha256绑定其已保存球面。输出固定为该worker目录中的
    saved_quality.private.json，须不存在。完整源码/输入清单前后复验，
    4个CPU线程、interop1，坐标仍为surface RAS/mm。失败保留phase/日志，
    非零退出；旧SIGSEGV不被改为成功。该质量步骤没有独立官方CLI。
    """
    import faulthandler
    faulthandler.enable(all_threads=True)
    payload = read_bound(args.payload, args.payload_sha256)
    output = Path(payload["output"]) / "saved_quality.private.json"
    if output.exists():
        raise FileExistsError(output)
    report = {"status": "running", "phase": "validation", "payload_sha256": args.payload_sha256,
              "surface_sha256": args.surface_sha256, "tool_sha256": sha(__file__),
              "execution": "fresh_exec_CPU_original_predicates", "started_monotonic": time.monotonic()}
    save(output, report)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("saved quality process must hide CUDA explicitly")
        verify_source(payload["source"], payload["source_manifest"], payload["source_manifest_sha256"])
        inputs = verify_inputs(payload["inputs_manifest"], payload["inputs_manifest_sha256"])
        if payload["row"] not in inputs["cases"]:
            raise ValueError("quality input is not in frozen production manifest")
        surface = regular(Path(payload["output"]) / (payload["row"]["hemisphere"] + ".sphere"))
        if sha(surface) != args.surface_sha256:
            raise ValueError("saved sphere changed before quality check")
        sys.path.insert(0, str(Path(payload["source"]) / "src"))
        import numba
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        numba.set_num_threads(4)
        report.update(phase="original_inspect_surface_running", threads={"torch": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(), "numba": numba.get_num_threads()},
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cuda_initialized=torch.cuda.is_initialized())
        save(output, report)
        report["quality"] = inspect_surface(surface, payload["row"]["inputs"]["inflated"]["path"])
        if sha(surface) != args.surface_sha256:
            raise ValueError("saved sphere changed during quality check")
        verify_source(payload["source"], payload["source_manifest"], payload["source_manifest_sha256"])
        verify_inputs(payload["inputs_manifest"], payload["inputs_manifest_sha256"])
        report.update(status="complete" if report["quality"]["passed"] else "failed_quality",
                      phase="original_inspect_surface_returned")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    report["ended_monotonic"] = time.monotonic()
    save(output, report)
    return 0 if report["status"] == "complete" else 1


def inspect_surface_fresh_process(*, payload_path, payload_sha256, python_interpreter):
    """新CPU解释器调用原QA并校验子报告绑定，返回(quality,receipt)。

    三个具名输入为本次worker的JSON路径/SHA及同一解释器路径。新QA报告
    和日志写在payload.output，不修改输入/旧报告。非零退出、缺少或不匹配
    的QA证据均抛RuntimeError；计时仅计外层验证，不纳入sphere API时间。
    """
    payload = read_bound(payload_path, payload_sha256)
    output = Path(payload["output"])
    report_path = output / "saved_quality.private.json"
    log_path = output / "saved_quality.log"
    if report_path.exists() or log_path.exists():
        raise FileExistsError("preserve previous saved-quality evidence")
    surface_sha256 = sha(regular(output / (payload["row"]["hemisphere"] + ".sphere")))
    command = [str(python_interpreter), "-X", "faulthandler", "-u", str(Path(__file__).absolute()),
               "quality-worker", "--payload", str(Path(payload_path).absolute()),
               "--payload-sha256", payload_sha256, "--surface-sha256", surface_sha256]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
    started = time.perf_counter()
    with log_path.open("x") as stream:
        process = subprocess.run(command, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    receipt = {"exit_code": process.returncode, "scope": "fresh exec CPU QA; outside GPU API time",
               "report": str(report_path), "log": str(log_path), "log_sha256": sha(log_path),
               "whole_quality_process_wall_seconds": time.perf_counter() - started,
               "report_sha256": sha(report_path) if report_path.exists() else None}
    save(output / "saved_quality_receipt.private.json", receipt)
    if process.returncode != 0 or not report_path.is_file():
        raise RuntimeError("saved-surface CPU quality process failed; preserve child receipt")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "complete" or report.get("payload_sha256") != payload_sha256
            or report.get("surface_sha256") != surface_sha256 or report.get("tool_sha256") != sha(__file__)
            or report.get("cuda_visible_devices") != "" or report.get("cuda_initialized") is not False
            or report.get("quality", {}).get("passed") is not True
            or sha(regular(output / (payload["row"]["hemisphere"] + ".sphere"))) != surface_sha256):
        raise RuntimeError("saved-surface QA evidence is incomplete or binding changed")
    return report["quality"], receipt


def worker(args):
    """一个独立冷进程执行一个完整sphere；失败也保存现场和非零退出。"""
    import faulthandler
    faulthandler.enable(all_threads=True)
    started = time.perf_counter()
    payload = read_bound(args.payload, args.payload_sha256)
    output = Path(payload["output"])
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    report_path = output / "report.private.json"
    report = {"status": "running", "mode": payload["mode"], "arm": payload["arm"],
              "case": payload["row"]["case"], "hemisphere": payload["row"]["hemisphere"],
              "payload_sha256": args.payload_sha256, "benchmark_script_sha256": sha(__file__),
              "python_version": sys.version, "python_executable": sys.executable,
              "python_executable_sha256": sha(Path(sys.executable).resolve()),
              "phase": "source_and_input_validation",
              "numba_boundscheck_environment": os.environ.get("NUMBA_BOUNDSCHECK"),
              "timing_qualification": "diagnostic_not_performance_sample" if os.environ.get("NUMBA_BOUNDSCHECK") == "1" else "normal",
              "overall_metric_equivalence": "not_assessed"}
    observer = None
    trace_updates, trace_finish = [], []
    try:
        source = verify_source(payload["source"], payload["source_manifest"], payload["source_manifest_sha256"])
        inputs = verify_inputs(payload["inputs_manifest"], payload["inputs_manifest_sha256"])
        if payload["row"] not in inputs["cases"]:
            raise ValueError("worker input row is not in frozen production manifest")
        sys.path.insert(0, str(Path(payload["source"]) / "src"))
        import numba
        import torch
        from fnit.recon_all import sphere_standard_run as operation
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        numba.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(payload["device"])
        if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
            raise ValueError("explicit available cuda:N required; no CPU fallback")
        if not Path(operation.__file__).resolve().is_relative_to(Path(payload["source"]).resolve()):
            raise ValueError("loaded sphere module is outside frozen source")
        kwargs = {"finish_device": str(device), "averaging_device": str(device),
                  "normals_device": None if payload["arm"] == "A" else str(device),
                  "bbox_mode": "source_float", "require_unfolded": True,
                  "overlap_diagnostics": True}
        if payload["mode"] == "trace":
            if sys.gettrace() is not None:
                raise RuntimeError("pre-existing Python trace is unsupported")
            observer, trace_updates, trace_finish = observe_complete_sphere(operation)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        free_before, total = torch.cuda.mem_get_info(device)
        report.update(code_version=source["commit"], source_manifest_sha256=payload["source_manifest_sha256"],
                      threads={"torch": torch.get_num_threads(), "numba": numba.get_num_threads(), "interop": torch.get_num_interop_threads()},
                      cpu_affinity=sorted(os.sched_getaffinity(0)), kwargs=kwargs,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      target_gpu_name=torch.cuda.get_device_name(device), target_gpu_free_before_bytes=free_before,
                      target_gpu_capacity_bytes=total, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      allocator_cache_disabled="PYTORCH_NO_CUDA_MEMORY_CACHING" in os.environ,
                      actual_precision={"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                                        "default_dtype": str(torch.get_default_dtype()),
                                        "autocast_cuda": torch.is_autocast_enabled("cuda"),
                                        "half_precision_requested": False})
        report["phase"] = "complete_sphere_api_running"
        save(report_path, report)
        tick = time.perf_counter()
        if observer is not None:
            sys.settrace(observer)
        try:
            api = operation.run_standard_sphere(
                inflated=payload["row"]["inputs"]["inflated"]["path"],
                smoothwm=payload["row"]["inputs"]["smoothwm"]["path"],
                output=output / (payload["row"]["hemisphere"] + ".sphere"), **kwargs)
            torch.cuda.synchronize(device)
            stage_seconds = time.perf_counter() - tick
        finally:
            if observer is not None:
                sys.settrace(None)
        report.update(stage_seconds_including_io_and_synchronization=stage_seconds, api=api,
                      timing_scope="complete API including read/write, first JIT, H2D/D2H; target CUDA synchronized immediately before/after; trace time excluded from ABBA")
        disabled = report["allocator_cache_disabled"]
        report["memory"] = {"target_device": str(device),
                            "pytorch_peak_allocated_bytes": None if disabled else torch.cuda.max_memory_allocated(device),
                            "pytorch_peak_reserved_bytes": None if disabled else torch.cuda.max_memory_reserved(device),
                            "pytorch_stats_status": "unavailable_cache_disabled" if disabled else "available",
                            "owned_simultaneous_driver_peak_bytes": None,
                            "owned_driver_status": "not_sampled_in_this_worker; parent lease sampler required",
                            "target_gpu_free_after_bytes": torch.cuda.mem_get_info(device)[0]}
        # 持久化真实API返回与同步计时后，再进入独立QA；后置JIT崩溃不能抹掉阶段证据。
        report["phase"] = "complete_sphere_api_returned"
        save(report_path, report)
        if observer is not None:
            if len(trace_updates) != len(api["updates"]) or len(trace_finish) != len(api["negative_counts"]):
                raise RuntimeError("observer missed actual update/finish steps")
            save(output / "trace.private.json", {"updates": trace_updates, "finish": trace_finish,
                                                   "terminal_overlap": api["overlap_diagnostics"]})
            report["trace_sha256"] = sha(output / "trace.private.json")
        report["phase"] = "saved_surface_quality_running"
        save(report_path, report)
        report["quality"], report["saved_quality_execution"] = inspect_surface_fresh_process(
            payload_path=args.payload, payload_sha256=args.payload_sha256,
            python_interpreter=sys.executable)
        report["phase"] = "saved_surface_quality_returned"
        save(report_path, report)
        verify_source(payload["source"], payload["source_manifest"], payload["source_manifest_sha256"])
        verify_inputs(payload["inputs_manifest"], payload["inputs_manifest_sha256"])
        report["status"] = "complete" if report["quality"]["passed"] else "failed_quality"
        report["phase"] = "completed_all_validation"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if hasattr(error, "diagnostics"):
            report["quality_failure"] = error.diagnostics
    if observer is not None and not (output / "trace.private.json").exists():
        save(output / "trace.private.json", {"updates": trace_updates, "finish": trace_finish,
                                               "partial_failure_trace": True})
        report["trace_sha256"] = sha(output / "trace.private.json")
    report["worker_seconds_including_validation_and_quality"] = time.perf_counter() - started
    save(report_path, report)
    return 0 if report["status"] == "complete" else 1


def run(args):
    """串行协调四个冻结输入的独立进程ABBA及独立trace，保留逐次结果。"""
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("NUMBA_BOUNDSCHECK") == "1":
        raise ValueError("NUMBA_BOUNDSCHECK=1 is diagnostic only; invoke an isolated worker payload, not ABBA")
    manifests = {arm: verify_source(getattr(args, name + "_source"), getattr(args, name + "_manifest"),
                                    getattr(args, name + "_manifest_sha256"))
                 for arm, name in (("A", "baseline"), ("B", "candidate"))}
    if manifests["A"]["commit"] != BASELINE:
        raise ValueError("baseline is not the declared 5ca07f99 source")
    # 本脚本仅验已有法向开关，不允许夹带sphere生产算法的其他改动。
    relevant = ("sphere_", "place_surface_normals", "mris_register_average", "mris_register_overlap")
    scope = {name for arm in manifests.values() for name in arm["files"]
             if name.startswith("src/fnit/recon_all/") and Path(name).name.startswith(relevant)}
    if any(manifests["A"]["files"].get(n) != manifests["B"]["files"].get(n) for n in scope):
        raise ValueError("sphere implementation differs; this runner isolates normals_device only")
    inputs = verify_inputs(args.inputs_manifest, args.inputs_manifest_sha256)
    args.output.mkdir(parents=True, mode=0o700)
    report = {"status": "running", "scope": "four same-production-input standard spheres; only Numba→existing CUDA normals; not recon-all",
              "baseline_commit": BASELINE, "candidate_commit": manifests["B"]["commit"],
              "inputs_manifest_sha256": args.inputs_manifest_sha256, "script_sha256": sha(__file__),
              "source_manifests": {"A": args.baseline_manifest_sha256, "B": args.candidate_manifest_sha256},
              "trace_timing_separate": True, "runs": [], "comparisons": [],
              "overall_metric_equivalence": "not_assessed", "official_reference": "not_read_by_this_benchmark"}
    summary = args.output / "summary.private.json"
    save(summary, report)
    try:
        for row in inputs["cases"]:
            identifier = row["case"] + "_" + row["hemisphere"]
            plans = [("timing", arm, i) for i, arm in enumerate("ABBA")]
            plans += [("trace", "A", 0), ("trace", "B", 1)]
            records = []
            for mode, arm, index in plans:
                name = ("baseline" if arm == "A" else "candidate")
                folder = args.output / identifier / f"{mode}-{index:02d}-{arm}"
                payload = {"row": row, "mode": mode, "arm": arm, "output": str(folder),
                           "source": str(getattr(args, name + "_source").absolute()),
                           "source_manifest": str(getattr(args, name + "_manifest").absolute()),
                           "source_manifest_sha256": getattr(args, name + "_manifest_sha256"),
                           "inputs_manifest": str(args.inputs_manifest.absolute()),
                           "inputs_manifest_sha256": args.inputs_manifest_sha256, "device": args.device}
                folder.parent.mkdir(parents=True, exist_ok=True)
                payload_path = folder.parent / (folder.name + ".payload.private.json")
                save(payload_path, payload)
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           NUMBA_CACHE_DIR=str(folder.parent / "numba-cache" / name),
                           TRITON_CACHE_DIR=str(folder.parent / "triton-cache" / name))
                # 两臂同样显式开启缓存；不将cacheoff的0统计误当实际零显存。
                env.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
                command = [args.python_interpreter, "-X", "faulthandler", "-u", str(Path(__file__).absolute()), "worker",
                           "--payload", str(payload_path), "--payload-sha256", sha(payload_path)]
                tick = time.perf_counter()
                with (folder.parent / (folder.name + ".log")).open("w") as stream:
                    process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
                record_path = folder / "report.private.json"
                record = json.loads(record_path.read_text()) if record_path.is_file() else {"status": "failed_before_report"}
                binding = {"case": row["case"], "hemisphere": row["hemisphere"], "mode": mode, "arm": arm,
                           "exit_code": process.returncode, "parent_full_wall_seconds": time.perf_counter() - tick,
                           "worker_report": str(record_path), "worker_report_sha256": sha(record_path) if record_path.exists() else None,
                           "status": record["status"]}
                report["runs"].append(binding)
                records.append((folder, record))
                save(summary, report)
                if process.returncode != 0 or record["status"] != "complete":
                    raise RuntimeError("stage failed; preserve all outputs: " + str(folder))
            controls = [r for _, r in records if r["mode"] == "timing" and r["arm"] == "A"]
            candidates = [r for _, r in records if r["mode"] == "timing" and r["arm"] == "B"]
            fingerprints = {(r["quality"]["sha256"], r["quality"]["coordinates_fp32_sha256"], r["quality"]["faces_sha256"])
                            for _, r in records}
            traces = [json.loads((folder / "trace.private.json").read_text()) for folder, r in records if r["mode"] == "trace"]
            import nibabel.freesurfer.io as fsio
            import numpy as np
            a, af = fsio.read_geometry(str(records[0][0] / (row["hemisphere"] + ".sphere")))
            b, bf = fsio.read_geometry(str(records[1][0] / (row["hemisphere"] + ".sphere")))
            if a.shape != b.shape or not np.array_equal(af, bf):
                raise RuntimeError("ordered correspondence changed; same-index distance is invalid")
            displacement = np.linalg.norm(b.astype(np.float64) - a.astype(np.float64), axis=1)
            errors = {"scope": "same input and identical ordered faces; candidate minus baseline",
                      "different_vertices": int(np.count_nonzero(np.any(a != b, axis=1))),
                      "coordinate_mae_mm": float(np.abs(b - a).mean()),
                      "vertex_distance_mean_mm": float(displacement.mean()),
                      "vertex_distance_p99_mm": float(np.percentile(displacement, 99)),
                      "vertex_distance_max_mm": float(displacement.max(initial=0))}
            medians = {"A": statistics.median(r["stage_seconds_including_io_and_synchronization"] for r in controls),
                       "B": statistics.median(r["stage_seconds_including_io_and_synchronization"] for r in candidates)}
            report["comparisons"].append({"case": row["case"], "hemisphere": row["hemisphere"],
                "abba_stage_median_seconds": medians, "observed_stage_reduction_percent": 100 * (1 - medians["B"] / medians["A"]),
                "all_six_saved_files_and_ordered_geometry_equal": len(fingerprints) == 1,
                "every_update_and_finish_trace_equal": traces[0] == traces[1],
                "same_index_geometry_error": errors,
                "all_quality_passed": all(r["quality"]["passed"] for _, r in records),
                "strict_reproduction": "passed" if len(fingerprints) == 1 and traces[0] == traces[1] else "difference_detected",
                "timing_limits": "one ABBA block, separate source caches, shared load; cold first-JIT and warm repeats separately retained; not whole-pipeline speedup"})
            save(summary, report)
        report["status"] = "complete"
        report["strict_reproduction"] = "passed" if all(r["strict_reproduction"] == "passed" for r in report["comparisons"]) else "difference_detected"
        report["optimization_regression"] = "not_observed_in_this_fixed_input_test" if report["strict_reproduction"] == "passed" else "requires_diagnosis"
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    save(summary, report)
    # 执行/质量与严格逐位复现分开；差异留给协调者按预定算子标准诊断。
    return 0 if report["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze", help="冻结源码完整Python SHA清单")
    frozen.add_argument("--source", type=Path, required=True)
    frozen.add_argument("--commit", required=True)
    frozen.add_argument("--output", type=Path, required=True)
    frozen.set_defaults(function=freeze)
    prepared = commands.add_parser("prepare", help="从两份完成的FNIT benchmark提取四份自产输入")
    prepared.add_argument("--benchmark", type=Path, action="append", required=True)
    prepared.add_argument("--output", type=Path, required=True)
    prepared.set_defaults(function=prepare)
    benchmark = commands.add_parser("run", help="独立进程ABBA+独立完整trace")
    for arm in ("baseline", "candidate"):
        benchmark.add_argument("--" + arm + "-source", type=Path, required=True)
        benchmark.add_argument("--" + arm + "-manifest", type=Path, required=True)
        benchmark.add_argument("--" + arm + "-manifest-sha256", required=True)
    benchmark.add_argument("--inputs-manifest", type=Path, required=True)
    benchmark.add_argument("--inputs-manifest-sha256", required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--python-interpreter", default=sys.executable)
    benchmark.set_defaults(function=run)
    child = commands.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("--payload", type=Path, required=True)
    child.add_argument("--payload-sha256", required=True)
    child.set_defaults(function=worker)
    quality = commands.add_parser("quality-worker", help=argparse.SUPPRESS)
    quality.add_argument("--payload", type=Path, required=True)
    quality.add_argument("--payload-sha256", required=True)
    quality.add_argument("--surface-sha256", required=True)
    quality.set_defaults(function=quality_worker)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
