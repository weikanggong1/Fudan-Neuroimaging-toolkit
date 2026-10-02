"""真实同输入球面配准：CPU 基线与 CPU/GPU averaging 候选独立进程配对。

--subject 为已有 FNIT 自产被试目录；--hemi 为 lh/rh；--atlas 为声明的
对应半球 folding TIFF。输入固定为 surf/H.sphere、H.smoothwm、H.sulc，
surface RAS/mm、相同顶点/有序面；图谱不是官方被试结果。两个 --*-source
指冻结 FNIT src 目录，--*-version 标记版本，实际源代码 SHA 另外保存。
--averaging-device 默认 cpu，可指定 cuda:N；只候选使用此设备，基线及
两者 overlap 均 CPU。--gpu-uuid 可核对明确目标，建议同时以 UUID 设置
CUDA_VISIBLE_DEVICES。Torch/Numba/OMP/BLAS 均固定 4 线程。--repetitions
默认 1，设 2 则 AB/BA 两轮。--threads 只接受 4；--output 必须是新目录，
各实现 JIT 缓存隔离。
--official-surface 仅比较已生成参考，不能作为配准输入；不执行官方程序。

返回 report.json、每轮各自输出 sphere.reg、完整 API JSON 和日志。独立
进程墙钟包含导入/加载/传输/读写/JIT；阶段墙钟包含 CUDA 初始化（若有）、
指定设备前后同步及 API 完整读写。报告保存源码/输入 SHA、实际设备 UUID、
线程、float32/TF32、allocator 峰值、有序几何/footer误差、dt/state轨迹。
若 API 仅保存选中步长，不编造未记录的 rejected trial。严格复现是诊断，
不另立数值/指标等效阈值；整例提速与指标等效均 not_assessed。
结构/运行失败保存 failed JSON、退出非零；严格尾差单独记录，不伪装为失败
或整体等效通过。对应 mris_register 的 sulc+smoothwm 球面配准。

具名参数示例（各行中文说明）：
    python benchmark_register_pair.py \\
      --subject /data/fnit_subject \\
      --hemi lh \\
      --atlas /data/fnit_assets/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif \\
      --baseline-source /data/frozen_baseline/src \\
      --baseline-version 764607c \\
      --candidate-source /data/frozen_candidate/src \\
      --candidate-version snapshot-with-source-sha256 \\
      --averaging-device cuda:0 \\
      --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \\
      --output /data/validation/register_pair_new_directory \\
      --threads 4 \\
      --repetitions 2
参数注释：subject=自产冻结输入；hemi=同半球；atlas=声明图谱；两个source
与version=计算代码快照/标识；averaging-device=仅候选平均内核设备；gpu-uuid=
明确设备核验；output=新诊断目录；threads=固定总线程；repetitions=反序配对次数。
无半精度选项。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time
import traceback


THREADS = 4


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_manifest(source: Path) -> dict:
    return {str(path.relative_to(source)): sha256(path)
            for path in sorted((source / "fnit" / "recon_all").rglob("*"))
            if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp")}


def inputs(args) -> dict[str, Path]:
    return {**{suffix: args.subject / "surf" / f"{args.hemi}.{suffix}"
               for suffix in ("sphere", "smoothwm", "sulc")}, "atlas": args.atlas}


def surface_footer(path: Path) -> bytes:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError(f"需要三角表面：{path}")
        stream.readline()
        stream.readline()
        vertices, faces = struct.unpack(">ii", stream.read(8))
        stream.seek(12 * (vertices + faces), 1)
        return stream.read()


def compare_geometry(reference: Path, candidate: Path) -> dict:
    """同网格才报告同索引 mm 误差；footer 单独比较，不产生新容差。"""
    import numpy as np
    from nibabel.freesurfer.io import read_geometry

    a, af = read_geometry(str(reference))
    b, bf = read_geometry(str(candidate))
    corresponding = a.shape == b.shape and np.array_equal(af, bf)
    row = {"reference": str(reference), "candidate": str(candidate),
           "reference_sha256": sha256(reference), "candidate_sha256": sha256(candidate),
           "reference_vertices": len(a), "candidate_vertices": len(b),
           "reference_faces": len(af), "candidate_faces": len(bf),
           "same_vertex_shape": a.shape == b.shape,
           "ordered_faces_equal": bool(np.array_equal(af, bf)),
           "index_correspondence": bool(corresponding),
           "coordinates_equal": bool(np.array_equal(a, b)),
           "finite_coordinates": bool(np.isfinite(a).all() and np.isfinite(b).all()),
           "footer_equal": surface_footer(reference) == surface_footer(candidate),
           "different_coordinates": None, "max_coordinate_error_mm": None,
           "p99_coordinate_error_mm": None, "max_vertex_displacement_mm": None,
           "p99_vertex_displacement_mm": None, "mean_vertex_displacement_mm": None}
    if corresponding:
        delta = np.abs(a - b)
        distance = np.linalg.norm(a - b, axis=1)
        row.update(different_coordinates=int(np.count_nonzero(a != b)),
                   max_coordinate_error_mm=float(delta.max(initial=0)),
                   p99_coordinate_error_mm=float(np.quantile(delta, .99)),
                   max_vertex_displacement_mm=float(distance.max(initial=0)),
                   p99_vertex_displacement_mm=float(np.quantile(distance, .99)),
                   mean_vertex_displacement_mm=float(np.mean(distance)))
    row["strict_reproduction"] = (row["coordinates_equal"] and row["ordered_faces_equal"]
                                  and row["footer_equal"] and row["finite_coordinates"])
    return row


def trajectory(report: dict) -> dict:
    """保留 API 实际保存的选步/状态/相交轨迹；排除耗时和设备元数据。"""
    result = {}
    for name in ("sulc_pass", "smoothwm_pass"):
        value = report.get(name, {})
        result[name] = {key: value[key] for key in
                       ("rigid_angles", "rigid_score", "rigid_evaluations", "last_iteration",
                        "seed_iteration", "negative_counts", "accepted", "rejected",
                        "accepted_steps", "rejected_steps", "decisions") if key in value}
        if "updates" in value:
            result[name]["updates"] = [
                {key: item for key, item in row.items()
                 if "second" not in key and "time" not in key}
                for row in value["updates"]]
    return result


def worker(args) -> int:
    """独立加载快照，计算一侧完整球面配准并记录设备/时间/输出结构。"""
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.candidate_source))
    files = inputs(args)
    report = {"status": "running", "subject": str(args.subject), "hemi": args.hemi,
              "source_version": args.candidate_version, "host": platform.node(),
              "python_version": platform.python_version(), "cpu": platform.processor(),
              "script_sha256": sha256(Path(__file__)),
              "source_sha256": None,
              "inputs": {name: str(path) for name, path in files.items()},
              "input_sha256": None,
              "averaging_device_requested": args.averaging_device,
              "overlap_device": "cpu", "threads_requested": THREADS,
              "gpu_uuid_requested": args.gpu_uuid,
              "thread_environment": {key: os.environ.get(key) for key in
                  ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "NUMBA_CACHE_DIR", "TRITON_CACHE_DIR",
                   "CUDA_VISIBLE_DEVICES", "PYTORCH_NO_CUDA_MEMORY_CACHING")},
              "whole_speedup": "not_assessed", "overall_metric_equivalence": "not_assessed"}
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        report["source_sha256"] = source_manifest(args.candidate_source)
        report["input_sha256"] = {name: sha256(path) for name, path in files.items()}
        import numba
        import numpy as np
        import torch
        from nibabel.freesurfer.io import read_geometry
        from fnit.recon_all.mris_register_run import run_register_sphere

        actual_source = Path(inspect.getsourcefile(run_register_sphere)).resolve()
        if not actual_source.is_relative_to(args.candidate_source.resolve()):
            raise RuntimeError(f"实际导入不是指定快照：{actual_source}")
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        numba.set_num_threads(THREADS)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        device = torch.device(args.averaging_device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index is None):
            raise ValueError("averaging-device 必须是 cpu 或显式 cuda:N")
        parameters = inspect.signature(run_register_sphere).parameters
        if device.type == "cuda" and "averaging_device" not in parameters:
            raise ValueError("该快照没有 averaging_device API，不能宣称执行了 GPU averaging")
        call_arguments = dict(**files, output=args.output / f"{args.hemi}.sphere.reg", overlap_device="cpu")
        if "averaging_device" in parameters:
            call_arguments["averaging_device"] = str(device)
        report.update(actual_api_source=str(actual_source), numpy_version=np.__version__,
                      numba_version=numba.__version__, torch_version=torch.__version__,
                      cuda_runtime=torch.version.cuda, torch_threads=torch.get_num_threads(),
                      torch_interop_threads=torch.get_num_interop_threads(),
                      numba_threads=numba.get_num_threads(),
                      numba_thread_capacity=numba.config.NUMBA_NUM_THREADS,
                      averaging_device_argument_supported="averaging_device" in parameters,
                      precision={"dtype": "float32", "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                                 "float16_or_bfloat16": False, "autocast_cuda": torch.is_autocast_enabled("cuda")},
                      gpu_uuid_actual=None, gpu_name=None,
                      memory_scope="target-device PyTorch allocator only; simultaneous process occupancy requires external monitor",
                      gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None,
                      cuda_allocation_cache_disabled=os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1")
        tick = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            actual_uuid = str(getattr(properties, "uuid", "unavailable"))
            report.update(gpu_uuid_actual=actual_uuid, gpu_name=properties.name)
            if args.gpu_uuid is not None:
                if actual_uuid == "unavailable":
                    raise RuntimeError("当前 Torch 不提供目标 UUID，无法核验 gpu-uuid")
                if actual_uuid.removeprefix("GPU-") != args.gpu_uuid.removeprefix("GPU-"):
                    raise RuntimeError("实际目标 GPU UUID 与指定 GPU 不符")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        report["initialization_before_api_seconds"] = time.perf_counter() - tick
        api = run_register_sphere(**call_arguments)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["wall_seconds_including_io_jit_sync_init"] = time.perf_counter() - tick
        report["stage_timing_scope"] = "CUDA init if used + explicit target synchronization before/after + full API IO/JIT/transfers; excludes Python imports"
        if device.type == "cuda" and not report["cuda_allocation_cache_disabled"]:
            report.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.output / "api_report.json", api)
        registered = Path(api["output"])
        in_vertices, in_faces = read_geometry(str(files["sphere"]))
        out_vertices, out_faces = read_geometry(str(registered))
        structure = {"same_vertex_shape_as_input": in_vertices.shape == out_vertices.shape,
                     "ordered_faces_equal_to_input": bool(np.array_equal(in_faces, out_faces)),
                     "finite_coordinates": bool(np.isfinite(out_vertices).all()),
                     "face_indices_valid": bool(((out_faces >= 0) & (out_faces < len(out_vertices))).all()),
                     "footer_equal_to_input": surface_footer(files["sphere"]) == surface_footer(registered)}
        if not all(structure.values()):
            raise RuntimeError(f"配准输出结构/几何尾部无效：{structure}")
        report.update(status="complete", started_utc=started_utc,
                      ended_utc=datetime.now(timezone.utc).isoformat(),
                      api_report=str(args.output / "api_report.json"),
                      api_report_sha256=sha256(args.output / "api_report.json"),
                      api_total_seconds_including_io=api.get("total_seconds_including_io"),
                      output=str(registered), output_sha256=sha256(registered), output_structure=structure,
                      trajectory=trajectory(api), trajectory_scope="selected dt/state updates and negative counts; rejected trials only when API records them",
                      actual_device_reports={name: api.get(name, {}).get("averaging_device", "cpu (legacy implicit)")
                                             for name in ("sulc_pass", "smoothwm_pass")},
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, exception=str(exc),
                      ended_utc=datetime.now(timezone.utc).isoformat())
        traceback.print_exc()
        write_json(args.output / "report.json", report)
        return 1
    write_json(args.output / "report.json", report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--baseline-version")
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--averaging-device", default="cpu")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(THREADS,), default=THREADS)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=1)
    parser.add_argument("--official-surface", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.baseline_source is None or args.baseline_version is None:
        parser.error("需要 baseline-source/baseline-version")
    args.subject = args.subject.resolve()
    args.atlas = args.atlas.resolve()
    args.baseline_source = args.baseline_source.resolve()
    args.candidate_source = args.candidate_source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    report = {"status": "running", "scope": "same-real-input-register-stage-pair;not-whole",
              "subject": str(args.subject), "hemi": args.hemi,
              "baseline_version": args.baseline_version, "candidate_version": args.candidate_version,
              "baseline_averaging_device": "cpu", "candidate_averaging_device": args.averaging_device,
              "overlap_device": "cpu", "threads": THREADS, "script_sha256": sha256(script),
              "input_sha256": None,
              "repetitions": args.repetitions, "timing_note": "AB/BA isolated processes; cache isolated by implementation, shared across repeats; cold first JIT included",
              "strict_reproduction_scope": "ordered geometry/footer and reported selected-step trajectory",
              "new_numerical_tolerance": None, "whole_speedup": "not_assessed",
              "overall_metric_equivalence": "not_assessed", "runs": []}
    write_json(args.output / "report.json", report)
    try:
        report["input_sha256"] = {name: sha256(path) for name, path in inputs(args).items()}
        write_json(args.output / "report.json", report)
        for repeat in range(args.repetitions):
            order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            pair = {"repeat": repeat, "order": list(order)}
            for implementation in order:
                is_baseline = implementation == "baseline"
                source = args.baseline_source if is_baseline else args.candidate_source
                version = args.baseline_version if is_baseline else args.candidate_version
                folder = args.output / f"round_{repeat}" / implementation
                folder.parent.mkdir(parents=True, exist_ok=True)
                cache = args.output / f"numba_cache_{implementation}"
                triton_cache = args.output / f"triton_cache_{implementation}"
                cache.mkdir(exist_ok=True)
                triton_cache.mkdir(exist_ok=True)
                env = dict(os.environ, PYTHONPATH=str(source), NUMBA_NUM_THREADS=str(THREADS),
                           OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
                           OPENBLAS_NUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS),
                           NUMBA_CACHE_DIR=str(cache), TRITON_CACHE_DIR=str(triton_cache))
                command = [sys.executable, str(script), "--worker", "--subject", str(args.subject),
                           "--hemi", args.hemi, "--atlas", str(args.atlas),
                           "--candidate-source", str(source), "--candidate-version", version,
                           "--averaging-device", "cpu" if is_baseline else args.averaging_device,
                           "--output", str(folder), "--threads", str(THREADS)]
                if args.gpu_uuid:
                    command.extend(["--gpu-uuid", args.gpu_uuid])
                tick = time.perf_counter()
                with (folder.parent / f"{implementation}.log").open("w") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                seconds = time.perf_counter() - tick
                path = folder / "report.json"
                child = json.loads(path.read_text()) if path.exists() else {"status": "failed"}
                pair[implementation] = {"command": command, "returncode": result.returncode,
                    "process_wall_seconds_including_imports": seconds,
                    "report": str(path), "report_sha256": sha256(path) if path.exists() else None,
                    "stage_report": child}
                if result.returncode or child["status"] != "complete":
                    report["runs"].append(pair)
                    raise RuntimeError(f"{implementation} 未完成：{folder}")
            comparison = compare_geometry(Path(pair["baseline"]["stage_report"]["output"]),
                                          Path(pair["candidate"]["stage_report"]["output"]))
            comparison["reported_trajectory_equal"] = (pair["baseline"]["stage_report"]["trajectory"] ==
                                                         pair["candidate"]["stage_report"]["trajectory"])
            pair["comparison"] = comparison
            pair["strict_reproduction"] = comparison["strict_reproduction"] and comparison["reported_trajectory_equal"]
            pair["optimization_degradation"] = ("not_detected_under_exact_comparison" if pair["strict_reproduction"] else "not_assessed")
            if args.official_surface is not None:
                pair["official_diagnostic"] = compare_geometry(args.official_surface.resolve(),
                                          Path(pair["candidate"]["stage_report"]["output"]))
                pair["official_diagnostic"]["overall_metric_equivalence"] = "not_assessed"
            pair["stage_wall_ratio_baseline_over_candidate"] = (
                pair["baseline"]["stage_report"]["wall_seconds_including_io_jit_sync_init"] /
                pair["candidate"]["stage_report"]["wall_seconds_including_io_jit_sync_init"])
            pair["process_wall_ratio_baseline_over_candidate"] = (
                pair["baseline"]["process_wall_seconds_including_imports"] /
                pair["candidate"]["process_wall_seconds_including_imports"])
            report["runs"].append(pair)
            write_json(args.output / "report.json", report)
        report.update(status="complete", strict_reproduction=all(row["strict_reproduction"] for row in report["runs"]))
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, exception=str(exc))
        traceback.print_exc()
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "status": report["status"],
                      "strict_reproduction": report.get("strict_reproduction")}, ensure_ascii=False))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
