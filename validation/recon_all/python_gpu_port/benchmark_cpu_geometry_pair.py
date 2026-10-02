"""真实冻结网格的 remesh/quick-sphere 配对回归及可选 CPU 剖析。

输入 --input 为真实 orig.premesh（remesh）或 inflated.nofix（quick-sphere）
三角表面，坐标 surface RAS/mm。--baseline-source/--candidate-source 指各自
冻结的 src 目录；--baseline-version/--candidate-version 标识代码版本，实际
源码 SHA 另行保存。--output 必须为空的新目录；--threads 默认 4，
--repetitions 默认 2，第二轮颠倒先后顺序。--profile 额外保存 pstats/text，
其计时含剖析开销，不用于无剖析性能结论。--official-output 可逐输入指定
已授权的官方冻结结果，仅用于诊断；不会作为算法输入。

每次使用独立 CPU 子进程，固定线程并分别隔离 Numba 缓存；第一轮的 JIT
计入耗时，后续轮可复用本实现缓存。保存包含读写/JIT的阶段墙钟、包含导入
的进程墙钟、RSS、输入/源码/输出 SHA、完整有序几何/尾部比较与阶段追踪。
回归门槛预先固定为坐标、有序面及尾部完全相同，quick trace 也须相同；
没有调整整例或指标等效门槛。异常写 failed JSON 并退出非零。
对应官方 mris_remesh --remesh --iters 3 或 mris_sphere -q；不调用官方程序。
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import struct
import subprocess
import sys
import time
import traceback


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def source_manifest(source: Path) -> dict:
    return {str(path.relative_to(source)): digest(path)
            for path in sorted((source / "fnit" / "recon_all").glob("*.py"))}


def footer(path: Path) -> bytes:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError("需要 FreeSurfer 三角表面")
        stream.readline()
        stream.readline()
        vertices, faces = struct.unpack(">ii", stream.read(8))
        stream.seek(12 * (vertices + faces), 1)
        return stream.read()


def geometry_compare(reference: Path, candidate: Path) -> dict:
    """同索引坐标/有序面/几何尾部严格比较；不匹配时不声称逐顶点误差。"""
    import numpy as np
    from nibabel.freesurfer.io import read_geometry

    a, af = read_geometry(str(reference))
    b, bf = read_geometry(str(candidate))
    same_vertices = a.shape == b.shape
    ordered_faces = bool(np.array_equal(af, bf))
    row = {"reference": str(reference), "candidate": str(candidate),
           "reference_sha256": digest(reference), "candidate_sha256": digest(candidate),
           "reference_vertices": len(a), "candidate_vertices": len(b),
           "reference_faces": len(af), "candidate_faces": len(bf),
           "same_vertex_shape": same_vertices, "ordered_faces_equal": ordered_faces,
           "coordinates_equal": bool(np.array_equal(a, b)),
           "footer_equal": footer(reference) == footer(candidate),
           "index_correspondence": same_vertices and ordered_faces,
           "max_coordinate_error_mm": None, "p99_coordinate_error_mm": None,
           "max_vertex_displacement_mm": None, "p99_vertex_displacement_mm": None,
           "different_coordinate_values": None}
    if row["index_correspondence"]:
        delta = np.abs(a - b)
        distance = np.linalg.norm(a - b, axis=1)
        row.update(different_coordinate_values=int(np.count_nonzero(a != b)),
                   max_coordinate_error_mm=float(delta.max(initial=0)),
                   p99_coordinate_error_mm=float(np.quantile(delta, .99)),
                   max_vertex_displacement_mm=float(distance.max(initial=0)),
                   p99_vertex_displacement_mm=float(np.quantile(distance, .99)))
    row["strict_pass"] = (row["coordinates_equal"] and ordered_faces and row["footer_equal"])
    return row


def worker(args) -> int:
    """独立进程加载指定源代码；阶段计时包含表面读写，导入另计外部墙钟。"""
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.candidate_source))
    report = {"status": "running", "stage": args.stage, "input": str(args.input[0]),
              "input_sha256": digest(args.input[0]), "version": args.candidate_version,
              "source_sha256": source_manifest(args.candidate_source),
              "script_sha256": digest(Path(__file__)), "host": platform.node(),
              "cpu": platform.processor(), "python": platform.python_version(),
              "threads_requested": args.threads, "device": "cpu", "gpu_used": False,
              "timing_scope": "stage includes reading/writing/JIT; process wall also includes imports",
              "profile_enabled": args.profile,
              "thread_environment": {key: os.environ.get(key) for key in
                  ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "NUMBA_CACHE_DIR")}, "substeps": []}
    destination = args.output / "surface"
    try:
        import numpy as np
        import numba
        import torch

        torch.set_num_threads(args.threads)
        report.update(numpy_version=np.__version__, numba_version=numba.__version__,
                      torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                      numba_threads=numba.get_num_threads(),
                      numba_thread_capacity=numba.config.NUMBA_NUM_THREADS)
        if args.stage == "remesh":
            from fnit.recon_all import mris_remesh_python as stage

            original_split, original_collapse, original_smooth = (
                stage.split_pass, stage.Mesh.collapse_pass, stage.smooth)

            def split(*values, **kwargs):
                started = time.perf_counter()
                result = original_split(*values, **kwargs)
                report["substeps"].append({"step": "split", "accepted": result,
                                           "seconds": time.perf_counter() - started})
                return result

            def collapse(mesh, *values, **kwargs):
                started = time.perf_counter()
                result = original_collapse(mesh, *values, **kwargs)
                report["substeps"].append({"step": "collapse", "accepted": result,
                                           "seconds": time.perf_counter() - started})
                return result

            def smooth(mesh, *values, **kwargs):
                started = time.perf_counter()
                result = original_smooth(mesh, *values, **kwargs)
                report["substeps"].append({"step": "smooth", "vertices": len(mesh.points),
                                           "faces": len(mesh.faces),
                                           "seconds": time.perf_counter() - started})
                return result

            stage.split_pass, stage.Mesh.collapse_pass, stage.smooth = split, collapse, smooth
            call = lambda: stage.remesh_surface(input_path=args.input[0],
                                                output_path=destination, iterations=3)
        else:
            from fnit.recon_all import sphere_quick_python as stage
            original = stage.quick_sphere_from_inflated

            def quick(*values, **kwargs):
                result, trace = original(*values, **kwargs)
                report["quick_trace"] = trace
                return result, trace

            stage.quick_sphere_from_inflated = quick
            call = lambda: stage.write_quick_sphere(input_path=args.input[0], output_path=destination)
        report["actual_stage_source"] = str(Path(stage.__file__).resolve())
        profiler = cProfile.Profile() if args.profile else None
        tick = time.perf_counter()
        if profiler:
            profiler.enable()
        try:
            call()
        finally:
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(args.output / "stage.pstats"))
                with (args.output / "profile.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                        "cumulative").print_stats(50)
        report["wall_seconds_including_io_jit"] = time.perf_counter() - tick
        report.update(status="complete", output=str(destination), output_sha256=digest(destination),
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, exception=str(exc))
        traceback.print_exc()
        write_json(args.output / "report.json", report)
        return 1
    write_json(args.output / "report.json", report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("remesh", "quick-sphere"), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--baseline-version")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-output", type=Path, nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1:
        parser.error("threads/repetitions 必须为正")
    if args.worker:
        if len(args.input) != 1:
            parser.error("worker 只接受一个输入")
        return worker(args)
    if args.baseline_source is None or args.baseline_version is None:
        parser.error("配对测试需要 baseline-source/baseline-version")
    if args.official_output and len(args.official_output) != len(args.input):
        parser.error("official-output 必须逐项对应 input")
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    report = {"status": "running", "scope": "real-frozen-input-cpu-stage-pair-not-whole",
              "stage": args.stage, "baseline_version": args.baseline_version,
              "candidate_version": args.candidate_version, "script_sha256": digest(script),
              "input_sha256": {str(path): digest(path) for path in args.input},
              "threads": args.threads, "profile": args.profile,
              "gate": "exact ordered coordinates/faces/footer; quick trace exact; no new tolerance",
              "overall_metrics_equivalence": "not_assessed", "runs": []}
    try:
        for index, input_path in enumerate(args.input):
            for repeat in range(args.repetitions):
                order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                pair = {"case": index, "input": str(input_path), "repeat": repeat, "order": list(order)}
                for implementation in order:
                    source = (args.baseline_source if implementation == "baseline" else args.candidate_source)
                    version = (args.baseline_version if implementation == "baseline" else args.candidate_version)
                    folder = args.output / f"case_{index}" / f"round_{repeat}" / implementation
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    cache = args.output / f"numba_cache_{implementation}"
                    cache.mkdir(exist_ok=True)
                    env = dict(os.environ, PYTHONPATH=str(source), NUMBA_NUM_THREADS=str(args.threads),
                               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                               OPENBLAS_NUM_THREADS=str(args.threads), NUMBA_CACHE_DIR=str(cache))
                    command = [sys.executable, str(script), "--worker", "--stage", args.stage,
                               "--input", str(input_path), "--candidate-source", str(source),
                               "--candidate-version", version, "--output", str(folder),
                               "--threads", str(args.threads)]
                    if args.profile:
                        command.append("--profile")
                    tick = time.perf_counter()
                    with (folder.parent / f"{implementation}.log").open("w") as log:
                        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    process_seconds = time.perf_counter() - tick
                    child_path = folder / "report.json"
                    child = json.loads(child_path.read_text()) if child_path.is_file() else {"status": "failed"}
                    pair[implementation] = {"command": command, "returncode": result.returncode,
                        "process_wall_seconds_including_imports": process_seconds,
                        "stage_report": str(child_path), "stage_report_sha256": digest(child_path) if child_path.exists() else None,
                        "stage": child}
                    if result.returncode:
                        report["runs"].append(pair)
                        raise RuntimeError(f"{implementation} {input_path} 阶段失败；查看 {folder.parent}")
                first = Path(pair["baseline"]["stage"]["output"])
                second = Path(pair["candidate"]["stage"]["output"])
                comparison = geometry_compare(first, second)
                if args.stage == "quick-sphere":
                    comparison["quick_trace_equal"] = (pair["baseline"]["stage"]["quick_trace"] ==
                                                         pair["candidate"]["stage"]["quick_trace"])
                    comparison["strict_pass"] &= comparison["quick_trace_equal"]
                else:
                    # 時間不屬於接受規則；兩條算法的每次拆邊/縮邊接受數及平滑尺寸須相同。
                    traces = [[{key: value for key, value in step.items() if key != "seconds"}
                               for step in pair[name]["stage"]["substeps"]] for name in ("baseline", "candidate")]
                    comparison["phase_decisions_equal"] = traces[0] == traces[1]
                    comparison["strict_pass"] &= comparison["phase_decisions_equal"]
                pair["comparison"] = comparison
                if args.official_output:
                    pair["official_diagnostic"] = geometry_compare(args.official_output[index], second)
                pair["phase_wall_ratio_baseline_over_candidate"] = (
                    pair["baseline"]["stage"]["wall_seconds_including_io_jit"] /
                    pair["candidate"]["stage"]["wall_seconds_including_io_jit"])
                report["runs"].append(pair)
                write_json(args.output / "report.json", report)
        report["strict_regression_passed"] = all(row["comparison"]["strict_pass"] for row in report["runs"])
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, exception=str(exc))
        traceback.print_exc()
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "status": report["status"],
                      "strict_regression_passed": report.get("strict_regression_passed")}, ensure_ascii=False))
    return int(report["status"] != "complete" or not report.get("strict_regression_passed", False))


if __name__ == "__main__":
    raise SystemExit(main())
