"""真实同输入 source marker 的只读剖析；不执行清理或更改判定。"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import resource
import time


def run(*, input_surface: Path, output: Path, threads: int = 4) -> dict:
    """完整加载表面，先原样标记、再同输入 cProfile，保留两次输出 SHA。

    input_surface 是真实自产三角表面，surface RAS/mm；output 须为新目录。
    threads 是 CPU 总线程预算；两调用顺序运行，CUDA 必须隐藏。
    报告的剖析墙钟包含 Python profiler 开销，不能当生产性能门。
    """
    tick = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    if threads < 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("positive CPU threads and hidden CUDA are required")
    output.mkdir(mode=0o700, parents=True)
    import nibabel.freesurfer.io as fsio
    import numba
    import numpy as np
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    numba.set_num_threads(threads)
    loading_tick = time.perf_counter()
    vertices, faces = fsio.read_geometry(input_surface)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    loading_seconds = time.perf_counter() - loading_tick
    def digest(value):
        return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
    input_hashes = dict(vertices=digest(vertices), faces=digest(faces))
    calls = []
    profiler = cProfile.Profile()
    for profiled in (False, True):
        call_tick = time.perf_counter()
        if profiled:
            profiler.enable()
        try:
            marked, count = mark_source_intersections(vertices, faces, predicate_backend="numba")
        finally:
            if profiled:
                profiler.disable()
        calls.append(dict(profiled=profiled, seconds=time.perf_counter()-call_tick,
                          intersecting_faces=count, marked_vertices=int(marked.sum()),
                          mark_sha256=digest(marked)))
    profiler.dump_stats(str(output/"marker.private.pstats"))
    stats = pstats.Stats(profiler)
    rows = [dict(module=Path(filename).name, function=function, line=line,
                 primitive_calls=primitive, total_calls=total,
                 own_seconds=own, cumulative_seconds=cumulative)
            for (filename, line, function), (primitive, total, own, cumulative, callers)
            in stats.stats.items()]
    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    result = dict(status="readonly_marker_profile", vertices=len(vertices), faces=len(faces),
                  input_file_sha256=hashlib.sha256(input_surface.read_bytes()).hexdigest(),
                  input_arrays=input_hashes, loading_seconds=loading_seconds, calls=calls,
                  same_outputs=(calls[0]["mark_sha256"] == calls[1]["mark_sha256"] and
                                calls[0]["intersecting_faces"] == calls[1]["intersecting_faces"]),
                  inputs_unchanged=input_hashes == dict(vertices=digest(vertices), faces=digest(faces)),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), threads=threads, gpu_used=False,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  profile_top_cumulative=rows[:35], profile_top_own=sorted(
                      rows, key=lambda row: row["own_seconds"], reverse=True)[:35],
                  wall_seconds=time.perf_counter()-tick,
                  whole_pipeline_equivalence="not_assessed")
    (output/"report.private.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"COMPLETE.json").write_text(json.dumps(dict(status=result["status"],
        same_outputs=result["same_outputs"], inputs_unchanged=result["inputs_unchanged"]))+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run(input_surface=args.input_surface, output=args.output, threads=args.threads)
    print(json.dumps(dict(status=result["status"], calls=result["calls"],
                         same_outputs=result["same_outputs"])), flush=True)


if __name__ == "__main__":
    main()
