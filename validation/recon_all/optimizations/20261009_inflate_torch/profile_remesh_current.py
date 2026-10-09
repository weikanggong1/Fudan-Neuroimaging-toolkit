"""复用现有remesh真实benchmark worker，分解当前自产orig.premesh阶段。

输入完整surface RAS/mm三角面、冻结src、全新输出和固定缓存/线程预算。
原算法/输出未修改；只在独立进程添加粗粒度时钟。--profile生成原工具的
完整cProfile，耗时不能与未profile成绩混用。--decision-trace额外核对每个
拆边/缩边的实际顺序与接受结果及每pass全数组SHA，属于精度诊断非纯计时。
原始工具的report.json不重写，额外字段写detail.json，未调用官方软件/GPU。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace


def main() -> int:
    entry = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--decision-trace", action="store_true")
    parser.add_argument("--scalar-storage", choices=("numpy", "python"), default="numpy")
    parser.add_argument("--gc-policy", choices=("inherit", "suspend"), default="inherit")
    args = parser.parse_args()
    if args.output.exists() or args.threads <= 0:
        raise ValueError("new output and positive thread budget required")
    sys.path.insert(0, str(args.source))
    import numpy as np
    from fnit.recon_all import mris_remesh_python as remesh
    from benchmark_cpu_geometry_pair import worker, digest
    if args.scalar_storage == "python":
        # Only bind the explicit public candidate parameter. The worker still
        # calls the same complete source function; no algorithm is replaced.
        surface = remesh.remesh_surface
        def run_surface(input_path, output_path, iterations=3):
            return surface(input_path=input_path, output_path=output_path,
                           iterations=iterations, scalar_storage="python")
        remesh.remesh_surface = run_surface

    nested = []
    for name in ("rebuild", "face_normals", "compact"):
        original = getattr(remesh.Mesh, name)
        def observe(mesh, *values, _function=original, _name=name, **kwargs):
            tick = time.perf_counter()
            result = _function(mesh, *values, **kwargs)
            nested.append({"name": _name, "seconds": time.perf_counter() - tick,
                           "vertices_after": len(mesh.points), "faces_after": len(mesh.faces)})
            return result
        setattr(remesh.Mesh, name, observe)

    stream = hashlib.sha256(); decisions = {"split": 0, "contract_called": 0, "contract_accepted": 0,
                                           "heap_pops": 0, "heap_pushes": 0}
    snapshots = []
    if args.decision_trace:
        pop, push = remesh.heapq.heappop, remesh.heapq.heappush
        def heap_pop(queue):
            value = pop(queue)
            caller = sys._getframe(1)
            if caller.f_code.co_filename == remesh.__file__:
                stream.update(b"P" + caller.f_code.co_name.encode() + struct.pack("<dq", *value))
                decisions["heap_pops"] += 1
            return value
        def heap_push(queue, value):
            result = push(queue, value)
            caller = sys._getframe(1)
            if caller.f_code.co_filename == remesh.__file__:
                stream.update(b"U" + caller.f_code.co_name.encode() + struct.pack("<dq", *value))
                decisions["heap_pushes"] += 1
            return result
        remesh.heapq.heappop, remesh.heapq.heappush = heap_pop, heap_push
        split_edge = remesh.split_edge
        contract = remesh.Mesh.contract
        def split(points, faces, edge_index, edge_vertices, edge_faces, face_edges, ei):
            a, b = edge_vertices[ei]
            stream.update(b"S" + struct.pack("<4q", ei, a, b, len(points)))
            result = split_edge(points, faces, edge_index, edge_vertices, edge_faces, face_edges, ei)
            stream.update(np.asarray(points[-1], dtype="<f8").tobytes())
            decisions["split"] += 1
            return result
        def collapse(mesh, ei, normals):
            a, b = mesh.edge_vertices[ei]
            result = contract(mesh, ei, normals)
            stream.update(b"C" + struct.pack("<3q?", ei, a, b, result))
            if result:
                stream.update(np.asarray(mesh.points[a], dtype="<f8").tobytes())
            decisions["contract_called"] += 1
            decisions["contract_accepted"] += int(result)
            return result
        remesh.split_edge, remesh.Mesh.contract = split, collapse
        def snapshot(name, points, faces, accepted):
            # Dynamic collapse contains removed faces before compaction; the
            # complete pass boundary is after its original compact/rebuild.
            xyz = np.asarray(points, dtype="<f8").reshape(-1, 3)
            triangles = np.asarray(faces, dtype="<i4").reshape(-1, 3)
            snapshots.append({"step": name, "accepted": accepted, "vertices": len(xyz),
                "faces": len(triangles), "float64_coordinates_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
                "ordered_faces_sha256": hashlib.sha256(triangles.tobytes()).hexdigest()})
        original_split, original_collapse, original_smooth = remesh.split_pass, remesh.Mesh.collapse_pass, remesh.smooth
        def split_pass(*items, **kwargs):
            accepted = original_split(*items, **kwargs)
            snapshot("split", items[0], items[1], accepted)
            return accepted
        def collapse_pass(mesh, *items, **kwargs):
            accepted = original_collapse(mesh, *items, **kwargs)
            snapshot("collapse", mesh.points, mesh.faces, accepted)
            return accepted
        def smooth(mesh, *items, **kwargs):
            result = original_smooth(mesh, *items, **kwargs)
            snapshot("smooth", mesh.points, mesh.faces, None)
            return result
        remesh.split_pass, remesh.Mesh.collapse_pass, remesh.smooth = split_pass, collapse_pass, smooth

    prior_gc = gc.isenabled(); gc_starts = {}; gc_rows = []
    def gc_observer(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_starts[generation] = time.perf_counter()
        else:
            start = gc_starts.pop(generation, None)
            if start is not None:
                gc_rows.append({"generation": generation, "seconds": time.perf_counter()-start,
                    "collected": info["collected"], "uncollectable": info["uncollectable"]})
    gc.callbacks.append(gc_observer)
    changed_gc = args.gc_policy == "suspend" and prior_gc
    deferred_gc_seconds = 0.0
    try:
        if changed_gc:
            gc.disable()
        status = worker(SimpleNamespace(output=args.output, candidate_source=args.source,
            candidate_version=args.version, stage="remesh", input=[args.input], threads=args.threads,
            profile=args.profile))
        if changed_gc:
            tick = time.perf_counter(); gc.collect()
            deferred_gc_seconds = time.perf_counter()-tick
    finally:
        gc.callbacks.remove(gc_observer)
        if changed_gc:
            gc.enable()
    detail = {"status": "complete" if status == 0 else "failed", "scope": "real same-input remesh detailed diagnostic; not whole pipeline or GPU port",
        "source_sha256": digest(Path(remesh.__file__)), "driver_sha256": digest(Path(__file__)),
        "reused_worker_sha256": digest(Path(sys.modules[worker.__module__].__file__)),
        "input_sha256": digest(args.input), "version": args.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "nested_steps": nested, "per_edge_trace_enabled": args.decision_trace,
        "per_edge_decisions": decisions if args.decision_trace else None,
        "per_edge_decision_stream_sha256": stream.hexdigest() if args.decision_trace else None,
        "complete_pass_snapshots": snapshots,
        "driver_seconds_including_setup_and_report_io": time.perf_counter() - entry,
        "gpu_used": False, "profile_enabled": args.profile,
        "scalar_storage": args.scalar_storage,
        "gc_policy": args.gc_policy, "gc_enabled_at_entry": prior_gc,
        "gc_enabled_after_restore": gc.isenabled(), "gc_observer": gc_rows,
        "deferred_gc_seconds_included_in_driver_wall": deferred_gc_seconds,
        "timing_scope": "coarse clock observers included; cProfile and per-edge trace diagnostic overhead recorded separately; cold/hot Numba cache selected by caller"}
    (args.output / "detail.json").write_text(json.dumps(detail, indent=2) + "\n")
    print("DONE remesh", args.version, args.output.name, status, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
