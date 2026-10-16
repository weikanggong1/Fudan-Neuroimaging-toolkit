"""对自产真实表面首缺陷的顺序边候选做只读剖析，不执行完整 GA。

输入为同一被试的 orig.nofix、qsphere.nofix、brain.mgz；输出是新目录内
profile.json 和可选的 cProfile 文件。坐标为 surface RAS 毫米，MRI 为
conformed uint8 网格。每次候选都重新建立同一初态；只测有序首候选，
它不代表 GA 实际抽取的首个候选，更不能代表最大缺陷。不写生产表面，
也不读取官方参考。失败会保留新目录并抛出异常。
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """读取三个具名输入，两次评估同一初态的顺序边候选，写 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-surface", type=Path, required=True)
    parser.add_argument("--input-qsphere", type=Path, required=True)
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--capture-output", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        raise ValueError("threads must be positive and output must not exist")
    inputs = {name: getattr(args, name) for name in ("input_surface", "input_qsphere", "brain")}
    input_sha256 = {name: _sha256(path) for name, path in inputs.items()}
    args.output.mkdir(parents=True, mode=0o700)

    from benchmark_recon_topology_gpu_first_ga import prepare
    from benchmark_recon_topology_candidate_state import dump_state_checkpoint
    from benchmark_recon_topology_prepared_cache import candidate_payload
    from fnit.recon_all.thread_budget import thread_budget
    from fnit.recon_all.topology_genetic_search import prepare_genetic_defect_evaluator

    runs = []
    with thread_budget(threads=args.threads):
        started = time.perf_counter()
        state, edges, candidate, preparation_seconds = prepare(args)
        initial_preparation_wall = time.perf_counter() - started

        def run_once(*, run_index: int, with_profiler: bool) -> dict:
            prepared = prepare_genetic_defect_evaluator(
                surface_state=state,
                defect_inputs=edges,
                candidate_inputs=candidate,
                face_detection_cache=True,
                blocker_scan_backend="numba",
                distance_backend="serial",
                state_backend="packed",
            )
            evaluator = prepared["evaluator"]
            retessellation = evaluator.retessellation_inputs
            profiler = cProfile.Profile() if with_profiler else None
            tick = time.perf_counter()
            if profiler is not None:
                profiler.enable()
            feedback = evaluator(
                ordering=np.arange(len(edges["edge_vertices"]), dtype=np.int32),
                discarded_defect_vertices=retessellation["inside_discarded"],
                vertex_rip=np.asarray(state["vertex_rip"], np.int32),
                use_counts=np.zeros(len(retessellation["inside_vertices"]), np.int32),
                vertex_fitness=np.zeros(len(retessellation["inside_vertices"]), np.float32),
            )
            if profiler is not None:
                profiler.disable()
            elapsed = time.perf_counter() - tick
            cycle = evaluator.last_result
            cand = cycle["candidate"]
            row = {
                "profiled": with_profiler,
                "factory_seconds": prepared["prepare_seconds"],
                "cycle_wall_seconds": elapsed,
                "cycle_subseconds": cycle["seconds"],
                "candidate_subseconds": cand["seconds"],
                "geometry_subseconds": cand["geometry"]["stage_seconds"],
                "fitness_subseconds": cand["fitness"]["seconds"],
                "fitness": float(feedback["fitness"]),
                "face_sha256": hashlib.sha256(np.ascontiguousarray(cand["geometry"]["faces"]).tobytes()).hexdigest(),
                "coordinate_sha256": hashlib.sha256(np.ascontiguousarray(cand["geometry"]["original_vertices"]).tobytes()).hexdigest(),
                "distance_policy": cand["distance_policy"],
                "edge_count": len(edges["edge_vertices"]),
                "selected_vertices": len(cand["geometry"]["selected_vertices"]),
            }
            if args.capture_output:
                candidate_arrays, control = candidate_payload(cand)
                candidate_path = args.output / f"candidate_{run_index}.private.npz"
                state_path = args.output / f"state_{run_index}.private.npz"
                np.savez_compressed(candidate_path, **candidate_arrays)
                dump_state_checkpoint(state_path, cycle["surface_state"])
                row["candidate_fields_sha256"] = {
                    key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for key, value in candidate_arrays.items()
                }
                with np.load(state_path, allow_pickle=False) as fields:
                    row["state_fields_sha256"] = {
                        key: hashlib.sha256(np.ascontiguousarray(fields[key]).tobytes()).hexdigest()
                        for key in fields.files
                    }
                row["control"] = control
            if profiler is not None:
                stats_path = args.output / "candidate.pstats"
                profiler.dump_stats(str(stats_path))
                stats = pstats.Stats(profiler)
                table = []
                for (filename, lineno, function), (primitive, calls, exclusive, cumulative, _) in stats.stats.items():
                    if "/fnit/recon_all/" in filename:
                        table.append({"file": filename.rsplit("/", 1)[-1], "line": lineno,
                                      "function": function, "calls": calls,
                                      "exclusive_seconds": exclusive, "cumulative_seconds": cumulative})
                row["profile_top_exclusive"] = sorted(table, key=lambda item: item["exclusive_seconds"], reverse=True)[:30]
                row["profile_top_cumulative"] = sorted(table, key=lambda item: item["cumulative_seconds"], reverse=True)[:30]
            return row

        runs.append(run_once(run_index=0, with_profiler=False))
        runs.append(run_once(run_index=1, with_profiler=args.profile))

    if (runs[0]["fitness"] != runs[1]["fitness"]
            or runs[0]["face_sha256"] != runs[1]["face_sha256"]
            or runs[0]["coordinate_sha256"] != runs[1]["coordinate_sha256"]
            or args.capture_output and any(
                runs[0][name] != runs[1][name]
                for name in ("candidate_fields_sha256", "state_fields_sha256", "control"))):
        raise RuntimeError("the two identical initial candidates did not reproduce")
    report = {
        "scope": "same-input ordered-edge candidate of defect 0 only; CPU distance; not complete GA or whole T1",
        "status": "complete",
        "host": os.uname().nodename,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "input_sha256": input_sha256,
        "script_sha256": _sha256(Path(__file__)),
        "loaded_source_sha256": {
            name: _sha256(Path(module.__file__)) for name, module in sys.modules.items()
            if name.startswith("fnit.recon_all.") and getattr(module, "__file__", "").endswith(".py")
        },
        "preparation_wall_seconds": initial_preparation_wall,
        "preparation_subseconds": preparation_seconds,
        "runs": runs,
    }
    (args.output / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "wall_seconds": [r["cycle_wall_seconds"] for r in runs],
                      "profile": str(args.output / "profile.json")}))


if __name__ == "__main__":
    main()
