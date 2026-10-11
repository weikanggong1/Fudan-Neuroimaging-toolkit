"""完整冻结候选序列的只读局部性/冷热剖析；不改变候选或 GA 算法。"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import inspect
import json
import os
from pathlib import Path
import pstats
import runpy
import sys
import time


def run(*, root: Path, workspace: Path, output: Path, threads: int = 4) -> dict:
    """同一已验证冻结序列完整执行；记录每核实际 N/F/patch/迭代和时间。

    root 是授权 FNIT 根；workspace 含本轮 overlay 五模块及源码清单；
    output 必须不存在。threads 固定总预算，CUDA 必须隐藏。读取已有
    阶段前置作冻结诊断，参考 after 只由成熟工具在自产输出保存后比较。
    不裁剪任何计算、修改输出、RNG、控制器、线程精度或参考初态。
    cProfile 增加 Python 开销，因此此报告不作为配对提速基线。
    """
    if output.exists() or threads < 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("fresh output, positive threads and explicitly hidden CUDA required")
    output.mkdir(mode=0o700, parents=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen = root / "workspaces/topology_ga_finalize_20261010_v2"
    base = root / "workspaces/topology_candidate_cycle_20261010_v2"
    for name, meta in json.loads((frozen / "PATCH_SOURCE.json").read_text())["payload"].items():
        if sha(frozen / name) != meta["sha256"] or (frozen / name).stat().st_size != meta["size"]:
            raise RuntimeError("original frozen payload changed")
    manifest = json.loads((workspace / "SOURCE_MANIFEST.json").read_text())
    for name, meta in manifest["payload"].items():
        if sha(workspace / name) != meta["sha256"] or (workspace / name).stat().st_size != meta["bytes"]:
            raise RuntimeError("profile source bundle changed")
    sys.path[:0] = [str(frozen / "tools"), str(base / "tools"), str(base / "src")]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(workspace / "src/fnit/recon_all"))
    fnit.recon_all.__path__.insert(0, str(frozen / "src/fnit/recon_all"))
    import numpy as np
    import numba
    numba.set_num_threads(threads)
    calls, profiles = [], []

    def observed(name, function, count):
        signature = inspect.signature(function)
        def measure(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            row = dict(function=name, ordinal=sum(c["function"] == name for c in calls),
                       **count(bound.arguments))
            tick = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            finally:
                row["seconds"] = time.perf_counter() - tick
                calls.append(row)
            return result
        return measure

    def xyz_faces(a):
        points = a.get("xyz", a.get("original_vertices"))
        return dict(vertices=len(points), faces=len(a["faces"]))

    from fnit.recon_all import topology_defect_smooth as smooth
    from fnit.recon_all import topology_defect_mri_match as match
    smooth._smooth_core = observed("smooth_compiled_25", smooth._smooth_core,
        lambda a: dict(**xyz_faces(a), inside_vertices=len(a["inside"]),
                       normal_faces_evaluated=25 * len(a["faces"]),
                       coordinate_elements_copied_at_least=26 * np.asarray(a["xyz"]).size))
    match._match = observed("match_compiled_40", match._match,
        lambda a: dict(**xyz_faces(a), inside_vertices=len(a["inside"]), iterations=a["iterations"],
                       normal_faces_evaluated=a["iterations"] * len(a["faces"]),
                       mri_samples=len(a["inside"]) * a["iterations"] * 3,
                       coordinate_elements_copied_at_least=(a["iterations"] + 1) * np.asarray(a["xyz"]).size))
    from fnit.recon_all import topology_candidate_normals as normal
    normal.candidate_face_normals = observed("complete_face_normals", normal.candidate_face_normals,
        lambda a: dict(**xyz_faces(a), selected_faces=len(a["selected_faces"])))
    from fnit.recon_all import topology_candidate_sff as sff
    sff.candidate_second_fundamental_form = observed("selected_sff", sff.candidate_second_fundamental_form,
        lambda a: dict(vertices=len(a["original_vertices"]), selected_vertices=len(a["patch_vertices"]),
                       full_neighbor_entries=len(a["neighbors"])))
    from fnit.recon_all import topology_candidate_geometry as geometry
    from fnit.recon_all import topology_retessellation_dynamic as retess
    # 对原 detect 的完整全域构造只计数/计时，不替换为新算法。
    retess.detect_patch_faces = observed("complete_detect_patch_faces", retess.detect_patch_faces,
        lambda a: dict(vertices=len(a["canonical"]), selected_vertices=len(a["vertices"]),
                       base_faces=len(a["external_faces"]), neighbor_rows_converted_to_sets=len(a["neighbors"]),
                       global_face_edge_entries=3 * len(a["external_faces"])))

    def profile(name, function):
        def invoke(*args, **kwargs):
            profiler = cProfile.Profile()
            index = sum(row["function"] == name for row in profiles)
            tick = time.perf_counter()
            try:
                return profiler.runcall(function, *args, **kwargs)
            finally:
                stats = pstats.Stats(profiler)
                rows = []
                for (file, line, symbol), (_, count, exclusive, cumulative, _) in stats.stats.items():
                    if "fnit/recon_all/" not in file:
                        continue
                    rows.append(dict(file="fnit/recon_all/" + file.split("fnit/recon_all/", 1)[1],
                                     line=line, function=symbol, calls=count,
                                     exclusive_seconds=exclusive, cumulative_seconds=cumulative))
                profiles.append(dict(function=name, ordinal=index, seconds=time.perf_counter()-tick,
                                     total_profile_calls=stats.total_calls,
                                     top_fnit_functions=sorted(rows, key=lambda r:r["cumulative_seconds"], reverse=True)[:25]))
        return invoke

    geometry.candidate_geometry = profile("candidate_geometry", geometry.candidate_geometry)
    program = frozen / "tools/benchmark_recon_topology_ga_finalize.py"
    namespace = runpy.run_path(str(program), run_name="fnit_locality_readonly_target")
    from fnit.recon_all import topology_candidate_cycle as cycle
    old_state = cycle._candidate_state
    cycle._candidate_state = profile("candidate_state", observed("full_state_assembly", old_state,
        lambda a: dict(vertices=len(a["initial"]["original_vertices"]),
                       ordered_rows_validated=3 * len(a["initial"]["original_vertices"]))))
    subject = root / "workspaces/recon_topology_source_mri_edges_20261010_v2/input/subject"
    reference = root / "runs/topology_source_fitness_observer_20261010_v2/repeat0"
    args = dict(before_retess=root / "runs/topology_source_retessellation_observer_20261010_v3/repeat0/retess.before.bin",
                initial_attributes=root / "runs/topology_source_match_observer_20261010_v2/repeat0/match.before.bin",
                initial_global_priors=reference / "statistics.bin", input_surface=subject / "surf/lh.orig.nofix",
                input_qsphere=subject / "surf/lh.qsphere.nofix", brain=subject / "mri/brain.mgz",
                after_statistics_reference=reference / "patch_statistics.bin",
                after_volume_reference=reference / "volume.bin", fitness_reference=reference / "fitness.json",
                output=output / "candidate", threads=threads)
    tick = time.perf_counter()
    result = namespace["run"](**args)
    loaded = {name:sha(module.__file__) for name, module in sorted(sys.modules.items())
              if name.startswith("fnit.") and getattr(module, "__file__", None) and str(module.__file__).endswith(".py")}
    for name in ("smooth_surface_python", "topology_candidate_cycle", "topology_candidate_geometry",
                 "topology_defect_smooth", "topology_defect_mri_match"):
        if loaded["fnit.recon_all." + name] != sha(workspace / "src/fnit/recon_all" / (name + ".py")):
            raise RuntimeError("actual tested source differs from declared profile overlay")
    report = dict(schema_version=1, status="complete_readonly_profile", calls=calls, profiles=profiles,
                  complete_sequence_seconds=time.perf_counter()-tick,
                  records=result["records"], source_first_gate=result["initial_candidate_strict_source_gate"],
                  final_repeat_gate=result["status"], threads=threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_memory_bytes=None,
                  original_target_sha256=sha(program), driver_sha256=sha(__file__),
                  source_manifest_sha256=sha(workspace / "SOURCE_MANIFEST.json"),
                  actual_fnit_loaded_source_sha256=loaded,
                  input_sha256={name:sha(value) for name,value in args.items() if isinstance(value, Path) and name != "output"},
                  reference_after_used_to_generate=False, complete_ga=False,
                  caveat="cProfile increases Python overhead; cold JIT call separated; no paired speedup claim")
    (output / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "COMPLETE.json").write_text(json.dumps(dict(status=report["status"])) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(root=args.root, workspace=args.workspace, output=args.output, threads=args.threads)
