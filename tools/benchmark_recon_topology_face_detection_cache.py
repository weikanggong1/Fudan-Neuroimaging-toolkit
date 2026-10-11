"""复用完整候选/末态验证工具，比较保留面索引缓存的同输入ABBA。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import benchmark_recon_topology_candidate_hotspots as original
from benchmark_recon_topology_candidate_incident_reuse import compare_fields

MODULES = ("smooth_surface_python", "topology_candidate_cycle", "topology_candidate_geometry",
           "topology_defect_smooth", "topology_defect_mri_match", "topology_retessellation_dynamic")
PATHS = ("finalization_workspace", "cycle_workspace", "baseline_overlay_source", "overlay_source", *original.INPUTS)


def worker(args):
    """完整已有序列，仅显式注入本次面索引上下文；准备/导入/写出计时。"""
    started = time.perf_counter()
    backend = args.worker
    selected = args.overlay_source if backend == "candidate" else args.baseline_overlay_source
    sys.path[:0] = [str(args.finalization_workspace / "tools"),
                   str(args.cycle_workspace / "tools"), str(args.cycle_workspace / "src")]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.finalization_workspace / "src/fnit/recon_all"))
    fnit.recon_all.__path__.insert(0, str(selected / "fnit/recon_all"))
    from fnit.recon_all import topology_retessellation_dynamic as dynamic
    from fnit.recon_all import topology_candidate_geometry as geometry
    from fnit.recon_all import topology_candidate_cycle
    calls, contexts = [], []
    if backend == "candidate":
        from fnit.recon_all.topology_face_detection_cache import prepare_face_detection_context
    compute = geometry.retessellate_and_detect_candidate

    def observed(*, canonical, **inputs):
        tick = time.perf_counter()
        prepared_seconds = 0.
        if backend == "candidate":
            import numpy as np
            context = next((value for value in contexts if value.vertex_count == len(canonical) and
                           value.faces.shape == inputs["base_faces"].shape and
                           np.array_equal(value.faces, inputs["base_faces"])), None)
            if context is None:
                before = time.perf_counter()
                context = prepare_face_detection_context(base_faces=inputs["base_faces"],
                                                         vertex_count=len(canonical))
                prepared_seconds = time.perf_counter()-before
                contexts.append(context)
            result = compute(canonical=canonical, face_detection_context=context, **inputs)
        else:
            result = compute(canonical=canonical, **inputs)
        calls.append(dict(seconds=time.perf_counter()-tick, context_prepare_seconds=prepared_seconds,
            vertices=len(canonical), base_faces=len(inputs["base_faces"]),
            actual_selected_vertices=len(result["selected_vertices"]),
            accepted_faces=len(result["patch_faces"])))
        return result

    geometry.retessellate_and_detect_candidate = observed
    # 成熟worker的candidate仅表示显式source覆盖；两个分支都使用固定fc78五模块。
    # 它先保存自产结果，再读取冻结after参考；不重新实现评分/恢复/末态写出。
    delegated = argparse.Namespace(**vars(args))
    delegated.worker = "candidate"
    delegated.overlay_source = selected
    receipt = original.worker(delegated)
    for name in MODULES:
        if receipt["actual_fnit_loaded_source_sha256"]["fnit.recon_all." + name] != original.sha(
                selected / "fnit/recon_all" / (name + ".py")):
            raise RuntimeError("unexpected actual selected source: " + name)
    receipt.update(backend=backend, face_detection_calls=calls, context_preparations=len(contexts),
        child_seconds_including_imports=time.perf_counter()-started,
        driver_sha256=original.sha(__file__), delegate_driver_sha256=original.sha(original.__file__),
        baseline_implementation="complete fc78 candidate and final stage",
        optimization="explicit immutable base face/edge indices; dynamic adjacency and decisions unchanged")
    original.write(args.output / "receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PATHS, "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--worker", choices=("control", "candidate"))
    args = parser.parse_args()
    if args.worker:
        worker(args); return
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started, records = time.perf_counter(), []
    for index, backend in enumerate(("control", "candidate", "candidate", "control")):
        folder = args.output / f"{index}-{backend}"
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in PATHS:
            command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
        command.extend(("--output", str(folder), "--threads", str(args.threads), "--worker", backend))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NUMBA_CACHE_DIR=str(args.output / f"cache-{backend}"))
        tick = time.perf_counter()
        with (args.output / f"{index}-{backend}.private.log").open("wb") as output:
            child = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        wall = time.perf_counter()-tick
        if child.returncode:
            original.write(args.output / "EXIT.json", dict(exit_code=child.returncode, failed_index=index))
            raise RuntimeError("full stage failed; diagnostics retained")
        receipt = json.loads((folder / "receipt.json").read_text())
        receipt.update(index=index, fresh_cli_seconds=wall); records.append(receipt)
        print(json.dumps(dict(index=index, backend=backend, fresh_cli_seconds=wall)), flush=True)
    checks = [dict(index=row["index"], groups=compare_fields(args.output / "0-control/candidate",
        args.output / f"{row['index']}-{row['backend']}" / "candidate")) for row in records]
    passed = all(all(group.get("equal", True) and group.get("keys_equal", True) and
        all(field["shape_equal"] and field["dtype_equal"] and field["byte_equal"]
            for field in group.get("fields", {}).values()) for group in row["groups"].values()) for row in checks)
    report = dict(status="passed" if passed else "failed", records=records, comparisons=checks,
        complete_driver_seconds=time.perf_counter()-started, order=["control", "candidate", "candidate", "control"],
        driver_sha256=original.sha(__file__), delegate_driver_sha256=original.sha(original.__file__),
        reference_after_used_to_generate=False, complete_ga=False, whole_recon_all="not_assessed",
        final_mesh_quality="not_assessed", overall_metric_equivalence="not_assessed",
        production_default_changed=False, scope="full same-input candidate/finalization ABBA, not full GA or raw-T1")
    original.write(args.output / "report.private.json", report)
    original.write(args.output / "EXIT.json", dict(exit_code=0 if passed else 1))
    if not passed:
        raise RuntimeError("full state/control comparison failed")


if __name__ == "__main__":
    main()
