"""冻结自产white的完整pial；只替换source清理标记，并保存实际清理前状态。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import nibabel.freesurfer.io as fs
import numpy as np
import torch

from benchmark_placement_full_pial import sha256


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-base-commit", required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--order", nargs="+", choices=("numba", "torch"),
                        default=["numba", "torch", "torch", "numba"])
    parser.add_argument("--legacy-baseline-report", type=Path)
    parser.add_argument("--legacy-baseline-surface", type=Path)
    args = parser.parse_args()
    if args.order not in (["numba", "torch"], ["torch", "numba"],
                          ["numba", "torch", "torch", "numba"],
                          ["torch", "numba", "numba", "torch"]):
        raise ValueError("complete pair or ABBA with both implementations required")
    if (args.threads < 1 or args.max_steps < 1 or args.output_directory.exists()
            or (args.legacy_baseline_report is None) != (args.legacy_baseline_surface is None)):
        raise ValueError("positive budgets, new output directory and paired baseline paths required")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("explicit CUDA device required")
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory.resolve()))
    from fnit.recon_all import place_pial_python as stage
    from fnit.recon_all.place_surface_intersection_marking import mark_source_intersections
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hemi = args.hemisphere
    inputs = [args.subject/f"surf/{hemi}.white", args.subject/f"surf/autodet.gw.stats.{hemi}.dat",
              args.subject/f"label/{hemi}.cortex.label", args.subject/f"label/{hemi}.cortex+hipamyg.label"]
    inputs += [args.subject/f"mri/{name}.mgz" for name in ("brain.finalsurfs", "wm", "aseg.presurf")]
    report = {"scope": "frozen_self_generated_complete_pial_source_cleanup_only_not_recon_all",
              "code_base_commit": args.code_base_commit, "hostname": platform.node(),
              "input_sha256": {p.name: sha256(p) for p in inputs}, "torch": torch.__version__,
              "threads": args.threads, "cpu_affinity_count": len(os.sched_getaffinity(0)),
              "device": str(device), "tf32_matmul": True, "tf32_cudnn": True, "half_precision": False,
              "cuda_allocator_environment": {key: os.environ.get(key) for key in
                  ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
              "fixed_placement_strategy": {"sampling_backend": "cpu", "regularization_backend": "cpu",
                  "candidate_backend": "torch_snapshot", "candidate_grid_cells_per_axis": 3,
                  "retained_mht_backend": "compiled"},
              "required_tolerance_before_run": {"all_trial_and_step_trace": 0,
                  "precleanup_arrays": 0, "source_cleanup_trace": 0, "ordered_final_coordinates_mm": 0},
              "timing_scope": "synchronized complete API with input validation, reads, trace callback, actual precleanup NPZ write and final surface write; imports/CUDA setup separate; independent QC outside",
              "order": args.order, "runs": [], "cleanup_replays": [],
              "sha256_helper_source_sha256": sha256(sys.modules[sha256.__module__].__file__),
              "overall_metric_equivalence": "not_assessed", "status": "started"}

    def save(status):
        report["status"] = status
        report["source_sha256"] = {Path(m.__file__).name: sha256(m.__file__) for name, m in list(sys.modules.items())
            if name.startswith("fnit.recon_all.place_") and getattr(m, "__file__", None)}
        report["script_sha256"] = sha256(__file__)
        tmp = args.output_directory/"report.tmp"
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        tmp.replace(args.output_directory/"report.json")

    save("started")
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    report["CUDA_setup_seconds"] = time.perf_counter()-started
    report["gpu"] = torch.cuda.get_device_name(device)
    original_cleanup = stage.repair_intersections
    try:
        for index, backend in enumerate(args.order):
            output = args.output_directory/f"{hemi}.pial.{index}.{backend}"
            captured = {}
            trace = []

            def observe_cleanup(vertices, faces, ripped, **kwargs):
                if captured:
                    raise RuntimeError("unexpected repeated final cleanup call")
                checkpoint = args.output_directory/f"precleanup-{index}.npz"
                np.savez(checkpoint, vertices=vertices, faces=faces, ripped=ripped)
                captured.update(path=str(checkpoint), array_sha256={"vertices": array_sha(vertices),
                    "faces": array_sha(faces), "ripped": array_sha(ripped)}, checkpoint_sha256=sha256(checkpoint),
                    ripped_vertices=int(np.count_nonzero(ripped)))
                return original_cleanup(vertices, faces, ripped, **kwargs)

            def callback(step, outer_pass, coordinates, diagnostics):
                trace.append({"step": step, "pass": outer_pass,
                              "coordinate_sha256": array_sha(coordinates), "diagnostics": diagnostics})
                print(json.dumps({"run": index, "backend": backend, "step": step, "pass": outer_pass}), flush=True)

            report["active_run"] = {"index": index, "backend": backend,
                                    "trace": trace, "precleanup": captured}
            save("running")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            stage.repair_intersections = observe_cleanup
            tick = time.perf_counter()
            try:
                result = stage.place_pial_t1(subject=args.subject, hemisphere=hemi, output=output,
                    max_steps=args.max_steps, sampling_backend="cpu", regularization_backend="cpu",
                    candidate_backend="torch_snapshot", candidate_grid_cells_per_axis=3,
                    retained_mht_backend="compiled", cleanup_marking_backend="source_"+backend,
                    cleanup_candidate_grid_cells_per_axis=3 if backend == "torch" else 2,
                    device=str(device), trace_callback=callback, profile=True)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter()-tick
            finally:
                stage.repair_intersections = original_cleanup
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            vertices, faces = fs.read_geometry(str(output))
            _, count = mark_source_intersections(vertices=vertices, faces=faces, predicate_backend="torch",
                                                 device=str(device), candidate_grid_cells_per_axis=3)
            row = {"backend": backend, "wall_seconds": elapsed, "stage": result, "trace": trace,
                   "precleanup": captured, "output_sha256": sha256(output),
                   "coordinate_sha256": array_sha(vertices), "ordered_faces_sha256": array_sha(faces),
                   "quality_not_in_timing": {"finite_coordinates": bool(np.isfinite(vertices).all()),
                       "source_intersecting_face_count": int(count)},
                   "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}
            report["runs"].append(row)
            del report["active_run"]
            save("running")

        first = report["runs"][0]
        report["strict_backend_reproduction"] = "passed" if all(
            r["trace"] == first["trace"] and r["precleanup"]["array_sha256"] == first["precleanup"]["array_sha256"]
            and r["coordinate_sha256"] == first["coordinate_sha256"]
            and r["ordered_faces_sha256"] == first["ordered_faces_sha256"]
            and r["stage"]["cleanup"] == first["stage"]["cleanup"] for r in report["runs"]) else "failed"
        report["mesh_quality_status"] = "passed" if all(r["quality_not_in_timing"]["finite_coordinates"]
            and r["quality_not_in_timing"]["source_intersecting_face_count"] == 0 for r in report["runs"]) else "failed"
        report["median_complete_API_seconds"] = {b: statistics.median(r["wall_seconds"] for r in report["runs"]
            if r["backend"] == b) for b in ("numba", "torch")}
        checkpoint = Path(first["precleanup"]["path"])
        # Frozen real precleanup arrays; repeat the entire existing cleanup, not only its marker.
        for index, backend in enumerate(("numba", "torch", "torch", "numba")):
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with np.load(checkpoint) as arrays:
                vertices, faces, ripped = (arrays[n].copy() for n in ("vertices", "faces", "ripped"))
            repaired, cleanup = original_cleanup(vertices, faces, ripped,
                marking_backend="source_"+backend, device=str(device),
                candidate_grid_cells_per_axis=3 if backend == "torch" else 2)
            replay_output = args.output_directory/f"replay-{index}-{backend}.surface"
            stage._write_vertices_like(args.subject/f"surf/{hemi}.white", replay_output, repaired)
            torch.cuda.synchronize(device)
            report["cleanup_replays"].append({"backend": backend,
                "full_read_cleanup_write_seconds": time.perf_counter()-tick, "cleanup": cleanup,
                "coordinate_sha256": array_sha(repaired), "surface_sha256": sha256(replay_output)})
            save("running_replays")
        replay0 = report["cleanup_replays"][0]
        report["strict_cleanup_replay"] = "passed" if all(r["cleanup"] == replay0["cleanup"]
            and r["coordinate_sha256"] == replay0["coordinate_sha256"] for r in report["cleanup_replays"]) else "failed"
        if args.legacy_baseline_report is not None:
            legacy = json.loads(args.legacy_baseline_report.read_text())
            lv, lf = fs.read_geometry(str(args.legacy_baseline_surface))
            current, cf = fs.read_geometry(str(Path(first["stage"]["output"])))
            same = lv.shape == current.shape and np.array_equal(lf, cf)
            report["source_semantics_change_vs_legacy_not_gpu_change"] = {
                "legacy_report_sha256": sha256(args.legacy_baseline_report),
                "same_input_sha256": legacy["input_sha256"] == report["input_sha256"],
                "same_optimization_trace": legacy["stages"]["torch"]["trace"] == first["trace"],
                "same_ordered_faces": same,
                "different_final_coordinate_elements": int(np.count_nonzero(lv != current)) if same else None,
                "max_final_distance_mm": float(np.linalg.norm(lv-current, axis=1).max()) if same else None,
                "legacy_cleanup": legacy["stages"]["torch"]["stage"]["cleanup"],
                "source_cleanup": first["stage"]["cleanup"]}
        report["input_sha256_after"] = {p.name: sha256(p) for p in inputs}
        if report["input_sha256_after"] != report["input_sha256"]:
            raise RuntimeError("frozen inputs changed")
        save("complete")
        if any(report[k] != "passed" for k in ("strict_backend_reproduction", "mesh_quality_status", "strict_cleanup_replay")):
            raise SystemExit(1)
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        save("failed")
        raise
    finally:
        stage.repair_intersections = original_cleanup


if __name__ == "__main__":
    main()
