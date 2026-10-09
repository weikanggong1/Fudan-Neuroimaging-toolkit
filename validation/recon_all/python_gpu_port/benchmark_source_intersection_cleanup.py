"""真实白质平滑orig：源有向标记/完整清理对照，不输出生产表面。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import nibabel.freesurfer.io as fs
import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="可选冻结orig平滑5次的NPZ；未提供时直接从orig重新计算")
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--device", help="显式Torch设备；另做静态标记回归")
    parser.add_argument("--cleanup-marking-backend", choices=("source_numba", "source_torch"),
                        default="source_numba", help="完整清理的有向标记后端，默认source_numba")
    args = parser.parse_args()
    if args.cleanup_marking_backend == "source_torch" and args.device is None:
        parser.error("source_torch requires --device")
    complete_started = time.perf_counter()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory.resolve()))
    from fnit.recon_all import place_surface_final_cleanup as cleanup
    from fnit.recon_all import place_surface_intersection_marking as marking
    from fnit.recon_all import mris_remove_intersection_python as legacy
    from fnit.recon_all import place_surface_collision as source_predicate
    from fnit.recon_all import place_surface_collision_torch as pair_predicate
    from fnit.recon_all.place_pial_python import _write_vertices_like
    from fnit.recon_all.place_surface_smoothing import average_vertex_positions
    xyz, orig_faces = fs.read_geometry(str(args.orig))
    fresh = average_vertex_positions(xyz, orig_faces, 5)
    if args.initial_checkpoint is not None:
        data = np.load(args.initial_checkpoint)
        initial, faces = data["initial"], data["faces"]
        if not np.array_equal(faces, orig_faces) or not np.array_equal(initial, fresh):
            raise RuntimeError("checkpoint is not the same orig -> five-step smoothing")
    else:
        initial, faces = fresh, orig_faces
    report = {"scope": "frozen_real_initial_white_cleanup_only_not_complete_white",
        "hostname": platform.node(), "code_commit": args.code_commit,
        "script_sha256": sha(__file__), "orig_sha256": sha(args.orig),
        "checkpoint_sha256": sha(args.initial_checkpoint) if args.initial_checkpoint is not None else None,
        "source_sha256": {
            module.__name__: sha(module.__file__) for module in (cleanup, marking, legacy, source_predicate, pair_predicate)},
        "native_sha256": sha(args.native_binary), "vertices": len(initial), "faces": len(faces),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "cuda_allocator_environment": {key: os.environ.get(key) for key in
            ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "cleanup_marking_backend": args.cleanup_marking_backend,
        "execution_status": "running", "source_calls": []}

    def save():
        path = args.output_directory / "report.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)

    save()
    cuda = None
    if args.cleanup_marking_backend == "source_torch":
        import torch
        cuda = torch.device(args.device)
        if cuda.type == "cuda":
            torch.cuda.synchronize(cuda)
            torch.cuda.reset_peak_memory_stats(cuda)
    tick = time.perf_counter()
    lm, lc = legacy.mark_intersections(initial, faces)
    report["legacy_initial"] = {"count": lc, "marked_vertices": int(lm.sum()), "seconds": time.perf_counter() - tick}
    tick = time.perf_counter()
    sm, sc = marking.mark_source_intersections(initial, faces,
        predicate_backend="torch" if args.cleanup_marking_backend == "source_torch" else "numba",
        device=args.device)
    if cuda is not None and cuda.type == "cuda":
        torch.cuda.synchronize(cuda)
    report["source_initial"] = {"count": sc, "marked_vertices": int(sm.sum()),
        "different_marked_vertices_from_legacy": int(np.count_nonzero(sm != lm)),
        "seconds": time.perf_counter() - tick}
    save()
    input_surface = args.output_directory / "smoothed-orig.diagnostic-surface"
    _write_vertices_like(args.orig, input_surface, initial)
    native_map = args.output_directory / "native-initial-mark.mgz"
    map_command = [str(args.native_binary.resolve()), "-map", str(input_surface.resolve()), str(native_map.resolve())]
    with (args.output_directory / "native-map.log").open("w") as stream:
        mapped = subprocess.run(map_command, stdout=stream, stderr=subprocess.STDOUT)
    report["native_initial_mark"] = {"command": map_command, "returncode": mapped.returncode}
    if mapped.returncode == 0:
        native_marks = np.asarray(nib.load(str(native_map)).dataobj).reshape(-1) != 0
        report["native_initial_mark"].update(marked_vertices=int(native_marks.sum()),
            different_source_marked_vertices=int(np.count_nonzero(sm != native_marks)),
            different_legacy_marked_vertices=int(np.count_nonzero(lm != native_marks)))
    save()
    original = marking.mark_source_intersections

    def observed(vertices, faces, **kwargs):
        tick = time.perf_counter()
        marked, count = original(vertices, faces, **kwargs)
        report["source_calls"].append({"count": count, "marked_vertices": int(marked.sum()),
            "coordinate_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
            "seconds": time.perf_counter() - tick})
        save()
        return marked, count

    marking.mark_source_intersections = observed
    file_api_started = time.perf_counter()
    reread, reread_faces = fs.read_geometry(str(input_surface))
    read_seconds = time.perf_counter() - file_api_started
    tick = time.perf_counter()
    result, stats = cleanup.repair_intersections(reread, reread_faces, np.zeros(len(initial), bool),
        marking_backend=args.cleanup_marking_backend, device=args.device)
    if cuda is not None and cuda.type == "cuda":
        torch.cuda.synchronize(cuda)
    report["source_cleanup"] = {"seconds": time.perf_counter() - tick, "diagnostics": stats}
    if cuda is not None and cuda.type == "cuda":
        report["source_cleanup"].update(allocated_peak_bytes=torch.cuda.max_memory_allocated(cuda),
            reserved_peak_bytes=torch.cuda.max_memory_reserved(cuda))
    source_surface = args.output_directory / "source-cleanup.diagnostic-surface"
    write_started = time.perf_counter()
    _write_vertices_like(args.orig, source_surface, result)
    report["source_file_api"] = {"seconds_including_read_compute_write": time.perf_counter() - file_api_started,
        "read_seconds": read_seconds, "write_seconds": time.perf_counter() - write_started,
        "scope": "warm_imported_file_API; CUDA_setup_and_initial_diagnostic_excluded; trace_IO_included"}
    np.savez(args.output_directory / "source_cleanup_checkpoint.npz", initial=initial, cleaned=result, faces=faces)
    save()
    if args.device is not None:
        import torch
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        gm, gc = original(initial, faces, predicate_backend="torch", device=str(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["torch_initial"] = {"device": str(device), "count": gc,
            "seconds": time.perf_counter() - tick, "different_marked_vertices": int(np.count_nonzero(gm != sm)),
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        save()
    command = [str(args.native_binary.resolve()), str(input_surface.resolve()),
               str((args.output_directory / "native-cleanup.diagnostic-surface").resolve())]
    tick = time.perf_counter()
    log = args.output_directory / "native.log"
    with log.open("w") as stream:
        run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    report["native"] = {"command": command, "returncode": run.returncode,
        "seconds": time.perf_counter() - tick,
        "cycle_counts": [int(value) for value in re.findall(r"\d{3}: (\d+) intersecting", log.read_text())],
        "terminal_count": re.findall(r"terminating search with (\d+) intersecting", log.read_text())}
    if run.returncode == 0:
        native, nf = fs.read_geometry(command[-1])
        ordered = np.array_equal(faces, nf) and result.shape == native.shape
        distances = np.linalg.norm(result - native, axis=1) if ordered else None
        report["source_native_comparison"] = {"ordered_faces_same": ordered,
            "different_coordinate_elements": int(np.count_nonzero(result != native)) if ordered else None,
            "p99_distance_mm": float(np.percentile(distances, 99)) if ordered else None,
            "max_distance_mm": float(distances.max()) if ordered else None}
    report["execution_status"] = "complete" if run.returncode == 0 else "failed_native"
    report["final_zero_intersection_gate"] = "passed" if stats["intersecting_faces_after"] == 0 else "failed"
    report["benchmark_total_wall_seconds"] = time.perf_counter() - complete_started
    save()
    raise SystemExit(0 if run.returncode == 0 else 1)


if __name__ == "__main__":
    main()
