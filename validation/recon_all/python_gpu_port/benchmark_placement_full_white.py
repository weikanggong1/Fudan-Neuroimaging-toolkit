"""固定真实 orig/MRI 的完整 white.preaparc 对照；只用于隔离 benchmark。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_geometry(first, second):
    a, af = fs.read_geometry(str(first))
    b, bf = fs.read_geometry(str(second))
    ordered = a.shape == b.shape and np.array_equal(af, bf)
    distances = np.linalg.norm(a - b, axis=1) if ordered else None
    return {
        "same_vertex_count_and_ordered_faces": ordered,
        "same_file_bytes": sha256(first) == sha256(second),
        "different_coordinate_elements": int(np.count_nonzero(a != b)) if ordered else None,
        "mean_vertex_distance_mm": float(distances.mean()) if ordered else None,
        "p99_vertex_distance_mm": float(np.percentile(distances, 99)) if ordered else None,
        "max_vertex_distance_mm": float(distances.max()) if ordered else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--candidate-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-base-commit", required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "torch"), default=("cpu", "torch"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--candidate-backend", choices=("tree", "snapshot", "torch_snapshot"), default="tree")
    parser.add_argument("--control-candidate-backend", choices=("tree", "snapshot", "torch_snapshot"))
    parser.add_argument("--candidate-regularization-backend", choices=("cpu", "torch"), default="torch")
    parser.add_argument("--sampling-backend", choices=("cpu", "torch", "triton"), default="cpu")
    parser.add_argument("--cleanup-marking-backend", choices=("legacy", "source_numba", "source_torch"), default="legacy")
    parser.add_argument("--official-binary", type=Path)
    parser.add_argument("--conda-binary", type=Path)
    parser.add_argument("--assets-directory", type=Path)
    parser.add_argument("--reference-repeat", type=int, default=2)
    parser.add_argument("--native-only", action="store_true", help="仅重跑具名原生参考，不执行Python候选")
    args = parser.parse_args()
    if args.threads < 1 or args.max_steps < 1 or args.reference_repeat < 1:
        parser.error("threads/max-steps/reference-repeat must be positive")
    if len(set(args.backends)) != len(args.backends):
        parser.error("backends cannot contain duplicates")
    native = [(name, binary) for name, binary in
              (("official", args.official_binary), ("conda", args.conda_binary)) if binary is not None]
    if native and args.assets_directory is None:
        parser.error("native reference requires --assets-directory")
    if args.native_only:
        if not native:
            parser.error("--native-only requires an explicit reference binary")
        args.backends = []
    device = torch.device(args.device)
    gpu_requested = ("torch" in args.backends or args.candidate_backend == "torch_snapshot"
                     or args.cleanup_marking_backend == "source_torch")
    if gpu_requested and device.type == "cuda" and device.index is None:
        parser.error("CUDA benchmarking requires an explicitly indexed device")
    args.subject = args.subject.resolve()
    args.output_directory = args.output_directory.resolve()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.candidate_directory.resolve()))
    from fnit.recon_all import place_white_preaparc_python as stage
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hemi = args.hemisphere
    orig = args.subject / f"surf/{hemi}.orig"
    stats = args.subject / f"surf/autodet.gw.stats.{hemi}.dat"
    brain, wm, seg = (args.subject / f"mri/{name}.mgz"
                      for name in ("brain.finalsurfs", "wm", "aseg.presurf"))
    inputs = (orig, stats, brain, wm, seg)
    report = {
        "scope": "frozen_same_input_complete_white_preaparc_only",
        "hostname": platform.node(), "platform": platform.platform(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "threads": args.threads, "thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "code_base_commit": args.code_base_commit,
        "source_version": "actual imported source_sha256, including any patch relative to commit",
        "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "half_precision": False,
        "input_sha256": {str(path.relative_to(args.subject)): sha256(path) for path in inputs},
        "script_sha256": sha256(__file__), "source_sha256": {},
        "python_order": args.backends, "python_runs": {}, "native_runs": {},
        "sampling_backend_candidate": args.sampling_backend,
        "control_candidate_backend": args.control_candidate_backend or args.candidate_backend,
        "candidate_candidate_backend": args.candidate_backend,
        "candidate_regularization_backend": args.candidate_regularization_backend,
        "cleanup_marking_backend": args.cleanup_marking_backend,
        "cuda_allocator_environment": {key: os.environ.get(key) for key in
            ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "gpu_process_memory_sampling": "not_measured; allocator counters are not total process memory",
        "whole_recon_all": "not_run", "overall_metric_equivalence": "not_assessed",
        "admission_requirement": "same ordered geometry and pass/trial decisions for backend replacement",
        "instrumentation": "coordinate SHA and incremental trace/report writes included in wall time",
        "strict_python_backend_reproduction": "not_run",
        "new_degradation_under_declared_exact_backend_gate": "not_assessed",
    }

    def current_sources():
        return {name: sha256(module.__file__) for name, module in list(sys.modules.items())
                if name.startswith("fnit.recon_all") and getattr(module, "__file__", None)
                and Path(module.__file__).is_file()}

    def save():
        actual_sources = current_sources()
        for name, digest in actual_sources.items():
            previous = report["source_sha256"].setdefault(name, digest)
            if previous != digest:
                report.setdefault("source_changes_during_run", {})[name] = {
                    "before": previous, "after": digest,
                }
        temp = args.output_directory / "report.json.tmp"
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temp.replace(args.output_directory / "report.json")
        if report.get("source_changes_during_run"):
            raise RuntimeError("frozen candidate source changed during benchmark")

    save()
    traces, outputs = {}, {}
    for backend in args.backends:
        output = args.output_directory / f"{hemi}.white.preaparc.{backend}"
        output_volume = args.output_directory / f"mrisps.wpa.{backend}.mgz"
        trace = []
        run_candidate = ((args.control_candidate_backend or args.candidate_backend)
                         if backend == "cpu" else args.candidate_backend)
        run_regularization = "cpu" if backend == "cpu" else args.candidate_regularization_backend
        run_sampling = "cpu" if backend == "cpu" else args.sampling_backend
        gpu_components = (run_regularization == "torch" or run_candidate == "torch_snapshot"
                          or run_sampling != "cpu" or args.cleanup_marking_backend == "source_torch")
        use_cuda = gpu_components and device.type == "cuda"
        row = {"status": "running", "trace": trace, "regularization_backend": run_regularization,
               "candidate_backend": run_candidate, "sampling_backend": run_sampling}
        report["python_runs"][backend] = row
        context_started = time.perf_counter()
        if use_cuda:
            try:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                report["gpu"] = torch.cuda.get_device_name(device)
            except Exception as exc:
                row.update(status="failed_cuda_setup", error=str(exc), traceback=traceback.format_exc(),
                           cuda_setup_seconds=time.perf_counter() - context_started)
                save()
                raise
        row["cuda_setup_seconds"] = time.perf_counter() - context_started if use_cuda else 0.
        row["wall_scope"] = "full stage API plus synchronized timing/trace IO; CUDA context setup reported separately"
        started = time.perf_counter()

        def callback(step, pass_index, coordinates, diagnostics):
            trace.append({"step": step, "pass": pass_index,
                          "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
                          "diagnostics": diagnostics})
            print(json.dumps({"backend": backend, **trace[-1]}, ensure_ascii=False), flush=True)
            save()

        try:
            result = stage.place_white_preaparc(
                subject_dir=args.subject, hemi=hemi, output=output, max_steps=args.max_steps,
                output_volume=output_volume, regularization_backend=run_regularization,
                sampling_backend=run_sampling, candidate_backend=run_candidate,
                cleanup_marking_backend=args.cleanup_marking_backend,
                device=str(device) if gpu_components else None,
                trace_callback=callback,
            )
            if use_cuda:
                torch.cuda.synchronize(device)
            row.update(status="complete", wall_seconds=time.perf_counter() - started,
                       stage=result, output_sha256=sha256(output), output_volume_sha256=sha256(output_volume),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if use_cuda else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if use_cuda else None)
            row["wall_with_cuda_setup_seconds"] = row["wall_seconds"] + row["cuda_setup_seconds"]
            outputs[backend], traces[backend] = output, trace
        except Exception as exc:
            row.update(status="failed", wall_seconds=time.perf_counter() - started,
                       error=str(exc), traceback=traceback.format_exc())
            if hasattr(exc, "intersection_cleanup"):
                row["intersection_cleanup"] = exc.intersection_cleanup
            if hasattr(exc, "partial_stage"):
                row["partial_stage"] = exc.partial_stage
            if hasattr(exc, "intersection_coordinates"):
                failed_path = args.output_directory / f"{backend}-failed-cleanup.diagnostic.npz"
                np.savez(failed_path, vertices=exc.intersection_coordinates, faces=exc.intersection_faces)
                row["failed_cleanup_checkpoint"] = {"path": str(failed_path), "sha256": sha256(failed_path)}
            if use_cuda:
                try:
                    torch.cuda.synchronize(device)
                    row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                except Exception as memory_error:
                    row["allocator_report_error"] = str(memory_error)
            save()
            raise
        after = {str(path.relative_to(args.subject)): sha256(path) for path in inputs}
        row["input_sha256_after"] = after
        save()
        if after != report["input_sha256"]:
            raise RuntimeError("frozen white inputs changed during benchmark")
    if "cpu" in outputs and "torch" in outputs:
        report["python_comparison"] = compare_geometry(outputs["cpu"], outputs["torch"])
        report["python_comparison"].update(
            same_trace=traces["cpu"] == traces["torch"],
            speed_ratio_cpu_over_torch=report["python_runs"]["cpu"]["wall_seconds"] /
                                      report["python_runs"]["torch"]["wall_seconds"],
        )
        comparison = report["python_comparison"]
        report["strict_python_backend_reproduction"] = "passed" if (
            comparison["same_vertex_count_and_ordered_faces"]
            and comparison["different_coordinate_elements"] == 0 and comparison["same_trace"]) else "failed"
        report["new_degradation_under_declared_exact_backend_gate"] = (
            "none_detected" if report["strict_python_backend_reproduction"] == "passed" else "detected")
        save()
    env = dict(os.environ, FREESURFER_HOME=str(args.assets_directory),
               SUBJECTS_DIR=str(args.subject.parent), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    for name, binary in native:
        runs = []
        report["native_runs"][name] = {"binary_sha256": sha256(binary), "runs": runs}
        for index in range(args.reference_repeat):
            output = args.output_directory / f"{hemi}.white.preaparc.{name}-{index}"
            command = [str(binary.resolve()), "--adgws-in", str(stats), "--wm", str(wm),
                       "--threads", str(args.threads), "--invol", str(brain), f"--{hemi}",
                       "--i", str(orig), "--o", str(output), "--white", "--seg", str(seg),
                       "--restore-255", "--nsmooth", "5", "--rip-bg-no-annot", "--rip-bg",
                       "--rip-bg-lof", "--restore-255", "--outvol",
                       str(args.output_directory / f"mrisps.wpa.{name}-{index}.mgz")]
            started = time.perf_counter()
            with (args.output_directory / f"{name}-{index}.log").open("w") as stream:
                completed = subprocess.run(command, cwd=args.output_directory, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
            runs.append({"command": command, "exit_code": completed.returncode,
                         "wall_seconds": time.perf_counter() - started, "output": str(output),
                         "output_sha256": sha256(output) if output.is_file() else None})
            save()
            if completed.returncode:
                raise RuntimeError(f"{name} reference failed; log/report preserved")
        report["native_runs"][name]["repeat_geometry"] = [compare_geometry(runs[0]["output"], row["output"])
                                                         for row in runs[1:]]
        report["native_runs"][name]["python_geometry"] = {backend: compare_geometry(output, runs[0]["output"])
                                                         for backend, output in outputs.items()}
        save()
    report["input_sha256_after"] = {str(path.relative_to(args.subject)): sha256(path) for path in inputs}
    report["source_sha256_after"] = current_sources()
    if report["input_sha256_after"] != report["input_sha256"]:
        report["status"] = "failed_input_changed"
        save()
        raise RuntimeError("frozen white inputs changed during native reference")
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
