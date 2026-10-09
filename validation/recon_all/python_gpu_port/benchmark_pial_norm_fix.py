"""单次完整 pial API 收据；控制/候选由外部 ABBA 顺序新进程执行。

仅 source_directory 中的限幅子函数发生变化，其余四轮、碰撞和清理固定。
原生参考只在生产 API 返回后读取以计算误差，不向放置函数传递参考。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import nibabel.freesurfer.io as fs
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--reference-surface", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--variant", choices=("control", "candidate"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--source-base-commit", required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError(args.output_directory)
    args.output_directory.mkdir(parents=True)
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0, str(args.source_directory))
    from fnit.recon_all import place_pial_python as stage
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    h, out = args.hemisphere, args.output_directory
    inputs = [Path(f"surf/{h}.white"), Path(f"surf/autodet.gw.stats.{h}.dat"),
        Path(f"label/{h}.cortex.label"), Path(f"label/{h}.cortex+hipamyg.label"),
        *[Path(f"mri/{n}.mgz") for n in ("brain.finalsurfs", "wm", "aseg.presurf")]]
    report = {"scope": "same_input_complete_pial_GPU_API_norm_rounding_only",
        "hostname": platform.node(), "threads": args.threads, "device": args.device,
        "cpu_affinity_count": len(os.sched_getaffinity(0)), "variant": args.variant,
        "source_base_commit": args.source_base_commit, "source_binding": "per_module_SHA256",
        "input_sha256": {str(p): sha(args.subject/p) for p in inputs},
        "script_sha256": sha(__file__), "tf32_matmul": True, "tf32_cudnn": True,
        "half_precision": False, "reference_surface_sha256": sha(args.reference_surface),
        "overall_metric_equivalence": "not_assessed", "process_tree_memory": "external sampler required",
        "allocator_environment": {k: os.environ.get(k) for k in
            ("PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")},
        "status": "started", "trace": []}
    def save():
        tmp = out/"report.tmp"
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        tmp.replace(out/"report.json")
    def callback(step, outer, coordinates, diagnostics):
        report["trace"].append({"step": step, "pass": outer,
            "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
            "diagnostics": diagnostics})
    save()
    d = torch.device(args.device)
    started = time.perf_counter()
    torch.cuda.synchronize(d)
    report["CUDA_setup_seconds"] = time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats(d)
    started = time.perf_counter()
    try:
        result = stage.place_pial_t1(subject=args.subject, hemisphere=h,
            output=out/f"{h}.pial.T1", max_steps=200, sampling_backend="cpu",
            regularization_backend="cpu", candidate_backend="torch_snapshot",
            candidate_grid_cells_per_axis=3, retained_mht_backend="compiled",
            cleanup_marking_backend="source_torch", cleanup_candidate_grid_cells_per_axis=3,
            device=args.device, trace_callback=callback, profile=True)
        torch.cuda.synchronize(d)
    except Exception as exc:
        report["status"]="failed";report["error_type"]=type(exc).__name__
        report["API_wall_seconds_to_failure"]=time.perf_counter()-started;save();raise
    report["API_wall_seconds"] = time.perf_counter()-started
    report["result"] = result
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(d)
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(d)
    report["source_sha256"] = {Path(m.__file__).name: sha(m.__file__)
        for name, m in list(sys.modules.items()) if name.startswith("fnit.recon_all.place_")
        and getattr(m, "__file__", None)}
    # 候选算法完成后才打开隔离的数值参考。
    x, f = fs.read_geometry(str(out/f"{h}.pial.T1"))
    ref, rf = fs.read_geometry(str(args.reference_surface))
    same = x.shape == ref.shape and np.array_equal(f, rf)
    errors = np.linalg.norm(x-ref, axis=1) if same else None
    report["native_comparison"] = {"same_vertex_count_and_ordered_faces": same,
        "different_coordinate_elements": None if not same else int(np.count_nonzero(x!=ref)),
        "mean_vertex_distance_mm": None if not same else float(errors.mean()),
        "p99_vertex_distance_mm": None if not same else float(np.percentile(errors,99)),
        "max_vertex_distance_mm": None if not same else float(errors.max()),
        "vertices_over_0_1_mm": None if not same else int(np.count_nonzero(errors>.1))}
    report["output_sha256"] = sha(out/f"{h}.pial.T1")
    report["input_sha256_after"] = {str(p): sha(args.subject/p) for p in inputs}
    report["status"] = "complete" if report["input_sha256"] == report["input_sha256_after"] else "input_changed"
    save()
    print(json.dumps({k:report[k] for k in ("status","variant","API_wall_seconds","native_comparison")}))


if __name__ == "__main__":
    main()
