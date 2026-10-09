"""真实冻结WM/aseg/LTA的现有完整fill剖析；与整例及跨主机耗时分开。"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import nibabel as nib
import numpy as np
import scipy
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-dir", type=Path, required=True)
    parser.add_argument("--colortable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--boundary-backend", choices=("python", "torch"), default="python")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--marching-backend", choices=("python", "numba"), default="python")
    parser.add_argument("--reference-filled", type=Path,
                        help="仅比较阶段读取的冻结参考；不传入候选算法")
    parser.add_argument("--module-dir", type=Path,
                        help="独立本轮FNIT recon模块目录；其余包只读冻结源码，不复制或修改")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    if args.module_dir is not None:
        import fnit.recon_all
        fnit.recon_all.__path__.insert(0, str(args.module_dir.resolve()))
    from fnit.recon_all import fill_aseg_python as aseg
    from fnit.recon_all import fill_cutting_plane_python as cutting
    original_functions = {}
    rows = []

    def timed(module, name):
        function = getattr(module, name)
        original_functions[(module.__name__, name)] = function
        @functools.wraps(function)
        def wrapped(*positional, **keywords):
            started = time.perf_counter()
            result = function(*positional, **keywords)
            row = {"function": name, "module": module.__name__, "seconds": time.perf_counter() - started}
            if name == "affine_transform":
                row["parameters"] = {key: str(value) for key, value in keywords.items()
                    if key not in {"offset", "output_shape"}}
            if name == "_voronoi_round":
                row["coordinates"] = len(positional[2]); row["radius"] = positional[3]
            if name == "_cc_outside_distance":
                row["target_label"] = int(positional[2])
            rows.append(row)
            return result
        setattr(module, name, wrapped)

    for name in ("_cc_outside_distance", "_voronoi_round", "_edited_on_votes", "_largest_then_fill_holes"):
        timed(aseg, name)
    for name in ("_replace_cc_with_wm", "compute_cc_cut_mask", "_fill_preclassified", "save_filled_mgz"):
        timed(cutting, name)
    for name in ("affine_transform", "convolve1d", "distance_transform_cdt", "label", "binary_fill_holes"):
        timed(cutting.ndi, name)
    input_files = {"wm": args.mri_dir / "wm.mgz", "aseg": args.mri_dir / "aseg.presurf.mgz",
                   "lta": args.mri_dir / "transforms/talairach.lta", "colortable": args.colortable}
    reference_file = args.reference_filled or args.mri_dir / "filled.mgz"
    output_file = args.output_dir / "filled.mgz"
    cc_profiles = []
    device = torch.device(args.device)
    sampler = None
    if device.type == "cuda":
        if args.boundary_backend != "torch":
            raise ValueError("CUDA device requires the explicit torch boundary backend")
        from recon_wm_aseg_torch import ProcessMemorySampler
        import subprocess
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        gpu_row = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name",
            "--format=csv,noheader"], text=True).splitlines()[device.index]
        gpu_index, gpu_uuid, gpu_name = [part.strip() for part in gpu_row.split(",", 2)]
        sampler = ProcessMemorySampler(gpu_uuid)
        sampler.start()
    started = time.perf_counter()
    seed = cutting.fill_mgz(
        wm_file=input_files["wm"], aseg_file=input_files["aseg"], lta_file=input_files["lta"],
        colortable_file=input_files["colortable"], output_file=output_file,
        cut_log_file=args.output_dir / "ponscc.cut.log", cc_profiles=cc_profiles,
        cc_boundary_backend=args.boundary_backend, device=str(device),
        cc_marching_backend=args.marching_backend)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    memory = sampler.finish() if sampler is not None else None
    reference_image, output_image = nib.load(reference_file), nib.load(output_file)
    reference, output = np.asarray(reference_image.dataobj), np.asarray(output_image.dataobj)
    dice = {}
    for label in np.union1d(np.unique(reference), np.unique(output)):
        lhs, rhs = reference == label, output == label
        dice[str(int(label))] = 2 * int(np.count_nonzero(lhs & rhs)) / int(np.count_nonzero(lhs) + np.count_nonzero(rhs))
    delta = np.abs(reference.astype(np.float64) - output.astype(np.float64))
    report = {
        "scope": "complete_existing_FNIT_fill_file_API_profile; frozen_inputs; not whole_recon timing",
        "code_commit": args.code_commit, "code_commit_role": "baseline plus executed module hashes",
        "module_overlay": args.module_dir is not None,
        "modules_sha256": {Path(module.__file__).name: sha(module.__file__) for module in (aseg, cutting)},
        "script_sha256": sha(__file__), "input_sha256": {key: sha(path) for key, path in input_files.items()},
        "reference_kind": "existing FNIT frozen same-input filled only; new native reference is separate",
        "reference_sha256": sha(reference_file), "output_sha256": sha(output_file),
        "host": platform.node(), "cpu_model": next((line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": args.threads,
        "thread_environment": {key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "nibabel": nib.__version__, "device": str(device),
        "cc_boundary_backend": args.boundary_backend,
        "cc_marching_backend": args.marching_backend,
        "precision": "existing NumPy/SciPy/Numba arithmetic unchanged; Torch boundary int64/bool/float32 constants; no half",
        "full_API_seconds": seconds, "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "profile_rows_nested_do_not_sum": rows, "cc_seed_voxel": [int(value) for value in seed],
        "cc_distance_phase_profiles": cc_profiles,
        "shape": [int(axis) for axis in output.shape], "dtype": str(output.dtype),
        "affine_equal_reference": bool(np.array_equal(reference_image.affine, output_image.affine)),
        "different_voxels": int(np.count_nonzero(delta)), "max_abs": float(delta.max()),
        "p99_abs": float(np.percentile(delta, 99)), "label_dice": dice,
        "scope_limit": "headcw diagnostic may differ from gpucw1 hardware/load; no direct speed comparison to 53.16s production measurement",
    }
    if args.boundary_backend == "torch":
        from fnit.recon_all import fill_boundary_torch
        report["modules_sha256"]["fill_boundary_torch.py"] = sha(fill_boundary_torch.__file__)
    if args.marching_backend == "numba":
        from fnit.recon_all import fill_marching_numba
        report["modules_sha256"]["fill_marching_numba.py"] = sha(fill_marching_numba.__file__)
    if device.type == "cuda":
        report.update({"gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "gpu_name": gpu_name,
                       "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                       "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                       "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
                       "process_memory_sampling": memory})
    (args.output_dir / "profile.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
