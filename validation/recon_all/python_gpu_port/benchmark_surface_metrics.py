"""冻结 FNIT white/pial 输入，比较官方、Conda 和已有 PyTorch 顶点指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel.freesurfer as fs
import numpy as np
import torch

from fnit.recon_all.surface_area_gpu import area_map
from fnit.recon_all.surface_curvature_gpu import curvature_map
from fnit.recon_all.surface_thickness_gpu import thickness_map


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--metric", choices=("area", "curv", "thickness"), required=True)
    parser.add_argument("--surface", choices=("white", "pial"), default="white")
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    white = args.subject / "surf" / f"{args.hemi}.white"
    pial = args.subject / "surf" / f"{args.hemi}.pial"
    surface = white if args.surface == "white" else pial
    xyz, faces = fs.read_geometry(str(white))
    pxyz, pfaces = fs.read_geometry(str(pial))
    if xyz.shape != pxyz.shape or not np.array_equal(faces, pfaces):
        raise ValueError("white/pial 顶点和有序面必须对应")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu = torch.device(args.device).type == "cuda"
    times = {}
    env = dict(os.environ, FREESURFER_HOME=str(args.assets),
               OMP_NUM_THREADS=str(args.threads))
    for name, binary in (("reference", args.reference_binary),
                         ("conda", args.native_binary)):
        output = args.output / f"{args.hemi}.{name}"
        if args.metric == "area":
            command = ["--area-map", str(surface), str(output)]
        elif args.metric == "curv":
            command = ["--curv-map", str(surface), "2", "10", str(output)]
        else:
            command = ["--thickness", str(white), str(pial), "20", "5", str(output)]
        tick = time.perf_counter()
        with (args.output / f"{name}.log").open("w") as stream:
            subprocess.run([str(binary), *command], env=env, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
        times[name] = time.perf_counter() - tick
    if gpu:
        torch.cuda.init()
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    tick = time.perf_counter()
    output = args.output / f"{args.hemi}.pytorch"
    if args.metric == "thickness":
        thickness_map(white_file=white, pial_file=pial,
                      output_file=output, device=args.device)
    elif args.metric == "area":
        area_map(surface=surface, output=output, device=args.device)
    else:
        curvature_map(surface=surface, output=output, device=args.device)
    if gpu:
        torch.cuda.synchronize(args.device)
    times["pytorch"] = time.perf_counter() - tick
    reference = fs.read_morph_data(str(args.output / f"{args.hemi}.reference"))
    absolute = .001 if args.metric == "area" else .005
    comparisons = {}
    for name in ("conda", "pytorch"):
        candidate = fs.read_morph_data(str(args.output / f"{args.hemi}.{name}"))
        delta = np.abs(reference.astype(np.float64) - candidate)
        failed = ~np.isfinite(delta) | (delta > absolute + .001 * np.abs(reference))
        comparisons[name] = {"pass": not bool(failed.any()),
                             "different_values": int(np.count_nonzero(delta)),
                             "outliers": int(np.count_nonzero(failed)),
                             "max_abs": float(delta.max()),
                             "p99_abs": float(np.quantile(delta, .99)),
                             "mae": float(delta.mean())}
    cache_disabled = os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
    report = {"code_commit": args.code_commit, "metric": args.metric,
              "surface": args.surface if args.metric != "thickness" else "white+pial",
              "subject": str(args.subject), "hemi": args.hemi,
              "device": args.device, "threads": args.threads,
              "matmul_tf32": True, "cudnn_tf32": True,
              "vertices": len(xyz), "faces": len(faces),
              "input_sha256": {"white": sha256(white), "pial": sha256(pial)},
              "program_sha256": {"reference": sha256(args.reference_binary),
                                  "conda": sha256(args.native_binary)},
              "script_sha256": sha256(Path(__file__)),
              "seconds_including_io": times, "comparisons": comparisons,
              "threshold": {"absolute": absolute, "relative": .001},
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device)
              if gpu and not cache_disabled else None,
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)
              if gpu and not cache_disabled else None,
              "timing_scope": "同步 GPU；包含函数/子进程读写，排除脚本导入和 CUDA 上下文初始化"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seconds": times, "comparisons": comparisons}))


if __name__ == "__main__":
    main()
