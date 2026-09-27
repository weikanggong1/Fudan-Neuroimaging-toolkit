"""Compare torch 6DOF/normmi with FSL FLIRT on identical b0 and T1 files."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import surfa as sf
import torch

from fnit.flirt import TorchFLIRT, flirt_to_world_affine


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--fsl-matrix", type=Path, required=True)
    parser.add_argument("--fsl-moved", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    moving = sf.load_volume(str(args.b0))
    fixed = sf.load_volume(str(args.t1))
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = TorchFLIRT(device=args.device, dof=6, cost="normmi")(moving, fixed)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    reference = np.loadtxt(args.fsl_matrix)
    reference_world = flirt_to_world_affine(
        reference, moving.geom.vox2world.matrix, fixed.geom.vox2world.matrix,
        moving.shape[:3], fixed.shape[:3], moving.geom.voxsize, fixed.geom.voxsize,
    )
    shape = np.asarray(moving.shape[:3])
    axes = [np.linspace(0, size - 1, 13) for size in shape]
    voxels = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    points = voxels @ moving.geom.vox2world.matrix[:3, :3].T + moving.geom.vox2world.matrix[:3, 3]
    mismatch = (points @ result.moving_to_fixed_world[:3, :3].T + result.moving_to_fixed_world[:3, 3]
                - points @ reference_world[:3, :3].T - reference_world[:3, 3])
    distances = np.linalg.norm(mismatch, axis=1)
    image_metrics = None
    if args.fsl_moved is not None:
        fsl_moved = np.asarray(sf.load_volume(str(args.fsl_moved)).data, dtype=np.float32)
        torch_moved = np.asarray(result.moved.data, dtype=np.float32)
        if fsl_moved.shape != torch_moved.shape:
            raise ValueError("FSL and PyTorch output image shapes differ")
        abs_error = np.abs(fsl_moved.astype(np.float64) - torch_moved.astype(np.float64))
        both = (fsl_moved != 0) & (torch_moved != 0)
        image_metrics = {
            "shape": list(fsl_moved.shape),
            "reference_nonzero": int(np.count_nonzero(fsl_moved)),
            "candidate_nonzero": int(np.count_nonzero(torch_moved)),
            "foreground_dice": float(2 * np.count_nonzero(both) /
                                     (np.count_nonzero(fsl_moved) + np.count_nonzero(torch_moved))),
            "pearson_both_nonzero": float(np.corrcoef(fsl_moved[both], torch_moved[both])[0, 1]),
            "mae_both_nonzero": float(abs_error[both].mean()),
            "mae_all": float(abs_error.mean()),
            "p95_abs_difference_both_nonzero": float(np.percentile(abs_error[both], 95)),
            "max_abs_difference": float(abs_error.max()),
        }
    private_inputs = (args.b0, args.t1, args.fsl_matrix)
    if args.fsl_moved is not None:
        private_inputs += (args.fsl_moved,)
    report = {
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in private_inputs},
        "device": args.device,
        "peak_torch_cuda_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
        "peak_torch_cuda_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                         if args.device.startswith("cuda") else None),
        "torch_seconds": elapsed,
        "fsl_matrix": reference.tolist(),
        "torch_matrix": result.matrix.tolist(),
        "fsl_world": reference_world.tolist(),
        "torch_world": result.moving_to_fixed_world.tolist(),
        "grid_displacement_mm": {"mean": float(distances.mean()), "p95": float(np.percentile(distances, 95)),
                                 "max": float(distances.max()), "rms": float(np.sqrt(np.mean(distances**2)))},
        "qc": result.qc,
        "moved_image": image_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["grid_displacement_mm"], indent=2))
    print("seconds", elapsed)


if __name__ == "__main__":
    main()
