"""将原始 FIRST VTK 的 PyTorch PVE 与固定版本 MRtrix 逐体素比较。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.first_mesh_pve import first_vtk_to_pve, read_first_vtk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = nib.load(args.reference)
    ref = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    _, faces = read_first_vtk(args.mesh)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    out = first_vtk_to_pve(
        vtk_path=args.mesh,
        template_affine=torch.tensor(image.affine, dtype=torch.float64),
        template_shape=image.shape,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute = time.perf_counter() - begin
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    candidate = out.cpu().numpy()
    error = np.abs(candidate - ref)
    fg = (candidate > 0) | (ref > 0)
    report = {
        "shape": list(image.shape),
        "faces": int(len(faces)),
        "reference_nonzero": int(np.count_nonzero(ref)),
        "candidate_nonzero": int(np.count_nonzero(candidate)),
        "reference_partial": int(np.count_nonzero((ref > 0) & (ref < 1))),
        "candidate_partial": int(np.count_nonzero((candidate > 0) & (candidate < 1))),
        "exact_mismatch_voxels": int(np.count_nonzero(error)),
        "mismatch_gt_1e-6": int(np.count_nonzero(error > 1e-6)),
        "max_absolute_error": float(error.max()),
        "mae_foreground_union": float(error[fg].mean()) if fg.any() else 0.0,
        "foreground_dice": float(2 * np.count_nonzero((candidate > 0) & (ref > 0)) / (np.count_nonzero(candidate > 0) + np.count_nonzero(ref > 0))),
        "compute_seconds": compute,
        "gpu_peak_allocated_gib": peak,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.candidate:
        nib.save(nib.Nifti1Image(candidate, image.affine, image.header), args.candidate)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
