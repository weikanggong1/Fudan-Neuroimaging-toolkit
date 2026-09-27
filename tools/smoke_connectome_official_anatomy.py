"""Exercise the public connectome API with an official FreeSurfer segmentation."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome import UKBConnectome
from fnit.flirt import flirt_to_world_affine


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("dwi", "bvals", "bvecs", "t1", "aparc-aseg", "atlas-dwi",
                 "fsl-matrix", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dwi, t1 = nib.load(str(args.dwi)), nib.load(str(args.t1))
    matrix = flirt_to_world_affine(
        np.loadtxt(args.fsl_matrix), dwi.affine, t1.affine, dwi.shape[:3],
        t1.shape[:3], dwi.header.get_zooms()[:3], t1.header.get_zooms()[:3],
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = UKBConnectome(device=args.device)(
        args.dwi, args.bvals, args.bvecs, args.t1,
        t1_segmentation=args.aparc_aseg,
        segmentation_source="freesurfer",
        atlas_dwi=args.atlas_dwi,
        dwi_to_t1_world=matrix,
        n_seeds=args.n_seeds,
        seed=0,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    matrices = {name: value.cpu().numpy() for name, value in result.matrices.items()}
    report = {
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.dwi, args.bvals, args.bvecs, args.t1,
                                   args.aparc_aseg, args.atlas_dwi, args.fsl_matrix)},
        "device": args.device,
        "peak_torch_cuda_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
        "peak_torch_cuda_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                         if args.device.startswith("cuda") else None),
        "segmentation_source": "official FreeSurfer aparc+aseg",
        "n_seeds": args.n_seeds,
        "accepted_streamlines": len(result.tractogram.paths),
        "tissue_voxels": {str(k): int((result.tissues == k).sum()) for k in range(4)},
        "region_count": len(result.region_labels),
        "matrices": {name: {"shape": list(value.shape),
                             "finite": bool(np.isfinite(value).all()),
                             "symmetric": bool(np.allclose(value, value.T)),
                             "sum": float(value.sum())}
                     for name, value in matrices.items()},
        "seconds": seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": report["accepted_streamlines"],
                      "tissue_voxels": report["tissue_voxels"],
                      "seconds": seconds}))


if __name__ == "__main__":
    main()
