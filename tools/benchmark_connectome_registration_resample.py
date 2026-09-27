"""Compare TorchFLIRT and FSL resampling with the same fixed 6-DOF matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np
import surfa as sf
import torch

from fnit.flirt import voxel_to_fsl_scaled_mm
from fnit.flirt.core import _resample_output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--fsl-matrix", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    reference_path = args.scratch / "b0_to_t1_fsl.nii.gz"
    start = time.perf_counter()
    subprocess.run([
        "/public/software/apps/FSL/6.0.7.4/bin/flirt", "-in", str(args.b0),
        "-ref", str(args.t1), "-applyxfm", "-init", str(args.fsl_matrix),
        "-out", str(reference_path),
    ], check=True, env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ",
                    "FSLDIR": "/public/software/apps/FSL/6.0.7.4"})
    reference_seconds = time.perf_counter() - start
    moving, fixed = sf.load_volume(str(args.b0)), sf.load_volume(str(args.t1))
    moving_affine = np.asarray(moving.geom.vox2world.matrix)
    fixed_affine = np.asarray(fixed.geom.vox2world.matrix)
    moving_sizes = tuple(float(x) for x in moving.geom.voxsize)
    fixed_sizes = tuple(float(x) for x in fixed.geom.voxsize)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = _resample_output(
        np.asarray(moving.data), fixed.shape[:3],
        voxel_to_fsl_scaled_mm(moving_affine, moving.shape[:3], moving_sizes),
        voxel_to_fsl_scaled_mm(fixed_affine, fixed.shape[:3], fixed_sizes),
        np.loadtxt(args.fsl_matrix), moving_sizes, fixed_sizes,
        torch.device(args.device),
    ).cpu().numpy()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    torch_seconds = time.perf_counter() - start
    reference = np.asarray(nib.load(str(reference_path)).dataobj)
    assert reference.shape == output.shape
    differences = np.abs(output.astype(np.float64) - reference.astype(np.float64))
    both = (reference != 0) & (output != 0)
    report = {
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.b0, args.t1, args.fsl_matrix, reference_path)},
        "shape": list(reference.shape),
        "device": args.device,
        "peak_torch_cuda_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
        "peak_torch_cuda_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                         if args.device.startswith("cuda") else None),
        "mae_all": float(differences.mean()),
        "mae_both_nonzero": float(differences[both].mean()),
        "p95_abs_difference_both_nonzero": float(np.percentile(differences[both], 95)),
        "max_abs_difference": float(differences.max()),
        "pearson_both_nonzero": float(np.corrcoef(output[both], reference[both])[0, 1]),
        "foreground_dice": float(2 * np.count_nonzero(both) /
                                 (np.count_nonzero(reference) + np.count_nonzero(output))),
        "seconds": {"fsl_applyxfm": reference_seconds, "torch_resample": torch_seconds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
