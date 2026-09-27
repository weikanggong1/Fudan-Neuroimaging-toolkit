"""Compare torch nearest-label resampling with MRtrix for one fixed T1/DWI pair."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.anatomy import resample_labels_nearest
from fnit.flirt import flirt_to_world_affine


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--atlas-t1", type=Path, required=True)
    parser.add_argument("--dwi-reference", type=Path, required=True)
    parser.add_argument("--fsl-matrix", type=Path, required=True)
    parser.add_argument("--mrtrix-atlas-dwi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source, target, reference = [nib.load(str(path)) for path in
                                 (args.atlas_t1, args.dwi_reference, args.mrtrix_atlas_dwi)]
    labels = np.asarray(source.dataobj).astype(np.int32)
    reference_labels = np.squeeze(np.asarray(reference.dataobj)).astype(np.int32)
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(reference.affine),
        nib.orientations.io_orientation(target.affine),
    )
    reference_labels = nib.orientations.apply_orientation(reference_labels, orientation)
    reference_affine = reference.affine @ nib.orientations.inv_ornt_aff(
        orientation, reference.shape[:3]
    )
    assert target.shape[:3] == reference_labels.shape, (target.shape, reference_labels.shape)
    assert np.allclose(target.affine, reference_affine, atol=1e-4), (target.affine, reference_affine)
    fsl_matrix = np.loadtxt(args.fsl_matrix)
    # FSL matrix maps DWI -> T1; this is precisely target -> source.
    world_transform = flirt_to_world_affine(
        fsl_matrix, target.affine, source.affine, target.shape[:3], source.shape,
        target.header.get_zooms()[:3], source.header.get_zooms()[:3],
    )
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = resample_labels_nearest(
        torch.as_tensor(labels, device=args.device),
        torch.as_tensor(source.affine, device=args.device),
        target.shape[:3], torch.as_tensor(target.affine, device=args.device),
        torch.as_tensor(world_transform, device=args.device),
    ).cpu().numpy()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mismatch = result != reference_labels
    foreground_a, foreground_b = result > 0, reference_labels > 0
    report = {
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (args.atlas_t1, args.dwi_reference, args.fsl_matrix, args.mrtrix_atlas_dwi)},
        "shape": list(result.shape),
        "device": args.device,
        "peak_torch_cuda_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                          if args.device.startswith("cuda") else None),
        "peak_torch_cuda_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                         if args.device.startswith("cuda") else None),
        "voxel_mismatch_count": int(mismatch.sum()),
        "voxel_mismatch_fraction": float(mismatch.mean()),
        "foreground_dice": float(2 * np.count_nonzero(foreground_a & foreground_b) /
                                 (foreground_a.sum() + foreground_b.sum())),
        "foreground_voxels": [int(foreground_a.sum()), int(foreground_b.sum())],
        "first_mismatches": [list(map(int, point)) for point in np.argwhere(mismatch)[:10]],
        "torch_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
