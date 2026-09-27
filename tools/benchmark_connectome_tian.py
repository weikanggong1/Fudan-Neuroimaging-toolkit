"""Matched-input FSL invwarp/applywarp versus PyTorch Tian atlas propagation."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit.applywarp import TorchApplyWarp
from fnit.connectome.atlas_tian import invert_fnirt_t1_warp


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _labels(image):
    return np.asarray(image.dataobj).round().astype(np.int32)


def _scores(candidate, reference):
    mismatch = candidate != reference
    support_a, support_b = candidate > 0, reference > 0
    labels = np.union1d(np.unique(candidate), np.unique(reference))
    dices = {}
    for label in labels:
        if label <= 0:
            continue
        a, b = candidate == label, reference == label
        dices[str(int(label))] = float(2 * (a & b).sum() / (a.sum() + b.sum()))
    return {
        "voxel_mismatch_count": int(mismatch.sum()),
        "voxel_mismatch_fraction": float(mismatch.mean()),
        "foreground_dice": float(2 * (support_a & support_b).sum() /
                                 (support_a.sum() + support_b.sum())),
        "label_dice_min": min(dices.values()),
        "n_labels": len(dices),
        "reference_foreground_voxels": int(support_b.sum()),
        "candidate_foreground_voxels": int(support_a.sum()),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tian-template", type=Path, required=True)
    parser.add_argument("--native-t1", type=Path, required=True)
    parser.add_argument("--forward-coefficients", type=Path, required=True)
    parser.add_argument("--fsl-inverse", type=Path, required=True)
    parser.add_argument("--fsl-atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = nib.load(str(args.fsl_atlas))
    reference_labels = _labels(reference)
    operator = TorchApplyWarp(device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    fixed_inverse_result = operator(args.tian_template, args.native_t1,
                                    warp=args.fsl_inverse, interpolation="nearest")
    fixed_inverse_path = args.output_dir / "torch_from_fsl_inverse_tian.nii.gz"
    fixed_inverse_result.save(fixed_inverse_path)
    fixed_inverse_seconds = perf_counter() - start
    fixed_inverse_peak = (torch.cuda.max_memory_allocated() / 2**30
                          if args.device.startswith("cuda") else None)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    inverse = invert_fnirt_t1_warp(args.forward_coefficients, args.native_t1,
                                   device=args.device)
    own_inverse_path = args.output_dir / "torch_mni_to_t1_inverse.nii.gz"
    nib.save(inverse, str(own_inverse_path))
    own_inverse_seconds = perf_counter() - start
    own_inverse_peak = (torch.cuda.max_memory_allocated() / 2**30
                        if args.device.startswith("cuda") else None)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    own_result = operator(args.tian_template, args.native_t1,
                          warp=own_inverse_path, interpolation="nearest")
    own_atlas_path = args.output_dir / "torch_full_tian.nii.gz"
    own_result.save(own_atlas_path)
    own_atlas_seconds = perf_counter() - start
    own_atlas_peak = (torch.cuda.max_memory_allocated() / 2**30
                      if args.device.startswith("cuda") else None)
    official_field = np.asarray(nib.load(str(args.fsl_inverse)).dataobj, dtype=np.float32)
    own_field = np.asarray(inverse.dataobj, dtype=np.float32)
    field_difference = np.abs(official_field - own_field)
    atlas_support_difference = field_difference[reference_labels > 0]
    report = {
        "dataset": "matched UK Biobank T1/DWI; newly generated FSL FNIRT warp",
        "warp_provenance": "FSL 6.0.7.4 T1_2_MNI152_2mm.cnf on same T1; not archived UKB warp",
        "input_sha256": {path.name: _sha256(path) for path in (
            args.tian_template, args.native_t1, args.forward_coefficients,
            args.fsl_inverse, args.fsl_atlas)},
        "source_sha256": {
            "pytorch_inverse": _sha256(Path(__file__).parents[1] / "src/fnit/connectome/atlas_tian.py"),
            "pytorch_applywarp": _sha256(Path(__file__).parents[1] / "src/fnit/applywarp/core.py"),
        },
        "inverse_field_shape": list(own_field.shape),
        "inverse_field_mae_mm": float(field_difference.mean()),
        "inverse_field_max_abs_error_mm": float(field_difference.max()),
        "inverse_field_p99_abs_error_mm": float(np.percentile(field_difference, 99)),
        "inverse_field_voxels_above_0p0001_mm": int(
            np.any(field_difference > 1e-4, axis=-1).sum()),
        "atlas_support_field_mae_mm": float(atlas_support_difference.mean()),
        "atlas_support_field_max_abs_error_mm": float(atlas_support_difference.max()),
        "atlas_support_field_p99_abs_error_mm": float(
            np.percentile(atlas_support_difference, 99)),
        "fsl_inverse_fixed_torch_applywarp": _scores(
            _labels(fixed_inverse_result.image), reference_labels),
        "torch_inverse_and_applywarp": _scores(_labels(own_result.image), reference_labels),
        "torch_fixed_inverse_applywarp_seconds_with_io": fixed_inverse_seconds,
        "torch_fixed_inverse_applywarp_peak_cuda_allocated_gib": fixed_inverse_peak,
        "torch_inverse_seconds_with_io": own_inverse_seconds,
        "torch_inverse_peak_cuda_allocated_gib": own_inverse_peak,
        "torch_full_tian_seconds_with_io": own_atlas_seconds,
        "torch_full_tian_peak_cuda_allocated_gib": own_atlas_peak,
        "output_sha256": {
            "torch_inverse": _sha256(own_inverse_path),
            "torch_fixed_inverse_tian": _sha256(fixed_inverse_path),
            "torch_full_tian": _sha256(own_atlas_path),
        },
    }
    (args.output_dir / "tian_atlas.internal.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
