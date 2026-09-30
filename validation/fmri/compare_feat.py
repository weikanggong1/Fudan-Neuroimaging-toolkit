"""Compare saved real-data FNIT and official pre-ICA FEAT outputs."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    images = [nib.load(str(root / "filtered_func_data.nii.gz")) for root in
              (args.candidate_dir, args.reference_dir)]
    if images[0].shape != images[1].shape or not np.allclose(
            images[0].affine, images[1].affine, atol=1e-5, rtol=0):
        raise ValueError("FEAT output grids differ")
    masks = []
    for root, image in zip((args.candidate_dir, args.reference_dir), images):
        mask_image = nib.load(str(root / "mask.nii.gz"))
        if mask_image.shape != image.shape[:3] or not np.allclose(
                mask_image.affine, image.affine, atol=1e-5, rtol=0):
            raise ValueError("FEAT mask grid differs")
        masks.append(np.asarray(mask_image.dataobj) > 0)
    mask = masks[0] & masks[1]
    if not mask.any():
        raise ValueError("empty mask intersection")
    arrays = [np.asarray(image.dataobj, dtype=np.float32) for image in images]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("nonfinite FEAT output")
    first, second = [array[mask].astype(np.float64) for array in arrays]
    difference = first - second
    spatial_temporal_r = float(np.corrcoef(first.ravel(), second.ravel())[0, 1])
    first -= first.mean(axis=1, keepdims=True)
    second -= second.mean(axis=1, keepdims=True)
    denominator = np.sqrt((first * first).sum(axis=1) * (second * second).sum(axis=1))
    valid = denominator > 1e-8
    temporal_r = (first[valid] * second[valid]).sum(axis=1) / denominator[valid]
    pooled_denominator = np.sqrt((first * first).sum() * (second * second).sum())
    report = {"schema_version": 1, "shape": list(images[0].shape),
              "same_shape_and_affine": True, "all_values_finite": True,
              "candidate_mask_voxels": int(masks[0].sum()),
              "reference_mask_voxels": int(masks[1].sum()),
              "intersection_voxels": int(mask.sum()),
              "mask_dice": float(2 * mask.sum() / (masks[0].sum() + masks[1].sum())),
              "filtered_bold": {"pearson_r": spatial_temporal_r,
                                "mae": float(np.abs(difference).mean()),
                                "rmse": float(np.sqrt(np.square(difference).mean())),
                                "pooled_time_demeaned_r": float((first * second).sum() / pooled_denominator),
                                "valid_voxel_time_correlations": int(valid.sum()),
                                "median_voxel_time_r": float(np.median(temporal_r)),
                                "mean_voxel_time_r": float(temporal_r.mean())},
              "scope": "Current full BIDS volume run's pre-ICA FEAT output versus existing original FSL 6.0.7.22 no-GDC/no-B0 reference; same raw BOLD and SBRef.",
              "privacy": "Anonymous scalar metrics only; no subject identifiers, image paths, input/output hashes or voxelwise results."}
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
