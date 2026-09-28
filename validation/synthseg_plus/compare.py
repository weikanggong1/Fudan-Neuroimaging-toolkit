"""Compare two SynthSeg+ hard-label volumes on the same physical grid."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = nib.load(args.official)
    fnit = nib.load(args.fnit)
    if official.shape != fnit.shape or not np.allclose(official.affine, fnit.affine, atol=1e-5):
        raise ValueError("Images must have the same shape and affine")
    x = np.asanyarray(official.dataobj).astype(np.int32)
    y = np.asanyarray(fnit.dataobj).astype(np.int32)
    voxel_mm3 = float(abs(np.linalg.det(official.affine[:3, :3])))
    labels = sorted((set(np.unique(x)) | set(np.unique(y))) - {0})
    per_label = {}
    for label in labels:
        a = x == label
        b = y == label
        count_a = int(a.sum())
        count_b = int(b.sum())
        overlap = int(np.count_nonzero(a & b))
        per_label[str(label)] = {
            "dice": 2 * overlap / (count_a + count_b) if count_a + count_b else 1.0,
            "official_voxels": count_a,
            "fnit_voxels": count_b,
            "volume_diff_mm3": (count_b - count_a) * voxel_mm3,
        }
    out = {
        "official": str(args.official), "fnit": str(args.fnit),
        "shape": list(x.shape),
        "max_abs_affine_diff": float(np.abs(official.affine - fnit.affine).max()),
        "voxel_agreement": float(np.mean(x == y)),
        "different_voxels": int(np.count_nonzero(x != y)),
        "per_label": per_label,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "voxel_agreement": out["voxel_agreement"],
        "different_voxels": out["different_voxels"],
        "min_dice": min(item["dice"] for item in per_label.values()),
        "median_dice": float(np.median([item["dice"] for item in per_label.values()])),
        "labels": len(per_label),
    }, indent=2))


if __name__ == "__main__":
    main()
