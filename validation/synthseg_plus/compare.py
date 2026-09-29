"""Compare two SynthSeg+ hard-label volumes on the same physical grid."""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--official-volumes", type=Path)
    parser.add_argument("--fnit-volumes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.official_volumes is None) != (args.fnit_volumes is None):
        parser.error("Both volume CSV files must be supplied together")

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
    if args.official_volumes is not None:
        with args.official_volumes.open(newline="") as stream:
            official_csv = list(csv.reader(stream))
        with args.fnit_volumes.open(newline="") as stream:
            fnit_csv = list(csv.reader(stream))
        if len(official_csv) != 2 or len(fnit_csv) != 2:
            raise ValueError("Expected one header and one subject row in each volume CSV")
        if official_csv[0] != fnit_csv[0]:
            raise ValueError("Volume CSV column names or order differ")
        a = np.asarray(official_csv[1][1:], dtype=np.float64)
        b = np.asarray(fnit_csv[1][1:], dtype=np.float64)
        error = np.abs(a - b)
        out["soft_volumes"] = {
            "columns": len(a),
            "max_abs_diff_mm3": float(error.max()),
            "mean_abs_diff_mm3": float(error.mean()),
            "max_diff_column": official_csv[0][int(error.argmax()) + 1],
        }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    summary = {
        "voxel_agreement": out["voxel_agreement"],
        "different_voxels": out["different_voxels"],
        "min_dice": min(item["dice"] for item in per_label.values()),
        "median_dice": float(np.median([item["dice"] for item in per_label.values()])),
        "labels": len(per_label),
    }
    if "soft_volumes" in out:
        summary["soft_volumes"] = out["soft_volumes"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
