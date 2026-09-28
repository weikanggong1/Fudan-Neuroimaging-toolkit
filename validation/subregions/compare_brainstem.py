"""Compare official brainstem labels with FNIT labels on the FNIT image grid."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--fnit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = nib.load(args.official)
    fnit = nib.load(args.fnit)
    aligned = resample_from_to(official, (fnit.shape[:3], fnit.affine), order=0)
    x = np.asanyarray(aligned.dataobj).round().astype(np.int32)
    y = np.asanyarray(fnit.dataobj).astype(np.int32)
    voxel_mm3 = float(abs(np.linalg.det(fnit.affine[:3, :3])))
    per_label = {}
    for label in (173, 174, 175, 178):
        a, b = x == label, y == label
        na, nb = int(a.sum()), int(b.sum())
        intersection = int(np.count_nonzero(a & b))
        per_label[str(label)] = {
            "dice": 2 * intersection / (na + nb) if na + nb else 1.0,
            "official_voxels": na,
            "fnit_voxels": nb,
            "official_volume_mm3_on_fnit_grid": na * voxel_mm3,
            "fnit_volume_mm3": nb * voxel_mm3,
            "volume_percent_difference": 100 * (nb - na) / na if na else None,
        }
    report = {
        "official": str(args.official), "fnit": str(args.fnit),
        "official_shape": [int(v) for v in official.shape],
        "comparison_shape": [int(v) for v in fnit.shape],
        "resampling": "nearest-neighbor official FSvoxelSpace to FNIT grid",
        "foreground_dice": 2 * np.count_nonzero((x != 0) & (y != 0)) /
                           max(1, np.count_nonzero(x != 0) + np.count_nonzero(y != 0)),
        "per_label": per_label,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["dice"] for k, v in per_label.items()}, indent=2))


if __name__ == "__main__":
    main()
