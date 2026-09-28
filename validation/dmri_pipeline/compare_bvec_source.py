"""Compare two single-subject bvec choices against the same UKB reference maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


MAPS = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def compare(candidate: Path, reference: Path) -> dict:
    first = np.asarray(nib.load(str(candidate)).dataobj, dtype=np.float32).squeeze()
    second = np.asarray(nib.load(str(reference)).dataobj, dtype=np.float32).squeeze()
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {candidate} and {reference}")
    valid = np.isfinite(first) & np.isfinite(second)
    valid &= (first != 0) | (second != 0)
    a = first[valid].astype(np.float64)
    b = second[valid].astype(np.float64)
    return {
        "voxels": int(valid.sum()),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "mae": float(np.mean(np.abs(a - b))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotated-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--official-native", type=Path, required=True)
    parser.add_argument("--official-tbss", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {}
    for name in MAPS:
        native_name = f"{'dti' if name in MAPS[:6] else 'NODDI'}_{name}.nii.gz"
        result[name] = {}
        for stage in ("native", "standard", "skeleton"):
            if stage == "native":
                official = args.official_native / native_name
            else:
                suffix = "" if stage == "standard" else "_skeletonised"
                official = args.official_tbss / "stats" / f"all_{name}{suffix}.nii.gz"
            filename = native_name if stage == "native" else f"{name}.nii.gz"
            folder = "native" if stage == "native" else f"registration/{stage}"
            rotated = compare(args.rotated_root / folder / filename, official)
            raw = compare(args.raw_root / folder / filename, official)
            result[name][stage] = {
                "rotated": rotated,
                "raw": raw,
                "delta_r_raw_minus_rotated": raw["pearson"] - rotated["pearson"],
            }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
