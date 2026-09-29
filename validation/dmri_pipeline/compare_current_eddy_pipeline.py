#!/usr/bin/env python3
"""Compare one completed FNIT pipeline with a same-grid reference pipeline."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


MAPS = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def _file(root, stage, name):
    if stage == "native":
        prefix = "NODDI_" if name in ("ICVF", "OD", "ISOVF") else "dti_"
        return root / "native" / f"{prefix}{name}.nii.gz"
    return root / "registration" / stage / f"{name}.nii.gz"


def _compare(left, right):
    a, b = nib.load(str(left)), nib.load(str(right))
    x = np.asarray(a.dataobj, dtype=np.float32)
    y = np.asarray(b.dataobj, dtype=np.float32)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if y.ndim == 4 and y.shape[-1] == 1:
        y = y[..., 0]
    if x.shape != y.shape:
        return {"pearson_r": None, "shape_equal": False, "candidate_shape": x.shape,
                "reference_shape": y.shape, "affine_equal": False}
    affine_equal = bool(np.allclose(a.affine, b.affine, rtol=0, atol=1e-5))
    mask = (x != 0) | (y != 0)
    if not np.any(mask):
        raise ValueError(f"empty union support: {left.name}")
    u, v = x[mask].astype(np.float64), y[mask].astype(np.float64)
    d = u - v
    return {
        "pearson_r": float(np.corrcoef(u, v)[0, 1]),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "union_voxels": int(mask.sum()),
        "shape_equal": a.shape == b.shape,
        "comparison_shape_equal": True,
        "affine_equal": affine_equal,
        "singleton_volume_squeezed": bool(a.shape != x.shape or b.shape != y.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-label")
    parser.add_argument("--official-native", type=Path)
    parser.add_argument("--matched-native-root", type=Path)
    parser.add_argument("--official-standard", type=Path)
    parser.add_argument("--official-skeleton", type=Path)
    parser.add_argument("--branch", choices=("tbss", "mmorf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.reference_root) != bool(args.reference_label):
        parser.error("--reference-root and --reference-label must be provided together")
    stages = ("native", "standard", "skeleton") if args.branch == "tbss" else ("native", "standard")
    report = {
        "branch": args.branch,
        "comparison_support": "nonzero union",
    }
    if args.reference_root:
        report["reference"] = args.reference_label
        report["maps"] = {
            stage: {
                name: _compare(_file(args.candidate_root, stage, name),
                               _file(args.reference_root, stage, name))
                for name in MAPS
            } for stage in stages
        }
    if args.official_native:
        report["official_native"] = {
            name: _compare(_file(args.candidate_root, "native", name),
                           args.official_native / _file(args.candidate_root, "native", name).name)
            for name in MAPS
        }
    if args.matched_native_root:
        report["matched_fsl_eddy_native"] = {
            name: _compare(_file(args.candidate_root, "native", name),
                           _file(args.matched_native_root, "native", name))
            for name in MAPS
        }
    if args.official_standard:
        report["official_standard"] = {
            name: _compare(_file(args.candidate_root, "standard", name),
                           args.official_standard / (f"all_{name}.nii.gz" if args.branch == "tbss"
                                                     else f"{name}.nii.gz"))
            for name in MAPS
        }
    if args.official_skeleton and args.branch == "tbss":
        report["official_skeleton"] = {
            name: _compare(_file(args.candidate_root, "skeleton", name),
                           args.official_skeleton / f"all_{name}_skeletonised.nii.gz")
            for name in MAPS
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({stage: {name: (round(values[name]["pearson_r"], 6)
                                    if values[name]["pearson_r"] is not None else None)
                              for name in MAPS} for stage, values in report.items()
                      if stage in ("official_native", "official_standard", "official_skeleton")}, indent=2))


if __name__ == "__main__":
    main()
