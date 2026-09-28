#!/usr/bin/env python3
"""Compare one TorchEDDY run with a matched FSL eddy_cuda10.2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(first: np.ndarray, second: np.ndarray) -> dict:
    valid = np.isfinite(first) & np.isfinite(second)
    a = first[valid].astype(np.float64)
    b = second[valid].astype(np.float64)
    diff = a - b
    return {
        "count": int(a.size),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def voxel_correlations(first: np.ndarray, second: np.ndarray) -> dict:
    a = first.astype(np.float64)
    b = second.astype(np.float64)
    finite = np.all(np.isfinite(a) & np.isfinite(b), axis=1)
    a = a[finite] - a[finite].mean(axis=1, keepdims=True)
    b = b[finite] - b[finite].mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    r = np.sum(a * b, axis=1)[denominator > 1e-8] / denominator[denominator > 1e-8]
    return {
        "count": int(r.size),
        "excluded_constant_or_nonfinite": int(first.shape[0] - r.size),
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "p05": float(np.quantile(r, 0.05)),
        "p95": float(np.quantile(r, 0.95)),
        "fraction_below_0_9": float(np.mean(r < 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-root", type=Path, required=True, help="TorchEDDY output basename")
    parser.add_argument("--fsl-root", type=Path, required=True, help="FSL eddy_cuda10.2 output basename")
    parser.add_argument("--mask", type=Path, required=True, help="identical input brain mask")
    parser.add_argument("--bvals", type=Path, required=True, help="identical input b-values")
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    args = parser.parse_args()

    paths = {name: Path(str(root) + ".nii.gz") for name, root in
             (("torch", args.torch_root), ("fsl", args.fsl_root))}
    images = {name: nib.load(str(path)) for name, path in paths.items()}
    mask_image = nib.load(str(args.mask))
    torch_image, fsl_image = images["torch"], images["fsl"]
    if torch_image.shape != fsl_image.shape or torch_image.shape[:3] != mask_image.shape:
        raise ValueError("EDDY and mask shapes differ")
    if not np.allclose(torch_image.affine, fsl_image.affine, atol=1e-5, rtol=0) or not np.allclose(
        torch_image.affine, mask_image.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("EDDY and mask grids differ")
    bvals = np.loadtxt(args.bvals).reshape(-1)
    if bvals.size != torch_image.shape[3]:
        raise ValueError("bvals and EDDY volume counts differ")
    mask = np.asarray(mask_image.dataobj) > 0
    a = np.asarray(torch_image.dataobj, dtype=np.float32)[mask]
    b = np.asarray(fsl_image.dataobj, dtype=np.float32)[mask]
    shells = {
        "all": np.ones(bvals.size, dtype=bool),
        "b0": bvals < 100,
        "b1000": (bvals >= 900) & (bvals < 1500),
        "b2000": bvals >= 1500,
    }
    result = {
        "reference": "FSL 6.0.7.4 eddy_cuda10.2 with matched input files",
        "shape": list(torch_image.shape),
        "mask_voxels": int(mask.sum()),
        "input_sha256": {"mask": sha256(args.mask), "bvals": sha256(args.bvals)},
        "output_sha256": {name: sha256(path) for name, path in paths.items()},
        "shells": {
            name: {
                "volumes": int(selected.sum()),
                "pooled": metrics(a[:, selected], b[:, selected]),
                "per_voxel_across_volumes": voxel_correlations(a[:, selected], b[:, selected]),
            }
            for name, selected in shells.items()
        },
    }
    torch_bvec = np.loadtxt(str(args.torch_root) + ".eddy_rotated_bvecs")
    fsl_bvec = np.loadtxt(str(args.fsl_root) + ".eddy_rotated_bvecs")
    if torch_bvec.shape != fsl_bvec.shape or torch_bvec.shape != (3, bvals.size):
        raise ValueError("rotated bvec shapes differ")
    directions = bvals >= 100
    dot = np.sum(torch_bvec[:, directions] * fsl_bvec[:, directions], axis=0)
    norm = np.linalg.norm(torch_bvec[:, directions], axis=0) * np.linalg.norm(
        fsl_bvec[:, directions], axis=0
    )
    angles = np.degrees(np.arccos(np.clip(np.abs(dot / norm), -1, 1)))
    result["rotated_bvec_unsigned_angle_degrees"] = {
        "mean": float(np.mean(angles)), "p95": float(np.quantile(angles, 0.95)),
        "maximum": float(np.max(angles)),
    }
    for suffix in ("eddy_parameters", "eddy_outlier_map"):
        fsl_file = Path(str(args.fsl_root) + "." + suffix)
        torch_file = Path(str(args.torch_root) + "." + suffix)
        if fsl_file.exists() and torch_file.exists():
            reference = np.loadtxt(fsl_file, ndmin=2, skiprows=int(suffix == "eddy_outlier_map"))
            candidate = np.loadtxt(torch_file, ndmin=2)
            if candidate.shape != reference.shape:
                raise ValueError(f"{suffix} shapes differ")
            if suffix == "eddy_parameters":
                result[suffix] = {
                    "translation_mae_mm": float(np.mean(np.abs(candidate[:, :3] - reference[:, :3]))),
                    "rotation_mae_rad": float(np.mean(np.abs(candidate[:, 3:6] - reference[:, 3:6]))),
                    "ec_parameter_mae": float(np.mean(np.abs(candidate[:, 6:] - reference[:, 6:]))),
                }
            else:
                result[suffix] = {
                    "torch_count": int(np.count_nonzero(candidate)),
                    "fsl_count": int(np.count_nonzero(reference)),
                    "overlap_count": int(np.count_nonzero((candidate != 0) & (reference != 0))),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
