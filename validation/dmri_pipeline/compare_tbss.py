"""Compare one FNIT TorchTBSS output with the matched UKB/FSL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

MAPS = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")


def _load(path):
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float64).squeeze()
    if data.ndim != 3:
        raise ValueError(f"expected one 3D map: {path}")
    return image, data


def _metrics(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    valid &= (first != 0) | (second != 0)
    x = first[valid]
    y = second[valid]
    return {
        "voxels": int(x.size),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean(np.square(x - y)))),
    }


def compare(fnit_dir, fsl_dir):
    fnit_dir = Path(fnit_dir)
    fsl_dir = Path(fsl_dir)
    report = {
        "standard_maps": {},
        "skeleton_maps": {},
        "grid_contract": {},
    }
    for name in MAPS:
        fnit_standard, fnit_data = _load(fnit_dir / "standard" / f"{name}.nii.gz")
        fsl_standard_name = "all_FA.nii.gz" if name == "FA" else f"all_{name}.nii.gz"
        fsl_standard, fsl_data = _load(fsl_dir / "stats" / fsl_standard_name)
        fnit_skeleton, fnit_skeleton_data = _load(
            fnit_dir / "skeleton" / f"{name}.nii.gz"
        )
        fsl_skeleton, fsl_skeleton_data = _load(
            fsl_dir / "stats" / f"all_{name}_skeletonised.nii.gz"
        )
        report["standard_maps"][name] = _metrics(fnit_data, fsl_data)
        report["skeleton_maps"][name] = _metrics(
            fnit_skeleton_data, fsl_skeleton_data
        )
        report["grid_contract"][name] = {
            "same_shape": fnit_standard.shape == fsl_standard.shape,
            "same_affine": bool(
                np.allclose(fnit_standard.affine, fsl_standard.affine, atol=1e-5, rtol=0)
            ),
            "same_dtype": fnit_standard.get_data_dtype() == fsl_standard.get_data_dtype(),
            "skeleton_same_shape": fnit_skeleton.shape == fsl_skeleton.shape,
            "skeleton_same_affine": bool(
                np.allclose(fnit_skeleton.affine, fsl_skeleton.affine, atol=1e-5, rtol=0)
            ),
            "skeleton_same_dtype": (
                fnit_skeleton.get_data_dtype() == fsl_skeleton.get_data_dtype()
            ),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnit-dir", required=True)
    parser.add_argument("--fsl-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = compare(args.fnit_dir, args.fsl_dir)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
