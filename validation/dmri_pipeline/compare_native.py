"""Compare FNIT native DTI/NODDI maps with matched official UKB outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

MAP_FILES = {
    "FA": "dti_FA.nii.gz",
    "MD": "dti_MD.nii.gz",
    "L1": "dti_L1.nii.gz",
    "L2": "dti_L2.nii.gz",
    "L3": "dti_L3.nii.gz",
    "MO": "dti_MO.nii.gz",
    "ICVF": "NODDI_ICVF.nii.gz",
    "OD": "NODDI_OD.nii.gz",
    "ISOVF": "NODDI_ISOVF.nii.gz",
}


def _metrics(first, second, mask):
    x = first[mask]
    y = second[mask]
    return {
        "voxels": int(x.size),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean(np.square(x - y)))),
    }


def compare(fnit_native, official_native):
    fnit_native = Path(fnit_native)
    official_native = Path(official_native)
    report = {}
    for name, filename in MAP_FILES.items():
        first_image = nib.load(str(fnit_native / filename))
        second_image = nib.load(str(official_native / filename))
        first = np.asarray(first_image.dataobj, dtype=np.float64).squeeze()
        second = np.asarray(second_image.dataobj, dtype=np.float64).squeeze()
        if first.ndim != 3 or second.ndim != 3:
            raise ValueError(f"expected a 3D or one-frame map: {filename}")
        finite = np.isfinite(first) & np.isfinite(second)
        report[name] = {
            "same_shape": first.shape == second.shape,
            "same_affine": bool(
                np.allclose(first_image.affine, second_image.affine, atol=1e-5, rtol=0)
            ),
            "union_support": _metrics(
                first, second, finite & ((first != 0) | (second != 0))
            ),
            "common_support": _metrics(
                first, second, finite & (first != 0) & (second != 0)
            ),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnit-native", required=True)
    parser.add_argument("--official-native", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = compare(args.fnit_native, args.official_native)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
