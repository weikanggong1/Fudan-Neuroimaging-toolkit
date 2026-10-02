"""比较真实 T1 的 MNI 非线性链输出，可分别记录同输入 CPU/CUDA 与官方参考。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


OUTPUTS = ("warp.to.mni152.1.0mm.1.0mm.nii.gz",
           "warp.to.mni152.1.0mm.1.0mm.inv.nii.gz", "test.nii.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-subject", type=Path, required=True)
    parser.add_argument("--candidate-subject", type=Path, required=True)
    parser.add_argument("--scope", choices=("frozen_same_input", "connected_vs_official"),
                        required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"code_commit": args.code_commit, "scope": args.scope,
              "reference_subject": str(args.reference_subject),
              "candidate_subject": str(args.candidate_subject),
              "outputs": {}}
    for name in OUTPUTS:
        relative = Path("mri/transforms/synthmorph.1.0mm.1.0mm") / name
        reference_image = nib.load(str(args.reference_subject / relative))
        candidate_image = nib.load(str(args.candidate_subject / relative))
        reference, candidate = (np.asarray(image.dataobj) for image in
                                (reference_image, candidate_image))
        if reference.shape != candidate.shape:
            report["outputs"][name] = {"reference_shape": reference.shape,
                                       "candidate_shape": candidate.shape}
            continue
        error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        report["outputs"][name] = {
            "shape": reference.shape, "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "affine_max_absolute_mm": float(np.abs(reference_image.affine -
                                                    candidate_image.affine).max()),
            "different_values": int(np.count_nonzero(error)),
            "maximum_absolute_value_difference": float(error.max()),
            "p99_absolute_value_difference": float(np.quantile(error, .99)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
