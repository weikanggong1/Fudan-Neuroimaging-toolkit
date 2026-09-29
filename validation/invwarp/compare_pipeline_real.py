"""Compare FSL and FNIT inverse masks for saved dMRI pipeline fields."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _load(path):
    image = nib.load(str(path))
    return image, np.asarray(image.dataobj, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tbss", "mmorf"), required=True)
    for name in ("fnit-forward", "fsl-inverse", "fnit-inverse", "fsl-mask",
                 "fnit-mask", "brain-mask", "source-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--fsl-forward")
    parser.add_argument("--fsl-convert-seconds", type=float)
    parser.add_argument("--fsl-inverse-seconds", type=float, required=True)
    parser.add_argument("--fsl-apply-seconds", type=float, required=True)
    args = parser.parse_args()
    fsl_inv, fsl_data = _load(args.fsl_inverse)
    fnit_inv, fnit_data = _load(args.fnit_inverse)
    _, brain = _load(args.brain_mask)
    _, fsl_mask = _load(args.fsl_mask)
    _, fnit_mask = _load(args.fnit_mask)
    if fsl_inv.shape != fnit_inv.shape or not np.allclose(
            fsl_inv.affine, fnit_inv.affine, atol=1e-5, rtol=0):
        raise ValueError("inverse fields are on different grids")
    fsl_mask, fnit_mask = fsl_mask > 0, fnit_mask > 0
    intersection = int(np.logical_and(fsl_mask, fnit_mask).sum())
    norm = np.linalg.norm(fsl_data - fnit_data, axis=-1)
    report = {
        "date": "2026-09-29", "backend": args.backend,
        "reference": "FSL 6.0.7.4", "candidate": "FNIT PyTorch GPU",
        "data": "one real UKB-format DWI; saved FNIT dMRI pipeline registration; JHU atlas label 9",
        "source_sha256": {
            name: hashlib.sha256((Path(args.source_root) / "fnit" / name).read_bytes()).hexdigest()
            for name in ("convertwarp/core.py", "invwarp/core.py", "applywarp/core.py")
        },
        "inverse": {
            "shape": list(fsl_inv.shape),
            "brain_vector_mean_mm": float(norm[brain > 0].mean()),
            "brain_vector_p95_mm": float(np.percentile(norm[brain > 0], 95)),
            "whole_grid_vector_mean_mm": float(norm.mean()),
            "fsl_complete_command_seconds": args.fsl_inverse_seconds,
        },
        "mask": {
            "fsl_voxels": int(fsl_mask.sum()), "fnit_voxels": int(fnit_mask.sum()),
            "intersection_voxels": intersection,
            "dice": float(2 * intersection / (fsl_mask.sum() + fnit_mask.sum())),
            "fsl_complete_command_seconds": args.fsl_apply_seconds,
        },
        "timing_boundary": "FSL one complete process invocation; FNIT branch Python calls reported separately",
    }
    if args.fsl_forward:
        fsl_fwd, fsl_forward = _load(args.fsl_forward)
        fnit_fwd, fnit_forward = _load(args.fnit_forward)
        if fsl_fwd.shape != fnit_fwd.shape or not np.allclose(
                fsl_fwd.affine, fnit_fwd.affine, atol=1e-5, rtol=0):
            raise ValueError("forward fields are on different grids")
        report["forward"] = {
            "shape": list(fsl_fwd.shape),
            "component_mae_mm": float(np.abs(fsl_forward - fnit_forward).mean()),
            "component_max_mm": float(np.abs(fsl_forward - fnit_forward).max()),
            "fsl_complete_command_seconds": args.fsl_convert_seconds,
        }
    else:
        report["forward_reference"] = (
            "FSL convertwarp cannot read MMORF reference-axis millimetre input; "
            "both inverse tools received the same FNIT-converted FSL dense field"
        )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
