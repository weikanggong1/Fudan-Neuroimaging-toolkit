#!/usr/bin/env python3
"""Run current TorchFNIRT on the fixed real-FA matched-input case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.dmri_pipeline.tbss import TBSSConfig
from fnit.flirt.coordinates import flirt_to_world_affine
from fnit.fnirt import TorchFNIRT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--affine", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    moving = nib.load(str(args.input))
    fixed = nib.load(str(args.reference))
    fsl_affine = np.loadtxt(args.affine, dtype=np.float64)
    moving_to_fixed_world = flirt_to_world_affine(
        fsl_affine,
        moving.affine,
        fixed.affine,
        moving.shape[:3],
        fixed.shape[:3],
        nib.affines.voxel_sizes(moving.affine),
        nib.affines.voxel_sizes(fixed.affine),
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = TorchFNIRT(device=device, config=TBSSConfig().fnirt)(
        moving,
        fixed,
        moving_to_fixed_world,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    outputs = {
        "coefficient": args.output_dir / "dti_FA_to_MNI_warp.nii.gz",
        "iout": args.output_dir / "dti_FA_to_MNI.nii.gz",
        "jacobian_nonlinear": args.output_dir / "jacobian_nonlinear.nii.gz",
        "jacobian_full_pull": args.output_dir / "jacobian_full_pull.nii.gz",
    }
    nib.save(result.coefficient_image, str(outputs["coefficient"]))
    nib.save(result.moved, str(outputs["iout"]))
    nib.save(result.nonlinear_jacobian, str(outputs["jacobian_nonlinear"]))
    nib.save(result.full_pull_jacobian, str(outputs["jacobian_full_pull"]))
    report = {
        "schema_version": 1,
        "input": {
            "image_sha256": _sha256(args.input),
            "reference_sha256": _sha256(args.reference),
            "affine_sha256": _sha256(args.affine),
        },
        "execution": {
            "device": str(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "synchronized_wall_seconds": elapsed,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "schedule": {
            "subsampling": list(TBSSConfig().fnirt.subsampling),
            "maximum_iterations": list(TBSSConfig().fnirt.maximum_iterations),
            "process_stages": list(TBSSConfig().fnirt.process_stages or ()),
            "warp_resolution_schedule_mm": [
                list(value)
                for value in (TBSSConfig().fnirt.warp_resolution_schedule_mm or ())
            ],
        },
        "affine_pull_determinant": result.affine_pull_determinant,
        "qc": result.qc,
        "output_sha256": {
            name: _sha256(path) for name, path in outputs.items()
        },
    }
    (args.output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
