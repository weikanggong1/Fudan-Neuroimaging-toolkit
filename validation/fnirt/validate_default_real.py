"""Run the FNIT no-config preset on one real image and compare saved FSL output."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.fnirt.standalone import run_fnirt


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_image(path):
    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite image: {path}")
    return image, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--fsl-warped", type=Path, required=True)
    parser.add_argument("--fsl-coeff", type=Path, required=True)
    parser.add_argument("--fsl-status", type=Path, required=True)
    parser.add_argument("--fsl-log", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = run_fnirt(
        input=args.moving,
        reference=args.reference,
        affine=args.affine,
        cout=output / "fnit_coeff.nii.gz",
        iout=output / "fnit_warped.nii.gz",
        refmask=None,
        config="default",
        device=args.device,
        overwrite=False,
    )
    seconds = time.perf_counter() - started
    reference, reference_data = checked_image(args.reference)
    fnit, fnit_data = checked_image(output / "fnit_warped.nii.gz")
    fsl, fsl_data = checked_image(args.fsl_warped)
    coefficient, coefficient_data = checked_image(output / "fnit_coeff.nii.gz")
    fsl_coefficient, fsl_coefficient_data = checked_image(args.fsl_coeff)
    if fnit.shape != fsl.shape or fnit.shape != reference.shape:
        raise ValueError("FNIT, FSL, and reference shapes differ")
    if not np.allclose(fnit.affine, reference.affine) or not np.allclose(
        fsl.affine, reference.affine
    ):
        raise ValueError("FNIT or FSL output grid differs from the reference")
    if int(coefficient.header["intent_code"]) != 2007 or coefficient_data.shape[-1] != 3:
        raise ValueError("FNIT coefficient format is invalid")
    if coefficient.shape != fsl_coefficient.shape or int(fsl_coefficient.header["intent_code"]) != 2007:
        raise ValueError("FNIT and FSL coefficient formats differ")
    brain = reference_data > 0
    fnit_support = fnit_data > 0.05 * np.percentile(fnit_data[brain & (fnit_data > 0)], 99)
    fsl_support = fsl_data > 0.05 * np.percentile(fsl_data[brain & (fsl_data > 0)], 99)
    fsl_log = args.fsl_log.read_text()
    fsl_seconds = next(
        (float(line.split("=", 1)[1].split()[0]) for line in fsl_log.splitlines()
         if line.startswith("fsl_wall_seconds=")), None
    )
    report = {
        "schema_version": 1,
        "fsl_exit_status": int(args.fsl_status.read_text().strip()),
        "fsl_wall_seconds": fsl_seconds,
        "fnit_exit_status": 0,
        "fnit_registration_and_output_seconds": seconds,
        "fnit_gpu_peak_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        ),
        "fnit_gpu_peak_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1e9 if device.type == "cuda" else None
        ),
        "fnit_preset": "default",
        "fnit_intensity_model": result.qc["global_intensity_model"],
        "fnit_level_count": len(result.qc["levels"]),
        "output_shape": list(fnit.shape),
        "coefficient_shape": list(coefficient.shape),
        "coefficient_intent": int(coefficient.header["intent_code"]),
        "output_grid_matches_reference": True,
        "gzip_crc_and_finite": True,
        "pearson_vs_fsl_same_input": float(np.corrcoef(fnit_data[brain], fsl_data[brain])[0, 1]),
        "warped_mae_vs_fsl_same_input": float(np.mean(np.abs(fnit_data[brain] - fsl_data[brain]))),
        "coefficient_pearson_vs_fsl": float(np.corrcoef(
            coefficient_data.ravel(), fsl_coefficient_data.ravel()
        )[0, 1]),
        "coefficient_mae_vs_fsl": float(np.mean(np.abs(
            coefficient_data - fsl_coefficient_data
        ))),
        "support_dice_vs_fsl_same_input": float(
            2 * np.count_nonzero(fnit_support & fsl_support)
            / (fnit_support.sum() + fsl_support.sum())
        ),
        "input_sha256": {
            "moving": sha256(args.moving),
            "reference": sha256(args.reference),
            "fsl_affine": sha256(args.affine),
            "fsl_warped": sha256(args.fsl_warped),
            "fsl_coefficients": sha256(args.fsl_coeff),
        },
        "output_sha256": {
            "fnit_coefficients": sha256(output / "fnit_coeff.nii.gz"),
            "fnit_warped": sha256(output / "fnit_warped.nii.gz"),
        },
        "source_sha256": {
            name: sha256(args.source_root / path)
            for name, path in {
                "fnirt_registration.py": "src/fnit/fnirt/registration.py",
                "fnirt_standalone.py": "src/fnit/fnirt/standalone.py",
                "validate_default_real.py": "validation/fnirt/validate_default_real.py",
            }.items()
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
