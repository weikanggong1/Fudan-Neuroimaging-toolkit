"""Compare FNIT applyxfm with FSL on original MNI152 1/2 mm templates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, env, *, expected_output):
    expected_output.unlink(missing_ok=True)
    start = time.perf_counter()
    process = subprocess.run(command, env=env, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if not expected_output.is_file():
        raise RuntimeError(f"command did not write output (status {process.returncode}): {process.stderr}")
    return seconds, process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsl-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["FSLDIR"] = str(args.fsl_dir)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["LD_LIBRARY_PATH"] = str(args.fsl_dir / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = str(source_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    records = []
    for source_mm, target_mm in ((1, 2), (2, 1)):
        moving = args.fsl_dir / "data" / "standard" / f"MNI152_T1_{source_mm}mm.nii.gz"
        fixed = args.fsl_dir / "data" / "standard" / f"MNI152_T1_{target_mm}mm.nii.gz"
        official = args.work_dir / f"fsl_{source_mm}to{target_mm}.nii.gz"
        candidate = args.work_dir / f"fnit_{source_mm}to{target_mm}.nii.gz"
        matrix = args.work_dir / f"fnit_{source_mm}to{target_mm}.mat"
        official_seconds, official_status = run([
            str(args.fsl_dir / "bin" / "flirt"), "-in", str(moving),
            "-ref", str(fixed), "-applyxfm", "-usesqform", "-out", str(official),
        ], env, expected_output=official)
        matrix.unlink(missing_ok=True)
        fnit_seconds, fnit_status = run([
            sys.executable, "-m", "fnit.flirt.cli", "-in", str(moving),
            "-ref", str(fixed), "-applyxfm", "-usesqform", "-out", str(candidate),
            "-omat", str(matrix), "--device", args.device,
        ], env, expected_output=candidate)
        if fnit_status != 0:
            raise RuntimeError(f"FNIT returned status {fnit_status}")
        fsl_image = nib.load(official)
        fnit_image = nib.load(candidate)
        reference = nib.load(fixed)
        fsl_data = fsl_image.get_fdata(dtype=np.float32)
        fnit_data = fnit_image.get_fdata(dtype=np.float32)
        difference = np.abs(fsl_data - fnit_data)
        union = (fsl_data != 0) | (fnit_data != 0)
        records.append({
            "direction": f"{source_mm}mm_to_{target_mm}mm",
            "moving_sha256": sha256(moving),
            "reference_sha256": sha256(fixed),
            "reference_shape": list(reference.shape),
            "output_shape_matches_reference": fsl_image.shape == fnit_image.shape == reference.shape,
            "fsl_affine_matches_reference": bool(np.allclose(fsl_image.affine, reference.affine)),
            "fnit_affine_matches_reference": bool(np.allclose(fnit_image.affine, reference.affine)),
            "pearson_all_voxels": float(np.corrcoef(fsl_data.ravel(), fnit_data.ravel())[0, 1]),
            "mae_all_voxels": float(difference.mean()),
            "mae_nonzero_union": float(difference[union].mean()),
            "max_absolute_difference": float(difference.max()),
            "fsl_wall_seconds": official_seconds,
            "fsl_exit_status": official_status,
            "fnit_wall_seconds": fnit_seconds,
            "fnit_fsl_matrix": np.loadtxt(matrix).tolist(),
        })
    report = {
        "benchmark": "FSL MNI152_T1 1mm/2mm applyxfm -usesqform",
        "candidate_device": args.device,
        "timing_scope": "fresh CLI process, image loading, resampling, gzip output",
        "fnit_core_sha256": sha256(source_root / "src" / "fnit" / "flirt" / "core.py"),
        "fnit_standalone_sha256": sha256(source_root / "src" / "fnit" / "flirt" / "standalone.py"),
        "cases": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
