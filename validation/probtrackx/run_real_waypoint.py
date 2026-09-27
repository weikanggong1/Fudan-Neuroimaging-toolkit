"""Private, matched FSL/FNIT waypoint benchmark on an authorized real DWI."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

import nibabel as nib
import numpy as np


def _paths(list_file):
    base = list_file.resolve().parent
    return [(Path(line) if Path(line).is_absolute() else base / line).resolve()
            for line in list_file.read_text().splitlines() if line.strip()]


def _choose_pair(roi_list, network_matrix):
    masks = _paths(roi_list)
    counts = np.atleast_2d(np.loadtxt(network_matrix))
    if len(masks) < 2 or counts.shape != (len(masks), len(masks)):
        raise ValueError("ROI list and prior FSL network matrix disagree")
    counts = counts.copy()
    np.fill_diagonal(counts, -np.inf)
    seed_index, waypoint_index = np.unravel_index(np.argmax(counts), counts.shape)
    if not np.isfinite(counts[seed_index, waypoint_index]) or counts[seed_index, waypoint_index] <= 0:
        raise ValueError("prior FSL network has no positive off-diagonal connection")
    return masks[seed_index], masks[waypoint_index], seed_index, waypoint_index


def _run(command, log, env):
    started = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, check=False)
    return {"exit_code": result.returncode,
            "wall_seconds_including_load_and_write": time.perf_counter() - started}


def _compare(reference, current):
    fsl_image = nib.load(str(reference / "fdt_paths.nii.gz"))
    fnit_image = nib.load(str(current / "fdt_paths.nii.gz"))
    if fsl_image.shape != fnit_image.shape or not np.allclose(fsl_image.affine, fnit_image.affine):
        raise ValueError("fdt_paths image geometry differs")
    a = np.asarray(fsl_image.dataobj, dtype=np.float64)
    b = np.asarray(fnit_image.dataobj, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("fdt_paths contains nonfinite values")
    support_a, support_b = a > 0, b > 0
    union = support_a | support_b
    common = support_a & support_b
    pearson = None
    if union.sum() > 1 and np.std(a[union]) > 0 and np.std(b[union]) > 0:
        pearson = float(np.corrcoef(a[union], b[union])[0, 1])
    waytotal_a = int(np.loadtxt(reference / "waytotal"))
    waytotal_b = int(np.loadtxt(current / "waytotal"))
    return {
        "fsl_waytotal": waytotal_a,
        "fnit_waytotal": waytotal_b,
        "waytotal_absolute_difference": abs(waytotal_a - waytotal_b),
        "both_waytotal_positive": waytotal_a > 0 and waytotal_b > 0,
        "fsl_support_voxels": int(support_a.sum()),
        "fnit_support_voxels": int(support_b.sum()),
        "common_support_voxels": int(common.sum()),
        "support_dice": float(2 * common.sum() / (support_a.sum() + support_b.sum()))
        if support_a.any() or support_b.any() else 1.0,
        "pearson_on_union_support": pearson,
        "mean_absolute_error_on_union_support":
            float(np.abs(a[union] - b[union]).mean()) if union.any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="private output directory on the authorized server")
    parser.add_argument("--bedpostx", type=Path, required=True)
    parser.add_argument("--roi-list", type=Path, required=True,
                        help="ordered list of real diffusion-space ROI NIfTIs")
    parser.add_argument("--selection-matrix", type=Path, required=True,
                        help="prior FSL five-ROI fdt_network_matrix")
    parser.add_argument("--fsl-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True,
                        help="Python with FNIT and nibabel installed")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="directory containing fnit/probtrackx")
    parser.add_argument("--nsamples", type=int, default=2000)
    args = parser.parse_args()
    if args.nsamples < 1 or args.output_dir.exists():
        parser.error("nsamples must be positive and output-dir must not exist")
    seed, waypoint, seed_index, waypoint_index = _choose_pair(
        args.roi_list, args.selection_matrix)
    args.output_dir.mkdir(parents=True)
    out = args.output_dir.resolve()
    source = args.source_dir.resolve()
    source_files = ("pipeline.py", "_triton.py", "cli.py", "matrix_io.py")
    source_sha256 = {
        name: hashlib.sha256((source / "fnit/probtrackx" / name).read_bytes()).hexdigest()
        for name in source_files
    }
    env = os.environ.copy()
    env["FSLDIR"] = str(args.fsl_dir.resolve())
    env["LD_LIBRARY_PATH"] = f"{env['FSLDIR']}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PYTHONPATH"] = str(source) + ":" + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "8"
    fsl_out, fnit_out = out / "fsl_cpu", out / "fnit_cpu"
    common_fsl = ["-s", str(args.bedpostx / "merged"),
                  "-m", str(args.bedpostx / "nodif_brain_mask.nii.gz"),
                  "-x", str(seed), f"--waypoints={waypoint}", "--opd",
                  "-P", str(args.nsamples), "-S", "400", "--steplength=0.5",
                  "--cthr=0.2", "--fibthresh=0.01", "--rseed=20260927"]
    fsl_command = [str(args.fsl_dir / "bin/probtrackx2"), *common_fsl,
                   f"--dir={fsl_out}", "--forcedir"]
    fnit_command = [str(args.python), "-m", "fnit.probtrackx.cli",
                    "--samples-dir", str(args.bedpostx), "--seed", str(seed),
                    "--waypoints", str(waypoint), "--output-dir", str(fnit_out),
                    "--device", "cpu", "--nsamples", str(args.nsamples),
                    "--nsteps", "400", "--steplength", "0.5", "--cthr", "0.2",
                    "--fibthresh", "0.01", "--batch-size", "2048",
                    "--rseed", "20260927"]
    fsl_run = _run(fsl_command, out / "fsl_cpu.log", env)
    for name in ("fdt_paths.nii.gz", "waytotal"):
        if not (fsl_out / name).is_file():
            raise RuntimeError(f"FSL output missing: {name}; see private log")
    if not re.search(r"\bfinished\b|\bTOTAL TIME\b",
                     (out / "fsl_cpu.log").read_text(errors="replace"), re.IGNORECASE):
        raise RuntimeError("FSL log did not report completion; see private log")
    fnit_run = _run(fnit_command, out / "fnit_cpu.log", env)
    if fnit_run["exit_code"] != 0:
        raise RuntimeError("FNIT failed; see private log")
    for name in ("fdt_paths.nii.gz", "waytotal"):
        if not (fnit_out / name).is_file():
            raise RuntimeError(f"FNIT output missing: {name}; see private log")
    metrics = _compare(fsl_out, fnit_out)
    report = {
        "reference": "FSL 6.0.7.22 probtrackx2 CPU",
        "selection": {
            "rule": "maximum positive off-diagonal entry in prior FSL five-ROI network matrix",
            "seed_roi_index_zero_based": int(seed_index),
            "waypoint_roi_index_zero_based": int(waypoint_index),
            "seed_mask": str(seed), "waypoint_mask": str(waypoint),
            "selection_matrix": str(args.selection_matrix.resolve()),
        },
        "settings": {"nsamples_per_seed_voxel": args.nsamples, "nsteps": 400,
                     "steplength_mm": 0.5, "cthr": 0.2, "fibthresh": 0.01,
                     "rseed": 20260927, "fnit_batch_size": 2048,
                     "cpu_threads": 8, "options": ["opd", "waypoints"]},
        "source_sha256": source_sha256,
        "wall_time": {"fsl": fsl_run, "fnit": fnit_run},
        "comparison": metrics,
    }
    (out / "report.private.json").write_text(json.dumps(report, indent=2) + "\n")
    if not metrics["both_waytotal_positive"]:
        raise RuntimeError("No accepted waypoint paths in one or both runs; see private report")
    print("Waypoint benchmark complete; private report remains in output-dir")


if __name__ == "__main__":
    main()
