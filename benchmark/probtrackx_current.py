"""Summarize matched FSL/FNIT ProbtrackX runs on real DWI."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from probtrackx_validation import compare


def _wall(path):
    for line in path.read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1].split()[0])
    raise ValueError(f"missing wall time in {path}")


def _length_image(reference, current):
    first = nib.load(str(reference / "fdt_paths_lengths.nii.gz"))
    second = nib.load(str(current / "fdt_paths_lengths.nii.gz"))
    if first.shape != second.shape or not np.allclose(first.affine, second.affine):
        raise ValueError("length image geometry differs")
    a = np.asarray(first.dataobj, dtype=np.float64)
    b = np.asarray(second.dataobj, dtype=np.float64)
    support_a = a > 0
    support_b = b > 0
    common = support_a & support_b
    union = support_a | support_b
    return {
        "common_voxels": int(common.sum()),
        "support_dice": float(2 * common.sum() / (support_a.sum() + support_b.sum()))
        if support_a.any() or support_b.any() else 1.0,
        "pearson_on_common_nonzero": float(np.corrcoef(a[common], b[common])[0, 1])
        if common.sum() > 1 else None,
        "mean_abs_error_mm_on_common": float(np.abs(a[common] - b[common]).mean())
        if common.any() else 0.0,
        "fsl_mean_mm_on_common": float(a[common].mean()) if common.any() else 0.0,
        "fnit_mean_mm_on_common": float(b[common].mean()) if common.any() else 0.0,
        "mean_abs_error_mm_on_union_including_missing":
        float(np.abs(a[union] - b[union]).mean()) if union.any() else 0.0,
    }



def _matrix(reference, current, name):
    a = np.atleast_2d(np.loadtxt(reference / name))
    b = np.atleast_2d(np.loadtxt(current / name))
    if a.shape != b.shape:
        raise ValueError(f"{name} shape differs")
    return {
        "fsl": a.tolist(), "fnit": b.tolist(),
        "mean_abs_error": float(np.abs(a - b).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("default", "pd_ompl"), default="pd_ompl")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir
    names = {
        "seed_cpu": ("fsl_cpu_genu_cc", "equiv_cpu_seed"),
        "seed_gpu": ("fsl_gpu_genu_cc", "equiv_gpu_seed"),
        "network_cpu": ("fsl_cpu_network_2000_20260927", "equiv_cpu_network"),
        "network_gpu": ("fsl_gpu_network_2000", "equiv_gpu_network"),
    }
    report = {
        "reference": "FSL 6.0.7.22 probtrackx2 / probtrackx2_gpu",
        "data": "one real UK Biobank DWI; same FSL BEDPOSTX three-fibre posterior",
        "settings": {"options": ["--opd", "--pd", "--ompl"]
                     if args.mode == "pd_ompl" else ["--opd"],
                     "seed_voxels": 7, "seed_nsamples": 200,
                     "network_rois": 5, "network_nsamples": 2000,
                     "nsteps": 400, "steplength_mm": 0.5,
                     "cthr": 0.2, "fibthresh": 0.01,
                     "rseed": 20260927, "fnit_batch_size": 16384,
                     "cpu_threads": 8},
        "source_sha256": {name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
                          for name in ("pipeline.py", "_triton.py", "cli.py", "matrix_io.py", "_fast_counts.py")},
        "cases": {},
    }
    for name, (fsl_name, fnit_name) in names.items():
        fsl = directory / fsl_name
        fnit = directory / fnit_name
        result = {
            "fsl_vs_fnit_density": compare(fsl, fnit),
            "wall_seconds_including_load_and_write": {
                "fsl": _wall(directory / f"{fsl_name}.time"),
                "fnit": _wall(directory / f"{fnit_name}.time")},
        }
        if args.mode == "pd_ompl":
            result["fsl_vs_fnit_mean_length"] = _length_image(fsl, fnit)
        if name.startswith("network"):
            result["matrix"] = _matrix(fsl, fnit, "fdt_network_matrix")
            if args.mode == "pd_ompl":
                result["mean_length_matrix"] = _matrix(
                    fsl, fnit, "fdt_network_matrix_lengths")
        report["cases"][name] = result
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
