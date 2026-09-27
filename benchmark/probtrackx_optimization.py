"""Summarize matched real-DWI ProbtrackX optimization runs without subject images."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from probtrackx_validation import compare

REGIONS = ("genu_cc", "cst_right", "cst_left", "slf_right", "slf_left")


def _wall(path):
    for line in path.read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1].split()[0])
    raise ValueError(f"missing wall time: {path}")


def _matrix(directory):
    return np.atleast_2d(np.loadtxt(directory / "fdt_network_matrix", dtype=int)).tolist()


def _exact(first, second):
    a = np.asarray(nib.load(str(first / "fdt_paths.nii.gz")).dataobj)
    b = np.asarray(nib.load(str(second / "fdt_paths.nii.gz")).dataobj)
    return bool(np.array_equal(a, b) and
                np.array_equal(np.loadtxt(first / "waytotal"),
                               np.loadtxt(second / "waytotal")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--optimized-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    b, o = args.baseline_dir, args.optimized_dir
    report = {
        "reference": "FSL 6.0.7.22 probtrackx2 / probtrackx2_gpu",
        "data": "one real UK Biobank DWI, shared FSL BEDPOSTX three-fibre posterior",
        "settings": {"seed_voxels_per_roi": 7, "seed_nsamples": 200,
                     "network_nsamples": 2000, "nsteps": 400,
                     "steplength_mm": 0.5, "cthr": 0.2,
                     "fibthresh": 0.01, "rseed": 20260927,
                     "old_batch_size": 256, "optimized_batch_size": 2048},
        "source_sha256": {name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
                          for name in ("pipeline.py", "_triton.py")},
        "loader_only_same_batch_exact": {
            device: _exact(o / f"baseline_{device}", o / f"optimized_{device}")
            for device in ("cpu", "cuda0")},
        "seed_cpu": {},
    }
    if not all(report["loader_only_same_batch_exact"].values()):
        raise AssertionError("loader-only optimization changed tracking output")
    for region in REGIONS:
        fsl = o / f"paired_fsl_cpu_{region}"
        fnit = o / f"paired_fnit_cpu_{region}"
        report["seed_cpu"][region] = {
            "fsl_vs_fnit": compare(fsl, fnit),
            "wall_seconds_including_load_and_write": {
                "fsl": _wall(o / f"paired_fsl_cpu_{region}.time"),
                "fnit": _wall(o / f"paired_fnit_cpu_{region}.time")}}
    fsl_cpu = b / "fsl_cpu_network_2000_20260927"
    cpu = o / "batch_2048_network"
    fsl_gpu = o / "fsl_gpu_network_2000"
    gpu = o / "integrated_network_2000"
    report["network_cpu"] = {
        "fsl_vs_optimized": compare(fsl_cpu, cpu),
        "fsl_matrix": _matrix(fsl_cpu),
        "optimized_matrix": _matrix(cpu),
        "wall_seconds_including_load_and_write": {
            "fsl": _wall(b / "fsl_cpu_network_2000_20260927.time"),
            "original_fnit": _wall(b / "fnit_cpu_network_2000.time"),
            "optimized_fnit": _wall(o / "batch_2048_network.time")}}
    report["network_gpu"] = {
        "fsl_vs_optimized": compare(fsl_gpu, gpu),
        "fsl_matrix": _matrix(fsl_gpu),
        "optimized_matrix": _matrix(gpu),
        "wall_seconds_including_load_and_write": {
            "fsl": _wall(o / "fsl_gpu_network_2000.time"),
            "pytorch_fnit": _wall(o / "optimized_gpu_network_2000.time"),
            "triton_fnit": _wall(o / "integrated_network.time")},
        "fnit_peak_cuda_gib": json.loads((o / "integrated_network.runtime.json").read_text())[
            "max_cuda_allocated_gib"]}
    report["seed_gpu_genu_cc"] = {
        "fsl_vs_optimized": compare(b / "fsl_gpu_genu_cc", o / "integrated_seed_genu_cc"),
        "wall_seconds_including_load_and_write": {
            "fsl": _wall(b / "fsl_gpu_genu_cc.time"),
            "triton_fnit": _wall(o / "integrated_seed.time")}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
