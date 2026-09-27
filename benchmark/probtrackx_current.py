"""Summarize matched FSL and current FNIT ProbtrackX runs on authorized data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probtrackx_validation import compare


def _wall(path):
    for line in path.read_text().splitlines():
        if line.startswith("wall_s="):
            return float(line.split("=", 1)[1].split()[0])
    raise ValueError(f"missing wall time in {path}")



def _network_time(directory, device):
    clean = directory / f"equiv_{device}_network_clean.time"
    return clean if clean.is_file() else directory / f"equiv_{device}_network.time"


def _matrix(directory):
    return np.atleast_2d(np.loadtxt(directory / "fdt_network_matrix", dtype=int)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    reference, current = args.reference_dir, args.current_dir
    cases = {
        "seed_cpu": (reference / "fsl_cpu_genu_cc", current / "equiv_cpu_seed",
                     reference / "fsl_cpu_genu_cc.time", current / "equiv_cpu_seed.time"),
        "seed_gpu": (reference / "fsl_gpu_genu_cc", current / "equiv_gpu_seed",
                     reference / "fsl_gpu_genu_cc.time", current / "equiv_gpu_seed.time"),
        "network_cpu": (reference / "fsl_cpu_network_2000_20260927", current / "equiv_cpu_network",
                        reference / "fsl_cpu_network_2000_20260927.time",
                        _network_time(current, "cpu")),
        "network_gpu": (current / "fsl_gpu_network_2000", current / "equiv_gpu_network",
                        current / "fsl_gpu_network_2000.time",
                        _network_time(current, "gpu")),
    }
    report = {
        "reference": "FSL 6.0.7.22 probtrackx2 / probtrackx2_gpu",
        "data": "one real UK Biobank DWI; same FSL BEDPOSTX three-fibre posterior",
        "settings": {"seed_voxels": 7, "seed_nsamples": 200,
                     "network_rois": 5, "network_nsamples": 2000,
                     "nsteps": 400, "steplength_mm": 0.5,
                     "cthr": 0.2, "fibthresh": 0.01,
                     "rseed": 20260927, "fnit_batch_size": 2048,
                     "cpu_threads": 8},
        "source_sha256": {name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
                          for name in ("pipeline.py", "_triton.py", "cli.py")},
        "cases": {},
    }
    for name, (fsl_dir, fnit_dir, fsl_time, fnit_time) in cases.items():
        result = {"fsl_vs_fnit": compare(fsl_dir, fnit_dir),
                  "wall_seconds_including_load_and_write": {
                      "fsl": _wall(fsl_time), "fnit": _wall(fnit_time)}}
        if name.startswith("network"):
            result["fsl_matrix"] = _matrix(fsl_dir)
            result["fnit_matrix"] = _matrix(fnit_dir)
        report["cases"][name] = result
    peak_file = current / "equiv_gpu_network.runtime.json"
    if peak_file.is_file():
        report["cases"]["network_gpu"]["fnit_peak_cuda_gib"] = json.loads(
            peak_file.read_text())["max_cuda_allocated_gib"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
