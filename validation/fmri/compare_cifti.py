"""Compare two real, identically indexed CIFTI time series without publishing data."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from benchmark_bids import sha256


def metrics(first, second):
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    difference = first - second
    first -= first.mean(axis=0)
    second -= second.mean(axis=0)
    denominator = np.sqrt((first * first).sum(axis=0) * (second * second).sum(axis=0))
    valid = denominator > 1e-8
    temporal_r = (first[:, valid] * second[:, valid]).sum(axis=0) / denominator[valid]
    return {"grayordinates": int(first.shape[1]), "valid_temporal_correlations": int(valid.sum()),
            "mean_temporal_r": float(temporal_r.mean()) if valid.any() else None,
            "median_temporal_r": float(np.median(temporal_r)) if valid.any() else None,
            "p05_temporal_r": float(np.percentile(temporal_r, 5)) if valid.any() else None,
            "mae": float(np.abs(difference).mean()),
            "maximum_absolute_difference": float(np.abs(difference).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    candidate, reference = (nib.load(str(p)) for p in (args.candidate, args.reference))
    if candidate.shape != reference.shape or not all(
            candidate.header.get_axis(index) == reference.header.get_axis(index) for index in (0, 1)):
        raise ValueError("CIFTI time or brain-model axes differ")
    first = np.asarray(candidate.dataobj, dtype=np.float32)
    second = np.asarray(reference.dataobj, dtype=np.float32)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("nonfinite CIFTI values")
    axis = candidate.header.get_axis(1)
    by_structure = {name: metrics(first[:, indices], second[:, indices])
                    for name, indices, _ in axis.iter_structures()}
    subcortex = np.array(["CORTEX" not in name for name in axis.name])
    report = {"schema_version": 1, "shape": list(candidate.shape),
              "time_and_brain_model_axes_exact": True,
              "candidate_sha256": sha256(args.candidate), "reference_sha256": sha256(args.reference),
              "by_structure": by_structure, "subcortex": metrics(first[:, subcortex], second[:, subcortex]),
              "scope": "Same current-code volume BOLD, reconstruction, HCP assets and surface API; only the registered spheres differ.",
              "reference": "Previously generated official newMSM spheres; projection and assembly use the same FNIT/Workbench path.",
              "limits": ["This isolates sphere correspondence differences; it is not an independent full fMRIPrep run.",
                         "No subject identifiers, image data or absolute paths are included."]}
    args.report_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cortex_left": by_structure["CIFTI_STRUCTURE_CORTEX_LEFT"],
                      "cortex_right": by_structure["CIFTI_STRUCTURE_CORTEX_RIGHT"],
                      "subcortex": report["subcortex"]}, indent=2))


if __name__ == "__main__":
    main()
