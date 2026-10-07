"""Check complete real network TSV and CIFTI outputs with independent algebra."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--censor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import nibabel as nib
    import numpy as np
    series = np.load(args.series, mmap_mode="r", allow_pickle=False)
    if series.shape != (490, 59412):
        raise ValueError("Independent output check uses the complete real cortical run")
    if args.censor:
        censor = np.loadtxt(args.censor)
        if censor.shape != (490,) or not np.isin(censor, [0, 1]).all():
            raise ValueError("Invalid full-length censor")
        series = series[censor.astype(bool)]
    labels = np.load(args.results / "labels_fslr32k_64984.npy", allow_pickle=False)
    with np.load(args.assets, allow_pickle=False) as assets:
        mask = assets["cortex_mask"].astype(bool)
    weights = (labels[mask, None] == np.arange(1, 18)[None]).astype(np.float64)
    counts = weights.sum(axis=0)
    expected = np.full((len(series), 17), np.nan, dtype=np.float64)
    present = counts > 0
    expected[:, present] = np.asarray(series, dtype=np.float64) @ weights[:, present] / counts[present]
    actual = np.loadtxt(args.results / "network_timeseries.tsv", skiprows=1)
    centered = expected - expected.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    normalized = centered / norm
    correlation = normalized.T @ normalized
    actual_correlation = np.loadtxt(args.results / "network_correlation.tsv", skiprows=1)
    cifti = nib.load(args.results / "labels_fslr32k.dlabel.nii")
    report = {"source_frames": 490, "output_frames": len(series),
              "network_timeseries_shape": list(actual.shape),
              "network_timeseries_max_absolute_error": float(np.nanmax(np.abs(actual - expected))),
              "network_correlation_max_absolute_error": float(np.nanmax(np.abs(actual_correlation - correlation))),
              "missing_network_nan_pattern_exact": bool(np.array_equal(np.isnan(actual), np.isnan(expected))),
              "cifti_shape": list(cifti.shape),
              "cifti_different_labels": int(np.count_nonzero(np.asarray(cifti.dataobj)[0] != labels[mask])),
              "labels_dtype": str(labels.dtype),
              "medial_wall_nonzero_labels": int(np.count_nonzero(labels[~mask])),
              "network_label_range_valid": bool(np.isin(labels, np.arange(18)).all())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
