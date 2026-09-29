"""Check subject-block sufficient statistics against direct OLS on all ABIDE subjects."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    with args.participants.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1748:
        raise ValueError("expected 1748 quality-screened ABIDE I+II subjects")
    sites = [column for column in rows[0] if column.startswith("site_")]
    design = np.array([[float(row[column]) for column in
                        ("case", "age", "sex", *sites)] + [1.0]
                       for row in rows], dtype=np.float64)
    if design.shape != (1748, 39) or np.linalg.matrix_rank(design) != 39:
        raise ValueError("unexpected or rank-deficient case/age/sex/site design")
    nvoxels = 112215
    seeds = (nvoxels//4, 3*nvoxels//4)
    values = np.empty((len(rows), 2*nvoxels), dtype=np.float64)
    start = perf_counter()
    for subject in range(len(rows)):
        series = np.load(args.cache_dir / f"subject-{subject}.npy", mmap_mode="r")
        if series.shape[1] != nvoxels:
            raise ValueError("wrong gray-matter cache width")
        correlation = (series[:, seeds].T @ series) / len(series)
        correlation[correlation > 0.9999] = 0
        values[subject] = np.arctanh(np.clip(correlation, -0.999999, 0.999999)).ravel()
    correlation_seconds = perf_counter()-start
    inverse = np.linalg.pinv(design.T @ design)
    degrees = len(rows)-design.shape[1]
    scale = np.sqrt(inverse[0, 0] / degrees)
    weights = inverse @ design.T

    start = perf_counter()
    z_direct = np.empty(values.shape[1], dtype=np.float64)
    for lo in range(0, values.shape[1], 8192):
        hi = min(lo+8192, values.shape[1])
        observed = values[:, lo:hi]
        beta = weights @ observed
        residual = observed-design @ beta
        sigma = np.sum(residual*residual, axis=0)
        t = beta[0] / (np.sqrt(sigma) * scale)
        z_direct[lo:hi] = stats.norm.ppf(stats.t.cdf(t, degrees-1))
    direct_seconds = perf_counter()-start

    start = perf_counter()
    xy = np.zeros((design.shape[1], values.shape[1]), dtype=np.float64)
    yy = np.zeros(values.shape[1], dtype=np.float64)
    for lo in range(0, len(rows), 16):
        hi = min(lo+16, len(rows))
        xy += design[lo:hi].T @ values[lo:hi]
        yy += np.sum(values[lo:hi]**2, axis=0)
    beta = inverse @ xy
    sigma = yy-np.sum(beta*xy, axis=0)
    t = beta[0] / (np.sqrt(sigma) * scale)
    z_blocked = stats.norm.ppf(stats.t.cdf(t, degrees-1))
    blocked_seconds = perf_counter()-start
    valid = np.isfinite(z_direct) & np.isfinite(z_blocked)
    for row, seed in enumerate(seeds):
        valid[row*nvoxels+seed] = False
    difference = np.abs(z_direct[valid]-z_blocked[valid])
    result = {
        "subjects": len(rows), "site_dummy_columns": len(sites),
        "gray_voxels": nvoxels, "seeds": len(seeds),
        "voxel_pairs_compared": int(valid.sum()),
        "z_mean_absolute_difference": float(difference.mean()),
        "z_max_absolute_difference": float(difference.max()),
        "cdt3_disagreements": int(np.count_nonzero(
            (np.abs(z_direct[valid]) > 3) != (np.abs(z_blocked[valid]) > 3))),
        "cdt5_disagreements": int(np.count_nonzero(
            (np.abs(z_direct[valid]) > 5) != (np.abs(z_blocked[valid]) > 5))),
        "correlation_seconds": correlation_seconds,
        "direct_ols_seconds": direct_seconds,
        "subject_block_ols_seconds": blocked_seconds,
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
