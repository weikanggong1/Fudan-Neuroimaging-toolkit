"""Compare full-cohort subject-block GLM with one-shot OLS on the same Fisher edges."""

import argparse
import csv
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import stats
import torch

from fnit.bwas import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with args.participants.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1748:
        raise ValueError("expected 1748 quality-screened ABIDE I+II subjects")
    sites = [column for column in rows[0] if column.startswith("site_")]
    design = np.array([[float(row[column]) for column in
                        ("case", "age", "sex", *sites)] + [1.0]
                       for row in rows], dtype=np.float32)
    if design.shape != (1748, 39) or np.linalg.matrix_rank(design) != 39:
        raise ValueError("unexpected or rank-deficient case/age/sex/site design")
    nvoxels = 112215
    seeds = (nvoxels//4, 3*nvoxels//4)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < len(rows)+128:
        resource.setrlimit(resource.RLIMIT_NOFILE, (len(rows)+128, hard))
    values = np.empty((len(rows), 2*nvoxels), dtype=np.float32)
    start = perf_counter()
    for subject in range(len(rows)):
        series = np.load(args.cache_dir / f"subject-{subject}.npy", mmap_mode="r")
        if series.shape[0] != nvoxels:
            raise ValueError("expected voxel-major gray-matter cache")
        correlation = series[list(seeds)] @ series.T / series.shape[1]
        correlation[correlation > 0.9999] = 0
        values[subject] = np.arctanh(
            np.clip(correlation, -0.999999, 0.999999)).ravel()
    correlation_seconds = perf_counter()-start

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.as_tensor(design, device=device)
    nuisance, _ = torch.linalg.qr(x[:, 1:], mode="reduced")
    phenotype = x[:, 0] - nuisance @ (nuisance.T @ x[:, 0])
    q = torch.column_stack((phenotype / torch.linalg.vector_norm(phenotype),
                            nuisance))
    q_cpu = q.cpu().numpy()
    df = len(rows)-q.shape[1]

    start = perf_counter()
    z_direct = np.empty(values.shape[1], dtype=np.float64)
    for lo in range(0, values.shape[1], 8192):
        hi = min(lo+8192, values.shape[1])
        observed = np.ascontiguousarray(values[:, lo:hi])
        beta = q_cpu.T @ observed
        residual = observed-q_cpu @ beta
        sigma = np.sum(residual*residual, axis=0)
        t = beta[0] / np.sqrt(sigma/df)
        z_direct[lo:hi] = stats.norm.ppf(stats.t.cdf(t, df-1))
    direct_seconds = perf_counter()-start

    start = perf_counter()
    xy = torch.zeros((q.shape[1], values.shape[1]), device=device)
    yy = torch.zeros(values.shape[1], device=device)
    for lo in range(0, len(rows), 16):
        hi = min(lo+16, len(rows))
        observed = torch.as_tensor(np.ascontiguousarray(values[lo:hi]),
                                   device=device)
        xy += q[lo:hi].T @ observed
        yy += (observed*observed).sum(dim=0)
    sigma = yy-(xy*xy).sum(dim=0)
    t = (xy[0] / torch.sqrt(sigma/df)).cpu().numpy()
    z_blocked = stats.norm.ppf(stats.t.cdf(t, df-1))
    blocked_seconds = perf_counter()-start

    valid = np.isfinite(z_direct) & np.isfinite(z_blocked)
    for row, seed in enumerate(seeds):
        valid[row*nvoxels+seed] = False
    difference = np.abs(z_direct[valid]-z_blocked[valid])
    result = {
        "subjects": len(rows), "site_dummy_columns": len(sites),
        "gray_voxels": nvoxels, "seeds": len(seeds),
        "voxel_pairs_compared": int(valid.sum()),
        "precision": "float32", "tf32": False,
        "z_mean_absolute_difference": float(difference.mean()),
        "z_max_absolute_difference": float(difference.max()),
        "cdt3_disagreements": int(np.count_nonzero(
            (np.abs(z_direct[valid]) > 3) != (np.abs(z_blocked[valid]) > 3))),
        "cdt5_disagreements": int(np.count_nonzero(
            (np.abs(z_direct[valid]) > 5) != (np.abs(z_blocked[valid]) > 5))),
        "correlation_seconds": correlation_seconds,
        "direct_ols_seconds": direct_seconds,
        "subject_block_ols_seconds": blocked_seconds,
        "fnit_core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
