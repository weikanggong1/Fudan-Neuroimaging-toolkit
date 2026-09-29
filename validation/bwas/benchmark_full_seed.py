"""Compare whole-gray-matter seed-to-voxel z values with original BWAS on ABIDE I+II."""

import argparse
import csv
import json
import resource
from pathlib import Path
from time import monotonic, perf_counter, sleep

import nibabel as nib
import numpy as np
from scipy import stats
import torch

from benchmark_abide import load_original
from fnit.bwas.core import _fisher_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subject-block-size", type=int, default=8)
    args = parser.parse_args()
    with (args.bids_root / "private_qc/participants_minvalid120000.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1748:
        raise ValueError("expected 1748 QC-passing ABIDE I+II subjects")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < len(rows)+128:
        resource.setrlimit(resource.RLIMIT_NOFILE, (len(rows)+128, hard))
    deadline = monotonic()+12*3600
    if args.cache_dir is None:
        while True:
            caches = list(args.result_root.glob("bwas-cache-*"))
            prepared = (args.run_log.is_file() and
                        "BWAS prepared 1748/1748 BOLD images" in args.run_log.read_text())
            if prepared and len(caches) == 1:
                break
            if monotonic() > deadline:
                raise TimeoutError("full-subject normalized BOLD cache was not completed")
            sleep(60)
        cache = caches[0]
    else:
        cache = args.cache_dir
    matrices = [np.load(cache / f"subject-{i}.npy", mmap_mode="r")
                for i in range(len(rows))]
    mask = np.asarray(nib.load(str(next(args.bids_root.glob(
        "*desc-qc120000GrayMatter_mask.nii.gz")))).dataobj) != 0
    voxels = int(mask.sum())
    seeds = (voxels//4, 3*voxels//4)
    covariates = [name for name in rows[0] if name.startswith("site_")]
    design = np.array([[float(row[name]) for name in
                        ("case", "age", "sex", *covariates)] + [1.0]
                       for row in rows], dtype=np.float32)
    original, source_hash = load_original(args.upstream_source)

    start = perf_counter()
    observed = np.empty((len(rows), len(seeds)*voxels), dtype=np.float32)
    for subject, series in enumerate(matrices):
        corr = original["BWAS_correlation"](series[:, seeds], series)
        corr[corr > 0.9999] = 0
        observed[subject] = original["BWAS_fisher_z"](corr).reshape(-1)
    beta = original["BWAS_regression_online1"](
        design, observed, 0, len(rows),
        np.zeros((design.shape[1], observed.shape[1]), dtype=np.float32))
    sigma = original["BWAS_regression_online2"](
        design, observed, 0, len(rows), beta,
        np.zeros((1, observed.shape[1]), dtype=np.float32))
    t = original["BWAS_regression_online3"](design, beta, sigma).reshape(-1)
    z_reference = stats.norm.ppf(stats.t.cdf(t, len(rows)-design.shape[1]-1))
    original_seconds = perf_counter()-start
    del observed

    start = perf_counter()
    device = torch.device(args.device)
    x = torch.as_tensor(design, dtype=torch.float64, device=device)
    inverse = torch.linalg.pinv(x.T @ x)
    xy = torch.zeros((x.shape[1], len(seeds)*voxels),
                     device=device, dtype=torch.float64)
    yy = torch.zeros(len(seeds)*voxels, device=device, dtype=torch.float64)
    for begin in range(0, len(rows), args.subject_block_size):
        end = min(begin+args.subject_block_size, len(rows))
        values = torch.cat([
            _fisher_block(matrices, begin, end, seed, 0, 1, voxels, device)
            for seed in seeds], dim=1)
        xy += x[begin:end].T @ values
        yy += (values*values).sum(0)
    beta = inverse @ xy
    sigma = yy-(beta*xy).sum(0)
    df = len(rows)-x.shape[1]
    t = (beta[0]/torch.sqrt(sigma*(inverse[0, 0]/df))).cpu().numpy()
    z_fnit = stats.norm.ppf(stats.t.cdf(t, int(df)-1))
    fnit_seconds = perf_counter()-start
    valid = np.isfinite(z_reference) & np.isfinite(z_fnit)
    for row, seed in enumerate(seeds):
        valid[row*voxels+seed] = False
    difference = np.abs(z_reference[valid]-z_fnit[valid])
    summary = {
        "dataset": "ABIDE I+II real preprocessed BOLD, 2 mm gray matter",
        "matched_subjects": 1778, "subjects": len(rows), "gray_voxels": voxels,
        "seeds": len(seeds), "voxel_pairs_compared": int(valid.sum()),
        "cdt": 5.0, "original_source_sha256": source_hash,
        "original_cpu_seconds": original_seconds,
        "fnit_elapsed_seconds": fnit_seconds,
        "z_mean_absolute_difference": float(difference.mean()),
        "z_max_absolute_difference": float(difference.max()),
        "cdt_disagreements": int(np.count_nonzero(
            (np.abs(z_reference[valid]) > 5) != (np.abs(z_fnit[valid]) > 5))),
        "original_suprathreshold_edges": int(np.count_nonzero(
            np.abs(z_reference[valid]) > 5)),
        "fnit_suprathreshold_edges": int(np.count_nonzero(
            np.abs(z_fnit[valid]) > 5)),
        "cdt3_disagreements": int(np.count_nonzero(
            (np.abs(z_reference[valid]) > 3) != (np.abs(z_fnit[valid]) > 3))),
        "original_cdt3_edges": int(np.count_nonzero(np.abs(z_reference[valid]) > 3)),
        "fnit_cdt3_edges": int(np.count_nonzero(np.abs(z_fnit[valid]) > 3)),
        "fnit_device": str(device),
        "fnit_subject_block_size": args.subject_block_size,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
