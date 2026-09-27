"""Compare GPU SIFT2 optimisation with MRtrix on the same sparse fixel map.

The NPZ must contain track_index, fixel_index, length_mm, target and
processing_mask from a fixed FMLS segmentation and fixed TCK. The original
command is ``tcksift2 tracks.tck wm_fod.mif weights.txt -act 5tt.mif``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from fnit.connectome.sift2_optimizer import optimize_sift2_fixels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--reference-weights", type=Path, required=True)
    parser.add_argument("--reference-mu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    paths = (args.records, args.reference_weights, args.reference_mu)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = np.loadtxt(args.reference_weights, comments="#", dtype=np.float64).reshape(-1)
    device = torch.device(args.device)
    with np.load(args.records) as records:
        required = ("track_index", "fixel_index", "length_mm", "target", "processing_mask")
        missing = set(required).difference(records.files)
        if missing:
            raise ValueError(f"missing sparse record arrays: {sorted(missing)}")
        data = {name: torch.as_tensor(records[name].copy(), device=device) for name in required}
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = optimize_sift2_fixels(*(data[name] for name in required), len(reference))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    candidate = result.weights.detach().cpu().numpy()
    if candidate.shape != reference.shape:
        raise ValueError("reference and candidate weight vectors differ in length")
    difference = candidate - reference
    report = {
        "comparison": "fixed tracks and FMLS/ACT sparse input; SIFT2 optimizer only",
        "input_sha256": {str(path): _sha256(path) for path in paths},
        "device": str(device),
        "streamlines": int(len(reference)),
        "sparse_records": int(len(data["track_index"])),
        "fixel_count_including_dummy": int(len(data["target"])),
        "mu_reference": float(args.reference_mu.read_text().strip()),
        "mu_candidate": result.mu,
        "iterations": result.iterations,
        "weights": {
            "pearson": float(np.corrcoef(reference, candidate)[0, 1]),
            "spearman": float(spearmanr(reference, candidate).statistic),
            "mae": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "max_abs": float(np.abs(difference).max()),
            "reference_sum": float(reference.sum()),
            "candidate_sum": float(candidate.sum()),
        },
        "runtime_seconds": elapsed,
        "peak_torch_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
