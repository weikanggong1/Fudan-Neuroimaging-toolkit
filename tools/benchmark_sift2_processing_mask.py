"""Compare PyTorch ACT-derived SIFT2 processing mask on real 5TT/FOD images."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fod", type=Path, required=True)
    parser.add_argument("--five-tt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--max-voxels", type=int)
    parser.add_argument("--batch-voxels", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    fod = nib.load(args.fod)
    five = nib.load(args.five_tt)
    reference = nib.load(args.reference)
    sh = np.asarray(fod.dataobj, dtype=np.float32)
    tt = np.asarray(five.dataobj, dtype=np.float32)
    official = np.asarray(reference.dataobj, dtype=np.float32)
    if args.max_voxels:
        active = np.flatnonzero(sh[..., 0].reshape(-1) != 0)
        rng = np.random.default_rng(20260927)
        sampled = rng.choice(active, min(args.max_voxels, len(active)), replace=False)
        keep = np.zeros(sh.shape[:3], dtype=bool)
        keep.reshape(-1)[sampled] = True
        sh[..., 0] *= keep
    else:
        keep = np.ones(sh.shape[:3], dtype=bool)
    device = torch.device(args.device)
    fod_tensor = torch.from_numpy(sh).to(device)
    tt_tensor = torch.from_numpy(tt).to(device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.perf_counter()
    candidate = processing_mask_from_5tt(
        fod_tensor, torch.from_numpy(fod.affine), tt_tensor,
        torch.from_numpy(five.affine), batch_voxels=args.batch_voxels,
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    actual = candidate.cpu().numpy()
    valid = keep & np.isfinite(official)
    error = np.abs(actual[valid] - official[valid])
    active = valid & (official > 0)
    candidate_support = valid & (actual > 0)
    intersection = np.count_nonzero(active & candidate_support)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "source": "MRtrix3 3.0.3-103-g026e850d SIFT/proc_mask.cpp ACT2pve, public ds004666 corrected FOD + FS5TT",
        "input_sha256": {"fod": sha(args.fod), "five_tt": sha(args.five_tt), "reference_proc_mask": sha(args.reference)},
        "shape": list(sh.shape[:3]), "tested_voxels": int(valid.sum()),
        "official_nonzero_voxels": int(active.sum()),
        "candidate_nonzero_voxels": int(candidate_support.sum()),
        "support_dice": float(2 * intersection / max(1, int(active.sum() + candidate_support.sum()))),
        "exact_voxels": int(np.count_nonzero(error == 0)),
        "differing_voxels": int(np.count_nonzero(error != 0)),
        "mae": float(error.mean()), "mae_official_nonzero": float(error[active[valid]].mean()),
        "max_abs": float(error.max()),
        "pearson_nonzero": float(np.corrcoef(actual[active], official[active])[0, 1]) if active.sum() > 1 else None,
        "seconds": elapsed, "device": str(device),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.mask_output:
        nib.save(nib.Nifti1Image(actual, fod.affine), args.mask_output)
        report["candidate_mask_sha256"] = sha(args.mask_output)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
