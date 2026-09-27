"""Compare PyTorch FMLS fixels with MRtrix SIFT2 debug images on real FOD data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.sift2_fixels import segment_fod_fixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fod", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--official-index", type=Path, required=True)
    parser.add_argument("--official-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lut-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    fod = nib.load(args.fod)
    wm_sh = torch.from_numpy(np.asarray(fod.dataobj, dtype=np.float32)).to(args.device)
    mask = torch.from_numpy(np.asarray(nib.load(args.mask).dataobj, dtype=np.float32)).to(args.device)
    official_count = np.asarray(nib.load(args.official_index).dataobj)[..., 0]
    official_target = np.asarray(nib.load(args.official_target).dataobj, dtype=np.float32)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    result = segment_fod_fixels(wm_sh, mask, batch_size=args.batch_size)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - start
    count = result.count_image.cpu().numpy()
    target = result.target_image.cpu().numpy()
    finite = np.isfinite(target) & np.isfinite(official_target)
    error = np.abs(target[finite] - official_target[finite])
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    np.savez_compressed(
        args.lut_output,
        voxel_ids=result.voxel_ids.cpu().numpy(),
        first_fixel_index=result.first_fixel_index.cpu().numpy(),
        count=result.count.cpu().numpy(),
        lookup_table=result.lookup_table.cpu().numpy(),
        fixel_integrals=result.fixel_integrals.cpu().numpy(),
    )
    report = {
        "method": "MRtrix3 3.0.3-103-g026e850d FMLS, batched PyTorch float64 on actual WM FOD",
        "input_sha256": {"fod": sha(args.fod), "processing_mask": sha(args.mask),
                         "official_index": sha(args.official_index), "official_target": sha(args.official_target)},
        "shape": list(count.shape), "masked_voxels": int(torch.count_nonzero(mask).item()),
        "count_exact_voxels": int(np.count_nonzero(count == official_count)),
        "count_total_voxels": int(count.size),
        "candidate_fixels": int(count.sum()), "official_fixels": int(official_count.sum()),
        "target_finite_voxels": int(finite.sum()),
        "target_finite_disagreements": int(np.count_nonzero(np.isfinite(target) != np.isfinite(official_target))),
        "target_mae": float(error.mean()), "target_max_abs": float(error.max()),
        "target_pearson": float(np.corrcoef(target[finite], official_target[finite])[0, 1]),
        "lut_shape": list(result.lookup_table.shape), "lut_dtype": str(result.lookup_table.dtype),
        "lut_sha256": sha(args.lut_output), "seconds": seconds,
        "device": args.device,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)) if args.device.startswith("cuda") else 0,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
