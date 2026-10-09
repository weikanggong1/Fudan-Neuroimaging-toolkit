"""Measure real-data FNIT ACT/iFOD2 tracking at a specified seed count."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import probabilistic_tractography


def _load(path: Path, device: str):
    image = nib.load(str(path))
    return (torch.as_tensor(np.asarray(image.dataobj).copy(), device=device, dtype=torch.float32),
            torch.as_tensor(image.affine, device=device, dtype=torch.float64), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "gmwmi", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    fod, fod_affine, fod_image = _load(args.fod, args.device)
    five, five_affine, five_image = _load(args.five_tissue, args.device)
    gmwmi, gmwmi_affine, _ = _load(args.gmwmi, args.device)
    if five.shape[:3] != gmwmi.shape or not torch.allclose(five_affine, gmwmi_affine):
        raise ValueError("5TT and GMWMI must share one real T1 grid")
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    tracks = probabilistic_tractography(
        wm_sh=fod,
        fod_affine=fod_affine,
        five_tissue=five,
        five_tissue_affine=five_affine,
        gmwmi=gmwmi,
        n_seeds=args.n_seeds,
        lmax=8,
        five_tissue_spacing_mm=five_image.header.get_zooms()[:3],
        seed=args.seed,
        batch_size=args.batch_size,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    # Freeze the synchronized tracking interval before hashing files or reading outputs.
    tracking_seconds = time.perf_counter() - start
    report = {
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                         (("fod", args.fod), ("five_tissue", args.five_tissue), ("gmwmi", args.gmwmi))},
        "fod_shape": list(fod_image.shape),
        "five_tissue_shape": list(five_image.shape),
        "device": args.device,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "n_seeds": args.n_seeds,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "accepted_streamlines": len(tracks.paths),
        "total_path_points": sum(len(path) for path in tracks.paths),
        "tracking_seconds_with_inputs_loaded": tracking_seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(args.device) / 2**30
                                     if args.device.startswith("cuda") else None),
        "peak_torch_reserved_gib": (torch.cuda.max_memory_reserved(args.device) / 2**30
                                    if args.device.startswith("cuda") else None),
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
