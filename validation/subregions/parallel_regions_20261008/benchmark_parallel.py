"""Run one reproducible serial/parallel segment_4_subregions comparison.

The caller must provide an atlas cache and the declared SynthSeg weights. This
script never reads an official result and never deletes an existing output.
Use an external ``nvidia-smi`` sampler when driver-level memory is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from fnit import segment_4_subregions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True, type=Path)
    parser.add_argument("--atlas-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parallel-regions", action="store_true")
    parser.add_argument("--max-parallel-regions", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    started = monotonic()
    result = segment_4_subregions(
        args.t1, atlas_root=args.atlas_root, structures="all",
        synthseg_weights=args.weights, synthseg_parc_weights=args.weights,
        device=args.device, threads=args.threads, optimization="fast",
        parallel_regions=args.parallel_regions,
        max_parallel_regions=args.max_parallel_regions,
        output_dir=args.output_dir, save_highres=True, save_posteriors=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    labels = np.asarray(result.labels.dataobj, dtype=np.int32)
    report = {
        "mode": "parallel" if args.parallel_regions else "serial",
        "wall_seconds": monotonic() - started,
        "timings": result.timings,
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "labels_shape": list(labels.shape),
        "nonzero_voxels": int(np.count_nonzero(labels)),
        "initialization": result.initialization,
    }
    (args.output_dir / "parallel_benchmark.json").write_text(
        json.dumps(report, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n",
        encoding="utf-8")
    print(json.dumps(report, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
