"""Measure peak PyTorch CUDA allocation for the real five-ROI ProbtrackX run."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fnit import TorchProbtrackX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--roi-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rois = [Path(line) for line in args.roi_list.read_text().splitlines() if line.strip()]
    if len(rois) != 5:
        raise ValueError("expected five real-data ROIs")
    torch.cuda.set_per_process_memory_fraction(0.2)
    torch.cuda.reset_peak_memory_stats()
    result = TorchProbtrackX(device="cuda:0", nsamples=2000, nsteps=400,
                             batch_size=16384, seed=20260927,
                             pathdist=True, mean_path_length=True).run(
                                 args.samples_dir, args.output_dir, regions=rois)
    report = {
        "purpose": "real DWI five-ROI network --pd --ompl peak-memory check",
        "source_sha256": {name: hashlib.sha256((args.source_dir / name).read_bytes()).hexdigest()
                          for name in ("pipeline.py", "_triton.py", "cli.py", "_fast_counts.py")},
        "device": torch.cuda.get_device_name(0),
        "measurement": "torch.cuda.max_memory_allocated() / 1024**3",
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "accepted_streamlines": result.accepted_streamlines,
        "elapsed_seconds_inside_process": result.elapsed_seconds,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
