#!/usr/bin/env python3
"""Measure single-subject SynthSeg+ calls on a real T1w image."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit import SynthSegPlus


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--save-combined", type=Path)
    parser.add_argument("--save-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--volumes", action="store_true")
    args = parser.parse_args()
    if args.save_csv is not None and not args.volumes:
        parser.error("--save-csv requires --volumes")

    torch.set_num_threads(args.threads)
    model = SynthSegPlus(weights=args.weights, parc_weights=args.weights,
                         device=args.device)
    official = nib.load(str(args.official)) if args.official is not None else None

    runs = []
    first = None
    for number in (1, 2):
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = model(args.input, keep_geometry=False, volumes=args.volumes)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        combined = result.combined
        labels = np.asanyarray(combined.dataobj)
        row = {"call": number, "seconds": elapsed,
               "shape": list(labels.shape), "dtype": str(labels.dtype)}
        if args.device.startswith("cuda"):
            row.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        if official is not None:
            if official.shape != combined.shape:
                raise ValueError("Official output has a different grid")
            row["official_max_affine_difference"] = float(
                np.max(np.abs(official.affine - combined.affine)))
            row["official_different_voxels"] = int(np.count_nonzero(
                np.asanyarray(official.dataobj) != labels))
        if first is not None:
            row["first_call_different_voxels"] = int(np.count_nonzero(first != labels))
        else:
            first = labels.copy()
        runs.append(row)
        if number == 1 and args.save_combined is not None:
            args.save_combined.parent.mkdir(parents=True, exist_ok=True)
            combined.save(args.save_combined)
        if number == 1 and args.save_csv is not None:
            result.write_volumes_csv(args.input, args.save_csv)

    report = {"input": str(args.input), "input_sha256": sha256(args.input),
              "weights": str(args.weights), "device": args.device,
              "torch_version": torch.__version__, "threads": args.threads,
              "volumes": args.volumes,
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
