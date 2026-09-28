"""固定真实 MRtrix TCK、权重、长度和 FA，核对 84 节点矩阵赋值。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.assignment import build_connectomes


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--lengths", type=Path, required=True)
    parser.add_argument("--fa", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = nib.load(args.atlas)
    atlas = torch.as_tensor(np.asarray(image.dataobj).astype(np.int32), device=args.device)
    paths = nib.streamlines.load(args.tracks).tractogram.streamlines
    endpoints = torch.as_tensor(np.stack([(p[0], p[-1]) for p in paths]),
                                device=args.device, dtype=torch.float32)
    values = [torch.as_tensor(np.loadtxt(path), device=args.device, dtype=torch.float32)
              for path in (args.weights, args.lengths, args.fa)]
    if any(value.shape != (len(paths),) for value in values):
        raise ValueError("weights, lengths and FA must match TCK streamline order")
    if endpoints.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    matrices = build_connectomes(
        endpoints=endpoints, atlas=atlas,
        affine=torch.as_tensor(image.affine, device=args.device),
        weights=values[0], lengths=values[1], fa=values[2], node_count=84,
    )
    if endpoints.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    metrics = {}
    for name in NAMES:
        expected = np.loadtxt(args.reference_dir / f"{name}.csv", delimiter=",")
        actual = matrices[name].cpu().numpy()
        if actual.shape != expected.shape or actual.shape != (84, 84):
            raise ValueError(f"{name} must match reference 84x84 geometry")
        metrics[name] = {
            "different_cells": int(np.count_nonzero(actual != expected)),
            "mean_abs_error": float(np.abs(actual - expected).mean()),
            "max_abs_error": float(np.abs(actual - expected).max()),
        }
    report = {"dataset": "ds004666", "fixed_input": "same TCK, DWI atlas, SIFT2 weights, length and FA", "tracks": len(paths), "node_count": 84, "device": args.device, "matrix_seconds": seconds, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
