"""同一真实 TCK/逐轨数值核对七套 UKB atlas 的 PyTorch 四矩阵赋值。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import NAMES, _sha256, _sync
from fnit.connectome.assignment import build_connectomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--lengths", required=True, type=Path)
    parser.add_argument("--fa", required=True, type=Path)
    parser.add_argument("--atlas-manifest", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    paths = nib.streamlines.load(str(args.tracks)).tractogram.streamlines
    endpoints = torch.as_tensor(np.stack([(path[0], path[-1]) for path in paths]),
                                device=device, dtype=torch.float32)
    weights, lengths, fa = (
        torch.as_tensor(np.loadtxt(path), device=device, dtype=torch.float32)
        for path in (args.weights, args.lengths, args.fa)
    )
    if any(item.shape != (len(paths),) for item in (weights, lengths, fa)):
        raise ValueError("weights, lengths and FA must have one value per TCK streamline")
    _sync(device)
    load_seconds = time.perf_counter() - start
    report = {
        "dataset": "OpenNeuro ds004666 sub-01/ses-2mm",
        "fixed_input": "same official 100k TCK, SIFT2 weights, lengths, FA and each DWI atlas",
        "input_sha256": {name: _sha256(path) for name, path in
                         (("tracks", args.tracks), ("weights", args.weights),
                          ("lengths", args.lengths), ("fa", args.fa))},
        "streamlines": len(paths), "device": str(device),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "load_seconds": load_seconds, "profiles": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for line in args.atlas_manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        profile, atlas_path, node_count_text = line.split("\t")
        atlas_path = Path(atlas_path)
        node_count = int(node_count_text)
        image = nib.load(str(atlas_path))
        atlas = torch.as_tensor(np.asarray(image.dataobj).astype(np.int32), device=device)
        affine = torch.as_tensor(image.affine, device=device, dtype=torch.float32)
        _sync(device)
        start = time.perf_counter()
        matrices = build_connectomes(
            endpoints=endpoints, atlas=atlas, affine=affine, weights=weights,
            lengths=lengths, fa=fa, node_count=node_count,
        )
        _sync(device)
        matrix_seconds = time.perf_counter() - start
        candidate = {name: matrices[name].cpu().numpy() for name in NAMES}
        reference = {name: np.loadtxt(args.reference_root / profile / f"{name}.csv",
                                      delimiter=",") for name in NAMES}
        metrics = {}
        for name in NAMES:
            actual, expected = candidate[name], reference[name]
            if actual.shape != expected.shape or actual.shape != (node_count, node_count):
                raise ValueError(f"{profile}/{name}: matrix geometry mismatch")
            difference = np.abs(actual - expected)
            metrics[name] = {
                "different_cells": int(np.count_nonzero(difference)),
                "mae": float(difference.mean()),
                "max_abs_error": float(difference.max()),
                "all_finite": bool(np.isfinite(actual).all()),
            }
        np.savez_compressed(args.output_dir / f"{profile}.npz",
                            **{f"fnit_{name}": candidate[name] for name in NAMES},
                            **{f"official_{name}": reference[name] for name in NAMES})
        report["profiles"][profile] = {
            "atlas_sha256": _sha256(atlas_path), "nodes": node_count,
            "matrix_seconds": matrix_seconds, "metrics": metrics,
        }
        print(f"{profile}: count mismatch {metrics['count']['different_cells']}; "
              f"{matrix_seconds:.2f} s", flush=True)
    report["peak_torch_allocated_gib"] = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
