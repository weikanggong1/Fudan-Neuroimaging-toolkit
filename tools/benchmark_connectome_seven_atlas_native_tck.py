"""FNIT 真实 100k TCK 一次计算 SIFT2/FA，复用到七套 atlas 四矩阵。"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import NAMES, _load, _sha256, _sync
from fnit.connectome.assignment import build_connectomes
from fnit.connectome.sift2 import estimate_sift2_weights
from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tracks", "fod", "five-tissue", "fa", "atlas-manifest", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    fod, fod_affine = _load(args.fod, device)
    five, five_affine = _load(args.five_tissue, device)
    fa, fa_affine = _load(args.fa, device)
    loaded = nib.streamlines.load(str(args.tracks)).tractogram.streamlines
    endpoints = torch.as_tensor(np.stack([(path[0], path[-1]) for path in loaded]),
                                device=device, dtype=torch.float32)
    lengths = torch.as_tensor(np.array([
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        for path in loaded
    ], dtype=np.float32), device=device)
    paths = tuple(torch.as_tensor(np.asarray(path), device=device) for path in loaded)
    _sync(device)
    load_seconds = time.perf_counter() - start
    voxel_mm = float(torch.linalg.vector_norm(fod_affine[:3, :3], dim=0).prod().pow(1 / 3))
    start = time.perf_counter()
    processing = processing_mask_from_5tt(fod, fod_affine, five, five_affine)
    weights = estimate_sift2_weights(
        paths=paths, wm_sh=fod, fod_affine=fod_affine, five_tissue=five,
        five_tissue_affine=five_affine, step_size_mm=voxel_mm / 2,
        processing_mask=processing,
    )
    _sync(device)
    sift2_seconds = time.perf_counter() - start
    start = time.perf_counter()
    fa_values = sample_streamline_mean_precise(paths, fa, fa_affine)
    _sync(device)
    fa_seconds = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "track_metrics.npz",
                        weights=weights.cpu().numpy(), lengths=lengths.cpu().numpy(),
                        mean_fa=fa_values.cpu().numpy())
    report = {
        "dataset": "OpenNeuro ds004666 sub-01/ses-2mm",
        "input_sha256": {name: _sha256(path) for name, path in
                         (("tracks", args.tracks), ("fod", args.fod),
                          ("five_tissue", args.five_tissue), ("fa", args.fa))},
        "streamlines": len(paths), "device": str(device),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "load_seconds": load_seconds, "sift2_seconds": sift2_seconds,
        "fa_seconds": fa_seconds, "profiles": {},
    }
    for line in args.atlas_manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        profile, atlas_path, node_count_text = line.split("\t")
        atlas_path = Path(atlas_path)
        image = nib.load(str(atlas_path))
        atlas = torch.as_tensor(np.asarray(image.dataobj).astype(np.int32), device=device)
        affine = torch.as_tensor(image.affine, device=device, dtype=torch.float32)
        _sync(device)
        start = time.perf_counter()
        matrices = build_connectomes(
            endpoints=endpoints, atlas=atlas, affine=affine,
            weights=weights, lengths=lengths, fa=fa_values,
            node_count=int(node_count_text),
        )
        _sync(device)
        seconds = time.perf_counter() - start
        matrix_arrays = {name: matrices[name].cpu().numpy() for name in NAMES}
        np.savez_compressed(args.output_dir / f"{profile}.npz", **matrix_arrays)
        report["profiles"][profile] = {
            "atlas_sha256": _sha256(atlas_path),
            "nodes": int(node_count_text), "matrix_seconds": seconds,
            "assigned_streamlines": int(np.triu(matrix_arrays["count"]).sum()),
        }
        print(f"{profile}: {seconds:.2f} s", flush=True)
    report["peak_torch_allocated_gib"] = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
