"""冻结真实种子和后续 RNG，单独检验 ACT 皮层单向播种对矩阵的影响。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_connectome_ifod2_initial_ab import NAMES, _load, _metrics, _sha256, _sync, _track
from fnit.connectome.assignment import build_connectomes
from fnit.connectome.sift2 import estimate_sift2_weights
from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise
from fnit.connectome.tracking import _act_seed_direction, _initial_directions, _sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "seeds", "fa", "atlas", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, action="append", required=True,
                        help="每个目录含官方四张 20×20 矩阵 CSV；重复三次")
    parser.add_argument("--five-spacing", type=float, nargs=3, required=True,
                        help="5TT NIfTI header 的三个体素尺寸，单位毫米")
    parser.add_argument("--seed", type=int, default=0, help="初始方向 RNG 种子；传播种子为 1729+seed")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    fod, fod_affine = _load(args.fod, device)
    five, five_affine = _load(args.five_tissue, device)
    fa, fa_affine = _load(args.fa, device)
    atlas, atlas_affine = _load(args.atlas, device, torch.int32)
    seeds = torch.as_tensor(np.loadtxt(args.seeds, dtype=np.float32), device=device)
    if seeds.shape != (10000, 3) or fod.shape[-1] != 45 or five.shape[-1] != 5:
        raise ValueError("expected 10000 seeds, lmax=8 FOD and five 5TT channels")
    coefficients = _sample(fod, seeds, torch.linalg.inv(fod_affine))
    initial, initial_valid, attempts = _initial_directions(
        coefficients, torch.Generator(device=device).manual_seed(args.seed), lmax=8, cutoff=.1,
    )
    effective = five_affine.clone()
    effective[:3, :3] *= torch.as_tensor(args.five_spacing, dtype=torch.float64, device=device) / \
        torch.linalg.vector_norm(effective[:3, :3], dim=0)
    act_valid, one_way, oriented = _act_seed_direction(five, seeds, initial, torch.linalg.inv(effective))
    valid = initial_valid & act_valid
    processing = processing_mask_from_5tt(fod, fod_affine, five, five_affine)
    reference = {
        directory.name: {name: np.loadtxt(directory / f"{name}.csv", delimiter=",") for name in NAMES}
        for directory in args.official_dir
    }
    report = {
        "dataset": "OpenNeuro ds004666 fixed FOD/5TT/FA/20-node atlas and 10000 GMWMI positions",
        "scope": "both arms share continuous initial draws, ACT seed validity, positions and downstream RNG; only cortical one-way orientation/path construction differs",
        "input_sha256": {name: _sha256(path) for name, path in
                         (("fod", args.fod), ("five_tissue", args.five_tissue),
                          ("seeds", args.seeds), ("fa", args.fa), ("atlas", args.atlas))},
        "official_sha256": {folder.name: {name: _sha256(folder / f"{name}.csv") for name in NAMES}
                            for folder in args.official_dir},
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "five_spacing_mm": args.five_spacing, "initial_rng_seed": args.seed,
        "downstream_rng_seed": 1729 + args.seed,
        "initial_valid": int(initial_valid.sum()), "act_seed_valid": int(valid.sum()),
        "act_one_way": int(one_way.sum()), "initial_attempts_mean": float(attempts.float().mean()),
        "arms": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, tangent, one_way_mask in (("bidirectional", initial, None),
                                         ("act_one_way", oriented, one_way)):
        _sync(device)
        started = time.perf_counter()
        paths, endpoints, lengths = _track(
            seeds, tangent, valid, fod, fod_affine, five, five_affine,
            rng_seed=1729 + args.seed, one_way=one_way_mask,
        )
        _sync(device)
        tracking_seconds = time.perf_counter() - started
        started = time.perf_counter()
        weights = estimate_sift2_weights(paths, fod, fod_affine, five, five_affine,
                                         step_size_mm=1.00639975, processing_mask=processing)
        fa_values = sample_streamline_mean_precise(paths, fa, fa_affine)
        matrices = build_connectomes(endpoints, atlas, atlas_affine, weights=weights,
                                     lengths=lengths, fa=fa_values)
        _sync(device)
        post_seconds = time.perf_counter() - started
        candidate = {name: matrices[name].cpu().numpy() for name in NAMES}
        for name, matrix in candidate.items():
            np.savetxt(args.output_dir / f"{label}_{name}.csv", matrix, delimiter=",")
        report["arms"][label] = {
            "accepted_streamlines": len(paths), "tracking_seconds": tracking_seconds,
            "postprocessing_seconds": post_seconds,
            "metrics": {ref: {name: _metrics(candidate[name], values[name],
                                            (candidate["count"] > 0, values["count"] > 0))
                              for name in NAMES} for ref, values in reference.items()},
        }
    report["peak_torch_allocated_gib"] = (torch.cuda.max_memory_allocated(device) / 2**30
                                          if device.type == "cuda" else None)
    args.output_dir.joinpath("report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
