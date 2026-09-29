"""真实 10k 固定种子经拒绝采样、SIFT2 和四张矩阵的重复对照。"""

import argparse
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import NAMES, _load, _metrics, _sha256, _sync
from fnit.connectome.assignment import build_connectomes
from fnit.connectome.sift2 import estimate_sift2_weights
from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise
from fnit.connectome.tracking import (_act_seed_direction, _grow, _ifod2_calibration,
                                      _initial_directions, _sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "seeds", "fa", "atlas", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, action="append", required=True)
    parser.add_argument("--five-spacing", type=float, nargs=3, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-tracks", action="store_true", help="另存世界毫米 TCK 供轨迹分布诊断")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    fod, fod_affine = _load(args.fod, device)
    five, five_affine = _load(args.five_tissue, device)
    fa, fa_affine = _load(args.fa, device)
    atlas, atlas_affine = _load(args.atlas, device, torch.int32)
    seeds = torch.as_tensor(np.loadtxt(args.seeds, dtype=np.float32), device=device)
    if seeds.shape != (10000, 3) or fod.shape[-1] != 45 or five.shape[-1] != 5:
        raise ValueError("expected 10000 real GMWMI positions, lmax=8 FOD and 5TT")
    fod_inverse = torch.linalg.inv(fod_affine)
    effective = five_affine.clone()
    effective[:3, :3] *= torch.as_tensor(args.five_spacing, dtype=torch.float64, device=device) / \
        torch.linalg.vector_norm(effective[:3, :3], dim=0)
    five_inverse = torch.linalg.inv(effective)
    voxel = float(torch.linalg.vector_norm(fod_affine[:3, :3], dim=0).prod().pow(1 / 3))
    calibration, ratio = _ifod2_calibration(8, voxel / 2, voxel, 45., .5, device)
    initial, initial_valid, attempts = _initial_directions(
        _sample(fod, seeds, fod_inverse),
        torch.Generator(device=device).manual_seed(args.seed), lmax=8, cutoff=.1,
    )
    act_valid, one_way, initial = _act_seed_direction(five, seeds, initial, five_inverse)
    valid = initial_valid & act_valid
    _sync(device)
    load_seconds = time.perf_counter() - load_start
    options = dict(lmax=8, proposals_per_step=16, step_mm=voxel / 2,
                   max_steps=math.ceil(250 / (voxel / 2)), max_angle_degrees=45.,
                   cutoff=.1, power=.5, calibration_local=calibration,
                   calibration_ratio=ratio)
    generator = torch.Generator(device=device).manual_seed(1729 + args.seed)
    started = time.perf_counter()
    forward, nf, gf, lf, wf = _grow(seeds, initial, fod, five, fod_inverse,
                                    five_inverse, generator, **options)
    backward, nb, gb, lb, wb = _grow(seeds, -initial, fod, five, fod_inverse,
                                     five_inverse, generator, seed_to_wm_initial=wf,
                                     **options)
    total = torch.where(one_way, lf, lf + lb)
    keep = torch.nonzero(valid & gf & (one_way | gb) & (wf | wb) &
                         (total >= 2 * voxel) & (total <= 250.),
                         as_tuple=False).flatten()
    paths = tuple(forward[i, :nf[i]] if bool(one_way[i]) else
                  torch.cat((backward[i, :nb[i]].flip(0), forward[i, 1:nf[i]]))
                  for i in keep.tolist())
    endpoints = torch.stack([torch.stack((path[0], path[-1])) for path in paths])
    _sync(device)
    tracking_seconds = time.perf_counter() - started
    if args.save_tracks:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        nib.streamlines.save(nib.streamlines.Tractogram(
            [path.cpu().numpy() for path in paths], affine_to_rasmm=np.eye(4)),
            str(args.output_dir / "tracks.tck"))
    started = time.perf_counter()
    processing = processing_mask_from_5tt(fod, fod_affine, five, five_affine)
    weights = estimate_sift2_weights(paths, fod, fod_affine, five, five_affine,
                                     step_size_mm=voxel / 2, processing_mask=processing)
    fa_values = sample_streamline_mean_precise(paths, fa, fa_affine)
    matrices = build_connectomes(endpoints, atlas, atlas_affine, weights=weights,
                                 lengths=total[keep], fa=fa_values)
    _sync(device)
    post_seconds = time.perf_counter() - started
    candidate = {name: matrices[name].cpu().numpy() for name in NAMES}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in candidate.items():
        np.savetxt(args.output_dir / f"{name}.csv", matrix, delimiter=",")
    official = {directory.name: {name: np.loadtxt(directory / f"{name}.csv", delimiter=",")
                                 for name in NAMES} for directory in args.official_dir}
    report = {
        "dataset": "OpenNeuro ds004666 real fixed FOD, 5TT, FA, atlas and 10000 GMWMI positions",
        "input_sha256": {name: _sha256(path) for name, path in
                         (("fod", args.fod), ("five_tissue", args.five_tissue),
                          ("seeds", args.seeds), ("fa", args.fa), ("atlas", args.atlas))},
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "initial_seed": args.seed, "tracking_seed": 1729 + args.seed,
        "initial_valid": int(initial_valid.sum()), "act_valid": int(valid.sum()),
        "act_one_way": int(one_way.sum()), "initial_attempts_mean": float(attempts.float().mean()),
        "accepted_streamlines": len(paths), "load_and_initial_seconds": load_seconds,
        "tracking_seconds": tracking_seconds, "postprocessing_seconds": post_seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
        "metrics": {label: {name: _metrics(candidate[name], reference[name],
                                           (candidate["count"] > 0, reference["count"] > 0))
                            for name in NAMES} for label, reference in official.items()},
    }
    args.output_dir.joinpath("report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
