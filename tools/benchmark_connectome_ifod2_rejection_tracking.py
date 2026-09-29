"""真实 FOD/5TT 固定种子的 iFOD2 拒绝采样传播计时。"""

import argparse
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tracking import (_act_seed_direction, _grow, _initial_directions,
                                      _sample, _ifod2_calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "seeds", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    fod_image = nib.load(str(args.fod))
    five_image = nib.load(str(args.five_tissue))
    fod = torch.as_tensor(np.asarray(fod_image.dataobj, dtype=np.float32).copy(), device=device)
    five = torch.as_tensor(np.asarray(five_image.dataobj, dtype=np.float32).copy(), device=device)
    seeds = torch.as_tensor(np.loadtxt(args.seeds, dtype=np.float32)[:args.count], device=device)
    fod_inverse = torch.linalg.inv(torch.as_tensor(fod_image.affine, dtype=torch.float64, device=device))
    five_inverse = torch.linalg.inv(torch.as_tensor(five_image.affine, dtype=torch.float64, device=device))
    five_spacing = torch.as_tensor(five_image.header.get_zooms()[:3], dtype=torch.float64, device=device)
    five_affine = torch.linalg.inv(five_inverse)
    five_affine[:3, :3] *= five_spacing / torch.linalg.vector_norm(five_affine[:3, :3], dim=0)
    five_inverse = torch.linalg.inv(five_affine)
    voxel = float(np.prod(fod_image.header.get_zooms()[:3]) ** (1 / 3))
    calibration, ratio = _ifod2_calibration(8, voxel / 2, voxel, 45., .5, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial, valid, attempts = _initial_directions(
        _sample(fod, seeds, fod_inverse), generator, lmax=8, cutoff=.1)
    act_valid, one_way, initial = _act_seed_direction(five, seeds, initial, five_inverse)
    valid &= act_valid
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    start = time.perf_counter()
    options = dict(lmax=8, proposals_per_step=16, step_mm=voxel / 2,
                   max_steps=math.ceil(250 / (voxel / 2)), max_angle_degrees=45.,
                   cutoff=.1, power=.5, calibration_local=calibration,
                   calibration_ratio=ratio)
    forward, nf, gf, lf, wf = _grow(
        seeds, initial, fod, five, fod_inverse, five_inverse,
        generator, **options)
    backward, nb, gb, lb, wb = _grow(
        seeds, -initial, fod, five, fod_inverse, five_inverse,
        generator, seed_to_wm_initial=wf, **options)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total = torch.where(one_way, lf, lf + lb)
    accepted = valid & gf & (one_way | gb) & (wf | wb) & \
        (total >= 2 * voxel) & (total <= 250.)
    result = {
        "dataset": "OpenNeuro ds004666 real WM FOD, 5TT, fixed GMWMI seeds",
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "count": len(seeds), "seed": args.seed,
        "initial_valid": int(valid.sum()), "one_way": int(one_way.sum()),
        "initial_attempts_mean": float(attempts.float().mean()),
        "accepted_streamlines": int(accepted.sum()),
        "forward_points_mean": float(nf.float().mean()),
        "backward_points_mean": float(nb.float().mean()),
        "accepted_length_mm_mean": float(total[accepted].mean()) if bool(accepted.any()) else None,
        "input_load_and_initial_seconds": load_seconds,
        "tracking_seconds": elapsed,
        "torch_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
