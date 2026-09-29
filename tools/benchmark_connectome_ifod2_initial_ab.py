"""固定真实 FOD、5TT、种子和后续 RNG，仅比较 iFOD2 初始方向选择。"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.assignment import build_connectomes
from fnit.connectome.fod import tracking_sh_precomputed
from fnit.connectome.sift2 import estimate_sift2_weights
from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise
from fnit.connectome.tracking import _grow, _initial_directions, _sample


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def _sphere(count, device):
    """Historical 128-direction grid used only for the A/B baseline."""
    index = torch.arange(count, device=device, dtype=torch.float32)
    z = 1 - 2 * (index + .5) / count
    angle = index * (math.pi * (3 - math.sqrt(5)))
    radial = (1 - z.square()).sqrt()
    return torch.stack((radial * angle.cos(), radial * angle.sin(), z), dim=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load(path, device, dtype=torch.float32):
    image = nib.load(str(path))
    data = torch.as_tensor(np.asarray(image.dataobj).copy(), dtype=dtype, device=device)
    affine = torch.as_tensor(image.affine, dtype=torch.float64, device=device)
    return data, affine


def _metrics(candidate, reference, support):
    edge = np.triu_indices(candidate.shape[0], 1)
    a, b = candidate[edge], reference[edge]
    common = support[0][edge] & support[1][edge]
    return {
        "pearson_upper": float(np.corrcoef(a, b)[0, 1]),
        "relative_l1_upper": float(np.abs(a - b).sum() / np.abs(b).sum()),
        "normalized_mae_common_support": (
            float(np.abs(a[common] - b[common]).mean() / np.abs(b[common]).mean())
            if common.any() and np.abs(b[common]).sum() else None
        ),
        "support_dice": float(2 * common.sum() / (support[0][edge].sum() + support[1][edge].sum())),
    }


def _track(seeds, initial, valid, fod, fod_affine, five, five_affine, *, rng_seed):
    device = seeds.device
    generator = torch.Generator(device=device).manual_seed(rng_seed)
    fod_inverse = torch.linalg.inv(fod_affine)
    five_inverse = torch.linalg.inv(five_affine)
    step = 1.00639975
    options = dict(lmax=8, proposals_per_step=16, step_mm=step,
                   max_steps=math.ceil(250 / step), max_angle_degrees=45., cutoff=.1, power=.5)
    forward, nf, gf, lf, wf = _grow(seeds, initial, fod, five, fod_inverse, five_inverse,
                                    generator, **options)
    backward, nb, gb, lb, wb = _grow(seeds, -initial, fod, five, fod_inverse, five_inverse,
                                     generator, **options)
    total = lf + lb
    keep = torch.nonzero(valid & gf & gb & (wf | wb) &
                         (total >= 4.025599) & (total <= 250.), as_tuple=False).flatten()
    paths = tuple(torch.cat((backward[i, :nb[i]].flip(0), forward[i, 1:nf[i]])) for i in keep.tolist())
    endpoints = torch.stack([torch.stack((path[0], path[-1])) for path in paths])
    return paths, endpoints, total[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "seeds", "fa", "atlas", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, action="append", required=True,
                        help="含 count/sift2_fbc/mean_length/mean_fa.csv 的官方参考目录；可重复")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="初始方向 RNG 种子；后续追踪用 1729+seed")
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
        raise ValueError("expected 10000 frozen seeds, lmax=8 FOD and five tissue channels")
    coefficients = _sample(fod, seeds, torch.linalg.inv(fod_affine))
    sphere = _sphere(128, device)
    scores = (coefficients @ tracking_sh_precomputed(sphere, 8).T).clamp_min(0)
    old_valid = (scores >= .1).any(-1)
    old_rng = torch.Generator(device=device).manual_seed(args.seed)
    old_initial = sphere[torch.multinomial(
        torch.where(old_valid[:, None], (scores >= .1).float(), torch.ones_like(scores)),
        1, generator=old_rng).squeeze(-1)]
    _sync(device)
    start = time.perf_counter()
    new_initial, new_valid, attempts = _initial_directions(
        coefficients, torch.Generator(device=device).manual_seed(args.seed), lmax=8, cutoff=.1,
    )
    _sync(device)
    initial_seconds = time.perf_counter() - start
    processing = processing_mask_from_5tt(fod, fod_affine, five, five_affine)
    reference = {
        directory.name: {name: np.loadtxt(directory / f"{name}.csv", delimiter=",") for name in NAMES}
        for directory in args.official_dir
    }
    report = {
        "dataset": "OpenNeuro ds004666 fixed real FOD/5TT/FA/20-node atlas and 10000 GMWMI seeds",
        "scope": "only initial directions and their valid masks vary; downstream propagation RNG is equal in both arms",
        "input_sha256": {name: _sha256(path) for name, path in
                         (("fod", args.fod), ("five_tissue", args.five_tissue),
                          ("seeds", args.seeds), ("fa", args.fa), ("atlas", args.atlas))},
        "official_sha256": {folder.name: {name: _sha256(folder / f"{name}.csv") for name in NAMES}
                            for folder in args.official_dir},
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "initial_rng_seed": args.seed, "downstream_rng_seed": 1729 + args.seed,
        "initial_direction_seconds": initial_seconds,
        "initial_valid": {"discrete128": int(old_valid.sum()), "continuous1000": int(new_valid.sum())},
        "continuous_attempts_mean": float(attempts.float().mean()),
        "arms": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, initial, valid in (("discrete128", old_initial, old_valid),
                                  ("continuous1000", new_initial, new_valid)):
        _sync(device)
        start = time.perf_counter()
        paths, endpoints, lengths = _track(
            seeds, initial, valid, fod, fod_affine, five, five_affine,
            rng_seed=1729 + args.seed,
        )
        _sync(device)
        tracking_seconds = time.perf_counter() - start
        start = time.perf_counter()
        weights = estimate_sift2_weights(paths, fod, fod_affine, five, five_affine,
                                         step_size_mm=1.00639975, processing_mask=processing)
        fa_values = sample_streamline_mean_precise(paths, fa, fa_affine)
        matrices = build_connectomes(endpoints, atlas, atlas_affine, weights=weights,
                                     lengths=lengths, fa=fa_values)
        _sync(device)
        post_seconds = time.perf_counter() - start
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
