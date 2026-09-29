"""真实 FOD/5TT 下高播种量 iFOD2→SIFT2→四矩阵同输入基准。"""

import argparse
import json
import resource
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
from fnit.connectome.tracking import probabilistic_tractography


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "gmwmi", "fa", "atlas", "official-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-arc", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    input_paths = (("fod", args.fod), ("five_tissue", args.five_tissue),
                   ("gmwmi", args.gmwmi), ("fa", args.fa), ("atlas", args.atlas))
    started = time.perf_counter()
    fod, fod_affine = _load(args.fod, device)
    five, five_affine = _load(args.five_tissue, device)
    gmwmi, gmwmi_affine = _load(args.gmwmi, device)
    fa, fa_affine = _load(args.fa, device)
    atlas, atlas_affine = _load(args.atlas, device, torch.int32)
    if five.shape[:3] != gmwmi.shape or not torch.allclose(five_affine, gmwmi_affine):
        raise ValueError("5TT and GMWMI must share one grid")
    if fod.shape[:3] != fa.shape or not torch.allclose(fod_affine, fa_affine, atol=1e-3):
        raise ValueError("FOD and FA must share one grid")
    _sync(device)
    load_seconds = time.perf_counter() - started
    spacing = nib.load(str(args.five_tissue)).header.get_zooms()[:3]
    voxel_mm = float(torch.linalg.vector_norm(fod_affine[:3, :3], dim=0).prod().pow(1 / 3))
    started = time.perf_counter()
    tracks = probabilistic_tractography(
        wm_sh=fod, fod_affine=fod_affine, five_tissue=five,
        five_tissue_affine=five_affine, gmwmi=gmwmi,
        five_tissue_spacing_mm=spacing, n_seeds=args.n_seeds,
        batch_size=args.batch_size, seed=args.seed, lmax=8,
        compile_arc=args.compile_arc,
    )
    _sync(device)
    tracking_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nib.streamlines.save(nib.streamlines.Tractogram(
        [path.cpu().numpy() for path in tracks.paths], affine_to_rasmm=np.eye(4)),
        str(args.output_dir / "tracks.tck"))
    (args.output_dir / "tracking.json").write_text(json.dumps({
        "accepted_streamlines": len(tracks.paths),
        "total_path_points": sum(len(path) for path in tracks.paths),
        "tracking_seconds": tracking_seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
    }, indent=2) + "\n")
    print(f"tracking: {len(tracks.paths)} accepted in {tracking_seconds:.2f} s", flush=True)
    started = time.perf_counter()
    processing = processing_mask_from_5tt(fod, fod_affine, five, five_affine)
    weights = estimate_sift2_weights(
        paths=tracks.paths, wm_sh=fod, fod_affine=fod_affine,
        five_tissue=five, five_tissue_affine=five_affine,
        step_size_mm=voxel_mm / 2, processing_mask=processing,
    )
    fa_values = sample_streamline_mean_precise(tracks.paths, fa, fa_affine)
    matrices = build_connectomes(
        endpoints=tracks.endpoints, atlas=atlas, affine=atlas_affine,
        weights=weights, lengths=tracks.lengths_mm, fa=fa_values,
    )
    _sync(device)
    post_seconds = time.perf_counter() - started
    candidate = {name: matrices[name].cpu().numpy() for name in NAMES}
    reference = {name: np.loadtxt(args.official_dir / f"{name}.csv", delimiter=",")
                 for name in NAMES}
    report = {
        "dataset": "OpenNeuro ds004666 corrected DWI; identical FOD, 5TT, GMWMI, FA and atlas",
        "input_sha256": {name: _sha256(path) for name, path in input_paths},
        "device": str(device), "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "n_seeds": args.n_seeds, "batch_size": args.batch_size, "seed": args.seed,
        "compile_arc": args.compile_arc,
        "accepted_streamlines": len(tracks.paths),
        "load_seconds": load_seconds, "tracking_seconds": tracking_seconds,
        "postprocessing_seconds": post_seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else None),
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "metrics": {name: _metrics(candidate[name], reference[name],
                                   (candidate["count"] > 0, reference["count"] > 0))
                    for name in NAMES},
    }
    for name, matrix in candidate.items():
        np.savetxt(args.output_dir / f"{name}.csv", matrix, delimiter=",")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
