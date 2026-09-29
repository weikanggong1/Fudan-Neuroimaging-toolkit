"""同一真实 FOD、5TT、种子和 RNG 下比较 FSL/TorchFLIRT 对最终矩阵的影响。"""

import argparse
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import NAMES, _load, _metrics, _sha256, _sync
from fnit.connectome.anatomy import resample_labels_nearest
from fnit.connectome.assignment import build_connectomes
from fnit.connectome.freesurfer_subject import fs_aparc_atlas
from fnit.connectome.sift2 import estimate_sift2_weights
from fnit.connectome.sift2_proc_mask import processing_mask_from_5tt
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise
from fnit.connectome.tracking import (
    _act_seed_direction, _grow, _ifod2_calibration, _initial_directions, _sample,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue-t1", "aparc-aseg", "seeds", "fa", "registration-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-seeds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n_seeds < 1:
        parser.error("--n-seeds must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.empty(1, device=device)
    fod, fod_affine = _load(args.fod, device)
    five, t1_affine = _load(args.five_tissue_t1, device)
    seg_image = nib.load(str(args.aparc_aseg))
    segmentation = torch.as_tensor(seg_image.get_fdata(dtype=np.float32), device=device)
    seg_affine = torch.as_tensor(seg_image.affine, dtype=torch.float64, device=device)
    fa, fa_affine = _load(args.fa, device)
    fsl_seeds = torch.as_tensor(np.loadtxt(args.seeds, dtype=np.float32)[:args.n_seeds], device=device)
    if len(fsl_seeds) != args.n_seeds or fod.shape[-1] != 45 or five.shape[-1] != 5:
        raise ValueError("input must supply n_seeds positions, 45 FOD and 5TT channels")
    atlas_t1, nodes = fs_aparc_atlas(segmentation=segmentation)
    transformations = json.loads(args.registration_report.read_text())
    voxel = float(torch.linalg.vector_norm(fod_affine[:3, :3], dim=0).prod().pow(1 / 3))
    step = voxel / 2
    calibration, ratio = _ifod2_calibration(8, step, voxel, 45., .5, device)
    fod_inverse = torch.linalg.inv(fod_affine)
    input_hashes = {name: _sha256(path) for name, path in (
        ("fod", args.fod), ("five_tissue_t1", args.five_tissue_t1),
        ("aparc_aseg", args.aparc_aseg), ("seeds", args.seeds), ("fa", args.fa),
        ("registration_report", args.registration_report),
    )}
    result = {"input_sha256": input_hashes, "device": str(device), "seed": args.seed,
              "n_seeds": len(fsl_seeds), "atlas_nodes": len(nodes), "arms": {},
              "seed_rule": "fixed T1 anatomical positions; each arm maps them into its DWI world"}
    fsl_transform = torch.as_tensor(transformations["fsl_world"], dtype=torch.float64, device=device)
    t1_seeds = fsl_seeds.double() @ fsl_transform[:3, :3].T + fsl_transform[:3, 3]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices_by_arm = {}
    atlases = {}
    for name in ("fsl", "torch"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        transform = torch.as_tensor(transformations[f"{name}_world"], dtype=torch.float64, device=device)
        inverse_transform = torch.linalg.inv(transform)
        seeds = (t1_seeds @ inverse_transform[:3, :3].T + inverse_transform[:3, 3]).float()
        five_affine = torch.linalg.inv(transform) @ t1_affine
        effective = five_affine.clone()
        spacing = torch.as_tensor(nib.load(str(args.five_tissue_t1)).header.get_zooms()[:3],
                                  dtype=torch.float64, device=device)
        effective[:3, :3] *= spacing / torch.linalg.vector_norm(effective[:3, :3], dim=0)
        five_inverse = torch.linalg.inv(effective)
        atlas = resample_labels_nearest(
            labels=atlas_t1, source_affine=seg_affine,
            target_shape=tuple(fod.shape[:3]), target_affine=fod_affine,
            target_to_source_world=transform,
        )
        atlases[name] = atlas
        nib.save(nib.Nifti1Image(atlas.cpu().numpy(), fod_affine.cpu().numpy()),
                 str(args.output_dir / f"{name}_atlas_dwi.nii.gz"))
        initial, initial_valid, _ = _initial_directions(
            _sample(fod, seeds, fod_inverse),
            torch.Generator(device=device).manual_seed(args.seed), lmax=8, cutoff=.1,
        )
        act_valid, one_way, initial = _act_seed_direction(five, seeds, initial, five_inverse)
        valid = initial_valid & act_valid
        generator = torch.Generator(device=device).manual_seed(1729 + args.seed)
        options = dict(lmax=8, proposals_per_step=16, step_mm=step,
                       max_steps=math.ceil(250 / step), max_angle_degrees=45.,
                       cutoff=.1, power=.5, calibration_local=calibration,
                       calibration_ratio=ratio)
        forward, nf, gf, lf, wf = _grow(
            seeds, initial, fod, five, fod_inverse, five_inverse, generator, **options,
        )
        backward, nb, gb, lb, wb = _grow(
            seeds, -initial, fod, five, fod_inverse, five_inverse, generator,
            seed_to_wm_initial=wf, **options,
        )
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
        started = time.perf_counter()
        processing = processing_mask_from_5tt(fod, fod_affine, five, five_affine)
        weights = estimate_sift2_weights(
            paths, fod, fod_affine, five, five_affine,
            step_size_mm=step, processing_mask=processing,
        )
        mean_fa = sample_streamline_mean_precise(paths, fa, fa_affine)
        matrices = build_connectomes(
            endpoints, atlas, fod_affine, weights=weights,
            lengths=total[keep], fa=mean_fa, node_count=len(nodes),
        )
        if name == "fsl":
            fixed_tracks = (endpoints, weights, total[keep], mean_fa)
        _sync(device)
        post_seconds = time.perf_counter() - started
        matrices_by_arm[name] = {key: value.cpu().numpy() for key, value in matrices.items()}
        for key in NAMES:
            np.savetxt(args.output_dir / f"{name}_{key}.csv", matrices_by_arm[name][key], delimiter=",")
        result["arms"][name] = {
            "act_valid": int(valid.sum()), "accepted_streamlines": len(paths),
            "seed_world_displacement_from_fsl_mm_mean": float(
                torch.linalg.vector_norm(seeds - fsl_seeds, dim=-1).mean()),
            "tracking_seconds": tracking_seconds, "postprocessing_seconds": post_seconds,
            "peak_torch_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else None,
        }
    count_fsl = matrices_by_arm["fsl"]["count"]
    count_torch = matrices_by_arm["torch"]["count"]
    result["atlas_voxel_xor"] = int((atlases["fsl"] != atlases["torch"]).sum())
    result["atlas_voxels"] = int(atlases["fsl"].numel())
    result["matrices_torch_vs_fsl"] = {
        key: _metrics(matrices_by_arm["torch"][key], matrices_by_arm["fsl"][key],
                      (count_torch > 0, count_fsl > 0)) for key in NAMES
    }
    endpoints, weights, lengths, mean_fa = fixed_tracks
    atlas_only = build_connectomes(
        endpoints, atlases["torch"], fod_affine, weights=weights,
        lengths=lengths, fa=mean_fa, node_count=len(nodes),
    )
    count_atlas_only = atlas_only["count"].cpu().numpy()
    result["fixed_fsl_tracks_torch_atlas_vs_fsl_atlas"] = {
        key: _metrics(atlas_only[key].cpu().numpy(), matrices_by_arm["fsl"][key],
                      (count_atlas_only > 0, count_fsl > 0)) for key in NAMES
    }
    for key in NAMES:
        np.savetxt(args.output_dir / f"fixed_tracks_torch_atlas_{key}.csv",
                   atlas_only[key].cpu().numpy(), delimiter=",")
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
