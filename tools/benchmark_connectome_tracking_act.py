"""Same-input tractography benchmark against MRtrix iFOD2 and ACT.

Example::

  python tools/benchmark_connectome_tracking_act.py \
    --fod wm_fod_norm.nii.gz --five-tissue five_tissue.nii.gz \
    --gmwmi gmwmi.nii.gz --reference official_tracks.tck \
    --reference-seeds official_successful_seeds.txt --output tracking_act.json

The reference command is documented in the JSON report. MRtrix and PyTorch
have distinct RNGs, so compare distributions and anatomy rather than pairs of
streamlines with the same nominal seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
import torch

from fnit.connectome.tracking import probabilistic_tractography


def _hash(path: Path) -> str:
    """Return the input file's hexadecimal SHA-256 provenance digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(tracks: list[np.ndarray]) -> dict:
    """Return length and endpoint arrays plus quantiles for RAS-mm tracks."""
    length = np.array([np.linalg.norm(np.diff(t, axis=0), axis=1).sum() for t in tracks])
    endpoints = np.concatenate([np.stack((t[0], t[-1])) for t in tracks])
    return {
        "streamlines": len(tracks),
        "length_mm_quantiles_0_10_25_50_75_90_100": np.quantile(
            length, [0, .1, .25, .5, .75, .9, 1]
        ).tolist(),
        "endpoint_center_ras_mm": endpoints.mean(axis=0).tolist(),
        "lengths": length,
        "endpoints": endpoints,
    }


def main() -> None:
    """Load paired real inputs, run PyTorch tracking and write JSON/figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "five-tissue", "gmwmi", "reference", "reference-seeds", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    fod_image = nib.load(args.fod)
    five_image = nib.load(args.five_tissue)
    gmwmi_image = nib.load(args.gmwmi)
    if five_image.shape[:3] != gmwmi_image.shape or not np.allclose(
        five_image.affine, gmwmi_image.affine
    ):
        raise ValueError("5TT and GMWMI must have identical geometry")
    five = np.asarray(five_image.dataobj, dtype=np.float32)
    reference_tracks = list(nib.streamlines.load(args.reference).tractogram.streamlines)
    reference = _summarize(reference_tracks)
    reference_seeds = np.loadtxt(args.reference_seeds, delimiter=",", comments="#",
                                 usecols=(2, 3, 4))
    if reference_seeds.ndim != 2 or reference_seeds.shape[1] != 3:
        raise ValueError("MRtrix -output_seeds must contain XYZ rows")

    device = torch.device(args.device)
    fod = torch.as_tensor(np.asarray(fod_image.dataobj, dtype=np.float32).copy(), device=device)
    five_tensor = torch.as_tensor(five.copy(), device=device)
    gmwmi_tensor = torch.as_tensor(np.asarray(gmwmi_image.dataobj, dtype=np.float32).copy(), device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = perf_counter()
    candidate = probabilistic_tractography(
        fod, torch.as_tensor(fod_image.affine, device=device, dtype=torch.float64),
        five_tensor, torch.as_tensor(five_image.affine, device=device, dtype=torch.float64),
        gmwmi_tensor, n_seeds=args.n_seeds, lmax=8, seed=args.seed,
        batch_size=args.batch_size, cutoff=.1, power=.5,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = perf_counter() - start
    candidate_tracks = [t.cpu().numpy() for t in candidate.paths]
    result = _summarize(candidate_tracks)
    ref_nn = cKDTree(reference["endpoints"])
    cand_nn = cKDTree(result["endpoints"])
    r_to_c = cand_nn.query(reference["endpoints"], workers=-1)[0]
    c_to_r = ref_nn.query(result["endpoints"], workers=-1)[0]
    seed_voxel = np.linalg.inv(five_image.affine) @ np.concatenate(
        [reference_seeds, np.ones((len(reference_seeds), 1))], axis=1
    ).T
    seed_index = np.rint(seed_voxel[:3].T).astype(int)
    inside = np.all((seed_index >= 0) & (seed_index < np.array(gmwmi_image.shape)), axis=1)
    seed_values = np.zeros(len(seed_index), dtype=np.float32)
    gmwmi = np.asarray(gmwmi_image.dataobj)
    ix = seed_index[inside]
    seed_values[inside] = gmwmi[ix[:, 0], ix[:, 1], ix[:, 2]]
    report = {
        "inputs_sha256": {n: _hash(getattr(args, n.replace("-", "_")))
                          for n in ("fod", "five-tissue", "gmwmi", "reference", "reference-seeds")},
        "reference_command": "tckgen -algorithm iFOD2 -seed_gmwmi GMWMI -act 5TT -seeds 10000 -select 0 -maxlength 250 -cutoff 0.1 -samples 3 -power 0.5 -nthreads 8 -output_seeds successful_seeds.txt FOD official_tracks.tck",
        "candidate_function": "probabilistic_tractography(wm_sh, fod_affine, five_tissue, five_tissue_affine, gmwmi, n_seeds=10000, lmax=8, cutoff=0.1, power=0.5)",
        "candidate_mode": "same 1 mm 5TT/GMWMI seed input and iFOD2 two-point curved FOD integration; continuous random cone with 16 finite proposals instead of MRtrix calibrated rejection sampling; SGM handling remains approximate",
        "n_seeds_attempted": args.n_seeds,
        "candidate_batch_size": args.batch_size,
        "reference_streamlines": reference["streamlines"],
        "candidate_streamlines": result["streamlines"],
        "reference_length_mm_quantiles": reference["length_mm_quantiles_0_10_25_50_75_90_100"],
        "candidate_length_mm_quantiles": result["length_mm_quantiles_0_10_25_50_75_90_100"],
        "length_ks_statistic": float(ks_2samp(reference["lengths"], result["lengths"]).statistic),
        "reference_seed_gmwmi_nearest_voxel_positive_fraction": float((seed_values > 0).mean()),
        "reference_seed_gmwmi_nearest_voxel_quantiles": np.quantile(seed_values, [0, .25, .5, .75, 1]).tolist(),
        "accepted_seed_centroid_ras_mm": candidate.accepted_seeds.mean(0).cpu().tolist() if len(candidate.accepted_seeds) else None,
        "reference_successful_seed_centroid_ras_mm": reference_seeds.mean(0).tolist(),
        "reference_to_candidate_endpoint_nn_mm_quantiles": np.quantile(r_to_c, [.5, .9, .99]).tolist(),
        "candidate_to_reference_endpoint_nn_mm_quantiles": np.quantile(c_to_r, [.5, .9, .99]).tolist(),
        "candidate_wall_seconds": elapsed,
        "candidate_cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
        "comparison_limit": "MRtrix RNG has no exposed tckgen seed; paired streamline identity cannot be required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.figure is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def density(tracks):
            """Count track vertices on the FOD voxel grid for example images."""
            counts = np.zeros(fod_image.shape[:3], dtype=np.int32)
            inverse = np.linalg.inv(fod_image.affine)
            for track in tracks:
                voxel = np.rint(track @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
                inside = np.all((voxel >= 0) & (voxel < np.array(counts.shape)), axis=1)
                ijk = voxel[inside]
                np.add.at(counts, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)
            return counts

        ref_density, cand_density = density(reference_tracks), density(candidate_tracks)
        z = int(np.argmax(ref_density.sum(axis=(0, 1))))
        lo, hi = max(0, z - 2), min(ref_density.shape[2], z + 3)
        ref_slice = np.log1p(ref_density[:, :, lo:hi].sum(-1))
        cand_slice = np.log1p(cand_density[:, :, lo:hi].sum(-1))
        vmax = max(np.quantile(ref_slice, .995), np.quantile(cand_slice, .995))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for axis, array, title in zip(axes[:2], (ref_slice, cand_slice),
                                      ("MRtrix iFOD2 + ACT", "PyTorch curved arcs + ACT")):
            axis.imshow(np.rot90(array), cmap="magma", vmin=0, vmax=vmax)
            axis.set_title(title)
            axis.set_xlabel(f"axial slab z={lo}:{hi - 1}")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[2].hist(reference["lengths"], bins=np.arange(0, 251, 5),
                     density=True, histtype="step", linewidth=1.8, label="MRtrix")
        axes[2].hist(result["lengths"], bins=np.arange(0, 251, 5),
                     density=True, histtype="step", linewidth=1.8, label="PyTorch")
        axes[2].set(xlabel="streamline length (mm)", ylabel="density")
        axes[2].legend(frameon=False)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
