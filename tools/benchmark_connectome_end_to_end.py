"""Run corrected real DWI through FNIT and compare four MRtrix matrices.

Use the same corrected DWI, FreeSurfer aparc+aseg, atlas, brain/FOD masks and
DWI-to-T1 RAS transform as the reference. MRtrix command provenance belongs
in the reference directory. The FSL BET mask and recon-all are fixed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import fnit.connectome as connectome_module
from fnit.connectome import UKBConnectome_pipeline


NAMES = ("count", "sift2_fbc", "mean_length", "mean_fa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
        raise ValueError("candidate and reference matrices must be the same square shape")
    triangle = np.triu_indices(candidate.shape[0], 1)
    a = candidate[triangle].astype(np.float64)
    b = reference[triangle].astype(np.float64)
    active = (a != 0) | (b != 0)
    da = np.abs(a - b)
    common = (a != 0) & (b != 0)
    common_a, common_b = a[common], b[common]
    reference_nonzero = np.abs(b[b != 0])
    common_correlation = (
        float(np.corrcoef(common_a, common_b)[0, 1])
        if len(common_a) > 1 and np.std(common_a) > 0 and np.std(common_b) > 0
        else None
    )
    return {
        "upper_triangle_edges": int(len(a)),
        "union_support_edges": int(active.sum()),
        "common_support_edges": int(common.sum()),
        "support_dice": float(2 * common.sum() / ((a != 0).sum() + (b != 0).sum())),
        "pearson_upper": float(np.corrcoef(a, b)[0, 1]),
        "mae_upper": float(da.mean()),
        "normalized_mae_upper": float(da.mean() / reference_nonzero.mean()),
        "relative_l1_upper": float(da.sum() / np.abs(b).sum()),
        "max_abs_upper": float(da.max()),
        "pearson_common_support": common_correlation,
        "mae_common_support": float(np.mean(np.abs(common_a - common_b))) if len(common_a) else None,
        "normalized_mae_common_support": (
            float(np.abs(common_a - common_b).mean() / np.abs(common_b).mean())
            if len(common_a) and np.abs(common_b).sum() else None
        ),
        "candidate_sum_upper": float(a.sum()),
        "reference_sum_upper": float(b.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "bvals", "bvecs", "t1-brain", "aparc-aseg", "atlas-dwi",
                 "brain-mask", "transform", "reference-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("response-mask", "fod-mask", "normalise-mask", "fa-map"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--shell-bvals", nargs="+", type=float)
    parser.add_argument("--n-seeds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    input_names = ("dwi", "bvals", "bvecs", "t1_brain", "aparc_aseg", "atlas_dwi",
                   "brain_mask", "transform", "response_mask", "fod_mask",
                   "normalise_mask", "fa_map")
    paths = {name: getattr(args, name) for name in input_names}
    for name, path in paths.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    transform = np.loadtxt(args.transform, comments="#")
    if transform.shape != (4, 4):
        raise ValueError("transform must contain a 4x4 RAS-mm matrix")
    reference_paths = {}
    for name in NAMES:
        direct = args.reference_dir / f"{name}.csv"
        published = args.reference_dir / f"connectome_{name}.csv"
        path = direct if direct.is_file() else published
        if not path.is_file():
            raise FileNotFoundError(f"reference {name}: {direct} or {published}")
        reference_paths[name] = path
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = UKBConnectome_pipeline(device=args.device)(
        args.dwi, args.bvals, args.bvecs, args.t1_brain,
        t1_segmentation=args.aparc_aseg, atlas_dwi=args.atlas_dwi,
        brain_mask=args.brain_mask, shell_bvals=args.shell_bvals,
        response_mask=args.response_mask, fod_mask=args.fod_mask,
        normalise_mask=args.normalise_mask, fa_map=args.fa_map,
        dwi_to_t1_world=transform, n_seeds=args.n_seeds, seed=args.seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in NAMES:
        candidate = result.matrices[name].detach().cpu().numpy()
        reference = np.loadtxt(reference_paths[name], delimiter=",")
        metrics[name] = _matrix_metrics(candidate, reference)
        np.savetxt(args.output_dir / f"candidate_{name}.csv", candidate, delimiter=",")
    tracks = [path.detach().cpu().numpy() for path in result.tractogram.paths]
    nib.streamlines.save(nib.streamlines.Tractogram(tracks, affine_to_rasmm=np.eye(4)),
                         args.output_dir / "candidate_tracks.tck")
    np.savetxt(args.output_dir / "candidate_sift2_weights.txt",
               result.sift2_weights.detach().cpu().numpy())
    np.savetxt(args.output_dir / "candidate_length_mm.txt",
               result.tractogram.lengths_mm.detach().cpu().numpy())
    np.savetxt(args.output_dir / "candidate_mean_fa.txt",
               result.tractogram.mean_fa.detach().cpu().numpy())
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm corrected AP DWI + paired T1",
        "reference_scope": "same corrected DWI, masks, FreeSurfer aparc+aseg, atlas and fixed RAS transform",
        "metric_policy": {
            "edges": "strict upper triangle; self connections excluded",
            "normalized_mae_upper": "MAE over all upper edges divided by mean absolute nonzero reference edge",
            "relative_l1_upper": "sum absolute error over upper edges divided by sum absolute reference upper edges",
            "normalized_mae_common_support": "MAE on edges nonzero in both matrices divided by mean absolute reference on those edges",
        },
        "device": args.device,
        "torch_version": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "accepted_streamlines": len(tracks),
        "time_seconds_full_call": compute_seconds,
        "peak_torch_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else None
        ),
        "input_sha256": {name: _sha256(path) for name, path in paths.items() if path is not None},
        "reference_sha256": {name: _sha256(path) for name, path in reference_paths.items()},
        "source_sha256": {
            "benchmark_script": _sha256(Path(__file__)),
            **{path.name: _sha256(path)
               for path in sorted(Path(connectome_module.__file__).parent.glob("*.py"))},
        },
        "matrices": metrics,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
