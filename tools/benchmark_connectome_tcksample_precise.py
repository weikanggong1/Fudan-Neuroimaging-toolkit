"""Compare real ds004666 PyTorch precise track FA means to MRtrix.

Original command: tcksample -precise -stat_tck mean tracks_10000.tck
fa_corrected.mif streamline_mean_fa.txt -nthreads 8
The MIF is converted once with mrconvert fa_corrected.mif fa_corrected.nii.gz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from fnit.connectome.tcksample_precise import sample_streamline_mean_precise


def _sha256(path: Path) -> str:
    """Return SHA-256 of one fixed real-data input file."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    """Load fixed tracks/FA/reference, compute precision/time, save JSON and PNG."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--fa", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    fa_image = nib.load(args.fa)
    fa = torch.as_tensor(np.asarray(fa_image.dataobj, dtype=np.float32).copy(), device=device)
    if fa.ndim != 3:
        raise ValueError("expected 3D float32 FA image")
    paths = [torch.as_tensor(np.asarray(path, dtype=np.float32).copy(), device=device)
             for path in nib.streamlines.load(args.tracks).streamlines]
    reference = np.loadtxt(args.reference, dtype=np.float32).reshape(-1)
    if len(paths) != len(reference):
        raise ValueError("TCK and official FA vector count differ")
    affine = torch.as_tensor(fa_image.affine, device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    candidate = sample_streamline_mean_precise(paths, fa, affine)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    candidate = candidate.cpu().numpy().astype(np.float64)
    ref = reference.astype(np.float64)
    finite = np.isfinite(ref) & np.isfinite(candidate)
    diff = candidate[finite] - ref[finite]
    figure = args.output.with_suffix(".png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    center = fa_image.shape[2] // 2
    axes[0].imshow(np.rot90(np.asarray(fa_image.dataobj[:, :, center])), cmap="gray", vmin=0, vmax=.8)
    axes[0].set_title(f"Fixed corrected FA, z={center}")
    axes[0].axis("off")
    axes[1].scatter(ref[finite], candidate[finite], s=3, alpha=.4)
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=1)
    axes[1].set(xlabel="MRtrix mean FA", ylabel="PyTorch mean FA", title="Same TCK + FA")
    axes[2].hist(diff, bins=80, color="#426e91")
    axes[2].set(xlabel="PyTorch − MRtrix FA", ylabel="Tracks", title="Per-track difference")
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm; 2758 fixed real MRtrix tracks and corrected FA",
        "original_command": "tcksample -precise -stat_tck mean tracks_10000.tck fa_corrected.mif streamline_mean_fa.txt -nthreads 8",
        "comparison": "identical TCK and FA image after lossless MIF-to-NIfTI float32 conversion",
        "input_sha256": {x.name: _sha256(x) for x in (args.tracks, args.fa, args.reference)},
        "n_tracks": len(paths), "n_finite_pairs": int(finite.sum()),
        "device": str(device), "dtype": "float32 inputs/output; float64 length sums; TF32 default on CUDA; no float16",
        "pytorch_seconds": seconds,
        "peak_cuda_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "official_tcksample_seconds": 0.03,
        "timing_note": "Official 0.03 s is command wall time in corrected_mrtrix_fs5tt_act_adapted/stage_times.tsv; PyTorch time starts after TCK/FA loading.",
        "reference_mean": float(np.mean(ref[finite])), "candidate_mean": float(np.mean(candidate[finite])),
        "pearson": float(np.corrcoef(ref[finite], candidate[finite])[0, 1]),
        "mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "abs_error_quantiles": {str(q): float(np.quantile(np.abs(diff), q)) for q in (.5, .9, .99, .999)},
        "figure": figure.name,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
