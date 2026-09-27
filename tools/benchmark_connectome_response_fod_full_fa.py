"""Compare full-brain PyTorch FA with MRtrix dwi2tensor/tensor2metric.

MRtrix reference:
dwi2tensor corrected.mif tensor.mif -mask synthseg_brain_mask_dwi.nii.gz
tensor2metric tensor.mif -fa fa_corrected.mif
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
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform
import numpy as np
import torch

from fnit.connectome.response import fit_mrtrix_dhollander_tensor


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aligned(path: Path, target: nib.spatialimages.SpatialImage) -> np.ndarray:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    orientation = ornt_transform(io_orientation(image.affine), io_orientation(target.affine))
    aligned_affine = image.affine @ inv_ornt_aff(orientation, data.shape[:3])
    if not np.allclose(aligned_affine, target.affine, atol=5e-5):
        raise ValueError(f"{path}: affine cannot align to DWI")
    return np.ascontiguousarray(apply_orientation(data, orientation))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "brain-mask", "ref-fa", "out", "png"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = nib.load(args.dwi)
    signal = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    gradient = np.loadtxt(args.grad)
    mask = _aligned(args.brain_mask, image).astype(bool)
    reference = _aligned(args.ref_fa, image).astype(np.float32)
    device = torch.device(args.device)
    data = torch.as_tensor(signal, device=device)
    mask_tensor = torch.as_tensor(mask, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fa, _ = fit_mrtrix_dhollander_tensor(data, gradient, mask_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    actual = fa.cpu().numpy()
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    finite = mask & np.isfinite(actual) & np.isfinite(reference)
    both_nan = mask & np.isnan(actual) & np.isnan(reference)
    nonfinite_mismatch = mask & (np.isfinite(actual) != np.isfinite(reference))
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm corrected DWI",
        "scope": "all original SynthSeg brain-mask voxels, all 105 DWI volumes",
        "reference_commands": __doc__.split("MRtrix reference:\n", 1)[1].strip(),
        "device": str(device), "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "shape": list(signal.shape), "masked_voxels": int(mask.sum()),
        "seconds": seconds,
        "peak_torch_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        ),
        "input_sha256": {
            name: _sha(getattr(args, name)) for name in ("dwi", "grad", "brain_mask", "ref_fa")
        },
        "fa": {
            "brain_finite_voxels": int(finite.sum()),
            "brain_matching_nan_voxels": int(both_nan.sum()),
            "brain_nonfinite_mismatch_voxels": int(nonfinite_mismatch.sum()),
            "brain_mae": float(error[finite].mean()),
            "brain_max_abs": float(error[finite].max()),
            "brain_pearson": float(np.corrcoef(actual[finite], reference[finite])[0, 1]),
            "brain_count_abs_gt_1e-6": int(np.count_nonzero(error[finite] > 1e-6)),
            "whole_grid_finite_mae": float(np.nanmean(error)),
            "whole_grid_finite_max_abs": float(np.nanmax(error)),
        },
    }
    report["fa"]["mismatch_voxels"] = [
        {"ijk": list(map(int, ijk)),
         "candidate": float(actual[tuple(ijk)]),
         "reference": float(reference[tuple(ijk)]),
         "signal_min": float(signal[tuple(ijk)].min()),
         "signal_max": float(signal[tuple(ijk)].max())}
        for ijk in np.argwhere(finite & (error > 1e-6))
    ]
    slices = [signal.shape[2] // 2,
              int(np.unravel_index(np.nanargmax(error), error.shape)[2])]
    report["figure_slices"] = slices
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, index in enumerate(slices):
        scale = max(1e-6, float(np.nanmax(error[:, :, index])))
        for ax, volume, title, maximum in zip(
            axes[row], (reference, actual, error),
            ("MRtrix FA", "PyTorch FA", "Absolute difference"),
            (1, 1, scale),
        ):
            ax.imshow(np.rot90(volume[:, :, index]), cmap="magma", vmin=0, vmax=maximum)
            ax.set_title(f"{title}, axial z={index}")
            ax.axis("off")
    fig.suptitle("ds004666 sub-01 ses-2mm; corrected DWI; full SynthSeg brain mask")
    fig.savefig(args.png, dpi=180)
    plt.close(fig)
    print(json.dumps(report["fa"] | {"seconds": seconds,
          "peak_torch_allocated_gib": report["peak_torch_allocated_gib"]}, indent=2))


if __name__ == "__main__":
    main()
