"""Compare PyTorch FOD peak metrics with Dhollander's MRtrix sh2peaks stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform
import numpy as np
import torch

from fnit.connectome.fod import mrtrix_fod_peak_amplitude


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
    for name in ("dwi", "wm2", "csf2", "metric2", "mask2", "selected2",
                 "wm6", "csf6", "metric6", "mask6", "selected6", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = nib.load(args.dwi)
    device = torch.device(args.device)
    paths = {key: getattr(args, key) for key in
             ("dwi", "wm2", "csf2", "metric2", "mask2", "selected2",
              "wm6", "csf6", "metric6", "mask6", "selected6")}
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm corrected DWI",
        "scope": "fixed official internal WM/CSF FOD; sh2peaks and peaks2amp metric only",
        "reference_commands": "sh2peaks abs_ewm2.mif - -num 1 -mask refined_wm.mif | peaks2amp - -; repeat on abs_ewm6.mif with refined_sfwm.mif",
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "input_sha256": {key: _sha(path) for key, path in paths.items()},
        "metrics": {},
    }
    for lmax in (2, 6):
        wm = torch.as_tensor(_aligned(getattr(args, f"wm{lmax}"), image), device=device)
        csf = torch.as_tensor(_aligned(getattr(args, f"csf{lmax}"), image), device=device)
        mask = _aligned(getattr(args, f"mask{lmax}"), image).astype(bool)
        selected = _aligned(getattr(args, f"selected{lmax}"), image).astype(bool)
        reference = _aligned(getattr(args, f"metric{lmax}"), image)
        coefficient = wm[torch.as_tensor(mask, device=device)].to(torch.float64)
        denominator = (coefficient[:, 0] + csf[torch.as_tensor(mask, device=device)].to(torch.float64))
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        peak = mrtrix_fod_peak_amplitude(coefficient, lmax)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        metric = (peak / denominator).cpu().numpy()
        truth = reference[mask]
        err = np.abs(metric - truth)
        indices = np.flatnonzero(mask.ravel())
        count_selected = int(selected.sum())
        chosen = np.zeros(mask.size, dtype=bool)
        chosen[indices[np.argpartition(metric, -count_selected)[-count_selected:]]] = True
        xor = np.count_nonzero(chosen ^ selected.ravel())
        report["metrics"][str(lmax)] = {
            "mask_voxels": int(mask.sum()), "selected_voxels": count_selected,
            "seconds": seconds, "metric_mae": float(err.mean()),
            "metric_max_abs": float(err.max()), "selected_xor": int(xor),
            "metric_pearson": float(np.corrcoef(metric, truth)[0, 1]),
            "peak_torch_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
            ),
        }
        print(lmax, json.dumps(report["metrics"][str(lmax)]), flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
