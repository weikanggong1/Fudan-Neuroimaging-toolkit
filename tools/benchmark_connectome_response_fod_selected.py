"""Benchmark PyTorch amp2response against MRtrix with identical selected voxels.

First retain the official Dhollander scratch using:
dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt -mask brain.nii.gz
    -scratch scratch -nocleanup -voxels selected.mif

Convert scratch/voxels_sfwm.mif, voxels_gm.mif, voxels_csf.mif and
safe_vecs.mif to NIfTI with mrconvert -strides 1,2,3,4. The command
amp2response dwi.mif voxels_sfwm.mif safe_vecs.mif wm.txt is the direct
anisotropic reference; append -isotropic for GM and CSF.
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

from fnit.connectome.response import estimate_mrtrix_selected_response


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "wm-mask", "gm-mask", "csf-mask", "directions",
                 "wm-response", "gm-response", "csf-response", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_"))
             for name in ("dwi", "grad", "wm-mask", "gm-mask", "csf-mask", "directions",
                          "wm-response", "gm-response", "csf-response")}
    start = time.perf_counter()
    dwi_image = nib.load(paths["dwi"])
    data = np.asanyarray(dwi_image.dataobj).astype(np.float32, copy=False)
    gradient = np.loadtxt(paths["grad"])
    directions_image = nib.load(paths["directions"])
    if not np.allclose(dwi_image.affine, directions_image.affine, atol=1e-5):
        raise ValueError("principal direction affine differs from DWI")
    directions = np.asanyarray(directions_image.dataobj).astype(np.float32, copy=False)
    device = torch.device(args.device)
    signal = torch.as_tensor(data, device=device)
    fibres = torch.as_tensor(directions, device=device)
    load_seconds = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = {}
    for tissue in ("wm", "gm", "csf"):
        mask_image = nib.load(paths[tissue + "_mask"])
        if not np.allclose(dwi_image.affine, mask_image.affine, atol=1e-5):
            raise ValueError(f"{tissue} mask affine differs from DWI")
        mask = np.asanyarray(mask_image.dataobj).astype(bool).reshape(data.shape[:3])
        response_path = paths[tissue + "_response"]
        header = response_path.open().readline().strip()
        if not header.startswith("# Shells:"):
            raise ValueError(f"missing MRtrix response shell header in {response_path}")
        shell = np.fromstring(header.split(":", 1)[1], sep=",")
        reference = np.loadtxt(response_path, comments="#", ndmin=2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()
        actual = estimate_mrtrix_selected_response(
            signal, gradient, shell, torch.as_tensor(mask, device=device),
            fibres, isotropic=tissue != "wm",
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - tic
        difference = actual.cpu().numpy() - reference
        results[tissue] = {
            "selected_voxels": int(mask.sum()),
            "seconds": seconds,
            "mae": float(np.mean(np.abs(difference))),
            "max_abs": float(np.max(np.abs(difference))),
            "reference": reference.tolist(),
            "candidate": actual.cpu().numpy().tolist(),
        }
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm, FSL TOPUP/EDDY corrected AP DWI",
        "scope": "fixed official Dhollander selected tissue voxels and principal directions; amp2response regression only",
        "reference_commands": {
            "wm": "amp2response dwi.mif voxels_sfwm.mif safe_vecs.mif response_wm.txt",
            "gm": "amp2response dwi.mif voxels_gm.mif safe_vecs.mif response_gm.txt -isotropic",
            "csf": "amp2response dwi.mif voxels_csf.mif safe_vecs.mif response_csf.txt -isotropic",
        },
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "dtype_signal": "float32",
        "dtype_solver": "float64",
        "load_seconds": load_seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                     if device.type == "cuda" else None),
        "input_sha256": {name: _sha(path) for name, path in paths.items()},
        "metrics": results,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"metrics": results, "peak_gib": report["peak_torch_allocated_gib"]},
                     indent=2))


if __name__ == "__main__":
    main()
