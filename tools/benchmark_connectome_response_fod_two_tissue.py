"""Compare both fixed-response two-tissue CSD fits inside Dhollander.

Reference commands:
dwi2fod msmt_csd dwi.mif ewmrf.txt abs_ewm2.mif response_csf.txt
    abs_csf2.mif -mask refined_wm.mif -lmax 2,0
dwi2fod msmt_csd dwi.mif ewmrf.txt abs_ewm6.mif response_csf.txt
    abs_csf6.mif -mask refined_sfwm.mif -lmax 6,0
"""

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

from fnit.connectome.fod import fit_mrtrix_two_tissue_csd


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


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "n_values": int(error.size),
        "mae": float(error.mean()),
        "max_abs": float(error.max()),
        "fraction_abs_le_1e-5": float(np.mean(error <= 1e-5)),
        "pearson": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "wm-response", "csf-response",
                 "mask2", "mask6", "ref-wm2", "ref-csf2",
                 "ref-wm6", "ref-csf6", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in
             ("dwi", "grad", "wm_response", "csf_response",
              "mask2", "mask6", "ref_wm2", "ref_csf2", "ref_wm6", "ref_csf6")}
    image = nib.load(args.dwi)
    dwi = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    grad = np.loadtxt(args.grad)
    wm_response = np.loadtxt(args.wm_response, ndmin=2)
    csf_response = np.loadtxt(args.csf_response, comments="#", ndmin=2)
    first = args.csf_response.open().readline().strip()
    if not first.startswith("# Shells:"):
        raise ValueError("CSF response lacks shell header")
    shells = np.fromstring(first.split(":", 1)[1], sep=",")
    device = torch.device(args.device)
    signal = torch.as_tensor(dwi, device=device)
    metrics = {}
    for lmax in (2, 6):
        mask = _aligned(getattr(args, f"mask{lmax}"), image).astype(bool)
        mask_tensor = torch.as_tensor(mask, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        wm, csf = fit_mrtrix_two_tissue_csd(
            signal, grad, shells, wm_response, csf_response, mask_tensor,
            lmax=lmax, batch_size=args.batch_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        wm_ref = _aligned(getattr(args, f"ref_wm{lmax}"), image)
        csf_ref = _aligned(getattr(args, f"ref_csf{lmax}"), image)
        if csf_ref.ndim == 4:
            csf_ref = csf_ref[..., 0]
        wm_values = wm.cpu().numpy()[mask]
        csf_values = csf.cpu().numpy()[mask]
        metrics[str(lmax)] = {
            "mask_voxels": int(mask.sum()),
            "seconds": seconds,
            "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                         if device.type == "cuda" else None),
            "wm": _metrics(wm_values, wm_ref[mask]),
            "csf": _metrics(csf_values, csf_ref[mask]),
        }
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm, corrected DWI",
        "scope": "fixed official empirical WM/CSF responses and official masks; Dhollander lmax2 and lmax6 two-tissue raw CSD only",
        "reference_commands": __doc__.split("Reference commands:\n", 1)[1].strip(),
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "shape": list(dwi.shape),
        "shell_bvals": shells.tolist(),
        "input_sha256": {key: _sha(path) for key, path in paths.items()},
        "metrics": metrics,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
