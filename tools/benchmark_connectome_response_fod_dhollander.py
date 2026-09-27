"""Real-DWI whole Dhollander response benchmark against MRtrix 3.0.3.

MRtrix: dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt
-mask synthseg_brain_mask_dwi.nii.gz -voxels voxels.mif -nocleanup.
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

from fnit.connectome.response import estimate_mrtrix_dhollander

MASK_NAMES = ("safe_mask", "crude_wm", "crude_gm", "crude_csf",
              "refined_wm", "refined_gm", "refined_csf",
              "voxels_gm", "voxels_csf", "refined_sfwm", "voxels_sfwm")
MAP_NAMES = ("safe_sdm", "fa", "metric_sfwm2", "metric_sfwm6")


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


def _numeric(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        actual, expected = actual[mask], expected[mask]
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    difference = np.abs(actual - expected)
    return {"values": int(difference.size), "mae": float(difference.mean()),
            "max_abs": float(difference.max()),
            "pearson": float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "mask", "reference-dir", "wm-response",
                 "gm-response", "csf-response", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = nib.load(args.dwi)
    signal = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    gradient = np.loadtxt(args.grad)
    brain_mask = _aligned(args.mask, image).astype(bool)
    first = args.wm_response.open().readline().strip()
    if not first.startswith("# Shells:"):
        raise ValueError("WM response lacks shell header")
    shells = np.fromstring(first.split(":", 1)[1], sep=",")
    device = torch.device(args.device)
    data = torch.as_tensor(signal, device=device)
    mask = torch.as_tensor(brain_mask, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    _, wm, gm, csf, masks = estimate_mrtrix_dhollander(
        data, gradient, shells, mask,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    paths = {"dwi": args.dwi, "grad": args.grad, "mask": args.mask}
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm corrected DWI",
        "scope": "whole default Dhollander response estimation; identical DWI, gradient, brain mask, shell centres",
        "reference_command": __doc__.split("MRtrix: ", 1)[1].strip(),
        "shape": list(signal.shape), "shell_bvals": shells.tolist(),
        "device": str(device), "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "seconds": seconds,
        "peak_torch_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        ),
        "responses": {}, "masks": {}, "maps": {},
    }
    for tissue, candidate in (("wm", wm), ("gm", gm), ("csf", csf)):
        path = getattr(args, tissue + "_response")
        paths[tissue + "_response"] = path
        reference = np.loadtxt(path, comments="#", ndmin=2)
        report["responses"][tissue] = _numeric(candidate.cpu().numpy(), reference)
        report["responses"][tissue]["candidate"] = candidate.cpu().numpy().tolist()
        report["responses"][tissue]["reference"] = reference.tolist()
    for name in MASK_NAMES:
        path = args.reference_dir / (name + ".nii.gz")
        paths["ref_" + name] = path
        reference = _aligned(path, image).astype(bool)
        candidate = masks[name].cpu().numpy()
        report["masks"][name] = {
            "candidate_voxels": int(candidate.sum()),
            "reference_voxels": int(reference.sum()),
            "xor_voxels": int(np.count_nonzero(candidate ^ reference)),
        }
    for name in MAP_NAMES:
        ref_name = {"fa": "safe_fa", "metric_sfwm2": "metric_sfwm2",
                    "metric_sfwm6": "metric_sfwm6"}.get(name, name)
        path = args.reference_dir / (ref_name + ".nii.gz")
        paths["ref_" + name] = path
        reference = _aligned(path, image)
        mask_name = ("refined_wm" if name == "metric_sfwm2" else "refined_sfwm"
                     if name == "metric_sfwm6" else "safe_mask")
        comparison_mask = _aligned(args.reference_dir / (mask_name + ".nii.gz"), image).astype(bool)
        report["maps"][name] = _numeric(
            masks[name].cpu().numpy(), reference, comparison_mask,
        )
    report["input_sha256"] = {key: _sha(path) for key, path in paths.items()}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("seconds", "peak_torch_allocated_gib",
          "responses", "masks", "maps")}, indent=2))


if __name__ == "__main__":
    main()
