"""Compare PyTorch Dhollander mask and SDM preparation with MRtrix on real DWI.

MRtrix reference is dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt
-mask mask.nii.gz -scratch dhollander_scratch -nocleanup. Its eroded_mask.mif,
safe_mask.mif, and safe_sdm.mif are converted to NIfTI with mrconvert.
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

from fnit.connectome.response import prepare_mrtrix_dhollander_sdm


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aligned(path: Path, target: nib.spatialimages.SpatialImage) -> np.ndarray:
    source = nib.load(path)
    data = np.asanyarray(source.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    orientation = ornt_transform(io_orientation(source.affine), io_orientation(target.affine))
    aligned_affine = source.affine @ inv_ornt_aff(orientation, data.shape)
    if not np.allclose(aligned_affine, target.affine, atol=5e-5):
        raise ValueError(f"{path}: affine cannot align by orientation")
    return np.ascontiguousarray(apply_orientation(data, orientation))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "response", "mask", "ref-eroded",
                 "ref-safe", "ref-sdm", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in
             ("dwi", "grad", "response", "mask", "ref_eroded", "ref_safe", "ref_sdm")}
    dwi_image = nib.load(args.dwi)
    dwi = np.asanyarray(dwi_image.dataobj).astype(np.float32, copy=False)
    mask = _aligned(args.mask, dwi_image).astype(bool)
    reference_eroded = _aligned(args.ref_eroded, dwi_image).astype(bool)
    reference_safe = _aligned(args.ref_safe, dwi_image).astype(bool)
    reference_sdm = _aligned(args.ref_sdm, dwi_image).astype(np.float32)
    grad = np.loadtxt(args.grad)
    first = args.response.open().readline().strip()
    if not first.startswith("# Shells:"):
        raise ValueError("response file lacks shell header")
    shells = np.fromstring(first.split(":", 1)[1], sep=",")
    device = torch.device(args.device)
    dwi_tensor = torch.as_tensor(dwi, device=device)
    mask_tensor = torch.as_tensor(mask, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    eroded, safe, sdm = prepare_mrtrix_dhollander_sdm(
        dwi_tensor, grad, shells, mask_tensor,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    eroded = eroded.cpu().numpy()
    safe = safe.cpu().numpy()
    sdm = sdm.cpu().numpy()
    error = np.abs(sdm.astype(np.float64) - reference_sdm.astype(np.float64))
    common = safe & reference_safe
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm, corrected DWI",
        "scope": "Dhollander six-neighbour mask erosion and shell-weighted signal decay metric only",
        "reference_command": "dwi2response dhollander corrected.mif response_wm.txt response_gm.txt response_csf.txt -mask brain_mask.nii.gz -scratch dhollander_scratch -nocleanup",
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "shape": list(dwi.shape),
        "shell_bvals": shells.tolist(),
        "seconds": seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                     if device.type == "cuda" else None),
        "input_sha256": {key: _sha(path) for key, path in paths.items()},
        "eroded": {
            "candidate_voxels": int(eroded.sum()),
            "reference_voxels": int(reference_eroded.sum()),
            "xor_voxels": int(np.count_nonzero(eroded ^ reference_eroded)),
        },
        "safe": {
            "candidate_voxels": int(safe.sum()),
            "reference_voxels": int(reference_safe.sum()),
            "xor_voxels": int(np.count_nonzero(safe ^ reference_safe)),
        },
        "sdm": {
            "all_voxels_mae": float(error.mean()),
            "all_voxels_max_abs": float(error.max()),
            "common_safe_voxels": int(common.sum()),
            "common_safe_mae": float(error[common].mean()) if common.any() else None,
            "common_safe_max_abs": float(error[common].max()) if common.any() else None,
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("seconds", "peak_torch_allocated_gib", "eroded", "safe", "sdm")},
                     indent=2))


if __name__ == "__main__":
    main()
