"""Compare PyTorch Dhollander FA/SDM tissue masks with official saved masks.

Reference: dwi2response dhollander corrected.mif wm.txt gm.txt csf.txt
-mask mask.nii.gz -scratch scratch -nocleanup.
This script fixes official FA, safe SDM, and safe mask, isolating tissue selection.
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

from fnit.connectome.response import segment_mrtrix_dhollander

NAMES = ("crude_wm", "crude_gm", "crude_csf", "refined_wm",
         "refined_gm", "refined_csf", "voxels_gm", "voxels_csf")


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
    for name in ("dwi", "safe-mask", "safe-sdm", "safe-fa", "ref-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    target = nib.load(args.dwi)
    mask = _aligned(args.safe_mask, target).astype(bool)
    sdm = _aligned(args.safe_sdm, target).astype(np.float32)
    fa = _aligned(args.safe_fa, target).astype(np.float32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    masks = segment_mrtrix_dhollander(
        torch.as_tensor(fa, device=device),
        torch.as_tensor(sdm, device=device),
        torch.as_tensor(mask, device=device),
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    metrics = {}
    paths = {"dwi": args.dwi, "safe_mask": args.safe_mask,
             "safe_sdm": args.safe_sdm, "safe_fa": args.safe_fa}
    for name in NAMES:
        reference_path = args.ref_dir / (name + ".nii.gz")
        reference = _aligned(reference_path, target).astype(bool)
        candidate = masks[name].cpu().numpy()
        metrics[name] = {
            "candidate_voxels": int(candidate.sum()),
            "reference_voxels": int(reference.sum()),
            "xor_voxels": int(np.count_nonzero(candidate ^ reference)),
            "dice": float(2 * np.count_nonzero(candidate & reference)
                          / (candidate.sum() + reference.sum())),
        }
        paths["ref_" + name] = reference_path
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm, corrected DWI",
        "scope": "Dhollander FA/SDM crude and refined tissue masks, plus final GM/CSF voxels; official FA and SDM fixed",
        "reference_command": __doc__.split("Reference: ", 1)[1].split("\nThis script", 1)[0].strip(),
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "shape": list(fa.shape),
        "seconds": seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                     if device.type == "cuda" else None),
        "input_sha256": {name: _sha(path) for name, path in paths.items()},
        "metrics": metrics,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("seconds", "peak_torch_allocated_gib", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
