"""Compare MRtrix Dhollander IWLS tensor FA and directions on corrected real DWI.

Reference:
dwi2tensor dwi.mif - -mask safe_mask.mif |
  tensor2metric - -fa safe_fa.mif -vector safe_vecs.mif
  -modulate none -mask safe_mask.mif
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

from fnit.connectome.response import fit_mrtrix_dhollander_tensor


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data(path: Path, target: nib.spatialimages.SpatialImage) -> np.ndarray:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    orientation = ornt_transform(io_orientation(image.affine), io_orientation(target.affine))
    aligned_affine = image.affine @ inv_ornt_aff(orientation, data.shape[:3])
    if not np.allclose(aligned_affine, target.affine, atol=5e-5):
        raise ValueError(f"{path}: affine cannot align to DWI by orientation")
    return np.ascontiguousarray(apply_orientation(data, orientation))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "safe-mask", "ref-fa",
                 "ref-vec", "ref-crude-wm", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in
             ("dwi", "grad", "safe_mask", "ref_fa", "ref_vec", "ref_crude_wm")}
    image = nib.load(args.dwi)
    data = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    grad = np.loadtxt(args.grad)
    mask = _data(args.safe_mask, image).astype(bool)
    ref_fa = _data(args.ref_fa, image)
    ref_vec = _data(args.ref_vec, image)
    ref_crude_wm = _data(args.ref_crude_wm, image).astype(bool)
    device = torch.device(args.device)
    signal = torch.as_tensor(data, device=device)
    safe = torch.as_tensor(mask, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fa, vec = fit_mrtrix_dhollander_tensor(signal, grad, safe, args.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    fa = fa.cpu().numpy()
    vec = vec.cpu().numpy()
    error = np.abs(fa[mask].astype(np.float64) - ref_fa[mask].astype(np.float64))
    candidate_crude = mask & (fa > 0.2)
    candidate_vec = vec[mask].astype(np.float64)
    reference_vec = ref_vec[mask].astype(np.float64)
    dot = np.einsum("ij,ij->i", candidate_vec, reference_vec)
    dot /= (np.linalg.norm(candidate_vec, axis=1)
            * np.linalg.norm(reference_vec, axis=1)).clip(1e-15)
    dot = np.clip(dot, -1, 1)
    report = {
        "dataset": "OpenNeuro ds004666 sub-01 ses-2mm, corrected DWI",
        "scope": "default dwi2tensor -iter 2 and tensor2metric FA/principal direction within official Dhollander safe mask",
        "reference_command": __doc__.split("Reference:\n", 1)[1].strip(),
        "device": str(device),
        "torch": torch.__version__,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "shape": list(data.shape),
        "masked_voxels": int(mask.sum()),
        "seconds": seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                     if device.type == "cuda" else None),
        "input_sha256": {name: _sha(path) for name, path in paths.items()},
        "fa": {
            "mae": float(error.mean()),
            "max_abs": float(error.max()),
            "fraction_abs_le_1e-5": float(np.mean(error <= 1e-5)),
            "pearson": float(np.corrcoef(fa[mask], ref_fa[mask])[0, 1]),
        },
        "principal_direction": {
            "mean_abs_cosine": float(np.abs(dot).mean()),
            "min_abs_cosine": float(np.abs(dot).min()),
            "fraction_abs_cosine_ge_0_999": float(np.mean(np.abs(dot) >= 0.999)),
        },
        "crude_wm_fa_gt_0_2": {
            "candidate_voxels": int(candidate_crude.sum()),
            "reference_voxels": int(ref_crude_wm.sum()),
            "xor_voxels": int(np.count_nonzero(candidate_crude ^ ref_crude_wm)),
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("seconds", "peak_torch_allocated_gib", "fa",
                       "principal_direction", "crude_wm_fa_gt_0_2")}, indent=2))


if __name__ == "__main__":
    main()
