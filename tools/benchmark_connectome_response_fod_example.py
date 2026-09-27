"""Generate a paired same-slice PyTorch versus MRtrix FOD example image.

Reference: dwi2fod msmt_csd corrected.mif response_wm.txt wm_fod.mif
    response_gm.txt gm.mif response_csf.txt csf.mif -mask brain_mask.nii.gz
Use the exact same response files, exported gradient scheme and DWI in both
arms. The input mask is reoriented into the DWI grid by its NIfTI affine.
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

from fnit.connectome.fod import fit_mrtrix_msmt_csd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dwi", "grad", "wm-response", "gm-response", "csf-response",
                 "mask", "ref-wm", "png", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--slice", type=int, default=36)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    dwi = nib.load(args.dwi)
    mask_image = nib.load(args.mask)
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    if mask.ndim == 4:
        mask = mask[..., 0]
    transform = ornt_transform(io_orientation(mask_image.affine), io_orientation(dwi.affine))
    aligned_affine = mask_image.affine @ inv_ornt_aff(transform, mask.shape)
    if not np.allclose(aligned_affine, dwi.affine, atol=5e-5):
        raise ValueError("mask cannot align to DWI by orientation")
    mask = np.ascontiguousarray(apply_orientation(mask, transform))
    z = args.slice
    signal = np.asanyarray(dwi.dataobj)[:, :, z:z+1, :].astype(np.float32)
    ref_image = nib.load(args.ref_wm)
    if not np.allclose(dwi.affine, ref_image.affine, atol=1e-5):
        raise ValueError("reference FOD affine differs from DWI")
    ref = np.asanyarray(ref_image.dataobj)[:, :, z, 0]
    responses = [np.loadtxt(path, comments="#", ndmin=2) for path in
                 (args.wm_response, args.gm_response, args.csf_response)]
    first = args.wm_response.open().readline().strip()
    if not first.startswith("# Shells:"):
        raise ValueError("WM response lacks shell header")
    shells = np.fromstring(first.split(":", 1)[1], sep=",")
    grad = np.loadtxt(args.grad)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    tic = time.perf_counter()
    candidate, _, _ = fit_mrtrix_msmt_csd(
        torch.as_tensor(signal, device=device),
        grad, shells, *responses,
        torch.as_tensor(mask[:, :, z:z+1], device=device),
        batch_size=4096,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - tic
    candidate = candidate[:, :, 0, 0].cpu().numpy()
    error = np.abs(candidate - ref)
    support = mask[:, :, z]
    scale = np.percentile(ref[support], 99)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)
    panels = [
        (signal[:, :, 0, 0], "Corrected b0", "gray", None),
        (ref, "MRtrix WM FOD c00", "viridis", (0, scale)),
        (candidate, "PyTorch WM FOD c00", "viridis", (0, scale)),
        (error, f"Absolute error (max {error[support].max():.2g})", "magma", None),
    ]
    for axis, (data, title, colour, limits) in zip(axes, panels):
        image = np.rot90(data)
        axis.imshow(image, cmap=colour, vmin=None if limits is None else limits[0],
                    vmax=None if limits is None else limits[1])
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle("ds004666 sub-01 ses-2mm, corrected DWI, axial slice " + str(z), fontsize=11)
    fig.savefig(args.png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    digest = hashlib.sha256(args.png.read_bytes()).hexdigest()
    report = {
        "scope": "same DWI, gradients, official response and affine-aligned mask; raw MSMT-CSD axial example",
        "device": str(device),
        "slice_index": z,
        "selected_voxels": int(support.sum()),
        "wm_c00_mae": float(error[support].mean()),
        "wm_c00_max_abs": float(error[support].max()),
        "seconds": seconds,
        "peak_torch_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                     if device.type == "cuda" else None),
        "png_sha256": digest,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
