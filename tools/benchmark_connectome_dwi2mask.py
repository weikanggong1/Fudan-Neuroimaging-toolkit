"""Compare the native legacy DWI mask with a pinned MRtrix reference on real DWI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.masks import dwi2mask_legacy
from fnit.connectome.response import mrtrix_shell_centres


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi", required=True, type=Path)
    parser.add_argument("--grad-mrtrix", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-mask", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(8)
    image = nib.load(str(args.dwi))
    reference = nib.load(str(args.reference_mask))
    if image.shape[:3] != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
        raise ValueError("DWI and reference mask grids differ")
    grad = np.loadtxt(args.grad_mrtrix, dtype=np.float64)
    if grad.shape != (image.shape[-1], 4):
        raise ValueError("MRtrix gradient shape differs from DWI")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    signal = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32), device=device)
    gradient = torch.as_tensor(grad, dtype=torch.float64, device=device)
    shell_means, _, _, _ = mrtrix_shell_centres(gradient)
    mask = dwi2mask_legacy(signal, gradient[:, 3], shell_means)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    actual = mask.cpu().numpy()
    if args.output_mask is not None:
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(actual.astype(np.uint8), image.affine, header),
                 str(args.output_mask))
    expected = np.asanyarray(reference.dataobj).astype(bool)
    xor = int(np.count_nonzero(actual ^ expected))
    intersection = int(np.count_nonzero(actual & expected))
    report = {
        "shape": list(actual.shape),
        "shell_bvalues": shell_means.cpu().tolist(),
        "reference_voxels": int(expected.sum()),
        "candidate_voxels": int(actual.sum()),
        "xor_voxels": xor,
        "dice": 2 * intersection / (int(expected.sum()) + int(actual.sum())),
        "wall_seconds_including_io": elapsed,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
