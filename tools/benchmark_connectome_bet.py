"""Compare native PyTorch BET with an independently generated real-image FSL mask."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.bet import bet_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-b0", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-mask", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fractional-threshold", type=float, default=0.2)
    parser.add_argument("--vertical-gradient", type=float, default=-0.05)
    parser.add_argument("--robust-center", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = perf_counter()
    image = nib.load(str(args.mean_b0))
    reference = nib.load(str(args.reference_mask))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
        raise ValueError("mean b0 and reference mask must share a NIfTI grid")
    if nib.aff2axcodes(image.affine) != ("L", "A", "S"):
        raise ValueError("this BET comparison currently requires an LAS NIfTI voxel grid")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    signal = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32), device=device)
    core_start = perf_counter()
    mask = bet_mask(mean_b0=signal, voxel_size=image.header.get_zooms()[:3],
                    fractional_threshold=args.fractional_threshold,
                    vertical_gradient=args.vertical_gradient, robust_center=args.robust_center)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    core_seconds = perf_counter() - core_start
    actual = mask.cpu().numpy()
    expected = np.asarray(reference.dataobj).astype(bool)
    total_seconds = perf_counter() - start
    if args.output_mask is not None:
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(actual.astype(np.uint8), image.affine, header),
                 str(args.output_mask))
    intersection = int(np.count_nonzero(actual & expected))
    report = {
        "shape": list(image.shape),
        "voxel_size_mm": [float(v) for v in image.header.get_zooms()[:3]],
        "fractional_threshold": args.fractional_threshold,
        "vertical_gradient": args.vertical_gradient,
        "robust_center": args.robust_center,
        "reference_voxels": int(expected.sum()),
        "candidate_voxels": int(actual.sum()),
        "xor_voxels": int(np.count_nonzero(actual ^ expected)),
        "dice": 2 * intersection / (int(actual.sum()) + int(expected.sum())),
        "core_seconds": core_seconds,
        "wall_seconds_including_io_and_comparison": total_seconds,
        "peak_gpu_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                    if device.type == "cuda" else None),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
