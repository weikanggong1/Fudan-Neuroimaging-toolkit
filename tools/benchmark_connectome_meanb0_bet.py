"""Validate Torch mean b0 and BET against the original real-DWI reference."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit.connectome.bet import bet_mask, mean_bzero, mrtrix_roundtrip_voxel_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi", required=True, type=Path)
    parser.add_argument("--bvals", required=True, type=Path)
    parser.add_argument("--reference-mean-b0", required=True, type=Path)
    parser.add_argument("--reference-mask", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-mean-b0", type=Path)
    parser.add_argument("--output-mask", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = perf_counter()
    image = nib.load(str(args.dwi))
    reference_mean = nib.load(str(args.reference_mean_b0))
    reference_mask = nib.load(str(args.reference_mask))
    if (len(image.shape) != 4 or image.shape[:3] != reference_mean.shape
            or image.shape[:3] != reference_mask.shape):
        raise ValueError("4D DWI and 3D references must have matching spatial shape")
    if nib.aff2axcodes(image.affine) != ("L", "A", "S"):
        raise ValueError("BET input must use an LAS NIfTI voxel grid")
    bvalues = np.asarray(np.loadtxt(args.bvals), dtype=np.float64).reshape(-1)
    if len(bvalues) != image.shape[-1]:
        raise ValueError("bval count differs from DWI volume count")
    bzero = bvalues < 50
    if not np.any(bzero):
        raise ValueError("no b=0 volumes selected")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    signal = torch.as_tensor(np.asarray(image.dataobj, dtype=np.float32), device=device)
    stage_start = perf_counter()
    mean_b0 = mean_bzero(dwi=signal, bvalues=torch.as_tensor(bvalues, device=device),
                          bzero_threshold=50.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mean_seconds = perf_counter() - stage_start
    mrtrix_zooms = mrtrix_roundtrip_voxel_size(image.header.get_zooms()[:3])
    bet_start = perf_counter()
    mask = bet_mask(mean_b0=mean_b0, voxel_size=mrtrix_zooms,
                    fractional_threshold=0.2, vertical_gradient=-0.05, robust_center=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    bet_seconds = perf_counter() - bet_start
    stage_seconds = perf_counter() - stage_start
    mean_array = mean_b0.cpu().numpy()
    actual = mask.cpu().numpy()
    expected_mean = np.asarray(reference_mean.dataobj, dtype=np.float32)
    expected_mask = np.asarray(reference_mask.dataobj).astype(bool)
    if args.output_mean_b0 is not None:
        args.output_mean_b0.parent.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_shape(image.shape[:3])
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(mean_array, image.affine, header), str(args.output_mean_b0))
    if args.output_mask is not None:
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_shape(image.shape[:3])
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(actual.astype(np.uint8), image.affine, header),
                 str(args.output_mask))
    delta = np.abs(mean_array.astype(np.float64) - expected_mean.astype(np.float64))
    intersection = int(np.count_nonzero(actual & expected_mask))
    report = {
        "spatial_shape": list(image.shape[:3]),
        "dwi_volumes": image.shape[3],
        "bzero_volumes": int(bzero.sum()),
        "bzero_threshold": 50,
        "mean_reduction": "float64 accumulate, float32 output",
        "compute_device": str(device),
        "bet_voxel_size_source": "MRtrix MIF 6-significant-digit then NIfTI float32",
        "input_reference_zoom_max_abs_mm": float(np.max(np.abs(np.asarray(image.header.get_zooms()[:3]) - np.asarray(reference_mean.header.get_zooms()[:3])))),
        "native_reference_zoom_max_abs_mm": float(np.max(np.abs(np.asarray(mrtrix_zooms) - np.asarray(reference_mean.header.get_zooms()[:3])))),
        "mean_b0_max_abs": float(delta.max()),
        "mean_b0_mae": float(delta.mean()),
        "mean_b0_exact_voxels": int(np.count_nonzero(delta == 0)),
        "reference_mask_voxels": int(expected_mask.sum()),
        "candidate_mask_voxels": int(actual.sum()),
        "mask_xor_voxels": int(np.count_nonzero(actual ^ expected_mask)),
        "mask_dice": 2 * intersection / (int(actual.sum()) + int(expected_mask.sum())),
        "mean_bzero_seconds": mean_seconds,
        "bet_mask_seconds": bet_seconds,
        "torch_mean_plus_bet_seconds": stage_seconds,
        "wall_seconds_including_io_and_comparison": perf_counter() - start,
        "peak_gpu_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                    if device.type == "cuda" else None),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
