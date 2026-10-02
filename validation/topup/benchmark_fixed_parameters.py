"""Render saved official TOPUP parameters on their original private b0 pair.

No field/motion optimization or external program is run. Only a JSON report is
written; input images and rendered arrays remain private and in memory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(candidate, reference, mask):
    x = candidate[mask].astype(np.float64).reshape(-1)
    y = reference[mask].astype(np.float64).reshape(-1)
    if x.size == 0:
        return {"values": 0}
    absolute = np.abs(x - y)
    return {"values": int(x.size), "pearson_r": float(np.corrcoef(x, y)[0, 1])
            if x.size > 1 and np.std(x) > 0 and np.std(y) > 0 else None,
            "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean((x-y)**2))),
            "p95_absdiff": float(np.percentile(absolute, 95)), "max_absdiff": float(absolute.max()),
            "array_equal": bool(np.array_equal(x, y)),
            "candidate_mean": float(x.mean()), "reference_mean": float(y.mean()),
            "least_squares_gain_candidate_vs_reference": float(np.dot(x-y.mean(), y-y.mean()) / np.dot(y-y.mean(), y-y.mean())) if np.std(y) > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imain", type=Path, required=True)
    parser.add_argument("--datain", type=Path, required=True)
    parser.add_argument("--coeff", type=Path, required=True)
    parser.add_argument("--movpar", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("report must be new")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.memory_limit_bytes / torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.synchronize(device)
    import fnit.topup.core as core
    started = time.perf_counter()
    rendered = core._render_fixed_topup(args.imain, args.datain, args.coeff, args.movpar, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    imain = nib.load(str(args.imain))
    raw = np.asarray(imain.dataobj, dtype=np.float32)
    mask = rendered["common_mask"]
    regions = {"full_fov": np.ones(raw.shape[:3], bool), "fixed_raw_signal": raw.mean(axis=3) > 100,
               "rendered_common_valid": mask, "outside_rendered_common_valid": ~mask}
    if args.brain_mask is not None:
        image = nib.load(str(args.brain_mask))
        if image.shape != raw.shape[:3] or not np.allclose(image.affine, imain.affine, atol=1e-5, rtol=0):
            raise ValueError("brain mask must use the original input voxel grid")
        regions["fixed_official_brain"] = np.asarray(image.dataobj) > .5
    comparison = {}
    outputs = {"fieldmap_fout.nii.gz": rendered["field_hz"],
               "fieldmap_iout.nii.gz": rendered["corrected"],
               "fieldmap_jacout_01.nii.gz": rendered["jacobians"][..., 0],
               "fieldmap_jacout_02.nii.gz": rendered["jacobians"][..., 1]}
    for name, x in outputs.items():
        image = nib.load(str(args.official_dir / name))
        y = np.asarray(image.dataobj, dtype=np.float32)
        is_jacobian = "jacout" in name
        geometry = x.shape == y.shape and np.allclose(image.header.get_zooms()[:3], imain.header.get_zooms()[:3], atol=1e-5, rtol=0)
        if not is_jacobian:
            geometry = geometry and np.allclose(image.affine, imain.affine, atol=1e-5, rtol=0)
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        row = {"shape": list(x.shape), "reference_shape": list(y.shape),
               "geometry_gate": bool(geometry), "finite": finite}
        if geometry and finite:
            masks = regions
            if is_jacobian and rendered["canonical_x_flip"]:
                masks = {key: value[::-1] for key, value in regions.items()}
            row["errors"] = {label: metrics(x, y, roi) for label, roi in masks.items()}
            if name == "fieldmap_iout.nii.gz":
                row["scan_errors"] = {str(scan): {label: metrics(x[..., scan], y[..., scan], roi) for label, roi in masks.items()} for scan in range(2)}
        comparison[name] = row
    reference_iout = np.asarray(nib.load(str(args.official_dir / "fieldmap_iout.nii.gz")).dataobj)
    reference_support = np.any(reference_iout != 0, axis=3)
    candidate_support = np.any(rendered["corrected"] != 0, axis=3)
    intersection = np.count_nonzero(mask & reference_support)
    denominator = np.count_nonzero(mask) + np.count_nonzero(reference_support)
    nonzero_denominator = np.count_nonzero(candidate_support) + np.count_nonzero(reference_support)
    payload = {"schema_version": 1, "reference_kind": "saved_official_parameters_vs_original_official_outputs",
               "scope": "identical original pair, acquisition file, saved official cubic coefficients and movement parameters; no estimation",
               "parameter_roundtrip_boundary": "official internal double coefficients are saved as float32 and movement as text; original output used internal parameters",
               "private_images_exported": False, "render_read_compute_seconds": elapsed,
               "threads": args.threads, "memory_limit_bytes": args.memory_limit_bytes,
               "input_sha256": sha256(args.imain), "datain_sha256": sha256(args.datain),
               "coefficient_sha256": sha256(args.coeff), "movement_sha256": sha256(args.movpar),
               "core_sha256": sha256(core.__file__), "torch": torch.__version__,
               "driver_sha256": sha256(__file__),
               "source_shape": rendered["source_shape"],
               "source_voxel_sizes": rendered["source_voxel_sizes"],
               "canonical_x_flip": rendered["canonical_x_flip"], "ssd_normalized": rendered["ssd"],
               "valid_voxels": rendered["valid_voxels"],
               "jacobian_le_005_voxels_previously_excluded": int(np.count_nonzero(np.any(rendered["jacobians"] <= .05, axis=3) & (mask[::-1] if rendered["canonical_x_flip"] else mask))),
               "geometric_mask_vs_reference_nonzero_support_dice": float(2 * intersection / denominator) if denominator else 1.0,
               "iout_support_dice": float(2 * np.count_nonzero(candidate_support & reference_support) / nonzero_denominator) if nonzero_denominator else 1.0,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
               "comparisons": comparison}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "complete", "seconds": elapsed, "valid_voxels": rendered["valid_voxels"],
                      "field": comparison["fieldmap_fout.nii.gz"]}), flush=True)


if __name__ == "__main__":
    main()
