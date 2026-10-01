"""Fit FNIT ICA to the original MELODIC input and compare original outputs.

The caller supplies an existing original MELODIC directory. This validator
does not invoke FSL and never places input paths or voxel data in its report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "min": float(values.min()),
            "max": float(values.max())}


def normalised_columns(values):
    values = np.asarray(values, dtype=np.float64).copy()
    values -= values.mean(axis=0)
    return values / np.maximum(np.linalg.norm(values, axis=0), 1e-30)


def load_core(path):
    if path is None:
        from fnit.melodic import ica
        return ica
    specification = importlib.util.spec_from_file_location("fnit_ica_control", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bold", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    core = load_core(args.core)
    core_path = Path(inspect.getfile(core))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    selected_device = torch.device(args.device)
    if selected_device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(selected_device)

    started = time.perf_counter()
    result = core.decompose_spatial_ica(
        args.input_bold, args.brain_mask, args.output_dir,
        n_components=None, device=args.device, random_state=0,
        max_iter=500, tolerance=0.001, mm_threshold=0.5,
    )
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    fit_wall_seconds = time.perf_counter() - started
    mixing = np.loadtxt(result.mixing, ndmin=2)
    reference_mixing = np.loadtxt(args.reference_dir / "melodic_mix", ndmin=2)
    temporal_correlation = normalised_columns(mixing).T @ normalised_columns(reference_mixing)
    candidate_columns, reference_columns = linear_sum_assignment(-np.abs(temporal_correlation))
    signed_r = temporal_correlation[candidate_columns, reference_columns]
    signs = np.sign(signed_r)
    aligned_mixing = mixing[:, candidate_columns] * signs
    aligned_reference = reference_mixing[:, reference_columns]
    temporal_scale = np.sum(aligned_mixing * aligned_reference, axis=0) / np.sum(aligned_mixing ** 2, axis=0)

    mask_image = nib.load(args.brain_mask)
    mask = np.asarray(mask_image.dataobj) > 0
    candidate_image = nib.load(result.component_maps)
    reference_image = nib.load(args.reference_dir / "melodic_IC.nii.gz")
    if (reference_image.shape[:3] != mask.shape or
            not np.allclose(reference_image.affine, mask_image.affine, atol=1e-4)):
        raise ValueError("original MELODIC maps and the input mask must share a grid")
    candidate_maps = np.asarray(candidate_image.dataobj)[mask][:, candidate_columns] * signs
    reference_maps = np.asarray(reference_image.dataobj)[mask][:, reference_columns]
    spatial_r = np.sum(normalised_columns(candidate_maps) * normalised_columns(reference_maps), axis=0)
    spatial_rmse = np.sqrt(np.mean((candidate_maps.astype(np.float64) - reference_maps) ** 2, axis=0))
    candidate_thresholded = np.asarray(nib.load(result.thresholded_maps).dataobj)[mask][:, candidate_columns]
    reference_thresholded = np.column_stack([
        np.asarray(nib.load(args.reference_dir / f"stats/thresh_zstat{column + 1}.nii.gz").dataobj)[mask]
        for column in reference_columns
    ])
    candidate_support, reference_support = candidate_thresholded != 0, reference_thresholded != 0
    dice = 2 * (candidate_support & reference_support).sum(axis=0) / np.maximum(
        1, candidate_support.sum(axis=0) + reference_support.sum(axis=0)
    )
    frequency = np.loadtxt(result.frequency_power, ndmin=2)[:, candidate_columns]
    reference_frequency = np.loadtxt(args.reference_dir / "melodic_FTmix", ndmin=2)[:, reference_columns]
    peak_bytes = (torch.cuda.max_memory_allocated(selected_device)
                  if selected_device.type == "cuda" else None)
    report = {
        "schema_version": 1,
        "protocol": "Same original pre-ICA BOLD and EPI mask; independent FNIT ICA fit only.",
        "input_shape": list(nib.load(args.input_bold).shape),
        "input_sha256": sha256(args.input_bold), "mask_sha256": sha256(args.brain_mask),
        "ica_source_sha256": sha256(core_path), "driver_sha256": sha256(__file__),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "nibabel": nib.__version__, "device": args.device,
                        "cuda": torch.version.cuda, "allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "precision": {"input_storage": "float32", "pca_covariance_whitening_and_fastica": "float64",
                      "output_spatial_maps": "float32", "gamma_histogram_coordinates": "float64",
                      "gamma_scalar_parameters_and_log_likelihood": "float32 as in FSL 2601.1"},
        "wall_seconds": fit_wall_seconds, "peak_cuda_allocated_bytes": peak_bytes,
        "components": {"fnit": result.n_components, "native": reference_mixing.shape[1]},
        "iterations": result.n_iterations, "converged": result.converged,
        "change": result.final_decorrelation_change, "resels": result.estimated_resels,
        "variance_explained": result.pca_variance_explained,
        "pairing": {"count": len(candidate_columns),
                    "same_order": bool(np.array_equal(candidate_columns, reference_columns)),
                    "all_positive_signs": bool((signs == 1).all())},
        "temporal_abs_pearson": distribution(np.abs(signed_r)),
        "temporal_matched_rmse": float(np.sqrt(np.mean((aligned_mixing - aligned_reference) ** 2))),
        "temporal_scale": distribution(temporal_scale),
        "spatial_abs_pearson": distribution(np.abs(spatial_r)), "spatial_rmse": distribution(spatial_rmse),
        "threshold_support_dice": distribution(dice),
        "frequency_power_matched_relative_l2_error": float(np.linalg.norm(frequency - reference_frequency) /
                                                          np.linalg.norm(reference_frequency)),
        "output_sha256": {name: sha256(getattr(result, name)) for name in
                          ("mixing", "frequency_power", "component_maps", "thresholded_maps", "posterior_maps")},
        "limits": ["One real run; component matching removes permutation and sign ambiguity.",
                   "Wall time includes FNIT decomposition and file writing; shared GPU load was not controlled.",
                   "No original FSL invocation is timed by this validator."],
        "privacy": "Anonymous summaries and file hashes only; no subject IDs, input paths or voxel data.",
    }
    np.savez(args.output_dir / "matched_components.private.npz", fnit=candidate_columns,
             native=reference_columns, sign=signs, temporal_r=np.abs(signed_r), spatial_r=spatial_r,
             threshold_dice=dice)
    report_path = args.output_dir / "report.public.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
