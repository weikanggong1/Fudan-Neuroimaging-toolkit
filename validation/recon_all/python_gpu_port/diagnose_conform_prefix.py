"""Compare the successive, real-data operations of the fixed T1 conform path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.conform_gpu import (
    _coronal_affine, _inverse32, _multiply32, _source_affine, _uchar_from_float,
)


def _volume(path: Path) -> tuple[nib.MGHImage, np.ndarray]:
    image = nib.load(str(path))
    if not isinstance(image, nib.MGHImage):
        raise ValueError(f"expected MGH/MGZ: {path}")
    return image, np.asarray(image.dataobj)


def _difference(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {"reference_shape": reference.shape, "candidate_shape": candidate.shape}
    error = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    nonzero = error > 0
    return {
        "different_voxels": int(nonzero.sum()),
        "maximum_absolute_difference": float(error.max()),
        "p99_absolute_difference": float(np.percentile(error, 99)),
    }


def _candidate_samples(source: np.ndarray, matrix: np.ndarray,
                       reference_float: np.ndarray, reference_orig: np.ndarray,
                       candidate_orig: np.ndarray, points: np.ndarray) -> list[dict]:
    """Record the current candidate's float32 sample coordinates at differing voxels."""
    samples = []
    for point in points:
        x, y, z = (np.float32(axis) for axis in point)
        ijk = [np.float32(np.float32(np.float32(matrix[row, 0] * x)
                                       + np.float32(matrix[row, 1] * y))
                          + np.float32(matrix[row, 2] * z)) + matrix[row, 3]
               for row in range(3)]
        base = np.floor(ijk).astype(int)
        fraction = np.asarray(ijk, dtype=np.float32) - base
        value = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    neighbor = base + (dx, dy, dz)
                    if np.any(neighbor < 0) or np.any(neighbor >= source.shape):
                        continue
                    weight = np.prod([float(fraction[i] if delta else 1.0 - fraction[i])
                                      for i, delta in enumerate((dx, dy, dz))])
                    value += weight * float(source[tuple(neighbor)])
        reference_value = float(reference_float[tuple(point)])
        samples.append({"target_voxel": point.tolist(), "source_float_coordinate":
                        [float(axis) for axis in ijk], "prequantized_double_sum": value,
                        "reference_prequantized_float": reference_value,
                        "prequantized_difference": value - reference_value,
                        "reference_uint8": int(reference_orig[tuple(point)]),
                        "candidate_uint8": int(candidate_orig[tuple(point)]),
                        "distance_to_half_integer": abs(value - np.floor(value) - 0.5)})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_raw", "candidate_raw", "reference_scaled",
                 "reference_float", "reference_orig", "candidate_orig", "report"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    args = parser.parse_args()
    ref_raw_image, ref_raw = _volume(args.reference_raw)
    cand_raw_image, cand_raw = _volume(args.candidate_raw)
    ref_scaled_image, ref_scaled = _volume(args.reference_scaled)
    ref_float_image, ref_float = _volume(args.reference_float)
    ref_orig_image, ref_orig = _volume(args.reference_orig)
    cand_orig_image, cand_orig = _volume(args.candidate_orig)
    candidate_scaled = _uchar_from_float(
        torch.from_numpy(cand_raw.astype(np.float32, copy=True))).numpy()
    affine = _coronal_affine(cand_raw_image, cand_orig.shape[0])
    transform = _multiply32(_inverse32(_source_affine(cand_raw_image)), affine)
    different = np.argwhere(ref_orig != cand_orig)
    predicted_reference = np.floor(np.clip(ref_float, 0, 255) + 0.5).astype(np.uint8)
    report = {
        "code_commit": args.code_commit, "host": platform.node(),
        "cpu": platform.processor(), "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "reference_binary_sha256": hashlib.sha256(args.reference_binary.read_bytes()).hexdigest(),
        "inputs_sha256": {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
                          for key in ("reference_raw", "candidate_raw")},
        "raw_voxels": _difference(ref_raw, cand_raw),
        "raw_geometry_max_mm": float(np.max(np.abs(ref_raw_image.affine - cand_raw_image.affine))),
        "scaled_source_voxels": _difference(ref_scaled, candidate_scaled),
        "scaled_source_geometry_max_mm": float(np.max(np.abs(ref_scaled_image.affine - cand_raw_image.affine))),
        "candidate_target_affine": affine.tolist(),
        "candidate_target_to_source_voxel": transform.tolist(),
        "conformed_voxels": _difference(ref_orig, cand_orig),
        "conformed_geometry_max_mm": float(np.max(np.abs(ref_orig_image.affine - cand_orig_image.affine))),
        "reference_float_geometry_max_mm": float(np.max(np.abs(ref_float_image.affine - ref_orig_image.affine))),
        "reference_float_roundtrip": _difference(ref_orig, predicted_reference),
        "output_dtype": {"reference": str(ref_orig.dtype), "candidate": str(cand_orig.dtype)},
        "different_voxel_samples": _candidate_samples(
            candidate_scaled, transform, ref_float, ref_orig, cand_orig, different),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("raw_voxels", "scaled_source_voxels", "conformed_voxels",
                       "reference_float_roundtrip")}, indent=2))


if __name__ == "__main__":
    main()
