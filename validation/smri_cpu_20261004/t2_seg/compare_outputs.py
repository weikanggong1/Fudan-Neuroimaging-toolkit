"""Compare real SynthSeg/WMH outputs without resampling or editing either arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(image):
    return {"shape": list(image.shape), "affine": image.affine.tolist(),
            "dtype": str(image.get_data_dtype()),
            "zooms": [float(v) for v in image.header.get_zooms()],
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"]),
            "extensions": [{"code": int(e.get_code()), "bytes": len(e._raw)}
                           for e in image.header.extensions]}


def compare_segmentation(reference_path, candidate_path):
    reference, candidate = nib.load(reference_path), nib.load(candidate_path)
    result = {"reference_sha256": sha256(reference_path),
              "candidate_sha256": sha256(candidate_path),
              "reference_geometry": geometry(reference),
              "candidate_geometry": geometry(candidate),
              "shape_equal": reference.shape == candidate.shape,
              "affine_max_abs_mm": float(np.abs(reference.affine - candidate.affine).max())}
    if not result["shape_equal"] or result["affine_max_abs_mm"] > 1e-5:
        result["comparison_status"] = "geometry_mismatch"
        return result
    x, y = np.asanyarray(reference.dataobj), np.asanyarray(candidate.dataobj)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite hard labels")
    if np.any(x != np.round(x)) or np.any(y != np.round(y)):
        raise ValueError("Noninteger hard labels")
    voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
    different = int(np.count_nonzero(x != y))
    per_label = []
    for label in sorted(set(int(v) for v in np.unique(x)) | set(int(v) for v in np.unique(y))):
        first, second = x == label, y == label
        ref_count, candidate_count = int(first.sum()), int(second.sum())
        overlap = int(np.count_nonzero(first & second))
        volume_difference = (candidate_count - ref_count) * voxel_volume
        per_label.append({"label": label, "reference_voxels": ref_count,
                          "candidate_voxels": candidate_count,
                          "dice": 2 * overlap / (ref_count + candidate_count),
                          "reference_hard_volume_mm3": ref_count * voxel_volume,
                          "candidate_hard_volume_mm3": candidate_count * voxel_volume,
                          "hard_volume_signed_difference_mm3": volume_difference,
                          "hard_volume_relative_difference":
                          (candidate_count - ref_count) / ref_count if ref_count else None})
    foreground = [row["dice"] for row in per_label if row["label"] != 0]
    result.update({"comparison_status": "compared", "different_voxels": different,
                   "voxel_count": int(x.size), "voxel_agreement": 1 - different / x.size,
                   "per_label": per_label, "foreground_minimum_dice": min(foreground),
                   "foreground_median_dice": float(np.median(foreground))})
    return result


def compare_csv(reference_path, candidate_path):
    def read(path):
        with Path(path).open(newline="") as stream:
            rows = list(csv.reader(stream))
        if len(rows) != 2 or len(rows[0]) != len(rows[1]):
            raise ValueError("Expected one header and one same-length data row")
        values = np.asarray(rows[1][1:], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite CSV volumes")
        return rows[0], values

    names_reference, reference = read(reference_path)
    names_candidate, candidate = read(candidate_path)
    result = {"reference_sha256": sha256(reference_path),
              "candidate_sha256": sha256(candidate_path),
              "column_names_and_order_equal": names_reference == names_candidate,
              "reference_columns": names_reference, "candidate_columns": names_candidate}
    if names_reference != names_candidate:
        result["comparison_status"] = "column_mismatch"
        return result
    difference = candidate - reference
    result.update({"comparison_status": "compared", "numeric_columns": len(reference),
                   "max_absolute_difference_mm3": float(np.abs(difference).max()),
                   "mean_absolute_difference_mm3": float(np.abs(difference).mean()),
                   "columns": [{"name": name, "reference_mm3": float(a),
                                "candidate_mm3": float(b),
                                "signed_difference_mm3": float(b - a),
                                "relative_difference": float((b - a) / a) if a else None}
                               for name, a, b in zip(names_reference[1:], reference, candidate)]})
    return result


def compare_probability(reference_path, candidate_path):
    reference, candidate = nib.load(reference_path), nib.load(candidate_path)
    result = {"reference_sha256": sha256(reference_path),
              "candidate_sha256": sha256(candidate_path),
              "shape_equal": reference.shape == candidate.shape,
              "affine_max_abs_mm": float(np.abs(reference.affine - candidate.affine).max())}
    if not result["shape_equal"] or result["affine_max_abs_mm"] > 1e-5:
        result["comparison_status"] = "geometry_mismatch"
        return result
    x = reference.get_fdata(dtype=np.float64)
    y = candidate.get_fdata(dtype=np.float64)
    difference = y - x
    rms = float(np.sqrt(np.mean(x * x)))
    result.update({"comparison_status": "compared", "finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
                   "different_voxels": int(np.count_nonzero(x != y)),
                   "max_absolute_difference": float(np.abs(difference).max()),
                   "mean_absolute_difference": float(np.abs(difference).mean()),
                   "rmse": float(np.sqrt(np.mean(difference * difference))),
                   "nrmse": float(np.sqrt(np.mean(difference * difference)) / rms) if rms else None,
                   "candidate_range": [float(y.min()), float(y.max())]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--reference-probability", type=Path)
    parser.add_argument("--candidate-probability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "fnit_smri_seg_comparison/v1",
              "comparison_script_sha256": sha256(__file__),
              "segmentation": compare_segmentation(args.reference, args.candidate)}
    if bool(args.reference_csv) != bool(args.candidate_csv):
        parser.error("CSV comparison requires both paths")
    if args.reference_csv:
        report["soft_volumes"] = compare_csv(args.reference_csv, args.candidate_csv)
    if bool(args.reference_probability) != bool(args.candidate_probability):
        parser.error("Probability comparison requires both paths")
    if args.reference_probability:
        report["lesion_probability"] = compare_probability(args.reference_probability,
                                                          args.candidate_probability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report["segmentation"].items()
                      if key in ("comparison_status", "different_voxels", "foreground_minimum_dice")}),
          flush=True)


if __name__ == "__main__":
    main()
