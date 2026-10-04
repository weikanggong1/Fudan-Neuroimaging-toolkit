"""Compare same-input saved SynthStrip/SynthSR outputs in their actual grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_output(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            data = archive["vol_data"]
        return data, None, str(data.dtype)
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj), image.affine, str(image.get_data_dtype())


def scalar_metrics(reference, candidate, region=None):
    first = np.ravel(reference)
    second = np.ravel(candidate)
    region = None if region is None else np.ravel(region)
    counts = n_equal = 0
    abs_sum = squared_sum = ref_squared_sum = signed_sum = 0.0
    maximum = 0.0
    for start in range(0, first.size, 1 << 20):
        stop = min(start + (1 << 20), first.size)
        a = first[start:stop].astype(np.float64)
        b = second[start:stop].astype(np.float64)
        if region is not None:
            keep = region[start:stop]
            a, b = a[keep], b[keep]
        if not a.size:
            continue
        delta = b - a
        counts += a.size
        n_equal += int(np.count_nonzero(delta == 0))
        abs_sum += float(np.abs(delta).sum())
        squared_sum += float(np.square(delta).sum())
        ref_squared_sum += float(np.square(a).sum())
        signed_sum += float(delta.sum())
        maximum = max(maximum, float(np.abs(delta).max()))
    if not counts:
        return {"voxel_count": 0, "status": "empty_region"}
    rmse = np.sqrt(squared_sum / counts)
    reference_rms = np.sqrt(ref_squared_sum / counts)
    return {"status": "measured", "voxel_count": int(counts),
            "different_voxels": int(counts - n_equal), "exact_fraction": n_equal / counts,
            "mae": abs_sum / counts, "rmse": float(rmse),
            "nrmse_reference_rms": float(rmse / reference_rms) if reference_rms else None,
            "max_abs_difference": maximum, "mean_signed_difference": signed_sum / counts}


def compare(reference_path, candidate_path, *, mask=False, region=None, rtol=None, atol=None):
    reference, reference_affine, reference_dtype = load_output(reference_path)
    candidate, candidate_affine, candidate_dtype = load_output(candidate_path)
    same_shape = reference.shape == candidate.shape
    both_npz = reference_affine is None and candidate_affine is None
    affine_max = (None if reference_affine is None or candidate_affine is None else
                  float(np.abs(reference_affine - candidate_affine).max()))
    same_geometry = both_npz or (affine_max is not None and affine_max <= 1e-6)
    report = {"reference_sha256": file_hash(reference_path),
              "candidate_sha256": file_hash(candidate_path),
              "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape),
              "reference_dtype": reference_dtype, "candidate_dtype": candidate_dtype,
              "shape_matches": same_shape, "geometry_matches": same_geometry,
              "dtype_matches": reference_dtype == candidate_dtype,
              "affine_max_abs_mm": affine_max,
              "reference_finite": bool(np.isfinite(reference).all()),
              "candidate_finite": bool(np.isfinite(candidate).all())}
    report["valid_direct_comparison"] = (same_shape and same_geometry and
                                          report["reference_finite"] and report["candidate_finite"])
    if not report["valid_direct_comparison"]:
        return report
    report["whole_grid"] = scalar_metrics(reference, candidate)
    if rtol is not None and atol is not None:
        report["fixed_allclose"] = {"rtol": rtol, "atol": atol,
                                     "passes": bool(np.allclose(candidate, reference, rtol=rtol, atol=atol))}
    if region is not None:
        if region.shape != reference.shape:
            raise ValueError("brain region shape differs from the compared image grid")
        report["mask_union_region"] = scalar_metrics(reference, candidate, region)
    else:
        report["nonzero_union_secondary"] = scalar_metrics(reference, candidate,
                                                            (reference != 0) | (candidate != 0))
    if mask:
        report["reference_binary"] = bool(np.isin(reference, (0, 1)).all())
        report["candidate_binary"] = bool(np.isin(candidate, (0, 1)).all())
        a, b = reference != 0, candidate != 0
        count_a, count_b = int(a.sum()), int(b.sum())
        report["reference_voxels"] = count_a
        report["candidate_voxels"] = count_b
        report["dice"] = 2 * int((a & b).sum()) / (count_a + count_b) if count_a + count_b else None
        report["both_nonempty"] = bool(count_a and count_b)
        voxel_mm3 = abs(float(np.linalg.det(reference_affine[:3, :3]))) if reference_affine is not None else None
        report["voxel_mm3"] = voxel_mm3
        report["reference_volume_mm3"] = count_a * voxel_mm3 if voxel_mm3 is not None else None
        report["candidate_volume_mm3"] = count_b * voxel_mm3 if voxel_mm3 is not None else None
        report["relative_volume_difference"] = (count_b - count_a) / count_a if count_a else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", choices=("synthstrip", "synthsr"), required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--candidate-image", type=Path, required=True)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--candidate-mask", type=Path)
    parser.add_argument("--reference-distance", type=Path)
    parser.add_argument("--candidate-distance", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate", choices=("measure_only", "strip_official", "strip_candidate",
                                            "sr_official", "sr_candidate"), default="measure_only")
    args = parser.parse_args()
    if args.gate != "measure_only" and not args.gate.startswith("strip_" if args.feature == "synthstrip" else "sr_"):
        parser.error("gate must match the selected feature")
    if args.report.exists():
        parser.error("preserve existing reports and use a fresh report path")
    if args.feature == "synthstrip" and not all((args.reference_mask, args.candidate_mask,
                                                  args.reference_distance, args.candidate_distance)):
        parser.error("SynthStrip comparison needs both mask and both distance outputs")
    region = None
    outputs = {}
    if args.feature == "synthstrip":
        outputs["mask"] = compare(args.reference_mask, args.candidate_mask, mask=True)
        if outputs["mask"]["valid_direct_comparison"]:
            region = (load_output(args.reference_mask)[0] != 0) | (load_output(args.candidate_mask)[0] != 0)
        outputs["distance"] = compare(args.reference_distance, args.candidate_distance, region=region,
                                      rtol=1e-5 if args.gate != "measure_only" else None,
                                      atol=1e-4 if args.gate != "measure_only" else None)
    image_atol = 1e-3 if args.gate == "sr_official" else 1e-4
    floating_sr = args.feature == "synthsr" and args.reference_image.suffix == ".npz"
    outputs["image"] = compare(args.reference_image, args.candidate_image, region=region,
                               rtol=1e-5 if floating_sr and args.gate != "measure_only" else None,
                               atol=image_atol if floating_sr and args.gate != "measure_only" else None)
    report = {"schema": "fnit.smri.cpu.strip_sr.comparison.v1", "feature": args.feature,
              "case": args.case, "worker_sha256": file_hash(__file__), "outputs": outputs,
              "all_direct_comparisons_valid": all(row["valid_direct_comparison"] for row in outputs.values()),
              "numerical_equivalence": "not_assessed_no_posthoc_tolerance"}
    if args.gate != "measure_only":
        valid = report["all_direct_comparisons_valid"] and all(row["dtype_matches"] for row in outputs.values())
        exact = lambda row: row.get("whole_grid", {}).get("different_voxels") == 0
        if args.feature == "synthstrip":
            mask_valid = outputs["mask"].get("reference_binary", False) and outputs["mask"].get("candidate_binary", False)
            mask_valid = mask_valid and outputs["mask"].get("reference_voxels", 0) > 0
            passed = (valid and mask_valid and exact(outputs["mask"]) and exact(outputs["image"])
                      and outputs["distance"].get("fixed_allclose", {}).get("passes", False))
            description = "mask and brain exactly equal; SDT rtol=1e-5, atol=1e-4 mm"
        elif floating_sr:
            passed = valid and outputs["image"].get("fixed_allclose", {}).get("passes", False)
            description = "floating SynthSR rtol=1e-5, atol=" + str(image_atol)
        elif args.gate == "sr_official":
            metrics = outputs["image"].get("whole_grid", {})
            passed = (valid and metrics.get("exact_fraction", 0) >= .9999
                      and metrics.get("max_abs_difference", float("inf")) <= 1
                      and metrics.get("mae", float("inf")) <= 1e-4)
            description = "quantization tolerance: exact fraction >=99.99%, max <=1, MAE <=1e-4"
        else:
            passed = valid and exact(outputs["image"])
            description = "quantized SynthSR exactly equal"
        report["fixed_gate"] = {"name": args.gate, "passes": bool(passed), "criterion": description}
        report["numerical_equivalence"] = "fixed_gate_passed" if passed else "fixed_gate_failed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": args.case, "status": "measured", "report_sha256": file_hash(args.report)}))
    if not report["all_direct_comparisons_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
