"""Read existing TOPUP outputs and diagnose support differences on CPU only.

Writes anonymous JSON, never images. Nonzero output support is explicitly
separate from geometric validity. Supply saved native-grid geometric masks
when available; this script never fits or renders a TOPUP model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _image(path):
    image = nib.load(str(path))
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("an input image has nonfinite values")
    return image, values


def _same_grid(image, reference):
    return (image.shape[:3] == reference.shape[:3]
            and np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0))


def _counts(candidate, reference, region):
    candidate, reference = candidate & region, reference & region
    common, union = candidate & reference, candidate | reference
    return {"region_voxels": int(region.sum()), "candidate": int(candidate.sum()),
            "reference": int(reference.sum()), "intersection": int(common.sum()),
            "union": int(union.sum()), "candidate_only": int((candidate & ~reference).sum()),
            "reference_only": int((reference & ~candidate).sum()),
            "xor": int((candidate ^ reference).sum()),
            "neither": int((region & ~union).sum())}


def _metrics(candidate, reference, region):
    voxels = int(region.sum())
    if not voxels:
        return {"spatial_voxels": 0, "scalar_values": 0, "compared": False,
                "reason": "empty region"}
    x = candidate[region].astype(np.float64).reshape(-1)
    y = reference[region].astype(np.float64).reshape(-1)
    delta = x - y
    absolute = np.abs(delta)
    centered_x, centered_y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    return {"spatial_voxels": voxels, "scalar_values": int(x.size), "compared": True,
            "exact": bool(np.array_equal(x, y)), "different_scalar_values": int(np.count_nonzero(delta)),
            "pearson_r": float(np.dot(centered_x, centered_y) / denominator) if denominator > 0 else None,
            "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(delta * delta))),
            "p95_absdiff": float(np.percentile(absolute, 95)), "percentile_sampling": False,
            "max_absdiff": float(absolute.max()), "sum_absdiff": float(absolute.sum()),
            "sum_squared_diff": float(np.sum(delta * delta))}


def _geometry_mask(path, reference):
    if path is None:
        return None
    image, values = _image(path)
    if len(image.shape) != 3 or not _same_grid(image, reference) or not np.isin(values, (0, 1)).all():
        raise ValueError("geometric mask must be native-grid 3D binary NIfTI")
    return values > 0.5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imain", type=Path, required=True, help="same selected b0 pair; only header/hash are read")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True, help="fixed official 3D brain mask on imain grid")
    parser.add_argument("--candidate-geometric-mask", type=Path)
    parser.add_argument("--official-geometric-mask", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("report must be new")
    if (args.candidate_geometric_mask is None) != (args.official_geometric_mask is None):
        raise ValueError("supply both geometric masks, or omit both")
    started = time.perf_counter()
    raw_image = nib.load(str(args.imain))
    if len(raw_image.shape) != 4 or raw_image.shape[3] != 2:
        raise ValueError("imain must be the same two-volume b0 pair")
    brain_image, brain_values = _image(args.brain_mask)
    if len(brain_image.shape) != 3 or not _same_grid(brain_image, raw_image):
        raise ValueError("official brain mask must match imain geometry")
    brain = brain_values > 0.5
    if not brain.any():
        raise ValueError("official brain mask is empty")
    path_candidate = args.candidate_dir / "fieldmap_iout.nii.gz"
    path_reference = args.official_dir / "fieldmap_iout.nii.gz"
    image_candidate, candidate = _image(path_candidate)
    image_reference, reference = _image(path_reference)
    if (candidate.shape != raw_image.shape or reference.shape != raw_image.shape
            or not _same_grid(image_candidate, raw_image) or not _same_grid(image_reference, raw_image)):
        raise ValueError("both iout arrays must match the original two-volume grid")
    geometric_candidate = _geometry_mask(args.candidate_geometric_mask, raw_image)
    geometric_reference = _geometry_mask(args.official_geometric_mask, raw_image)
    flipped = bool(np.linalg.det(raw_image.affine[:3, :3]) > 0)
    regions = {"full_fov": np.ones(brain.shape, bool), "fixed_official_brain": brain}
    nonzero_candidate, nonzero_reference = candidate != 0, reference != 0
    supports = {
        "nonzero_any_scan": {name: _counts(nonzero_candidate.any(3), nonzero_reference.any(3), roi)
                             for name, roi in regions.items()},
        "nonzero_all_scans": {name: _counts(nonzero_candidate.all(3), nonzero_reference.all(3), roi)
                              for name, roi in regions.items()},
        "per_scan_nonzero": [{"scan_index": scan,
                               **{name: _counts(nonzero_candidate[..., scan], nonzero_reference[..., scan], roi)
                                  for name, roi in regions.items()}} for scan in range(2)],
    }
    errors = []
    for scan in range(2):
        nz_c, nz_r = nonzero_candidate[..., scan], nonzero_reference[..., scan]
        scan_regions = {"fixed_official_brain": brain,
                        "brain_common_nonzero": brain & nz_c & nz_r,
                        "brain_candidate_only_nonzero": brain & nz_c & ~nz_r,
                        "brain_reference_only_nonzero": brain & nz_r & ~nz_c,
                        "brain_neither_nonzero": brain & ~nz_c & ~nz_r}
        if geometric_candidate is not None:
            scan_regions["brain_common_geometric"] = brain & geometric_candidate & geometric_reference
        row = {"scan_index": scan, "regions": {
            name: _metrics(candidate[..., scan], reference[..., scan], roi)
            for name, roi in scan_regions.items()}}
        total = row["regions"]["fixed_official_brain"]["sum_squared_diff"]
        mismatch_energy = sum(row["regions"][name].get("sum_squared_diff", 0)
                              for name in ("brain_candidate_only_nonzero", "brain_reference_only_nonzero"))
        row["nonzero_support_mismatch_squared_error_fraction"] = mismatch_energy / total if total else 0.0
        errors.append(row)
    pooled_regions = {"fixed_official_brain": brain,
                      "brain_common_nonzero_all_scans": brain & nonzero_candidate.all(3) & nonzero_reference.all(3)}
    if geometric_candidate is not None:
        pooled_regions["brain_common_geometric"] = brain & geometric_candidate & geometric_reference
    pooled_errors = {name: _metrics(candidate, reference, roi)
                     for name, roi in pooled_regions.items()}
    total_energy = sum(row["regions"]["fixed_official_brain"]["sum_squared_diff"] for row in errors)
    support_energy = sum(row["regions"][name].get("sum_squared_diff", 0)
                         for row in errors
                         for name in ("brain_candidate_only_nonzero", "brain_reference_only_nonzero"))
    if geometric_candidate is None:
        geometry = {"status": "not_available",
                    "reason": "explicit geometric masks not supplied; nonzero signal is not geometric validity"}
    else:
        geometry = {"status": "compared",
                    "mask_rule": "caller-supplied native-grid common geometric masks; no render/fitting by diagnostic",
                    "supports": {name: _counts(geometric_candidate, geometric_reference, roi)
                                 for name, roi in regions.items()},
                    "nonzero_any_vs_geometry": {
                        side: {name: _counts(nonzero, geometric, roi) for name, roi in regions.items()}
                        for side, nonzero, geometric in (
                            ("candidate", nonzero_candidate.any(3), geometric_candidate),
                            ("reference", nonzero_reference.any(3), geometric_reference))}}
    jacobian_errors = {}
    jacobian_regions = {name: roi[::-1] if flipped else roi for name, roi in regions.items()}
    if geometric_candidate is not None:
        common = brain & geometric_candidate & geometric_reference
        jacobian_regions["brain_common_geometric"] = common[::-1] if flipped else common
    input_hashes = {"selected_b0_pair": _digest(args.imain), "fixed_official_brain_mask": _digest(args.brain_mask),
                    "candidate_iout": _digest(path_candidate), "reference_iout": _digest(path_reference)}
    for scan in range(2):
        name = f"fieldmap_jacout_{scan + 1:02d}.nii.gz"
        candidate_path, reference_path = args.candidate_dir / name, args.official_dir / name
        if not candidate_path.is_file() or not reference_path.is_file():
            jacobian_errors[str(scan)] = {"status": "not_available"}
            continue
        _, jacobian_candidate = _image(candidate_path)
        _, jacobian_reference = _image(reference_path)
        if jacobian_candidate.shape != brain.shape or jacobian_reference.shape != brain.shape:
            raise ValueError("Jacobian arrays must have target spatial shape")
        jacobian_errors[str(scan)] = {"status": "compared", "roi_x_flip": flipped,
                                      "regions": {label: _metrics(jacobian_candidate, jacobian_reference, roi)
                                                  for label, roi in jacobian_regions.items()}}
        input_hashes[f"candidate_jacobian_{scan}"] = _digest(candidate_path)
        input_hashes[f"reference_jacobian_{scan}"] = _digest(reference_path)
    for role, path in (("candidate_geometric_mask", args.candidate_geometric_mask),
                       ("reference_geometric_mask", args.official_geometric_mask)):
        if path is not None:
            input_hashes[role] = _digest(path)
    payload = {"schema_version": 1, "subjects": 1,
               "scope": "completed candidate and official estimate outputs; CPU output-support diagnosis; no model render/fitting",
               "input_shape": list(raw_image.shape), "canonical_x_flip": flipped,
               "fixed_official_brain_voxels": int(brain.sum()),
               "decoded_dtype": "float32", "support_rule": "decoded output value !=0, evaluated separately by scan and by any/all scans; not geometry",
               "jacobian_orientation": "canonical FSL Analyze storage; raw/brain ROIs flipped in x only when input affine det>0",
               "geometry": geometry, "nonzero_supports": supports,
               "iout_errors": errors, "iout_pooled_errors": pooled_errors,
               "brain_nonzero_support_mismatch_squared_error_fraction": support_energy / total_energy if total_energy else 0.0,
               "jacobian_errors": jacobian_errors,
               "input_sha256": input_hashes,
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__},
               "source_sha256": _digest(__file__), "diagnostic_elapsed_seconds": time.perf_counter() - started}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "canonical_x_flip": flipped,
                      "brain_nonzero_supports": supports["nonzero_any_scan"]["fixed_official_brain"],
                      "geometry_status": geometry["status"]}), flush=True)


if __name__ == "__main__":
    main()
