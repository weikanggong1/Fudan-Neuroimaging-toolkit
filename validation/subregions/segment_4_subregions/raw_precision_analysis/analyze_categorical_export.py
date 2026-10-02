"""CPU audit of existing categorical exports; no fitting or phase selection."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(image):
    return {"shape": [int(size) for size in image.shape], "affine": image.affine.tolist(),
            "voxel_sizes_mm": np.linalg.norm(image.affine[:3, :3], axis=0).tolist()}


def resample(image, grid):
    return np.asarray(resample_from_to(image, grid, order=0).dataobj, np.int32)


def metrics(truth, prediction, identifiers):
    maximum = max(int(truth.max()), int(prediction.max()), max(identifiers))
    expected = np.bincount(truth.ravel(), minlength=maximum + 1)
    observed = np.bincount(prediction.ravel(), minlength=maximum + 1)
    intersection = np.bincount(truth[truth == prediction], minlength=maximum + 1)
    rows = []
    for label in identifiers:
        n, m, k = int(expected[label]), int(observed[label]), int(intersection[label])
        if n + m:
            rows.append({"label": label, "reference_voxels": n, "prediction_voxels": m,
                         "intersection_voxels": k, "dice": 2 * k / (n + m)})
    denominator = sum(row["reference_voxels"] for row in rows)
    return {"reference_voxels": denominator, "evaluated_labels": len(rows),
            "mean": float(np.mean([row["dice"] for row in rows])),
            "weighted": sum(row["reference_voxels"] * row["dice"] for row in rows) / denominator,
            "prediction_voxels": sum(row["prediction_voxels"] for row in rows), "labels": rows}


def union_grid(first, second):
    corners = []
    for image in (first, second):
        xyz = np.array(list(product(*[(0, int(size) - 1) for size in image.shape])))
        mapping = np.linalg.inv(first.affine) @ image.affine
        corners.append(xyz @ mapping[:3, :3].T + mapping[:3, 3])
    xyz = np.concatenate(corners)
    low = np.floor(xyz.min(0) - 1e-6).astype(int)
    high = np.ceil(xyz.max(0) + 1e-6).astype(int)
    affine = first.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ low
    return tuple((high - low + 1).tolist()), affine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    raw_path = root.parent.parent / "examples/data/sub-01_T1w.nii.gz"
    expected_hash = "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a"
    assert raw_path.stat().st_size == 3847853 and digest(raw_path) == expected_hash
    full = root / "segment4_precision_final_full_raw_20261002"
    probe = root / "precision_proxy_controls_20261002/thalamus_bias_own_grid"
    full_report = json.loads((full / "api_report.json").read_text())
    probe_report = json.loads((probe / "report.json").read_text())
    raw = nib.load(raw_path)
    raw_grid = raw.shape, raw.affine
    full_native = np.asarray(nib.load(full_report["files"]["labels"]).dataobj, np.int32)
    processing_geometry = full_report["initialization"]["shared_preprocessing"]["intensity_preprocessing"]["processing_geometry"]
    processing_grid = tuple(processing_geometry["shape"]), np.asarray(processing_geometry["affine"])
    coarse = nib.load(root / "raw_precision_controls_20261001/shared_cache/coarse.nii.gz")
    coarse_processing = resample(coarse, processing_grid)
    reference_root = root.parent / "fnit_subregions_plus_20260928"
    result = {"kind": "CPU_existing_public_example_categorical_export_audit",
              "public_example": {"dataset": "OpenNeuro ds000114", "license": "CC0",
                                 "defaced_published_sha256": expected_hash, "bytes_verified": 3847853},
              "analysis_script_sha256": digest(__file__) if Path(__file__).is_file() else os.environ["FNIT_AUDIT_SCRIPT_SHA256"],
              "native_geometry": geometry(raw), "processing_geometry": processing_geometry,
              "method": "Fixed saved scanner-RAS affines; nearest-neighbor hard labels; no fitting, GPU or phase selection.",
              "thalamus_fit_history": [], "families": {}}
    for phase, key in (("synthetic", "segmentation_fit"), ("intensity", "mesh_solver")):
        current = full_report["initialization"]["thalamus"][key]
        previous = probe_report["initialization"]["thalamus"][key]
        if phase == "synthetic":
            current, previous = current["mesh_solver"], previous["mesh_solver"]
        assert len(current["stages"]) == len(previous["stages"])
        for index, (a, b) in enumerate(zip(current["stages"], previous["stages"]), 1):
            first, second = a["objective_history"], b["objective_history"]
            result["thalamus_fit_history"].append({"phase": phase, "stage": index,
                "identical": first == second, "length": len(first),
                "history_sha256": hashlib.sha256(json.dumps(first, separators=(",", ":")).encode()).hexdigest(),
                "mesh_steps_full_probe": [a["mesh_steps"], b["mesh_steps"]],
                "mesh_evaluations_full_probe": [a["mesh_evaluations"], b["mesh_evaluations"]]})
    for family, folder, stem, support_ids in (
            ("thalamus", "official_thalamus_gpucw1_full_sub01_20260929", "ThalamicNuclei", (10, 49)),
            ("hippo-amygdala-left", "official_hippo_gpucw1_full_sub01_20260929", "lh.hippoAmygLabels", (17, 18)),
            ("hippo-amygdala-right", "official_hippo_gpucw1_full_sub01_20260929", "rh.hippoAmygLabels", (53, 54))):
        identifiers = sorted(int(k) for k, value in full_report["labels"].items() if value["source"] == family)
        fine_path = Path(full_report["files"]["highres/" + family])
        fine = nib.load(fine_path)
        fine_data = np.asarray(fine.dataobj, np.int32)
        official_high_path = reference_root / folder / (stem + ".mgz")
        official_native_path = reference_root / folder / (stem + ".FSvoxelSpace.mgz")
        official_high = nib.load(official_high_path)
        high_grid = union_grid(official_high, fine)
        truth_native = resample(nib.load(official_native_path), raw_grid)
        truth_high = resample(official_high, high_grid)
        if family.endswith("right"):
            truth_native[truth_native != 0] += 10000
            truth_high[truth_high != 0] += 10000
        native = np.where(np.isin(full_native, identifiers), full_native, 0)
        direct = resample(fine, raw_grid)
        on_processing = resample(fine, processing_grid)
        # Recipe support is five processing voxels for thalamus and two for HippoSF.
        support = ndimage.binary_dilation(np.isin(coarse_processing, support_ids),
                                          iterations=5 if family == "thalamus" else 2)
        on_processing[~support] = 0
        two_step = resample(nib.Nifti1Image(on_processing, processing_grid[1]), raw_grid)
        raw_support = resample(nib.Nifti1Image(support.astype(np.uint8), processing_grid[1]), raw_grid).astype(bool)
        guarded_direct = direct.copy()
        guarded_direct[~raw_support] = 0
        observed_high = resample(fine, high_grid)
        entry = {"highres_geometry": geometry(fine),
                 "official_highres_geometry": geometry(official_high),
                 "highres_union_geometry": {"shape": list(high_grid[0]), "affine": high_grid[1].tolist()},
                 "input_file_sha256": {"full_highres": digest(fine_path),
                                       "official_highres": digest(official_high_path),
                                       "official_native": digest(official_native_path)},
                 "highres_array_sha256": hashlib.sha256(fine_data.tobytes()).hexdigest(),
                 "sampling_metrics": {"full_direct_native": metrics(truth_native, native, identifiers),
                                      "highres_direct_raw_no_support": metrics(truth_native, direct, identifiers),
                                      "highres_direct_raw_with_support": metrics(truth_native, guarded_direct, identifiers),
                                      "reconstructed_processing_native_then_raw": metrics(truth_native, two_step, identifiers),
                                      "highres_vs_official_highres": metrics(truth_high, observed_high, identifiers)},
                 "changed_native_voxels": {"full_vs_reconstructed_processing_then_raw": int(np.count_nonzero(native != two_step)),
                                           "direct_vs_processing_then_raw": int(np.count_nonzero(direct != two_step)),
                                           "full_vs_direct_with_support": int(np.count_nonzero(native != guarded_direct))}}
        if family == "thalamus":
            probe_path = Path(probe_report["files"]["highres/thalamus"])
            probe_fine = nib.load(probe_path)
            probe_data = np.asarray(probe_fine.dataobj, np.int32)
            entry["full_vs_probe_highres"] = {"same_shape": fine.shape == probe_fine.shape,
                "affine_max_difference": float(np.abs(fine.affine - probe_fine.affine).max()),
                "changed_voxels": int(np.count_nonzero(fine_data != probe_data)),
                "probe_file_sha256": digest(probe_path),
                "probe_array_sha256": hashlib.sha256(probe_data.tobytes()).hexdigest()}
            probe_native = resample(nib.load(probe_report["files"]["labels"]), raw_grid)
            entry["sampling_metrics"]["saved_probe_processing_native_then_raw"] = metrics(truth_native, probe_native, identifiers)
            entry["changed_native_voxels"]["reconstructed_vs_saved_probe_processing_then_raw"] = int(np.count_nonzero(two_step != probe_native))
        result["families"][family] = entry
    result["limitations"] = ["Only this verified public example and existing saved outputs are evaluated.",
                             "HR posterior volumes and hard categorical labels are separate quantities.",
                             "Processing support is reconstructed from the verified shared cache; full merged labels can also change through inter-structure competition.",
                             "Saved NIfTI affine precision may affect nearest-neighbor ties, especially 0.33333mm grids.",
                             "This diagnostic is not a new fitting or performance benchmark."]
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
