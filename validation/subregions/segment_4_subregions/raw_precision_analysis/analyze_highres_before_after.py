"""Compare saved before/after HR labels on one fixed official HR grid per atlas."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(image):
    return {"shape": [int(size) for size in image.shape], "affine": image.affine.tolist(),
            "voxel_sizes_mm": np.linalg.norm(image.affine[:3, :3], axis=0).tolist()}


def fixed_union_grid(official, images):
    corners = []
    for image in [official, *images]:
        xyz = np.array(list(product(*[(0, int(size) - 1) for size in image.shape])))
        mapping = np.linalg.inv(official.affine) @ image.affine
        corners.append(xyz @ mapping[:3, :3].T + mapping[:3, 3])
    xyz = np.concatenate(corners)
    low = np.floor(xyz.min(0) - 1e-6).astype(int)
    high = np.ceil(xyz.max(0) + 1e-6).astype(int)
    affine = official.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ low
    return tuple((high - low + 1).tolist()), affine


def sample(image, grid):
    mapping = np.linalg.inv(image.affine) @ grid[1]
    return ndimage.affine_transform(np.asarray(image.dataobj, np.int32),
        mapping[:3, :3], offset=mapping[:3, 3], output_shape=grid[0],
        order=0, mode="constant", cval=0, prefilter=False, output=np.int32)


def aggregate(rows):
    valid = [row for row in rows if row["dice"] is not None]
    denominator = sum(row["reference_voxels"] for row in valid)
    return {"evaluated_labels": len(valid), "both_empty_labels": len(rows) - len(valid),
            "reference_voxels": denominator,
            "reference_hard_mm3": sum(row["reference_hard_mm3"] for row in valid),
            "mean": float(np.mean([row["dice"] for row in valid])) if valid else None,
            "weighted": sum(row["reference_voxels"] * row["dice"] for row in valid) / denominator if denominator else None,
            "strict_accepted_labels": sum(row["strict_accepted"] is True for row in valid)}


def label_metrics(truth, prediction, metadata, voxel_volume):
    identifiers = sorted(metadata)
    maximum = max(int(truth.max()), int(prediction.max()), max(identifiers))
    rc = np.bincount(truth.ravel(), minlength=maximum + 1)
    pc = np.bincount(prediction.ravel(), minlength=maximum + 1)
    ic = np.bincount(truth[truth == prediction], minlength=maximum + 1)
    rows = []
    for label in identifiers:
        reference, observed, intersection = int(rc[label]), int(pc[label]), int(ic[label])
        dice = 2 * intersection / (reference + observed) if reference + observed else None
        volume_error = (observed - reference) / reference if reference else None
        rows.append({"label": label, "name": metadata[label]["name"],
                     "parent": metadata[label]["parent"], "reference_voxels": reference,
                     "prediction_voxels": observed, "intersection_voxels": intersection,
                     "reference_hard_mm3": reference * voxel_volume,
                     "prediction_hard_mm3": observed * voxel_volume, "dice": dice,
                     "hard_volume_relative_difference": volume_error,
                     "strict_accepted": (dice >= .95 and abs(volume_error) <= .05)
                        if dice is not None and volume_error is not None else False if dice is not None else None,
                     "hard_evaluation_status": "both_empty" if dice is None else "evaluated"})
    return {"all": aggregate(rows), "by_parent": {
        parent: aggregate([row for row in rows if row["parent"] == parent])
        for parent in sorted({row["parent"] for row in rows})}, "labels": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    raw_path = root.parent.parent / "examples/data/sub-01_T1w.nii.gz"
    expected = "f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a"
    assert raw_path.stat().st_size == 3847853 and sha256(raw_path) == expected
    runs = {"before_raw": "segment4_backtracking_final_full_raw_20261001",
            "before_stage": "segment4_backtracking_final_full_stage_20261001",
            "after_raw": "segment4_precision_export_final_full_raw_20261002",
            "after_stage": "segment4_precision_export_final_full_stage_20261002"}
    sources = {"before": "source_thalamus_backtracking_final_20261001",
               "after": "source_precision_export_final_20261002"}
    result = {"kind": "CPU_public_example_fixed_highres_before_after_audit",
              "analysis_script_sha256": sha256(__file__) if Path(__file__).is_file() else os.environ["FNIT_AUDIT_SCRIPT_SHA256"],
              "public_example": {"dataset": "OpenNeuro ds000114", "license": "CC0",
                                 "defaced_published_sha256": expected, "bytes_verified": 3847853},
              "method": {"grid": "One fixed union FOV per atlas across official HR and all four before/after raw/stage outputs; original official axes, voxel spacing and integer phase.",
                         "interpolation": "Scanner-RAS affine nearest-neighbor hard labels; constant zero background.",
                         "scope": "Existing complete-run outputs only; no fitting, GPU, translation/phase selection or timing claim.",
                         "mean": "Both-empty labels excluded; matched before/after label sets separately reported.",
                         "weighted": "Official reference voxel weighted per-label Dice; not whole foreground Dice.",
                         "strict_rule": "Dice >=0.95 and absolute hard-volume relative difference <=0.05"},
              "sources": {}, "runs": {}, "structures": {}}
    reports = {}
    for version, directory in sources.items():
        manifest_path = root / directory / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["files"]:
            path = root / directory / record["path"]
            assert path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]
        result["sources"][version] = {"directory": directory, "manifest_sha256": sha256(manifest_path),
                                     "label": manifest["label"], "base_commit": manifest["base_commit"],
                                     "verified_files": len(manifest["files"]), "all_size_sha256_match": True}
    for role, directory in runs.items():
        api_path = root / directory / "api_report.json"
        report = json.loads(api_path.read_text())
        reports[role] = report
        input_path = Path(report["input"])
        if role.endswith("raw"):
            assert sha256(input_path) == expected
        result["runs"][role] = {"directory": directory, "api_report_sha256": sha256(api_path),
                                "input_name": input_path.name, "input_sha256": sha256(input_path),
                                "source_version": role.split("_", 1)[0]}
    reference_root = root.parent / "fnit_subregions_plus_20260928"
    for structure, folder, stem in (
            ("thalamus", "official_thalamus_gpucw1_full_sub01_20260929", "ThalamicNuclei"),
            ("hippo-amygdala-left", "official_hippo_gpucw1_full_sub01_20260929", "lh.hippoAmygLabels"),
            ("hippo-amygdala-right", "official_hippo_gpucw1_full_sub01_20260929", "rh.hippoAmygLabels")):
        official_path = reference_root / folder / (stem + ".mgz")
        official = nib.load(official_path)
        paths = {role: Path(report["files"]["highres/" + structure]) for role, report in reports.items()}
        images = {role: nib.load(path) for role, path in paths.items()}
        label_tables = {role: {int(k): value for k, value in report["labels"].items() if value["source"] == structure}
                        for role, report in reports.items()}
        assert all(table == label_tables["before_raw"] for table in label_tables.values())
        metadata = label_tables["before_raw"]
        grid = fixed_union_grid(official, list(images.values()))
        truth = sample(official, grid)
        if structure.endswith("right"):
            truth[truth != 0] += 10000
        voxel_volume = abs(float(np.linalg.det(grid[1][:3, :3])))
        values = {"official": {"file_name": official_path.name, "bytes": official_path.stat().st_size,
                                "file_sha256": sha256(official_path), "geometry": geometry(official)},
                  "measurement_grid": {"shape": list(grid[0]), "affine": grid[1].tolist(), "voxel_volume_mm3": voxel_volume},
                  "identical_label_metadata": True, "outputs": {}, "matched_evaluation": {}}
        sampled = {}
        for role, image in images.items():
            observed = sample(image, grid)
            sampled[role] = observed
            values["outputs"][role] = {"file_name": str(paths[role].relative_to(root)),
                                       "bytes": paths[role].stat().st_size, "file_sha256": sha256(paths[role]),
                                       "native_array_sha256": hashlib.sha256(np.asarray(image.dataobj, np.int32).tobytes()).hexdigest(),
                                       "measurement_array_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                                       "geometry": geometry(image), "metrics": label_metrics(truth, observed, metadata, voxel_volume)}
        for mode in ("raw", "stage"):
            before = values["outputs"]["before_" + mode]["metrics"]["labels"]
            after = values["outputs"]["after_" + mode]["metrics"]["labels"]
            common = {row["label"] for row in before if row["dice"] is not None} & {row["label"] for row in after if row["dice"] is not None}
            values["matched_evaluation"][mode] = {"labels": sorted(common),
                "before": aggregate([row for row in before if row["label"] in common]),
                "after": aggregate([row for row in after if row["label"] in common]),
                "prediction_changed_voxels": int(np.count_nonzero(sampled["before_" + mode] != sampled["after_" + mode]))}
        result["structures"][structure] = values
    result["limitations"] = ["One verified public development case; no clinical generalization.",
                             "Hard HR labels and soft posterior volumes are different quantities.",
                             "Saved affine precision and nearest-neighbor boundary sampling remain part of this fixed-grid measurement.",
                             "This audit measures segmentation changes independently of standard-1mm categorical export; it does not allocate improvements to individual combined algorithm changes."]
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
