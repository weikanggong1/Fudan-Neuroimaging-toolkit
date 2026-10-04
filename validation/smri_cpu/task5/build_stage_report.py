"""Publish stage metrics without server paths, argv, images or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUN_FIELDS = ("status", "wall_seconds", "returncode", "hostname", "max_cpu_threads",
              "cpu_affinity", "maximum_sampled_tree_rss_bytes",
              "maximum_sampled_tree_threads", "load_before", "load_after")

REGION_FIELDS = ("label", "first_voxels", "second_voxels", "intersection_voxels",
                 "different_voxels", "dice", "jaccard", "hard_status",
                 "hard_volume_difference_mm3", "hard_volume_relative_difference",
                 "first_soft_volume_mm3", "second_soft_volume_mm3",
                 "soft_volume_difference_mm3", "soft_volume_relative_difference",
                 "hard_volume_relative_to_official", "preexisting_gate_passed")
GEOMETRY_FIELDS = ("shape", "affine", "zooms", "dtype", "voxel_volume_mm3")
LABEL_FILE_FIELDS = ("method", "id", "label_image_sha256", "label_image_bytes",
                     "resampled_array_sha256", "label_offset")
CHECK_FIELDS = ("expected_official_label_offset", "declared_label_ids",
                "official_observed_label_ids_after_offset", "candidate_observed_requested_label_ids",
                "official_unexpected_label_ids_after_offset", "candidate_unexpected_label_ids",
                "source_shape_affine_exact", "source_direction_spacing_exact",
                "source_voxel_volume_exact", "source_origin_offset_in_official_voxels",
                "foreground_centroid_distance_mm", "soft_volumes_present_for_all_regions",
                "remaining_difference_scope", "cause_localization_status")


def _seconds(values):
    """Only publish measured numerical timing fields, never paths or arbitrary metadata."""
    return {key: value for key, value in (values or {}).items()
            if key.endswith("_seconds") and isinstance(value, (int, float))
            and not isinstance(value, bool)}


def build(comparison, metadata, source_manifest):
    worker = metadata["candidate_worker"]
    report = metadata["candidate_report"]
    grids = {}
    multiple_families = len({group["family"] for group in comparison["groups"]}) > 1
    for group in comparison["groups"]:
        pair = next(row for row in group["pairs"] if row["kind"] == "cross_method")
        rows = []
        for original in pair["regions"]:
            row = {key: original.get(key) for key in REGION_FIELDS}
            row["name"] = report["labels"][str(row["label"])]["name"]
            if row["first_voxels"] == row["second_voxels"] == 0:
                row["preexisting_gate_passed"] = None
            rows.append(row)
        # Left and right HA are distinct families; neither may overwrite the other.
        grid_key = group["id"] if multiple_families else group["space"]
        grids[grid_key] = {
            key: pair[key] for key in ("different_voxels", "foreground_dice",
                                      "mean_label_dice", "evaluated_labels", "both_empty_labels")}
        grids[grid_key].update(
            family=group["family"], space=group["space"],
            regions=rows,
            gate_pass_count=sum(row["preexisting_gate_passed"] is True for row in rows),
            gate_fail_count=sum(row["preexisting_gate_passed"] is False for row in rows),
            gate_not_assessed_count=sum(row["preexisting_gate_passed"] is None for row in rows),
            shape=group["grid"]["shape"], affine=group["grid"]["affine"],
            voxel_volume_mm3=group["grid"]["voxel_volume_mm3"],
            scoring_grid="fixed scanner-RAS grid; nearest-neighbor only, no fitted registration",
            label_files=[{key: run[key] for key in LABEL_FILE_FIELDS if key in run}
                         for run in group["runs"]])
        if group["id"] in metadata.get("output_geometry", {}):
            grids[grid_key]["source_geometry"] = {
                method: {key: geometry[key] for key in GEOMETRY_FIELDS}
                for method, geometry in metadata["output_geometry"][group["id"]].items()
                if method in ("official", "candidate")}
        if group["id"] in metadata.get("scoring_checks", {}):
            check = metadata["scoring_checks"][group["id"]]
            grids[grid_key]["scoring_checks"] = {key: check[key] for key in CHECK_FIELDS}
    return {
        "date": "2026-10-04", "scope": "same_norm_aseg_wmparc_cpu8_stage",
        "structures": report["structures"],
        "source_label": source_manifest["source_label"],
        "source_archive_sha256": source_manifest["archive_sha256"],
        "source_head_commit": source_manifest["head_commit"],
        "source_includes_reviewed_uncommitted_changes": source_manifest["includes_reviewed_uncommitted_changes"],
        "cpu_candidate": {key: metadata["candidate_record"].get(key) for key in RUN_FIELDS},
        "cpu_official": {key: metadata["official_record"].get(key) for key in RUN_FIELDS},
        "api_total_seconds": worker["api_total_seconds"], "timings": _seconds(worker["timings"]),
        "recipe_timings": {name: {"recipe_total_seconds": data.get("seconds"),
                                  "timing_seconds": _seconds(data.get("timing_seconds", {})),
                                  "mesh_solver_timings": _seconds(data.get("mesh_solver", {}))}
                           for name, data in report["initialization"].items()
                           if name in report["structures"]},
        "fit_min_jacobians": {name: report["fit_min_jacobians"][name]
                              for name in report["structures"]},
        "input_raw_t1_sha256": "eb2bc2ff1f30441b0aff54685cfdd7f196bd02dccad4698100a8bad40421de22",
        "threshold": comparison["threshold"], "both_empty_dice": None,
        "audit_helper_sha256": comparison["audit_helper_sha256"],
        "grids": grids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.comparison.read_text()), json.loads(args.metadata.read_text()),
                   json.loads(args.source_manifest.read_text()))
    result["comparison_sha256"] = hashlib.sha256(args.comparison.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, grid in result["grids"].items():
        print(name, grid["gate_pass_count"], grid["gate_fail_count"], grid["gate_not_assessed_count"])


if __name__ == "__main__":
    main()
