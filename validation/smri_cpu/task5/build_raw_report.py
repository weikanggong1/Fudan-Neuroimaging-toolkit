"""Publish the measured raw-T1 all-subregion run and saved-reference scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_stage_report import build, _seconds


FAMILIES = {"brainstem": 4, "thalamus": 50,
            "hippo-amygdala-left": 28, "hippo-amygdala-right": 28}
UPSTREAM_FIELDS = ("fresh_sha256", "legacy_sha256", "fresh_bytes", "legacy_bytes",
                   "fresh_shape", "legacy_shape", "fresh_dtype", "legacy_dtype",
                   "array_shape_exact", "array_dtype_exact", "array_exact",
                   "different_voxels", "max_abs_difference", "mean_abs_difference",
                   "affine_exact", "affine_max_abs_difference", "fresh_affine",
                   "legacy_affine", "fresh_zooms", "legacy_zooms", "zooms_exact")
INTENSITY_FIELDS = ("method", "mask_rule", "wm_rule", "minimum_wm_samples",
                    "intensity_dtype", "bias_correction_seconds", "seconds",
                    "intensity_scale", "threads", "applied", "grid_rule",
                    "original_wm_samples", "restored_wm_samples", "brain_mask_voxels",
                    "wm_median_before_scale", "tissue_means", "fallback_reason")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_audit(comparison, report):
    """Read saved geometries/namespaces; never refit or alter the scored maps."""
    import nibabel as nib
    import numpy as np

    geometry, checks = {}, {}
    all_declared = {int(key) for key in report["labels"]}
    for group in comparison["groups"]:
        declared = sorted(int(key) for key, row in report["labels"].items()
                          if row["source"] == group["family"])
        sources, observed, centroids = {}, {}, {}
        for run in group["runs"]:
            role = "official" if run["method"] == "official" else "candidate"
            image = nib.load(run["labels"])
            affine = image.affine
            values = np.asanyarray(image.dataobj)
            offset = int(run.get("label_offset", 0))
            values = np.where(values != 0, values + offset, values) if offset else values
            observed[role] = sorted(int(label) for label in np.unique(values) if label != 0)
            sources[role] = {"shape": list(image.shape), "affine": affine.tolist(),
                             "zooms": list(map(float, image.header.get_zooms()[:3])),
                             "dtype": str(image.get_data_dtype()),
                             "voxel_volume_mm3": float(abs(np.linalg.det(affine[:3, :3])))}
            points = np.argwhere(np.isin(values, declared))
            centroids[role] = (affine[:3, :3] @ points.mean(axis=0) + affine[:3, 3]
                               if len(points) else None)
        first, second = sources["official"], sources["candidate"]
        first_affine, second_affine = np.asarray(first["affine"]), np.asarray(second["affine"])
        pair = next(row for row in group["pairs"] if row["kind"] == "cross_method")
        centroid_distance = (float(np.linalg.norm(centroids["official"] - centroids["candidate"]))
                             if all(point is not None for point in centroids.values()) else None)
        allowed_candidate = all_declared if group["space"] == "native" else set(declared)
        checks[group["id"]] = {
            "expected_official_label_offset": 10000 if group["family"].endswith("-right") else 0,
            "declared_label_ids": declared,
            "official_observed_label_ids_after_offset": observed["official"],
            "candidate_observed_requested_label_ids": sorted(set(observed["candidate"]) & set(declared)),
            "official_unexpected_label_ids_after_offset": sorted(set(observed["official"]) - set(declared)),
            "candidate_unexpected_label_ids": sorted(set(observed["candidate"]) - allowed_candidate),
            "source_shape_affine_exact": first["shape"] == second["shape"] and bool(np.array_equal(first_affine, second_affine)),
            "source_direction_spacing_exact": bool(np.array_equal(first_affine[:3, :3], second_affine[:3, :3])),
            "source_voxel_volume_exact": first["voxel_volume_mm3"] == second["voxel_volume_mm3"],
            "source_origin_offset_in_official_voxels": (np.linalg.inv(first_affine) @ second_affine)[:3, 3].tolist(),
            "foreground_centroid_distance_mm": centroid_distance,
            "soft_volumes_present_for_all_regions": all(row["first_soft_volume_mm3"] is not None and row["second_soft_volume_mm3"] is not None for row in pair["regions"]),
            "remaining_difference_scope": "FNIT automatic preprocessing and subregion fits versus saved official outputs; not an isolated same-input solver comparison",
            "cause_localization_status": "output-only audit; no fitted registration or additional fitting",
        }
        geometry[group["id"]] = sources
    return geometry, checks


def build_raw(comparison, metadata, source_manifest, upstream):
    if comparison["comparison_scope"] != "fnit_raw_end_to_end_vs_official_saved_stage_outputs":
        raise ValueError("A stage comparison cannot be published as a raw end-to-end run")
    groups = {(group["family"], group["space"]): group for group in comparison["groups"]}
    if len(groups) != 8 or set(groups) != {(name, space) for name in FAMILIES for space in ("native", "hr")}:
        raise ValueError("Expected all four families on native and high-resolution grids")
    for (family, _), group in groups.items():
        pair = next(row for row in group["pairs"] if row["kind"] == "cross_method")
        if len(pair["regions"]) != FAMILIES[family]:
            raise ValueError("Missing declared regions: " + family)
    worker, report, receipt = (metadata[key] for key in
                               ("candidate_worker", "candidate_report", "candidate_record"))
    if receipt["status"] != "complete" or receipt["returncode"] != 0:
        raise ValueError("The complete measured process must succeed before export")
    if worker["device"] != "cpu" or worker["threads"] != 8 or receipt["max_cpu_threads"] != 8:
        raise ValueError("This report requires the measured CPU8 invocation")
    if worker["stage_input_checkpoint"] or worker["reference_labels_used_for_fitting"]:
        raise ValueError("This run used reference fitting inputs")
    shared = report["initialization"]["shared_preprocessing"]
    if shared["model_calls"] != {"SynthSegPlus": 1, "SynthSeg": 0}:
        raise ValueError("Unexpected automatic model calls")
    result = build(comparison, {**metadata, "official_record": {}}, source_manifest)
    del result["cpu_official"]
    result.update(
        scope="raw_t1_cpu8_end_to_end_vs_official_saved_stage_outputs",
        official_whole_raw_wall_seconds=None, official_speedup=None,
        official_clock_scope="No single measured official raw-T1 all-subregion invocation; separate recon/stage clocks are not summed",
        official_reference_scope=("Saved official stage outputs use legacy norm/aseg/wmparc; their decoded arrays and geometry were separately compared with this round's same-T1 official CPU recon"),
        accuracy_scope="Full FNIT automatic preprocessing plus all subregion fits; not an isolated same-input GEMS solver comparison",
        process_clock_scope="Fresh process including imports, weights, automatic preprocessing, all fits and saving; existing compilation caches not reset",
        shared_preprocessing={"total_seconds": shared["seconds"],
                              "timings": _seconds(shared),
                              "model_calls": {key: shared["model_calls"][key] for key in ("SynthSegPlus", "SynthSeg")},
                              "coarse_source": shared["coarse_source"],
                              "cortical_parcellation_source": shared["cortical_parcellation_source"],
                              "wmparc_source": shared["wmparc_source"]},
        provided_reference_checkpoints=False,
        reference_labels_used_for_fitting=False,
        threads_restored=worker["threads_before"] == worker["threads_after"],
        threads_before={key: worker["threads_before"][key] for key in ("torch_intraop", "torch_interop", "numba_mask")},
        threads_after={key: worker["threads_after"][key] for key in ("torch_intraop", "torch_interop", "numba_mask")},
        number_of_regions_per_space=sum(FAMILIES.values()), number_of_grids=8,
        saved_highres_and_posteriors=True,
        official_upstream_identity={
            "all_array_geometry_exact": upstream["all_array_geometry_exact"],
            "additional_fitting": False,
            "images": {name: {key: upstream["images"][name][key] for key in UPSTREAM_FIELDS}
                       for name in ("norm", "aseg", "wmparc")}},
    )
    intensity = shared.get("intensity_preprocessing", {})
    result["shared_preprocessing"]["intensity_preprocessing"] = {
        key: intensity[key] for key in INTENSITY_FIELDS if key in intensity}
    for key in ("original_geometry", "processing_geometry"):
        if key in intensity:
            result["shared_preprocessing"]["intensity_preprocessing"][key] = {
                name: intensity[key][name] for name in ("shape", "affine", "voxel_sizes_mm")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--official-upstream-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    comparison = read(args.comparison)
    metadata = {"candidate_report": read(args.candidate_dir / "report.json"),
                "candidate_worker": read(args.candidate_dir / "worker.json"),
                "candidate_record": read(args.candidate_record)}
    for family in FAMILIES:
        for suffix in (".nii.gz", "_posterior.nii.gz"):
            if not (args.candidate_dir / "highres" / (family + suffix)).is_file():
                raise ValueError("Requested high-resolution label/posterior output is missing")
    geometry, checks = output_audit(comparison, metadata["candidate_report"])
    metadata.update(output_geometry=geometry, scoring_checks=checks)
    result = build_raw(comparison, metadata, read(args.source_manifest),
                       read(args.official_upstream_identity))
    result["evidence_sha256"] = {
        "comparison": digest(args.comparison), "worker": digest(args.candidate_dir / "worker.json"),
        "report": digest(args.candidate_dir / "report.json"), "process_receipt": digest(args.candidate_record),
        "source_manifest": digest(args.source_manifest),
        "official_upstream_identity": digest(args.official_upstream_identity),
        "exporter": digest(__file__), "stage_exporter": digest(Path(__file__).with_name("build_stage_report.py")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, grid in result["grids"].items():
        print(name, grid["gate_pass_count"], grid["gate_fail_count"], grid["gate_not_assessed_count"])


if __name__ == "__main__":
    main()
