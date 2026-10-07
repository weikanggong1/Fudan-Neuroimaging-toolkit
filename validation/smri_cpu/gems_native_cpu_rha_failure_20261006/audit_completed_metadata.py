"""Read only: verify completed RHA label metadata, grids and score coverage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts", "like", "bindings", "scored-results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve prior reports; choose a fresh output")
    started = perf_counter()
    binding = json.loads(args.bindings.read_text())
    score = json.loads(args.scored_results.read_text())
    if score["status"] != "full_recipe_scored_after_fitting":
        raise RuntimeError("complete recipe must be independently scored first")
    if score["threshold"] != {"per_region_Dice_minimum": .95, "hard_volume_relative_to_official_maximum": .05}:
        raise RuntimeError("declared acceptance thresholds changed")
    if score["like_sha256"] != binding["inputs_sha256"]["norm"] or sha(args.like) != score["like_sha256"]:
        raise RuntimeError("native comparison grid is not the bound norm input")
    if score["bindings_sha256"] != sha(args.bindings):
        raise RuntimeError("scoring binding changed")
    run = score["full_run_scalar_receipt"]
    if run["source_sha256"] != binding["GEMS_sources_sha256"] or run["input_sha256"] != binding["inputs_sha256"]:
        raise RuntimeError("completed runtime source/input bindings differ")
    api_path = args.artifacts / "report.json"
    if sha(api_path) != score["fixed_grid_audit"]["candidate_report_sha256"]:
        raise RuntimeError("scored metadata changed")
    api = json.loads(api_path.read_text())
    if api["structures"] != ["hippo-amygdala-right"]:
        raise RuntimeError("acceptance is one right HA recipe")
    metadata = {int(key): row for key, row in api["labels"].items()}
    if not metadata or any(row["id"] != key or row["hemisphere"] != "right" or
                           row["source"] != "hippo-amygdala-right" for key, row in metadata.items()):
        raise RuntimeError("label metadata is not a complete right-hemisphere map")
    names = [row["name"] for row in metadata.values()]
    if len(set(names)) != len(names):
        raise RuntimeError("name to label reverse mapping is not unique")
    with (args.artifacts / "labels.tsv").open() as stream:
        table_rows = list(csv.DictReader(stream, delimiter="\t"))
        table = {int(row["label_id"]): row for row in table_rows}
    with (args.artifacts / "volumes.tsv").open() as stream:
        volume_rows = list(csv.DictReader(stream, delimiter="\t"))
        volumes = {int(row["label_id"]): row for row in volume_rows}
    if len(table_rows) != len(metadata) or len(volume_rows) != len(metadata) or set(table) != set(metadata) or set(volumes) != set(metadata):
        raise RuntimeError("table and volume labels do not cover all declared metadata")
    if any(table[key]["name"] != row["name"] or volumes[key]["name"] != row["name"]
           for key, row in metadata.items()):
        raise RuntimeError("table reverse names changed")
    like = nib.load(args.like)
    images = {}
    for space, relative in (("native", "subregions_native.nii.gz"),
                            ("hr", "highres/hippo-amygdala-right.nii.gz")):
        path = args.artifacts / relative
        receipt = run["output_files"]["artifacts/" + relative]
        if path.stat().st_size != receipt["bytes"] or sha(path) != receipt["sha256"]:
            raise RuntimeError("scored label image changed")
        image = nib.load(path)
        values = np.asanyarray(image.dataobj)
        if values.ndim != 3 or not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise RuntimeError("labels must remain finite three-dimensional integers")
        used, counts = np.unique(values, return_counts=True)
        unexpected = set(int(value) for value in used if value != 0) - set(metadata)
        if unexpected:
            raise RuntimeError("saved FNIT labels are absent from the declared table")
        if space == "native" and (image.shape != like.shape or not np.allclose(image.affine, like.affine, atol=1e-5, rtol=0)):
            raise RuntimeError("native output grid differs from bound input")
        rows = [row for row in score["regions"] if row["space"] == space]
        if len(rows) != len(metadata) or set(row["label"] for row in rows) != set(metadata):
            raise RuntimeError("scorer omitted or duplicated a declared region")
        if any(row["name"] != metadata[row["label"]]["name"] for row in rows):
            raise RuntimeError("score labels and metadata names differ")
        nonempty = [row for row in rows if row["nonempty"]]
        both_empty = [row for row in rows if not row["nonempty"]]
        if not nonempty:
            raise RuntimeError("all-empty groups cannot establish equivalence")
        if any(row["official_voxels"] or row["FNIT_voxels"] or row["dice"] is not None
               for row in both_empty):
            raise RuntimeError("both-empty Dice/count semantics changed")
        if space == "native":
            for row in rows:
                if not np.isclose(float(volumes[row["label"]]["hard_volume_mm3"]),
                                  row["FNIT_hard_volume_mm3"], atol=1e-6, rtol=0):
                    raise RuntimeError("native hard volume table differs from saved-score counts")
        images[space] = {
            "shape": list(image.shape), "affine": image.affine.tolist(),
            "header_dtype": str(image.get_data_dtype()),
            "qform_code": int(image.header["qform_code"]), "sform_code": int(image.header["sform_code"]),
            "scanner_RAS_axis_codes": list(nib.aff2axcodes(image.affine)),
            "voxel_volume_mm3": float(abs(np.linalg.det(image.affine[:3, :3]))),
            "label_image_sha256": sha(path), "declared_regions": len(metadata),
            "used_label_voxels": {str(int(value)): int(count) for value, count in zip(used, counts)},
            "scored_nonempty_regions": len(nonempty),
            "both_empty_region_ids": [row["label"] for row in both_empty],
            "all_nonempty_passed": all(row["passed"] for row in nonempty),
        }
    report = {"status": "completed_RHA_metadata_and_grids_verified",
              "label_reverse_mapping_unique_and_complete": True,
              "declared_metadata": metadata, "images": images,
              "like_matches_bound_norm_SHA256": True,
              "source25_and_input3_bindings_match": True,
              "score_result_sha256": sha(args.scored_results),
              "program_sha256": sha(__file__), "read_only_seconds": perf_counter() - started,
              "new_fitting_or_registration": False, "original_Dice_and_volume_thresholds_changed": False}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "images": images}))


if __name__ == "__main__":
    main()
