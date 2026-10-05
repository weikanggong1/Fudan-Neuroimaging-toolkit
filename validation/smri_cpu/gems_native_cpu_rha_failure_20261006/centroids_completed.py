"""Read only: centroid differences on actual and predeclared scored RAS grids."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moments(image, ids, offset):
    values = np.asarray(image.dataobj, dtype=np.int32)
    if offset:
        values = np.where(values != 0, values + offset, 0)
    coordinates = np.argwhere(np.isin(values, ids))
    codes = np.searchsorted(ids, values[tuple(coordinates.T)])
    counts = np.bincount(codes, minlength=len(ids))
    sums = np.stack([np.bincount(codes, weights=coordinates[:, axis], minlength=len(ids))
                     for axis in range(3)], axis=1)
    center = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] != 0)
    ras = nib.affines.apply_affine(image.affine, center)
    encoded = np.zeros(values.shape, dtype=np.uint16)
    encoded[tuple(coordinates.T)] = codes + 1
    return {"count": counts, "RAS_mm": ras,
            "compact_array_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
            "voxel_volume_mm3": float(abs(np.linalg.det(image.affine[:3, :3])))}


def point(stat, index):
    return stat["RAS_mm"][index].tolist() if stat["count"][index] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts", "official-root", "scored-results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve old reports; use a new output")
    started = perf_counter()
    score = json.loads(args.scored_results.read_text())
    if score["status"] != "full_recipe_scored_after_fitting":
        raise RuntimeError("independent complete-recipe scoring required")
    spaces = {}
    for group in score["fixed_grid_audit"]["groups"]:
        space = group["space"]
        rows = sorted([row for row in score["regions"] if row["space"] == space], key=lambda row: row["label"])
        ids = np.asarray([row["label"] for row in rows], dtype=np.int32)
        if len(ids) != 28 or len(set(ids)) != 28:
            raise RuntimeError("this frozen RHA recipe declares 28 regions")
        official_name = "rh.hippoAmygLabels" + (".FSvoxelSpace.mgz" if space == "native" else ".mgz")
        paths = {"official": args.official_root / "hippo-amygdala" / official_name,
                 "fnit": args.artifacts / ("subregions_native.nii.gz" if space == "native"
                                             else "highres/hippo-amygdala-right.nii.gz")}
        grid = (tuple(group["grid"]["shape"]), np.asarray(group["grid"]["affine"], dtype=np.float64))
        records, own, fixed = {}, {}, {}
        for method, path in paths.items():
            previous = next(row for row in group["runs"] if row["method"] == method)
            if sha(path) != previous["label_image_sha256"]:
                raise RuntimeError("scored label image changed")
            image = nib.load(path)
            offset = 10000 if method == "official" else 0
            own[method] = moments(image, ids, offset)
            common = image
            if image.shape != grid[0] or not np.allclose(image.affine, grid[1], atol=1e-5, rtol=0):
                common = resample_from_to(image, grid, order=0, mode="constant", cval=0)
            fixed[method] = moments(common, ids, offset)
            if fixed[method]["compact_array_sha256"] != previous["resampled_array_sha256"]:
                raise RuntimeError("centroid grid array differs from the original fixed-grid score")
            records[method] = {"label_image_sha256": sha(path), "shape": list(image.shape),
                "affine": image.affine.tolist(), "voxel_volume_mm3": own[method]["voxel_volume_mm3"],
                "scanner_RAS_axis_codes": list(nib.aff2axcodes(image.affine)),
                "fixed_grid_compact_array_sha256": fixed[method]["compact_array_sha256"]}
        differences, weights, result_rows = [], [], []
        for index, row in enumerate(rows):
            if int(fixed["official"]["count"][index]) != row["official_voxels"] or int(fixed["fnit"]["count"][index]) != row["FNIT_voxels"]:
                raise RuntimeError("saved fixed-grid count changed")
            centers = {method: point(stat, index) for method, stat in fixed.items()}
            delta = (np.subtract(centers["fnit"], centers["official"])
                     if all(value is not None for value in centers.values()) else None)
            if delta is not None:
                differences.append(delta)
                weights.append(row["official_hard_volume_mm3"])
            original_centers = {method: point(stat, index) for method, stat in own.items()}
            own_delta = (np.subtract(original_centers["fnit"], original_centers["official"])
                         if all(value is not None for value in original_centers.values()) else None)
            result_rows.append({**row, "fixed_grid_centroids_RAS_mm": centers,
                "fixed_grid_FNIT_minus_official_RAS_mm": delta.tolist() if delta is not None else None,
                "fixed_grid_centroid_distance_mm": float(np.linalg.norm(delta)) if delta is not None else None,
                "own_image_centroids_RAS_mm": original_centers,
                "own_image_FNIT_minus_official_RAS_mm": own_delta.tolist() if own_delta is not None else None,
                "own_image_voxels": {method: int(stat["count"][index]) for method, stat in own.items()},
                "own_image_hard_volume_mm3": {method: float(stat["count"][index]*stat["voxel_volume_mm3"]) for method, stat in own.items()}})
        if not differences or not sum(weights):
            raise RuntimeError("empty centroid comparison")
        differences = np.asarray(differences)
        weighted = np.average(differences, axis=0, weights=weights)
        distances = np.linalg.norm(differences, axis=1)
        spaces[space] = {"source_images": records, "fixed_grid": group["grid"], "regions": result_rows,
            "same_fixed_grid_all_compact_arrays_and_counts_exact": True,
            "centroid_distance_mm": {"minimum": float(distances.min()), "median": float(np.median(distances)), "maximum": float(distances.max())},
            "mean_FNIT_minus_official_RAS_mm": differences.mean(0).tolist(),
            "official_volume_weighted_FNIT_minus_official_RAS_mm": weighted.tolist(),
            "fraction_of_regions_with_positive_dot_to_weighted_shift": float(np.mean(differences @ weighted > 0)) if np.linalg.norm(weighted) else None}
    result = {"status": "completed_saved_label_centroids_read_only",
        "RAS_axes": ["right", "anterior", "superior"], "spaces": spaces,
        "score_result_sha256": sha(args.scored_results), "program_sha256": sha(__file__),
        "read_only_seconds": perf_counter() - started, "new_fitting_or_registration": False,
        "interpretation": "centroid/volume observations quantify final geometry; they do not prove the initial affine caused a difference"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "spaces": {key: {k:v for k,v in value.items() if k not in ["regions", "source_images", "fixed_grid"]} for key,value in spaces.items()}}))


if __name__ == "__main__":
    main()
