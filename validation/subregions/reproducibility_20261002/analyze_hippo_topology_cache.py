"""CPU topology experiments on server-local rawfit labels, without fitting.

All rules are fixed before scoring: original 6-connected LCC, closing with a
radius-one 6-neighbour ball, or dilation with that ball followed by LCC. Each
selection is intersected with the original foreground; no label is added or
changed. The official image is read only to score these predefined outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def largest(mask):
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels, sizes, int(sizes.argmax()) if count else 0


def sample(array, affine, grid, dtype):
    image = nib.Nifti1Image(np.asarray(array, dtype=dtype), np.asarray(affine))
    return np.asarray(resample_from_to(image, (tuple(grid["shape"]), np.asarray(grid["affine"])),
                                       order=0).dataobj, dtype=dtype)


def native_output(highres, cache, support):
    processing = cache["processing_grid"]
    output = sample(highres, cache["fit_affine"], processing, np.int32)
    output[~support] = 0
    exported = cache["export_grid"]
    same = processing["shape"] == exported["shape"] and np.allclose(
        processing["affine"], exported["affine"], atol=1e-5, rtol=0)
    if not same:
        output = sample(output, processing["affine"], exported, np.int32)
        exported_support = sample(support, processing["affine"], exported, np.uint8).astype(bool)
        output[~exported_support] = 0
    return output


def score(reference, candidate, affine, allowed, offset):
    reference = nib.load(str(reference))
    if reference.shape != candidate.shape or not np.allclose(reference.affine, affine, atol=1e-5):
        reference = resample_from_to(reference, (candidate.shape, affine), order=0)
    truth = np.asarray(reference.dataobj, dtype=np.int32).copy()
    truth[truth != 0] += offset
    rows = []
    for label in sorted(allowed):
        ma, mb = truth == label + offset, candidate == label + offset
        nr, nc = int(ma.sum()), int(mb.sum())
        rows.append({"atlas_label_id": label, "output_label_id": label + offset,
                     "reference_voxels": nr, "candidate_voxels": nc,
                     "dice": 2 * int(np.count_nonzero(ma & mb)) / (nr + nc) if nr + nc else None,
                     "both_empty": nr + nc == 0})
    families = {}
    for name, selected in (("hippocampus", [r for r in rows if r["atlas_label_id"] < 7000]),
                           ("amygdala", [r for r in rows if r["atlas_label_id"] >= 7000])):
        denominator = sum(r["reference_voxels"] for r in selected if r["dice"] is not None)
        families[name] = sum(r["dice"] * r["reference_voxels"] for r in selected
                             if r["dice"] is not None) / denominator if denominator else None
    return {"weighted_dice": families, "labels": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-metadata", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--original-native", required=True, type=Path)
    parser.add_argument("--original-highres", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cache = json.loads(args.cache_metadata.read_text())
    for key in ("rawfit_identity", "support_identity"):
        recorded = cache[key]
        if identity(recorded["path"]) != recorded:
            raise ValueError("Server-local cache content changed")
    raw = np.asarray(nib.load(cache["rawfit_identity"]["path"]).dataobj, dtype=np.int32)
    support = np.asarray(nib.load(cache["support_identity"]["path"]).dataobj).astype(bool)
    allowed = set(cache["foreground_ids"])
    mask = np.isin(raw, list(allowed))
    components, sizes, main_id = largest(mask)
    retained = components == main_id if main_id else np.zeros(raw.shape, bool)
    spacing = np.linalg.norm(np.asarray(cache["fit_affine"])[:3, :3], axis=0)
    voxel_volume = float(abs(np.linalg.det(np.asarray(cache["fit_affine"])[:3, :3])))
    detached_aaa = (raw == 7010) & ~retained
    gap = {"detached_AAA_voxels": int(detached_aaa.sum())}
    if np.any(detached_aaa) and main_id:
        taxicab = ndimage.distance_transform_cdt(~retained, metric="taxicab")
        euclidean = ndimage.distance_transform_edt(~retained, sampling=spacing)
        distance = int(taxicab[detached_aaa].min())
        gap.update(minimum_taxicab_centre_distance_voxels=distance,
                   minimum_empty_voxels_in_taxicab_route=max(0, distance - 1),
                   minimum_euclidean_centre_distance_mm=float(euclidean[detached_aaa].min()))
    removed = []
    for component_id in np.argsort(sizes)[::-1]:
        component_id = int(component_id)
        if component_id in (0, main_id) or not sizes[component_id]:
            continue
        ids, counts = np.unique(raw[components == component_id], return_counts=True)
        removed.append({"component_id": component_id, "voxels": int(sizes[component_id]),
                        "volume_mm3": float(sizes[component_id] * voxel_volume),
                        "labels": [{"atlas_label_id": int(label), "voxels": int(count),
                                    "name": cache["label_names"].get(str(int(label)))}
                                   for label, count in zip(ids, counts)]})
    ball = ndimage.generate_binary_structure(3, 1)
    rules = {"original_lcc6": mask,
             "closing_ball1_selection_only": mask | ndimage.binary_closing(mask, structure=ball, iterations=1),
             "dilation_ball1_selection_only": ndimage.binary_dilation(mask, structure=ball, iterations=1)}
    offset = int(cache["offset"])
    reference_highres = Path(str(args.reference).replace(".FSvoxelSpace", ""))
    native_saved = np.asarray(nib.load(str(args.original_native)).dataobj, dtype=np.int32)
    highres_saved = np.asarray(nib.load(str(args.original_highres)).dataobj, dtype=np.int32)
    variants = []
    for name, selection_mask in rules.items():
        variant_components, _, variant_main = largest(selection_mask)
        selection = mask & (variant_components == variant_main) if variant_main else np.zeros(mask.shape, bool)
        highres = np.where(selection, raw + offset, 0).astype(np.int32)
        assert not np.any(selection & ~mask)
        output = native_output(highres, cache, support)
        record = {"rule": name, "selected_original_foreground_voxels": int(selection.sum()),
                  "new_labeled_voxels": 0, "changed_label_ids": 0,
                  "label_counts_hr": {str(label + offset): int(np.count_nonzero(highres == label + offset))
                                      for label in sorted(allowed)},
                  "native_score": score(args.reference, output, np.asarray(cache["export_grid"]["affine"]), allowed, offset)}
        if reference_highres.exists():
            record["highres_score"] = score(reference_highres, highres, np.asarray(cache["fit_affine"]), allowed, offset)
        if name == "original_lcc6":
            record["changed_native_voxels_vs_driver"] = int(np.count_nonzero(output != native_saved))
            record["changed_highres_voxels_vs_driver"] = int(np.count_nonzero(highres != highres_saved))
        variants.append(record)
    result = {"scope": "server-local cached real rawfit labels; CPU only, no fitting or parameter selection by reference",
              "structure": cache["structure"], "cache_metadata_identity": identity(args.cache_metadata),
              "reference_identity": identity(args.reference),
              "predeclared_rule": "6-neighbour ball radius 1 HR voxel; select largest component then intersect original foreground; never add labels",
              "score_grid_scope": {"native": "original export grid, same recipe resampling/support and validation driver convention",
                                   "highres": "candidate fit grid for local rule comparison; final fixed-official-axis/union-FOV evaluation is separate"},
              "helper_identity": identity(Path(__file__)), "grid_spacing_mm": spacing.tolist(),
              "foreground_voxels": int(mask.sum()), "component_count": int(len(sizes) - 1),
              "main_component_voxels": int(sizes[main_id]) if main_id else 0,
              "detached_AAA_gap": gap, "removed_components": removed, "variants": variants}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"structure": cache["structure"], "gap": gap,
                      "variant_weighted_native": {r["rule"]: r["native_score"]["weighted_dice"] for r in variants}}))


if __name__ == "__main__":
    main()
