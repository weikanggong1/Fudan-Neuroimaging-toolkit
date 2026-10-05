"""CPU label-wise repeatability audit on fixed, real-image comparison grids.

Input schema (paths may be absolute or relative to the manifest):
  {"schema_version": 1, "groups": [{
      "id": "right_stage_native", "family": "hippo-amygdala-right",
      "space": "stage_native", "label_ids": [10240, 17006],
      "label_names": {"10240": "Right-CA3-body"},
      "grid": {"like": "norm.mgz"},
      "official": [{"id": "official_1", "labels": "rh.labels.mgz",
                    "label_offset": 10000,
                    "soft_volumes_mm3": {"10240": 100.0},
                    "provenance": {"input_sha256": "...", "config": {}}}],
      "fnit": [{"id": "fnit_1", "labels": "labels.nii.gz",
                "soft_volumes_mm3": {"10240": 99.0}}]
  }]}

For HR comparisons, grid={"union_like":"official_hr.mgz"} keeps official
axes, spacing and integer phase, expanding its field to cover every run.
Alternatively grid={"shape":[...],"affine":[[...],...]} supplies a fixed grid.
No translation, registration or phase is fitted to any output.

All within-method pairs and all cross-method pairs are retained. Both-empty
hard labels have undefined Dice, rather than perfect agreement. Three runs
give an observed range, not a population range or confidence interval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values) -> dict:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"n": len(values), "min": min(values), "median": float(np.median(values)),
            "mean": float(np.mean(values)), "max": max(values)}


def common_foreground_crop(encoded: dict) -> tuple[dict, list[int], list[int]]:
    """Remove only voxels that are background in every fixed-grid output."""
    shape = next(iter(encoded.values())).shape
    lower, upper = np.asarray(shape), np.zeros(3, dtype=int)
    for values in encoded.values():
        for axis in range(3):
            positions = np.flatnonzero(values.any(axis=tuple(other for other in range(3) if other != axis)))
            if positions.size:
                lower[axis] = min(lower[axis], int(positions[0]))
                upper[axis] = max(upper[axis], int(positions[-1]) + 1)
    if np.any(upper <= lower):
        # Both-empty groups still retain one background voxel for bincount.
        lower, upper = np.zeros(3, dtype=int), np.ones(3, dtype=int)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    for key in encoded:
        encoded[key] = encoded[key][crop].copy()
    return encoded, lower.tolist(), upper.tolist()


def pair_statistics(first: np.ndarray, second: np.ndarray, count: int,
                    label_ids: list[int], voxel_volume: float, first_soft: dict,
                    second_soft: dict) -> tuple[list[dict], dict]:
    """Use one compact confusion matrix, retaining background disagreements."""
    width = count + 1
    confusion = np.bincount(first.reshape(-1).astype(np.int64) * width
                            + second.reshape(-1), minlength=width * width).reshape(width, width)
    first_counts, second_counts = confusion.sum(1), confusion.sum(0)
    intersections = confusion.diagonal()
    rows = []
    for index, label in enumerate(label_ids, 1):
        a, b, intersection = (int(first_counts[index]), int(second_counts[index]),
                              int(intersections[index]))
        total, union = a + b, a + b - intersection
        sa, sb = first_soft.get(label), second_soft.get(label)
        soft_difference = abs(sa - sb) if sa is not None and sb is not None else None
        soft_mean = (sa + sb) / 2 if sa is not None and sb is not None else None
        rows.append({"label": label, "first_voxels": a, "second_voxels": b,
                     "intersection_voxels": intersection,
                     "different_voxels": total - 2 * intersection,
                     "dice": 2 * intersection / total if total else None,
                     "jaccard": intersection / union if union else None,
                     "hard_status": "both_empty" if total == 0 else "evaluated",
                     "hard_volume_difference_mm3": abs(a - b) * voxel_volume,
                     "hard_volume_relative_difference": 2 * abs(a - b) / total if total else None,
                     "first_soft_volume_mm3": sa, "second_soft_volume_mm3": sb,
                     "soft_volume_difference_mm3": soft_difference,
                     "soft_volume_relative_difference": (soft_difference / soft_mean
                         if soft_mean is not None and soft_mean > 0 else None)})
    total = int(first_counts[1:].sum() + second_counts[1:].sum())
    foreground_intersection = int(confusion[1:, 1:].sum())
    weights = (first_counts[1:] + second_counts[1:]) / 2
    evaluated = [row for row in rows if row["dice"] is not None]
    weighted = sum(weights[index] * row["dice"] for index, row in enumerate(rows)
                   if row["dice"] is not None) / weights.sum() if weights.sum() else None
    reference_weighted = sum(first_counts[index + 1] * row["dice"] for index, row in enumerate(rows)
                             if row["dice"] is not None) / first_counts[1:].sum() if first_counts[1:].sum() else None
    return rows, {"different_voxels": int(confusion.sum() - intersections.sum()),
                  "foreground_dice": 2 * foreground_intersection / total if total else None,
                  "symmetric_volume_weighted_label_dice": float(weighted) if weighted is not None else None,
                  "first_reference_volume_weighted_label_dice": float(reference_weighted) if reference_weighted is not None else None,
                  "mean_label_dice": float(np.mean([row["dice"] for row in evaluated])) if evaluated else None,
                  "evaluated_labels": len(evaluated),
                  "both_empty_labels": len(rows) - len(evaluated)}


def decide_geometry(official_rows: list[dict], fnit_rows: list[dict], cross_rows: list[dict],
                    official_counts: list[int], fnit_counts: list[int]) -> dict:
    official = describe(row["dice"] for row in official_rows)
    fnit = describe(row["dice"] for row in fnit_rows)
    cross = describe(row["dice"] for row in cross_rows)
    if len(official_counts) < 2 or len(fnit_counts) < 2:
        status = "insufficient_repeats"
        repeat_ok = cross_ok = None
    elif not any(official_counts) and not any(fnit_counts):
        status = "both_empty_hard_label"
        repeat_ok = cross_ok = None
    elif not any(official_counts):
        status = "outside_official_absent_hard_label"
        repeat_ok, cross_ok = None, False
    elif not any(fnit_counts):
        status = "outside_fnit_absent_hard_label"
        repeat_ok, cross_ok = None, False
    elif not official_rows or not fnit_rows or official["min"] is None or fnit["min"] is None or cross["min"] is None:
        status = "insufficient_repeats"
        repeat_ok = cross_ok = None
    else:
        # Only tolerance for floating representation; no anatomical tolerance.
        repeat_ok = fnit["min"] + 1e-12 >= official["min"]
        cross_ok = cross["min"] + 1e-12 >= official["min"]
        status = "within_observed_official_range" if repeat_ok and cross_ok else "outside_observed_official_range"
    return {"official_repeat_dice": official, "fnit_repeat_dice": fnit,
            "cross_method_dice": cross, "fnit_repeat_not_below_official": repeat_ok,
            "cross_not_below_official": cross_ok, "hard_geometry_status": status}


def _path(value, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _shape(image):
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D label image, got {image.shape}")
    return tuple(int(value) for value in image.shape)


def _grid(spec: dict, images: list, base: Path):
    if "shape" in spec:
        shape = tuple(int(value) for value in spec["shape"])
        affine = np.asarray(spec["affine"], dtype=np.float64)
    else:
        reference = nib.load(str(_path(spec.get("like", spec.get("union_like")), base)))
        shape, affine = _shape(reference), reference.affine.copy()
        if "union_like" in spec:
            inverse = np.linalg.inv(affine)
            lower, upper = np.zeros(3), np.asarray(shape, dtype=float) - 1
            for image in images:
                corners = np.asarray(list(itertools.product(*[(0, n - 1) for n in _shape(image)])))
                transformed = nib.affines.apply_affine(inverse @ image.affine, corners)
                lower = np.minimum(lower, transformed.min(0))
                upper = np.maximum(upper, transformed.max(0))
            # Avoid one-voxel expansion from harmless MGZ affine rounding.
            lower, upper = np.floor(lower + 1e-5), np.ceil(upper - 1e-5)
            affine[:3, 3] += affine[:3, :3] @ lower
            shape = tuple(int(value) for value in upper - lower + 1)
    if len(shape) != 3 or min(shape) <= 0 or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid fixed comparison grid")
    return shape, affine


def _soft_volumes(run: dict, names: dict[int, str], base: Path) -> dict[int, float]:
    result = {int(key): float(value) for key, value in run.get("soft_volumes_mm3", {}).items()}
    files = run.get("soft_volumes_files", [])
    if "soft_volumes_file" in run:
        files = [run["soft_volumes_file"], *files]
    names_to_ids = {}
    for label, name in names.items():
        names_to_ids[name] = label
        names_to_ids[name.removeprefix("Left-").removeprefix("Right-")] = label
    for filename in files:
        path = _path(filename, base)
        if path.suffix == ".tsv":
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    label = int(row.get("label_id", row.get("label")))
                    if label in names:
                        result[label] = float(row["soft_volume_mm3"])
        else:
            for line in path.read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[0] in names_to_ids:
                    result[names_to_ids[fields[0]]] = float(fields[1])
    if any(not np.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Soft volumes must be finite nonnegative numbers")
    return result


def audit_group(group: dict, base: Path) -> dict:
    labels = [int(value) for value in group["label_ids"]]
    if not labels or 0 in labels or len(set(labels)) != len(labels):
        raise ValueError("label_ids must be unique nonzero labels")
    names = {label: group.get("label_names", {}).get(str(label), str(label)) for label in labels}
    runs = [(method, run) for method in ("official", "fnit") for run in group[method]]
    if not group["official"] or not group["fnit"]:
        raise ValueError("At least one official and one FNIT run are required")
    if len({(method, run["id"]) for method, run in runs}) != len(runs):
        raise ValueError("Run IDs must be unique within each method")
    images = [nib.load(str(_path(run["labels"], base))) for _, run in runs]
    shape, affine = _grid(group["grid"], images, base)
    encoded, records, soft = {}, {}, {}
    sorted_labels = np.sort(np.asarray(labels, dtype=np.int64))
    code_for_sorted = np.asarray([labels.index(int(label)) + 1 for label in sorted_labels])
    for (method, run), image in zip(runs, images):
        key = (method, run["id"])
        if image.shape != shape or not np.allclose(image.affine, affine, atol=1e-5, rtol=0):
            image = resample_from_to(image, (shape, affine), order=0, mode="constant", cval=0)
        values = np.asarray(image.dataobj)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError(f"Labels must be finite integers: {run['labels']}")
        values = values.astype(np.int64)
        offset = int(run.get("label_offset", 0))
        if offset:
            values = np.where(values != 0, values + offset, values)
        indices = np.searchsorted(sorted_labels, values)
        bounded = np.minimum(indices, len(labels) - 1)
        included = (indices < len(labels)) & (sorted_labels[bounded] == values)
        # Restrict to requested atlas labels, preserving outside as background.
        compact = np.where(included, code_for_sorted[bounded], 0).astype(np.uint16 if len(labels) < 65535 else np.uint32)
        encoded[key] = compact
        soft[key] = _soft_volumes(run, names, base)
        path = _path(run["labels"], base)
        counts = np.bincount(compact.reshape(-1), minlength=len(labels) + 1)
        records[key] = {"method": method, "id": run["id"], "labels": str(path),
                        "label_image_sha256": sha256(path), "label_image_bytes": path.stat().st_size,
                        "resampled_array_sha256": hashlib.sha256(compact.tobytes()).hexdigest(),
                        "label_offset": offset, "provenance": run.get("provenance", {}),
                        "label_voxels": {str(label): int(counts[index]) for index, label in enumerate(labels, 1)},
                        "soft_volumes_mm3": {str(label): value for label, value in soft[key].items()}}
        del compact, values, indices, bounded, included
    encoded, metric_crop_start, metric_crop_stop = common_foreground_crop(encoded)
    method_keys = {method: [(method, run["id"]) for run in group[method]] for method in ("official", "fnit")}
    pairs, label_pairs = [], {label: {kind: [] for kind in ("official_repeat", "fnit_repeat", "cross_method")} for label in labels}
    definitions = [("official_repeat", list(itertools.combinations(method_keys["official"], 2))),
                   ("fnit_repeat", list(itertools.combinations(method_keys["fnit"], 2))),
                   ("cross_method", list(itertools.product(method_keys["official"], method_keys["fnit"])))]
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    for kind, choices in definitions:
        for first, second in choices:
            rows, overall = pair_statistics(encoded[first], encoded[second], len(labels), labels,
                                           voxel_volume, soft[first], soft[second])
            pairs.append({"kind": kind, "first": records[first]["id"], "second": records[second]["id"],
                          **overall, "regions": rows})
            for row in rows:
                label_pairs[row["label"]][kind].append(row)
    regions = []
    for label in labels:
        rows = label_pairs[label]
        official_counts = [records[key]["label_voxels"][str(label)] for key in method_keys["official"]]
        fnit_counts = [records[key]["label_voxels"][str(label)] for key in method_keys["fnit"]]
        region = {"label": label, "name": names[label], "official_voxels": official_counts,
                  "fnit_voxels": fnit_counts,
                  **decide_geometry(rows["official_repeat"], rows["fnit_repeat"], rows["cross_method"], official_counts, fnit_counts)}
        for kind, current in rows.items():
            region[kind + "_different_voxels"] = describe(row["different_voxels"] for row in current)
            region[kind + "_jaccard"] = describe(row["jaccard"] for row in current)
            region[kind + "_soft_relative_difference"] = describe(row["soft_volume_relative_difference"] for row in current)
            region[kind + "_soft_difference_mm3"] = describe(row["soft_volume_difference_mm3"] for row in current)
        for method in ("official", "fnit"):
            values = [soft[key].get(label) for key in method_keys[method]]
            region[method + "_soft_volumes_mm3"] = values
            valid = [value for value in values if value is not None]
            region[method + "_soft_cv"] = (float(np.std(valid, ddof=1) / np.mean(valid))
                if len(valid) >= 2 and np.mean(valid) > 0 else None)
        official_bound = region["official_repeat_soft_relative_difference"]["max"]
        fnit_bound = region["fnit_repeat_soft_relative_difference"]["max"]
        cross_bound = region["cross_method_soft_relative_difference"]["max"]
        region["soft_within_observed_official_range"] = (fnit_bound <= official_bound + 1e-12 and cross_bound <= official_bound + 1e-12
            if official_bound is not None and fnit_bound is not None and cross_bound is not None else None)
        regions.append(region)
    warnings = []
    for method in ("official", "fnit"):
        if len(group[method]) < 3:
            warnings.append(f"{method}: fewer than three independent runs")
        if any(not run.get("provenance") for run in group[method]):
            warnings.append(f"{method}: run provenance missing; caller must verify identical input/source/config")
    return {"id": group["id"], "family": group["family"], "space": group["space"],
            "grid": {"shape": list(shape), "affine": affine.tolist(), "voxel_volume_mm3": voxel_volume,
                     "definition": group["grid"], "metric_crop_start": metric_crop_start,
                     "metric_crop_stop": metric_crop_stop,
                     "metric_crop_rule": "Common union foreground bbox; removed voxels are background in every run, so all label counts/Dice/Jaccard/mismatch metrics are unchanged."},
            "runs": list(records.values()), "warnings": warnings, "pairs": pairs,
            "pair_summary": {kind: {"different_voxels": describe(pair["different_voxels"] for pair in pairs if pair["kind"] == kind),
                                     "weighted_label_dice": describe(pair["symmetric_volume_weighted_label_dice"] for pair in pairs if pair["kind"] == kind),
                                     "first_reference_weighted_label_dice": describe(pair["first_reference_volume_weighted_label_dice"] for pair in pairs if pair["kind"] == kind)}
                             for kind, _ in definitions},
            "regions": regions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Only schema_version=1 is supported")
    result = {"schema_version": 1, "manifest_sha256": sha256(args.manifest),
              "analysis_script_sha256": sha256(Path(__file__)),
              "methods": {"comparison": "Fixed-grid nearest-neighbor in scanner RAS; no fitted registration or phase.",
                          "range": "Observed min/max from independent runs; not confidence intervals or a population bound.",
                          "both_empty": "Undefined hard Dice/Jaccard, not 1; soft volumes evaluated separately.",
                          "hard_geometry_status": "Within only if every FNIT repeat and every cross-method pair Dice >= minimum observed official repeat Dice; floating tolerance 1e-12.",
                          "volumes": "Pair relative differences are symmetric: abs(a-b)/((a+b)/2).",
                          "weighted_dice": "Per-label Dice weighted by symmetric mean label voxel volume; not foreground Dice.",
                          "scope": "Same-stage and end-to-end comparisons remain separate. Stability alone cannot establish agreement with official outputs."},
              "metadata": manifest.get("metadata", {}), "groups": []}
    for group in manifest["groups"]:
        result["groups"].append(audit_group(group, args.manifest.parent))
        print(f"audited {group['id']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".tsv").open("w", newline="") as handle:
        columns = ["group", "family", "space", "label", "name", "hard_geometry_status",
                   "official_repeat_min_dice", "fnit_repeat_min_dice", "cross_min_dice",
                   "official_repeat_max_different_voxels", "fnit_repeat_max_different_voxels",
                   "cross_max_different_voxels", "soft_within_observed_official_range"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for group in result["groups"]:
            for region in group["regions"]:
                row = {key: group[key] for key in ("family", "space")}
                row.update(group=group["id"], **{key: region[key] for key in ("label", "name", "hard_geometry_status", "soft_within_observed_official_range")})
                for kind, abbreviation in (("official_repeat", "official_repeat"), ("fnit_repeat", "fnit_repeat"), ("cross_method", "cross")):
                    row[abbreviation + "_min_dice"] = region[kind + "_dice"]["min"]
                    row[abbreviation + "_max_different_voxels"] = region[kind + "_different_voxels"]["max"]
                writer.writerow(row)


if __name__ == "__main__":
    main()
