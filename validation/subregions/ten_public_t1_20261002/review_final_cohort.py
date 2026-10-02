#!/usr/bin/env python3
"""Independently audit final real cohort accuracy from text, using stdlib only.

No F8/analyzer calculation functions are imported. No image, GPU, network or
fitting operation occurs. The six final analysis files must already exist and
their status must confirm the complete predetermined ten subjects. The review
does not score unfinished snapshots or manufacture missing observations.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path


IDS = tuple(f"sub-{i:02d}" for i in range(1, 11))
SPACES = ("raw_native", "raw_hr", "stage_native", "stage_hr")
FAMILY_SIZES = {"brainstem": 4, "thalamus": 50, "hippocampus_left": 19,
                "hippocampus_right": 19, "amygdala_left": 9, "amygdala_right": 9}
ROI_METRICS = ("dice", "jaccard", "hard_volume_difference_mm3", "hard_volume_relative_difference",
               "soft_volume_difference_mm3", "soft_volume_relative_difference")
FAMILY_METRICS = ("reference_weighted_dice", "mean_label_dice", "mean_label_jaccard", "foreground_dice",
                  "different_voxels", "hard_volume_difference_mm3", "hard_volume_relative_difference",
                  "soft_volume_difference_mm3", "soft_volume_relative_difference", "mean_roi_soft_relative_difference")
CASE_METRICS = ("reference_weighted_dice", "mean_label_dice", "mean_label_jaccard", "micro_label_dice",
                "mean_roi_soft_relative_difference")
FINAL_FILES = ("cohort_analysis.json", "analysis_status.json", "cohort_summary.json",
               "cohort_roi.tsv", "cohort_family.tsv", "cohort_case.tsv")
ANALYZER_SHA = "ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc"
HELPER_SHA = "beef69670191f89ebf3a0d71395c274b54b7c61d02ed096abebebd7a5172f842"
MANIFEST_SHA = "a06a0a9580e93d0afef9873422efdb0bf5b651685b08837a5f42004631ba3623"
SOURCE_SHA = "d8c8f7aecdae521751f30fdd09bea6acbffd2c0a50c9fb70e1dc97ad74717b07"
ABS_TOL = 1e-9
REL_TOL = 1e-11


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")
    return json.loads(path.read_bytes(), parse_constant=reject)


def identity(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def keyed(rows, fields, name):
    keys = [tuple(row[key] for key in fields) for row in rows]
    require(len(keys) == len(set(keys)), f"Duplicate {name} keys")
    return dict(zip(keys, rows))


def integer(value, pointer, positive=False):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= int(positive),
            f"Nonnegative integer required at {pointer}")
    return value


def finite(value, pointer, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value >= 0 and (not positive or value > 0), f"Finite nonnegative scalar required at {pointer}")
    return value


class Arithmetic:
    def __init__(self):
        self.checked = 0
        self.max_absolute_error = 0.0

    def equal(self, observed, expected, pointer):
        self.checked += 1
        if expected is None:
            require(observed is None, f"Expected explicit NA at {pointer}")
        elif isinstance(expected, bool):
            require(observed is expected, f"Boolean differs at {pointer}")
        elif isinstance(expected, int):
            require(integer(observed, pointer) == expected, f"Integer arithmetic differs at {pointer}")
        else:
            require(isinstance(observed, (int, float)) and not isinstance(observed, bool)
                    and math.isfinite(observed), f"Finite comparison required at {pointer}")
            error = abs(observed - expected)
            self.max_absolute_error = max(self.max_absolute_error, error)
            require(math.isclose(observed, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                    f"Arithmetic differs at {pointer}: observed={observed!r}, independently_computed={expected!r}")


def mean(values):
    return math.fsum(values) / len(values) if values else None


def stats(values, planned):
    x = sorted(float(value) for value in values if value is not None)
    require(len(x) <= planned and all(math.isfinite(v) for v in x), "Invalid distribution input")
    n = len(x)
    average = math.fsum(x) / n if n else None
    median = (x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2) if n else None
    sd = math.sqrt(math.fsum((v - average) ** 2 for v in x) / (n - 1)) if n > 1 else None
    return {"planned_subjects": planned, "defined_subjects": n, "missing_or_na_subjects": planned - n,
            "mean": average, "median": median, "min": min(x) if n else None, "max": max(x) if n else None,
            "std": sd, "std_sample": sd, "std_ddof": 1}


def check_stats(observed, computed, arithmetic, pointer):
    require(set(observed) == set(computed), f"Distribution fields differ: {pointer}")
    for key in computed:
        arithmetic.equal(observed[key], computed[key], f"{pointer}/{key}")


def canonical_family(entry):
    return entry["parent"] + ("_" + entry["hemisphere"] if entry["source"].startswith("hippo-amygdala") else "")


def check_tsv(path, expected_rows, fields, omitted=()):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"Duplicate TSV columns {path}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"Unexpected extra TSV cells {path}")
    # The frozen writer leaves absent/None cells empty, with Python bool spelling.
    expected = [{key: value for key, value in row.items() if key not in omitted} for row in expected_rows]
    encode = lambda value: "" if value is None else str(value)
    encoded = [{key: encode(value) for key, value in row.items()} for row in expected]
    actual_index = keyed(rows, fields, str(path))
    expected_index = keyed(encoded, fields, str(path) + " JSON")
    require(set(actual_index) == set(expected_index), f"TSV key set differs {path}")
    columns = set().union(*(set(row) for row in expected))
    require(set(rows[0]) == columns, f"TSV column union differs {path}")
    for key in actual_index:
        require(actual_index[key] == {name: expected_index[key].get(name, "") for name in columns},
                f"TSV cell differs at {path}/{key}")
    return len(rows)


def roi_calculation(row, metadata, arithmetic):
    prefix = f"ROI/{row['case_id']}/{row['space']}/{row['label']}"
    require(row["measurement_status"] == "evaluated", f"Final ROI unavailable: {prefix}")
    require(row["name"] == metadata["name"] and row["source"] == metadata["source"]
            and row["family"] == canonical_family(metadata), f"ROI atlas identity differs: {prefix}")
    require(row["development_seen"] is (row["case_id"] == "sub-01"), f"ROI development flag {prefix}")
    require(row["space"] == f"{row['mode']}_{row['resolution']}", f"ROI mode/grid metadata {prefix}")
    a, b, overlap = (integer(row[name], f"{prefix}/{name}") for name in
                     ("official_voxels", "fnit_voxels", "intersection_voxels"))
    grid_n = integer(row["measurement_grid_total_voxels"], prefix + "/grid_size", positive=True)
    valid = integer(row["mode_input_finite_voxels"], prefix + "/input_finite", positive=True)
    nonzero = integer(row["mode_input_positive_finite_voxels"], prefix + "/input_positive")
    require(overlap <= min(a, b) and max(a, b) <= grid_n and nonzero <= valid, f"Invalid count bounds {prefix}")
    union = a + b - overlap
    require(union <= grid_n, f"Union larger than grid: {prefix}")
    voxel = finite(row["voxel_volume_mm3"], prefix + "/voxel_mm3", positive=True)
    sa, sb = (finite(row[name], f"{prefix}/{name}") for name in ("official_soft_volume_mm3", "fnit_soft_volume_mm3"))
    total = a + b
    soft_total = sa + sb
    hard_status = ("both_empty_hard_label" if not total else "official_present_fnit_hard_label_absent" if not b
                   else "official_hard_label_absent_fnit_present" if not a else "evaluated")
    require(row["hard_status"] == hard_status and
            row["na_reason"] == ("Both hard labels empty; Dice/Jaccard undefined." if not total else None),
            f"Both/one empty policy differs: {prefix}")
    require(row["soft_na_reason"] == ("Both soft volumes zero." if not soft_total else None), f"Soft NA policy {prefix}")
    computed = {"union_voxels": union, "different_voxels": total - 2 * overlap,
                "dice": 2 * overlap / total if total else None, "jaccard": overlap / union if union else None,
                "official_hard_volume_mm3": a * voxel, "fnit_hard_volume_mm3": b * voxel,
                "hard_volume_difference_mm3": abs(a - b) * voxel,
                "hard_volume_relative_difference": 2 * abs(a - b) / total if total else None,
                "hard_volume_reference_relative_difference": abs(a - b) / a if a else None,
                "soft_volume_difference_mm3": abs(sa - sb),
                "soft_volume_relative_difference": 2 * abs(sa - sb) / soft_total if soft_total else None,
                "soft_volume_reference_relative_difference": abs(sa - sb) / sa if sa else None}
    for key, value in computed.items():
        arithmetic.equal(row[key], value, prefix + "/" + key)
    return {**row, **computed}


def determinant3(matrix):
    a, b, c = matrix[0][:3], matrix[1][:3], matrix[2][:3]
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def family_calculation(group, rows, case_record, arithmetic):
    prefix = f"family/{group['case_id']}/{group['space']}/{group['family']}"
    require(group["measurement_status"] == "evaluated" and
            group["development_seen"] is (group["case_id"] == "sub-01"), f"Incomplete family {prefix}")
    require(len(rows) == FAMILY_SIZES[group["family"]], f"Family ROI denominator {prefix}")
    grid = group["grid"]
    require(len(grid["shape"]) == 3 and len(grid["affine"]) == 4 and all(len(row) == 4 for row in grid["affine"]),
            f"Grid dimensions {prefix}")
    for n in grid["shape"]:
        integer(n, prefix + "/shape", positive=True)
    require(all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
                for line in grid["affine"] for value in line) and grid["affine"][3] == [0, 0, 0, 1], f"Grid affine {prefix}")
    voxel = abs(determinant3(grid["affine"]))
    arithmetic.equal(grid["voxel_volume_mm3"], voxel, prefix + "/determinant_voxel_mm3")
    grid_n = math.prod(grid["shape"])
    for roi in rows:
        arithmetic.equal(roi["measurement_grid_total_voxels"], grid_n, prefix + "/grid_size")
        arithmetic.equal(roi["voxel_volume_mm3"], voxel, prefix + "/ROI_voxel_mm3")
        input_scan = case_record["fnit"][group["mode"]]["input_grid_and_valid_voxels"]
        arithmetic.equal(roi["mode_input_finite_voxels"], input_scan["finite_voxels"], prefix + "/input_finite")
        arithmetic.equal(roi["mode_input_positive_finite_voxels"], input_scan["positive_finite_voxels"], prefix + "/input_positive")
    a = sum(row["official_voxels"] for row in rows)
    b = sum(row["fnit_voxels"] for row in rows)
    overlap = sum(row["intersection_voxels"] for row in rows)
    total = a + b
    difference = integer(group["different_voxels"], prefix + "/categorical_different")
    sum_onehot_difference = sum(row["different_voxels"] for row in rows)
    # Family categorical mismatches count a foreground label swap once;
    # the sum of one-hot ROI mismatches counts it twice. Recover foreground
    # overlap algebraically, retaining the distinct saved categorical count.
    cross_label = sum_onehot_difference - difference
    foreground_overlap = a + b - overlap - difference
    require(max(a, b) <= grid_n and 0 <= cross_label <= min(a - overlap, b - overlap)
            and overlap <= foreground_overlap <= min(a, b)
            and a + b - foreground_overlap <= grid_n and difference <= grid_n,
            f"Family confusion/marginal bounds {prefix}")
    dice = [row["dice"] for row in rows if row["dice"] is not None]
    jaccard = [row["jaccard"] for row in rows if row["jaccard"] is not None]
    soft_relative = [row["soft_volume_relative_difference"] for row in rows if row["soft_volume_relative_difference"] is not None]
    sa, sb = (math.fsum(row[key] for row in rows) for key in ("official_soft_volume_mm3", "fnit_soft_volume_mm3"))
    hard_a, hard_b = a * grid["voxel_volume_mm3"], b * grid["voxel_volume_mm3"]
    computed = {"reference_weighted_dice": math.fsum(row["official_voxels"] * row["dice"] for row in rows if row["dice"] is not None) / a if a else None,
                "mean_label_dice": mean(dice), "mean_label_jaccard": mean(jaccard),
                "foreground_dice": 2 * foreground_overlap / total if total else None,
                "evaluated_labels": len(dice), "both_empty_labels": len(rows) - len(dice),
                "official_hard_volume_mm3": hard_a, "fnit_hard_volume_mm3": hard_b,
                "hard_volume_difference_mm3": abs(hard_a - hard_b),
                "hard_volume_relative_difference": 2 * abs(hard_a - hard_b) / (hard_a + hard_b) if hard_a + hard_b else None,
                "official_soft_volume_mm3": sa, "fnit_soft_volume_mm3": sb,
                "soft_volume_difference_mm3": abs(sa - sb),
                "soft_volume_relative_difference": 2 * abs(sa - sb) / (sa + sb) if sa + sb else None,
                "mean_roi_soft_relative_difference": mean(soft_relative)}
    for key, value in computed.items():
        arithmetic.equal(group[key], value, prefix + "/" + key)
    require(len(group["runs"]) == 2, f"Official/FNIT run pair {prefix}")
    runs = keyed(group["runs"], ("method",), prefix + " runs")
    require(set(runs) == {("official",), ("fnit",)}, f"One official/FNIT run required: {prefix}")
    for method, count_key, soft_key in (("official", "official_voxels", "official_soft_volume_mm3"),
                                       ("fnit", "fnit_voxels", "fnit_soft_volume_mm3")):
        run = runs[(method,)]
        require(run["provenance"]["case_id"] == group["case_id"], f"Actual run case identity {prefix}")
        if method == "official":
            source = rows[0]["source"]
            require(run["label_offset"] == (10000 if source.endswith("right") else 0)
                    and run["provenance"]["reference_only_after_fit"] is True, f"Official offset policy {prefix}")
            artifact = case_record["official_outputs"][source]["arrays"][group["resolution"]]
        else:
            require(run["label_offset"] == 0 and run["provenance"]["mode"] == group["mode"], f"FNIT run policy {prefix}")
            map_key = "labels" if group["resolution"] == "native" else "highres/" + rows[0]["source"]
            artifact = case_record["fnit"][group["mode"]]["arrays"][map_key]
        require(run["label_image_sha256"] == artifact["sha256"] and run["label_image_bytes"] == artifact["bytes"],
                f"Actual saved run artifact identity {prefix}/{method}")
        require(set(run["label_voxels"]) == {str(row["label"]) for row in rows}, f"Run label count keys {prefix}")
        for row in rows:
            arithmetic.equal(run["label_voxels"][str(row["label"])], row[count_key], prefix + f"/{method}/label_count")
            arithmetic.equal(run["soft_volumes_mm3"][str(row["label"])], row[soft_key], prefix + f"/{method}/soft_volume")
    micro = 2 * overlap / total if total else None
    if micro is not None:
        require(computed["foreground_dice"] + ABS_TOL >= micro, f"Foreground below micro-label Dice {prefix}")
    return {**group, **computed, "micro_label_dice": micro,
            "foreground_intersection_voxels_derived": foreground_overlap,
            "cross_family_label_mismatch_voxels_derived": cross_label}


def case_calculation(score, rows, arithmetic):
    prefix = f"case/{score['case_id']}/{score['space']}"
    require(len(rows) == 110 and score["measurement_status"] == "evaluated" and
            score["development_seen"] is (score["case_id"] == "sub-01"), f"Case complete label scope {prefix}")
    a = sum(row["official_voxels"] for row in rows)
    b = sum(row["fnit_voxels"] for row in rows)
    overlap = sum(row["intersection_voxels"] for row in rows)
    defined = [row for row in rows if row["dice"] is not None]
    soft = [row["soft_volume_relative_difference"] for row in rows if row["soft_volume_relative_difference"] is not None]
    computed = {"total_official_voxels": a, "total_fnit_voxels": b, "total_intersection_voxels": overlap,
                "reference_weighted_dice": math.fsum(row["official_voxels"] * row["dice"] for row in defined) / a if a else None,
                "mean_label_dice": mean([row["dice"] for row in defined]),
                "mean_label_jaccard": mean([row["jaccard"] for row in defined]),
                "micro_label_dice": 2 * overlap / (a + b) if a + b else None,
                "evaluated_labels": len(defined), "both_empty_labels": 110 - len(defined), "reported_roi_rows": 110,
                "mean_roi_soft_relative_difference": mean(soft), "eligible_for_case_ranking": bool(a)}
    require(score["na_reason"] is None and score["space"] == f"{score['mode']}_{score['resolution']}", f"Case scope/NA {prefix}")
    for key, value in computed.items():
        arithmetic.equal(score[key], value, prefix + "/" + key)
    return {**score, **computed}


def review(args):
    analysis_dir = args.analysis_dir.resolve()
    manifest_path = args.manifest.resolve()
    source_path = args.source_manifest.resolve()
    input_paths = {name: analysis_dir / name for name in FINAL_FILES}
    identities_before = {name: identity(path) for name, path in input_paths.items()}
    identities_before["cohort_manifest.json"] = identity(manifest_path)
    identities_before["source_manifest.json"] = identity(source_path)
    status = read_json(input_paths["analysis_status.json"])
    require(status.get("state") == "completed" and status.get("final_outcome_ready") is True
            and status.get("completed_subjects") == status.get("planned_subjects") == 10,
            "Only the final complete real ten-subject analysis may be reviewed")
    for name in FINAL_FILES:
        if name == "analysis_status.json":
            continue
        declaration = status["artifacts"][name]
        require(all(identities_before[name][key] == declaration[key] for key in ("bytes", "sha256")), f"Final artifact SHA/size differs: {name}")
    analysis, summary = (read_json(input_paths[name]) for name in ("cohort_analysis.json", "cohort_summary.json"))
    manifest, source = read_json(manifest_path), read_json(source_path)
    require(identities_before["cohort_manifest.json"]["sha256"] == analysis["manifest_sha256"] == MANIFEST_SHA,
            "Exact predetermined manifest differs")
    require(identities_before["source_manifest.json"]["sha256"] == SOURCE_SHA and source["base_commit"] ==
            "ac692bb4f7868a24ea4bd67180162e81726de9b4", "Frozen accuracy source identity differs")
    require(analysis["analysis_script_sha256"] == ANALYZER_SHA and analysis["comparison_helper_sha256"] == HELPER_SHA
            and status["analysis_script_sha256"] == ANALYZER_SHA, "Frozen comparison code identity differs")
    require(analysis["final_outcome_ready"] is True and analysis["planned_subjects"] == 10 and
            analysis["planned_roi_space_rows"] == 4400 and not analysis["not_completed_attempts"], "Final scope incomplete")
    require(analysis["source_audit"]["source_manifest"]["sha256"] == SOURCE_SHA and
            analysis["source_audit"]["verified_files"] == 432 and analysis["source_audit"]["verified_runtime_python_files"] == 428,
            "Source audit scope differs")
    for key, value in analysis.items():
        if key not in ("cases", "roi_rows", "family_rows"):
            require(summary[key] == value, f"Final summary metadata differs: {key}")
    require([row["id"] for row in manifest["cases"]] == list(IDS), "Predetermined subject order differs")
    canonical = manifest["canonical_label_metadata"]
    require(len(canonical) == 110, "Canonical 110 labels required")
    families = {family: {int(label) for label, row in canonical.items() if canonical_family(row) == family}
                for family in FAMILY_SIZES}
    require(all(len(families[name]) == size for name, size in FAMILY_SIZES.items()), "Family canonical label counts")
    records = keyed(analysis["cases"], ("case_id",), "case records")
    require(set(records) == {(case,) for case in IDS}, "Ten case records required")
    for case in manifest["cases"]:
        record = records[(case["id"],)]
        require(record["status"] == "completed" and record["development_seen"] is (case["id"] == "sub-01") and
                record["numeric_gates_passed_for_completed_modes"] is True, "Every final planned subject must complete")
        require(all(record["raw_t1"][key] == case["raw_t1"][key] for key in ("path", "bytes", "sha256")), "Real public T1 identity")
    roi_rows = analysis["roi_rows"]
    roi_index = keyed(roi_rows, ("case_id", "space", "label"), "4400 ROI rows")
    expected_roi = {(case, space, int(label)) for case in IDS for space in SPACES for label in canonical}
    require(len(roi_rows) == 4400 and set(roi_index) == expected_roi, "Complete 10×4×110 unique ROI grid required")
    family_index = keyed(analysis["family_rows"], ("case_id", "space", "family"), "240 family rows")
    require(len(family_index) == 240 and set(family_index) == {(case, space, name) for case in IDS for space in SPACES for name in FAMILY_SIZES},
            "Complete 10×4×6 family grid required")
    case_index = keyed(analysis["case_rows"], ("case_id", "space"), "40 ALL110 case rows")
    require(len(case_index) == 40 and set(case_index) == {(case, space) for case in IDS for space in SPACES}, "Complete forty case-space rows")
    arithmetic = Arithmetic()
    independently_calculated_roi = {key: roi_calculation(row, canonical[str(key[2])], arithmetic) for key, row in roi_index.items()}
    independently_calculated_family = {}
    for key, group in family_index.items():
        rows = [independently_calculated_roi[(key[0], key[1], label)] for label in sorted(families[key[2]])]
        independently_calculated_family[key] = family_calculation(group, rows, records[(key[0],)], arithmetic)
    independently_calculated_case = {key: case_calculation(score, [independently_calculated_roi[(key[0], key[1], int(label))]
                                      for label in canonical], arithmetic) for key, score in case_index.items()}
    tsv_counts = {"ROI": check_tsv(input_paths["cohort_roi.tsv"], roi_rows, ("case_id", "space", "label")),
                  "family": check_tsv(input_paths["cohort_family.tsv"], analysis["family_rows"], ("case_id", "space", "family"), ("runs", "grid")),
                  "case": check_tsv(input_paths["cohort_case.tsv"], analysis["case_rows"], ("case_id", "space"))}
    distribution_count = 0
    derived_family_micro = {}
    for cohort, selected in (("cohort_all", IDS), ("cohort_new_subjects", IDS[1:])):
        declared = summary[cohort]
        require(declared["case_ids"] == list(selected) and declared["planned_subjects"] == declared["completed_subjects"] == len(selected)
                and declared["failed_or_unavailable_subjects"] == [], f"Cohort denominator/failures: {cohort}")
        require([row["case_id"] for row in declared["case_status"]] == list(selected) and all(
                row["status"] == "completed" and row["development_seen"] is (row["case_id"] == "sub-01")
                for row in declared["case_status"]), f"Cohort statuses {cohort}")
        declared_rois = keyed(declared["roi"], ("space", "label"), cohort + " ROI distributions")
        require(set(declared_rois) == {(space, int(label)) for space in SPACES for label in canonical}, f"All 440 planned ROI distributions {cohort}")
        for key, item in declared_rois.items():
            rows = [independently_calculated_roi[(case, key[0], key[1])] for case in selected]
            require(item["planned_subjects"] == item["completed_subjects"] == len(selected) and
                    item["name"] == canonical[str(key[1])]["name"] and item["family"] == rows[0]["family"], f"ROI distribution metadata {cohort}/{key}")
            statuses = {name: sum(row["hard_status"] == name for row in rows) for name in {row["hard_status"] for row in rows}}
            require(item["hard_status_counts"] == statuses and sum(statuses.values()) == len(selected), "Hard status denominators")
            require(set(item["measurements"]) == set(ROI_METRICS), "ROI distribution metric inventory")
            for measure in ROI_METRICS:
                check_stats(item["measurements"][measure], stats([row[measure] for row in rows], len(selected)), arithmetic, f"{cohort}/ROI/{key}/{measure}")
                distribution_count += 1
        declared_families = keyed(declared["family"], ("space", "family"), cohort + " family distributions")
        require(set(declared_families) == {(space, name) for space in SPACES for name in FAMILY_SIZES}, "All 24 planned family distributions")
        derived_family_micro[cohort] = []
        for key, item in declared_families.items():
            rows = [independently_calculated_family[(case, key[0], key[1])] for case in selected]
            require(item["planned_subjects"] == item["completed_subjects"] == len(selected) and
                    set(item["measurements"]) == set(FAMILY_METRICS), "Family distribution denominator/metric inventory")
            for measure in FAMILY_METRICS:
                check_stats(item["measurements"][measure], stats([row[measure] for row in rows], len(selected)), arithmetic, f"{cohort}/family/{key}/{measure}")
                distribution_count += 1
            derived_family_micro[cohort].append({"space": key[0], "family": key[1],
                                               "micro_label_dice": stats([row["micro_label_dice"] for row in rows], len(selected))})
        declared_cases = keyed(declared["case_scores"], ("space",), cohort + " ALL110 case distributions")
        require(set(declared_cases) == {(space,) for space in SPACES}, "Four planned ALL110 space distributions")
        for key, item in declared_cases.items():
            rows = [independently_calculated_case[(case, key[0])] for case in selected]
            require(set(item["measurements"]) == set(CASE_METRICS), "Case distribution metric inventory")
            for measure in CASE_METRICS:
                check_stats(item["measurements"][measure], stats([row[measure] for row in rows], len(selected)), arithmetic, f"{cohort}/ALL110/{key}/{measure}")
                distribution_count += 1
    ranking = sorted(({"case_id": case, "reference_weighted_dice": independently_calculated_case[(case, "raw_native")]["reference_weighted_dice"]}
                      for case in IDS if independently_calculated_case[(case, "raw_native")]["eligible_for_case_ranking"]),
                     key=lambda item: (item["reference_weighted_dice"], item["case_id"]))
    selection = analysis["figure_selection"]
    require([row["case_id"] for row in selection["full_ascending_raw_native_ranking"]] == [row["case_id"] for row in ranking], "Actual case ranking differs")
    for observed, computed in zip(selection["full_ascending_raw_native_ranking"], ranking):
        arithmetic.equal(observed["reference_weighted_dice"], computed["reference_weighted_dice"], "Figure ranking score")
    require(selection["median_subject"]["case_id"] == ranking[(len(ranking) - 1) // 2]["case_id"]
            and selection["worst_subject"]["case_id"] == ranking[0]["case_id"], "Predeclared lower-central median/worst selection")
    for name, path in input_paths.items():
        require(identity(path) == identities_before[name], f"Final input changed while reviewing {name}")
    require(identity(manifest_path) == identities_before["cohort_manifest.json"] and
            identity(source_path) == identities_before["source_manifest.json"], "Manifest changed while reviewing")
    report = {"schema_version": 1, "state": "passed", "final_real_cohort_review": True,
              "reviewed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "script": identity(Path(__file__)),
              "no_images_read": True, "no_gpu_or_fitting": True, "no_network_requests": True,
              "no_F8_or_analyzer_calculation_functions_used": True, "inputs": identities_before,
              "verified_counts": {"subjects_all": 10, "subjects_new": 9, "ROI_rows": 4400, "family_rows": 240,
                                  "ALL110_case_rows": 40, "TSV_rows": tsv_counts,
                                  "cohort_ROI_distributions": 880, "cohort_family_distributions": 48,
                                  "cohort_case_distributions": 8, "accuracy_measurement_distributions": distribution_count},
              "arithmetic": {"checked_fields": arithmetic.checked, "max_observed_absolute_difference": arithmetic.max_absolute_error,
                             "relative_tolerance": REL_TOL, "absolute_tolerance": ABS_TOL, "tolerance_scope": "Floating arithmetic only; integer counts and unique keys are exact.",
                             "std_ddof": 1, "NA_policy": "Both hard empty: Dice/Jaccard NA; one empty: zero; NA retained in planned 10/9 denominators."},
              "ROI_status_counts": {status: sum(row["hard_status"] == status for row in roi_rows) for status in sorted({row["hard_status"] for row in roi_rows})},
              "independent_family_micro_label_dice": derived_family_micro,
              "figure_ranking": ranking,
              "foreground_scope": "Foreground Dice recomputed from ROI marginals/diagonal intersections plus the saved categorical family different_voxels; cross-label mismatch is sum(one-hot ROI differences)-family categorical differences. Spatial confusion counts are not re-read from images.",
              "ALL110_scope": "Reported-grid label voxel counts weight the ALL110 case score; no replacement with physical-volume weighting. HR families may use different voxel sizes.",
              "excluded_scope": "Timing distributions and algorithm repeatability were not re-audited here."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Mirror containing the final six analysis files")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "cohort_manifest.json")
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).resolve().parent / "expected_source_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New review JSON; existing output will not be overwritten")
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve an existing independent review output")
    report = review(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": report["state"], "verified_counts": report["verified_counts"], "output": identity(args.output)}, indent=2))


if __name__ == "__main__":
    main()
