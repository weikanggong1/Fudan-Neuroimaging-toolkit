#!/usr/bin/env python3
"""Audit exact official-HR label-count conservation on the frozen union grid.

Only final, complete real-subject outputs are read. No fitting, GPU, external
neuroimaging command, figure or result correction is performed. The 40 original
official HR arrays supply original counts; audited analyzer ROI rows supply
their order-zero mapped counts on the exact recorded union grids. --resample
also independently repeats that mapping. Any count difference is a failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "2"

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


ANALYZER_SHA256 = "ffb42cb57c7130cf4cedcb2e54ad4e4d75b70fb1425ba783e42123658519d2bc"
HELPER_SHA256 = "beef69670191f89ebf3a0d71395c274b54b7c61d02ed096abebebd7a5172f842"
STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
MODES = ("raw", "stage")
GRID_ROUNDING_TOLERANCE_VOXELS = 1e-5
AXIS_TOLERANCE_MM = 1e-12
GRID_AFFINE_TOLERANCE_MM = 1e-10
PHASE_TOLERANCE_VOXELS = 1e-5
COUNT_TOLERANCE_VOXELS = 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def identity(specification, base):
    path = Path(specification["path"])
    if not path.is_absolute():
        path = base / path
    actual = artifact(path)
    for key in ("bytes", "sha256"):
        if key in specification and actual[key] != specification[key]:
            raise ValueError(f"Declared {key} changed: {path}")
    return actual


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def family(entry):
    return entry["parent"] + ("_" + entry["hemisphere"] if entry["source"].startswith("hippo-amygdala") else "")


def wait_for_final(analysis_dir, wait, timeout, poll):
    started = time.monotonic()
    while True:
        try:
            status = read_json(analysis_dir / "analysis_status.json")
            analysis = read_json(analysis_dir / "cohort_analysis.json")
            if (status.get("state") in {"completed", "completed_with_failures_or_unavailable"}
                    and analysis.get("final_outcome_ready") is True):
                return status, analysis
            reason = f"analysis state={status.get('state')}, final_outcome_ready={analysis.get('final_outcome_ready')}"
        except (FileNotFoundError, json.JSONDecodeError) as error:
            reason = f"analysis not ready: {type(error).__name__}"
        if not wait or time.monotonic() - started >= timeout:
            raise RuntimeError("Do not audit incomplete official/stage outputs: " + reason)
        time.sleep(poll)


def independent_union_grid(reference_shape, reference_affine, contributors):
    """Reproduce the frozen grid's geometry using metadata, without voxel reads."""
    inverse = np.linalg.inv(reference_affine)
    lower = np.zeros(3)
    upper = np.asarray(reference_shape, dtype=float) - 1
    for geometry in contributors:
        corners = np.asarray(list(itertools.product(*[(0, size - 1) for size in geometry["shape"]])))
        transformed = nib.affines.apply_affine(inverse @ np.asarray(geometry["affine"]), corners)
        lower = np.minimum(lower, transformed.min(axis=0))
        upper = np.maximum(upper, transformed.max(axis=0))
    lower = np.floor(lower + GRID_ROUNDING_TOLERANCE_VOXELS).astype(np.int64)
    upper = np.ceil(upper - GRID_ROUNDING_TOLERANCE_VOXELS).astype(np.int64)
    affine = reference_affine.copy()
    affine[:3, 3] += affine[:3, :3] @ lower
    return tuple(map(int, upper - lower + 1)), affine, lower.tolist(), upper.tolist()


def shared_recorded_grid(case_id, labels, canonical, analysis):
    families = {family(canonical[str(label)]) for label in labels}
    rows = [row for row in analysis["family_rows"] if row["case_id"] == case_id
            and row["family"] in families and row["space"] in {"raw_hr", "stage_hr"}]
    if len(rows) != 2 * len(families):
        raise ValueError("Need one recorded family HR grid per completed raw/stage mode")
    if len({(row["family"], row["space"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate family/mode grid records")
    first = rows[0]["grid"]
    for row in rows[1:]:
        other = row["grid"]
        if first["shape"] != other["shape"] or not np.array_equal(np.asarray(first["affine"]), np.asarray(other["affine"])):
            raise ValueError("Raw/stage or hippocampus/amygdala recorded union grids differ")
    return first, [{"family": row["family"], "space": row["space"],
                    "metric_crop_start": row["grid"].get("metric_crop_start"),
                    "metric_crop_stop": row["grid"].get("metric_crop_stop")} for row in rows]


def label_counts(values, labels, offset):
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all() or np.any(values < 0):
        raise ValueError("Official HR label array must contain finite nonnegative integers")
    # Query original disk IDs directly; this is exactly one nonzero +10000
    # namespace translation, without touching background or double offsetting.
    return {label: int(np.count_nonzero(values == label - offset)) for label in labels}


def boundary_counts(values, labels, offset):
    """Retain face counts to diagnose interpolation-boundary loss if it occurs."""
    result = {label: {} for label in labels}
    for axis in range(3):
        for side, index in (("first", 0), ("last", values.shape[axis] - 1)):
            face = np.take(values, index, axis=axis)
            unique, counts = np.unique(face, return_counts=True)
            counts = dict(zip(unique.tolist(), counts.tolist()))
            for label in labels:
                result[label][f"axis_{axis}_{side}"] = int(counts.get(label - offset, 0))
    return result


def audit_map(case, record, structure, canonical, analysis, base, do_resample):
    labels = sorted(int(key) for key, entry in canonical.items() if entry["source"] == structure)
    source = identity(record["official_outputs"][structure]["arrays"]["hr"], base)
    if Path(source["path"]) != Path(case["official"]["subregions"][structure]["hr"]):
        raise ValueError("Official HR source differs from the predetermined cohort manifest")
    image = nib.load(source["path"])
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError("Invalid original official HR geometry")
    grid, grid_sources = shared_recorded_grid(case["id"], labels, canonical, analysis)
    shape = tuple(grid["shape"])
    target = np.asarray(grid["affine"], dtype=np.float64)
    contributors = [record["fnit"][mode]["arrays"][f"highres/{structure}"]["geometry"] for mode in MODES]
    recomputed_shape, recomputed_affine, lower, upper = independent_union_grid(image.shape, image.affine, contributors)
    source_to_target = np.linalg.inv(target) @ image.affine
    target_to_source = np.linalg.inv(image.affine) @ target
    phase = target_to_source[:3, 3]
    phase_residual = phase - np.rint(phase)
    axes_error = float(np.max(np.abs(target[:3, :3] - image.affine[:3, :3])))
    union_error = float(np.max(np.abs(target - recomputed_affine)))
    geometry_passed = (shape == recomputed_shape and axes_error <= AXIS_TOLERANCE_MM
        and union_error <= GRID_AFFINE_TOLERANCE_MM
        and np.max(np.abs(phase_residual)) <= PHASE_TOLERANCE_VOXELS
        and np.array_equal(np.rint(phase).astype(np.int64), lower))
    values = np.asarray(image.dataobj)
    offset = 10000 if structure.endswith("right") else 0
    original_counts = label_counts(values, labels, offset)
    original_boundary_counts = boundary_counts(values, labels, offset)
    already_offset_nonzero_voxels = int(np.count_nonzero(np.isin(values, labels))) if offset else 0
    if offset and already_offset_nonzero_voxels:
        raise ValueError("Official right HR image already contains global +10000 requested IDs; do not offset a second time")
    by_label_mode = {}
    for row in analysis["roi_rows"]:
        if row["case_id"] == case["id"] and row["label"] in labels and row["space"] in {"raw_hr", "stage_hr"}:
            key = row["label"], row["mode"]
            if key in by_label_mode or row["measurement_status"] != "evaluated":
                raise ValueError("Repeated or unevaluated official-HR ROI measurement")
            by_label_mode[key] = row
    if len(by_label_mode) != 2 * len(labels):
        raise ValueError("Every requested label needs both raw-HR and stage-HR mapped official counts")
    repeated_counts = None
    resample_skipped_as_same_grid = None
    if do_resample:
        resample_skipped_as_same_grid = image.shape == shape and np.allclose(image.affine, target, atol=1e-5, rtol=0)
        mapped_image = image if resample_skipped_as_same_grid else resample_from_to(image, (shape, target), order=0, mode="constant", cval=0)
        repeated_counts = label_counts(np.asarray(mapped_image.dataobj), labels, offset)
    if artifact(source["path"]) != source:
        raise ValueError("Official HR source changed while reading/counting; do not compare moving inputs")
    roi = []
    for label in labels:
        raw = by_label_mode[label, "raw"]["official_voxels"]
        stage = by_label_mode[label, "stage"]["official_voxels"]
        original = original_counts[label]
        repeated = repeated_counts[label] if repeated_counts is not None else None
        conserved = raw == original and stage == original
        repeats_match = repeated is None or repeated == raw == stage
        roi.append({"label": label, "name": canonical[str(label)]["name"], "disk_label_id": label - offset,
            "original_voxels": original, "analyzer_raw_hr_voxels": raw, "analyzer_stage_hr_voxels": stage,
            "original_boundary_face_voxels": original_boundary_counts[label],
            "raw_hr_count_delta": raw - original, "stage_hr_count_delta": stage - original,
            "independently_resampled_voxels": repeated,
            "independent_resample_count_delta": repeated - original if repeated is not None else None,
            "both_original_and_mapped_empty": original == raw == stage == 0,
            "counts_exactly_conserved": conserved, "independent_resample_matches_analyzer": repeats_match})
    return {"case_id": case["id"], "structure": structure,
        "state": "passed" if geometry_passed and all(row["counts_exactly_conserved"] and row["independent_resample_matches_analyzer"] for row in roi) else "failed",
        "source": source, "original_shape": list(image.shape), "original_affine": image.affine.tolist(),
        "original_spacing_mm": np.linalg.norm(image.affine[:3, :3], axis=0).tolist(),
        "union_shape": list(shape), "union_affine": target.tolist(),
        "union_spacing_mm": np.linalg.norm(target[:3, :3], axis=0).tolist(),
        "recorded_grid_family_sources": grid_sources,
        "fnit_grid_contributors": {mode: {key: value for key, value in record["fnit"][mode]["arrays"][f"highres/{structure}"].items()
                                          if key in ("path", "bytes", "sha256", "geometry")} for mode in MODES},
        "geometry": {"passed": bool(geometry_passed), "target_voxels_to_original_voxels": target_to_source.tolist(),
            "original_voxels_to_target_voxels": source_to_target.tolist(), "integer_lower": lower, "integer_upper": upper,
            "measured_integer_phase": np.rint(phase).astype(np.int64).tolist(), "phase_residual_voxels": phase_residual.tolist(),
            "axis_affine_max_abs_error_mm": axes_error, "recomputed_union_affine_max_abs_error_mm": union_error},
        "nonzero_namespace_offset": offset, "already_offset_requested_voxels_on_disk": already_offset_nonzero_voxels,
        "mapped_counts_source": "Frozen cohort_analysis ROI official_voxels after order=0 on its recorded union grid; background-only metric crop cannot alter label counts",
        "independent_resampling_repeated": do_resample, "same_grid_interpolation_skipped": resample_skipped_as_same_grid,
        "roi": roi}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resample", action="store_true", help="Also independently repeat all forty order-zero resamplings")
    parser.add_argument("--wait", action="store_true", help="Wait for final outcome-ready analysis before reading any image")
    parser.add_argument("--timeout-seconds", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.timeout_seconds <= 0:
        raise ValueError("poll-seconds must be 1–60; timeout must be positive")
    root = args.root.resolve()
    analysis_dir = args.analysis_dir or root / "analysis"
    output = args.output or root / "official_hr_grid_conservation.json"
    if output.exists():
        raise ValueError("Preserve existing audit; choose a fresh --output path")
    affinity_before = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if affinity_before:
        os.sched_setaffinity(0, affinity_before[-2:])
    status, analysis = wait_for_final(analysis_dir, args.wait, args.timeout_seconds, args.poll_seconds)
    manifest_path = root / "cohort_manifest.json"
    manifest = read_json(manifest_path)
    analyzer_path = root / "analyze_cohort.py"
    helper_path = root.parent / "reproducibility_20261002/analyze_repeatability.py"
    if sha256(analyzer_path) != ANALYZER_SHA256 or sha256(helper_path) != HELPER_SHA256:
        raise ValueError("Frozen analyzer/helper changed; do not silently audit another policy")
    analysis_path = analysis_dir / "cohort_analysis.json"
    analysis_identity = identity({**status["artifacts"]["cohort_analysis.json"], "path": str(analysis_path)}, root)
    # Bind the JSON actually inspected to the finalized artifact fingerprint.
    analysis = read_json(analysis_path)
    if sha256(analysis_path) != analysis_identity["sha256"]:
        raise ValueError("Final analysis changed during metadata loading")
    if (analysis["analysis_script_sha256"] != ANALYZER_SHA256 or analysis["comparison_helper_sha256"] != HELPER_SHA256
            or analysis["manifest_sha256"] != sha256(manifest_path)):
        raise ValueError("Final analysis is not tied to the frozen implementation and manifest")
    expected_ids = [f"sub-{index:02d}" for index in range(1, 11)]
    records = {record["case_id"]: record for record in analysis["cases"]}
    if ([case["id"] for case in manifest["cases"]] != expected_ids or sorted(records) != expected_ids
            or any(record["status"] != "completed" or set(record.get("fnit", {})) != set(MODES) for record in records.values())
            or analysis.get("not_completed_attempts")):
        raise ValueError("Require ten complete official/raw/stage outcomes; no partial cohort conservation claim")
    canonical = manifest["canonical_label_metadata"]
    if (len(canonical) != 110 or {entry["source"] for entry in canonical.values()} != set(STRUCTURES)
            or any(int(key) <= 0 or int(entry["id"]) != int(key) for key, entry in canonical.items())
            or [sum(entry["source"] == structure for entry in canonical.values()) for structure in STRUCTURES] != [4, 50, 28, 28]
            or any(int(key) < 10000 for key, entry in canonical.items() if entry["source"].endswith("right"))):
        raise ValueError("Need exactly the declared 110 labels across the four structures")
    result = {"schema_version": 1, "state": "auditing", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit_script": artifact(Path(__file__)), "frozen_analyzer": artifact(analyzer_path), "comparison_helper": artifact(helper_path),
        "cohort_manifest": artifact(manifest_path), "cohort_analysis": analysis_identity,
        "analysis_status": artifact(analysis_dir / "analysis_status.json"),
        "planned_subjects": 10, "planned_official_hr_maps": 40, "planned_label_map_rows": 1100,
        "cpu_only": True, "no_fitting": True, "no_figures": True, "no_result_corrections": True,
        "cpu_limit": {"environment": {key: os.environ[key] for key in THREAD_VARIABLES},
            "allowed_cores": sorted(os.sched_getaffinity(0)) if affinity_before else None},
        "tolerances": {"grid_rounding_voxels": GRID_ROUNDING_TOLERANCE_VOXELS,
            "axis_affine_mm": AXIS_TOLERANCE_MM, "union_affine_mm": GRID_AFFINE_TOLERANCE_MM,
            "integer_phase_voxels": PHASE_TOLERANCE_VOXELS, "label_counts_voxels": COUNT_TOLERANCE_VOXELS},
        "scope": "Read every original official HR array. Compare all requested original label counts against frozen analyzer raw_hr and stage_hr counts on the same official-axis/spacing/integer-phase union grid. Exact integer count equality required, including empty labels.",
        "independent_order_zero_resampling_requested": args.resample, "maps": []}
    write_json(output, result)
    started = time.monotonic()
    cpu_started = time.process_time()
    for case in manifest["cases"]:
        for structure in STRUCTURES:
            try:
                audit = audit_map(case, records[case["id"]], structure, canonical, analysis, root, args.resample)
            except Exception as error:
                audit = {"case_id": case["id"], "structure": structure, "state": "failed",
                         "error": f"{type(error).__name__}: {error}", "roi": []}
            result["maps"].append(audit)
            write_json(output, result)
            print(json.dumps({"case_id": case["id"], "structure": structure, "state": audit["state"],
                "changed_labels": sum(not row["counts_exactly_conserved"] for row in audit["roi"]),
                "error": audit.get("error")}), flush=True)
    rows = [row for mapping in result["maps"] for row in mapping["roi"]]
    failed_maps = [mapping for mapping in result["maps"] if mapping["state"] != "passed"]
    metadata_unchanged = sha256(analysis_path) == analysis_identity["sha256"] and sha256(manifest_path) == result["cohort_manifest"]["sha256"]
    result["summary"] = {"audited_maps": len(result["maps"]), "fully_counted_label_map_rows": len(rows),
        "failed_maps": [{key: mapping.get(key) for key in ("case_id", "structure", "error")} for mapping in failed_maps],
        "nonconserved_label_map_rows": sum(not row["counts_exactly_conserved"] for row in rows),
        "independent_resample_disagreements": sum(not row["independent_resample_matches_analyzer"] for row in rows),
        "both_original_and_mapped_empty_rows": sum(row["both_original_and_mapped_empty"] for row in rows),
        "metadata_unchanged": metadata_unchanged}
    passed = len(result["maps"]) == 40 and len(rows) == 1100 and not failed_maps and metadata_unchanged
    result.update(state="passed" if passed else "failed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  elapsed_wall_audit_seconds=time.monotonic() - started,
                  elapsed_cpu_process_seconds=time.process_time() - cpu_started,
                  nibabel_version=nib.__version__, numpy_version=np.__version__)
    write_json(output, result)
    print(json.dumps({"state": result["state"], "summary": result["summary"], "output_sha256": sha256(output)}), flush=True)
    if not passed:
        raise RuntimeError("Official-HR conservation audit FAILED; differences preserved, no results repaired or ignored")


if __name__ == "__main__":
    main()
