#!/usr/bin/env python3
"""Independently review and curate the final regression's text artifacts only.

This script reads no image, imports no imaging/GPU library, contacts no server,
and launches no jobs. Saved server audit assertions are checked for internal
consistency, source/input binding, unique complete rows and independently
recomputed arithmetic. MRI array comparison was performed by the separate
server auditor; this text review does not repeat that image examination.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil


IDS = [f"sub-{i:02d}" for i in range(1, 11)]
BASE_COMMIT = "ac692bb4f7868a24ea4bd67180162e81726de9b4"
MAIN_COMMIT = "f436de588647a0de80735e4a98d53df5d88e502d"
SUMMARY_SHA = "cd497c2dd130c4238c1314a5b5cf7cb9f5f59a59b7910646459f03e1baafebf4"
BASE_GATE_SHA = "820774cb5af76224b427f7495d8bdcbfc75f87473e3d95d3b5018c6fdb476059"
TIME_FIELDS = (
    "api_compute_seconds", "api_total_seconds", "output_save_seconds",
    "process_wall_seconds", "context_observer_seconds", "input_wait_seconds",
    "gpu_budget_wait_seconds", "preflight_identity_seconds",
)
MAPS = {"labels", "highres/brainstem", "highres/thalamus",
        "highres/hippo-amygdala-left", "highres/hippo-amygdala-right"}
CONTEXT_ARRAYS = {"data", "coarse_segmentation", "cortical_parcellation",
                  "wmparc_proxy", "brain_mask", "image_geometry/affine_identity",
                  "image_geometry/shape", "image_geometry/affine"}
PRESERVED = ("raw_launch.json", "audit_launch.json", "source_manifest.json",
             "source_diff.json", "source_export_identity.json", "source_metadata_copy.json")
COPIES = {
    "cohort_manifest.json": "cohort_manifest.json",
    "fnit_raw_queue.json": "fnit_raw_queue.json",
    "raw_regression_audit_launch.json": "raw_regression_audit_launch.json",
    "raw_regression_audit/summary.json": "audit/summary.json",
    "raw_regression_audit/case.tsv": "audit/case.tsv",
    "raw_regression_audit/map_pairs.tsv": "audit/map_pairs.tsv",
    "raw_regression_audit/volume_pairs.tsv": "audit/volume_pairs.tsv",
    "raw_regression_audit/context_pairs.tsv": "audit/context_pairs.tsv",
}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def identity(path):
    value = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest()}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def match_identity(path, declared):
    observed = identity(path)
    require(all(observed[key] == declared[key] for key in ("bytes", "sha256")),
            f"Identity mismatch: {path}")
    return observed


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"Duplicate columns: {path}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"Extra cells in {path}")
    return rows


def keyed(rows, fields, label):
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(keys) == len(set(keys)), f"Duplicate keys in {label}")
    return dict(zip(keys, rows))


def equal_tsv(row, expected, label):
    require(set(row) == set(expected), f"Column mismatch: {label}")
    for key, value in expected.items():
        actual = row[key]
        if isinstance(value, (list, dict)):
            require(json.loads(actual) == value, f"Serialized object mismatch: {label}/{key}")
        elif value is None:
            require(actual == "", f"Expected empty NA: {label}/{key}")
        elif isinstance(value, bool):
            require(actual == str(value), f"Boolean mismatch: {label}/{key}")
        elif isinstance(value, (float, int)):
            require(math.isfinite(float(actual)) and float(actual) == float(value),
                    f"Numeric mismatch: {label}/{key}")
        else:
            require(actual == value, f"Text mismatch: {label}/{key}")


def independent_stats(values, planned):
    x = sorted(float(value) for value in values if value is not None)
    require(len(x) <= planned and all(math.isfinite(value) for value in x), "Invalid denominator/value")
    n = len(x)
    mean = math.fsum(x) / n if n else None
    median = (x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2) if n else None
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in x) / (n - 1)) if n > 1 else None
    return {"planned_subjects": planned, "n": n, "missing_or_na": planned - n,
            "mean": mean, "median": median, "std_sample": sd, "std_ddof": 1,
            "min": min(x) if n else None, "max": max(x) if n else None}


def check_stats(actual, expected, pointer):
    require(set(actual) == set(expected), f"Stat fields mismatch: {pointer}")
    for key, value in actual.items():
        if isinstance(value, float):
            require(isinstance(expected[key], (int, float)) and
                    math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-10),
                    f"Independent arithmetic differs: {pointer}/{key}")
        else:
            require(value == expected[key], f"Denominator/NA mismatch: {pointer}/{key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    src = args.evidence_root.resolve()
    validation = args.validation_root.resolve()
    dest = validation / "latest_main_regression"
    preserved_before = {name: identity(dest / name) for name in PRESERVED}
    fetched_ids = {name: identity(src / name) for name in COPIES}
    require(fetched_ids["raw_regression_audit/summary.json"]["sha256"] == SUMMARY_SHA and
            fetched_ids["raw_regression_audit/summary.json"]["bytes"] == 897617, "Pinned summary changed")
    summary = read_json(src / "raw_regression_audit/summary.json")
    main_manifest = read_json(src / "cohort_manifest.json")
    old_manifest = read_json(validation / "cohort_manifest.json")
    main_queue = read_json(src / "fnit_raw_queue.json")
    base_gate = read_json(validation / "raw_all_live_audit.json")
    require(identity(validation / "raw_all_live_audit.json")["sha256"] == BASE_GATE_SHA, "Baseline gate changed")
    match_identity(validation / "cohort_manifest.json", summary["manifests"]["baseline"])
    match_identity(src / "cohort_manifest.json", summary["manifests"]["main"])
    match_identity(src / "fnit_raw_queue.json", summary["final_queue"])
    match_identity(validation / "audit_main_raw_regression.py", summary["audit_script"])
    match_identity(validation / "analyze_cohort.py", summary["frozen_analyzer"])
    require(summary["baseline_commit"] == BASE_COMMIT and summary["main_commit"] == MAIN_COMMIT, "Wrong commits")
    require(summary["state"] == "zero_voxel_equivalence_passed" and
            summary["zero_voxel_equivalence_passed"] is True and
            summary["final_main_regression_outcomes_ready"] is True and
            summary["final_source_asset_metadata_error"] is None, "Incomplete final audit")
    for key in ("cpu_only", "no_fitting", "no_official_images_read", "no_result_corrections", "raw_output_equivalence_only"):
        require(summary[key] is True, f"Wrong scope: {key}")
    require(summary["official_accuracy_compared"] is False and summary["primary_full_cohort_benchmark"] is False,
            "This is a raw version regression, not final official comparison")
    require(summary["cpu_resource_limit"]["threads"] == 2, "Auditor thread count")
    require(summary["planned_subjects"] == 10 and summary["planned_new_subjects"] == 9 and
            (summary["planned_map_pairs"], summary["planned_volume_pairs"], summary["planned_context_pairs"]) == (50, 1100, 160),
            "Planned denominators changed")
    sources = {"baseline": read_json(validation / "expected_source_manifest.json"),
               "main": read_json(dest / "source_manifest.json")}
    source_ids = {}
    for version, filename, commit, count, runtime_count in (
        ("baseline", validation / "expected_source_manifest.json", BASE_COMMIT, 432, 428),
        ("main", dest / "source_manifest.json", MAIN_COMMIT, 439, 435),
    ):
        metadata = sources[version]
        entries = keyed(metadata["files"], ("path",), f"{version} source")
        require(len(entries) == count and metadata["base_commit"] == commit, "Source file/commit count")
        runtime = {entry["path"].removeprefix("src/fnit/"): entry["sha256"] for entry in metadata["files"]
                   if entry["path"].startswith("src/fnit/") and entry["path"].endswith(".py")}
        require(len(runtime) == runtime_count and json_sha(runtime) == summary["runtime_map_sha256"][version], "Runtime hash map")
        source_ids[version] = match_identity(filename, summary["source_audits"][version]["source_manifest"])
        require(summary["source_audits"][version]["verified_files"] == count and
                summary["source_audits"][version]["verified_runtime_python_files"] == runtime_count, "Source audit count")
    old_files = {entry["path"]: entry for entry in sources["baseline"]["files"]}
    new_files = {entry["path"]: entry for entry in sources["main"]["files"]}
    gems = {path for path in old_files if path.startswith("src/fnit/gems/")}
    require(len(gems) == 24 and gems == {p for p in new_files if p.startswith("src/fnit/gems/")} and
            all(old_files[path] == new_files[path] for path in gems), "All 24 GEMS identities must match")
    require([case["id"] for case in main_manifest["cases"]] == IDS and
            [case["id"] for case in old_manifest["cases"]] == IDS, "Fixed case order changed")
    for key in ("dataset", "snapshot", "license", "dataset_doi", "canonical_label_metadata", "fnit_configuration"):
        require(main_manifest[key] == old_manifest[key], f"Fixed manifest field changed: {key}")
    canonical = main_manifest["canonical_label_metadata"]
    require(len(canonical) == 110, "110 ROI denominator")
    for before, after in zip(old_manifest["cases"], main_manifest["cases"]):
        for key in ("raw_t1", "official", "development_seen"):
            require(before[key] == after[key], f"Input/reference/development differs: {before['id']}/{key}")
        require(after["development_seen"] == (after["id"] == "sub-01"), "Only sub-01 development-seen")
        require(before["case_root"] != after["case_root"], "Separate version output trees required")
    require(main_manifest["reference_used_for_raw_fnit_fitting"] is False and
            old_manifest["reference_used_for_raw_fnit_fitting"] is False, "Reference used in raw fitting")
    require(main_manifest["source"]["base_commit"] == MAIN_COMMIT and
            main_manifest["source"]["manifest_sha256"] == source_ids["main"]["sha256"], "Main manifest source binding")
    require(main_queue["state"] == summary["final_queue_state"] == "completed" and
            main_queue["planned_cases"] == main_queue["successful_runs"] == 10 and
            main_queue["failed_or_blocked_runs"] == 0 and main_queue["modes"] == ["raw"], "Raw queue incomplete")
    require(main_queue["source_base_commit"] == MAIN_COMMIT and main_queue["runtime_python_files"] == 435 and
            main_queue["source"]["sha256"] == source_ids["main"]["sha256"], "Queue source binding")
    match_identity(src / "cohort_manifest.json", main_queue["manifest"])
    require(main_queue["optimization"] == "fast" and main_queue["threads_per_process"] == 4 and
            main_queue["own_memory_limit_mib"] == 19073 and main_queue["reference_used_for_fitting"] is False and
            main_queue["observer_changes_solver_options"] is False, "Default runtime configuration changed")
    assets = keyed(main_queue["assets"], ("path",), "fixed assets")
    require(len(assets) == summary["verified_asset_files"] == 40 and
            json_sha(sorted(main_queue["assets"], key=lambda row: row["path"])) == summary["fixed_asset_inventory_sha256"],
            "Fixed asset inventory identity")
    require((src / "raw_regression_audit_launch.json").read_bytes() == (dest / "audit_launch.json").read_bytes(),
            "Final auditor launch differs from retained launch")
    cases = keyed(summary["cases"], ("case_id",), "summary cases")
    runs = keyed(main_queue["runs"], ("case_id",), "main runs")
    gates = keyed(base_gate["cases"], ("case_id",), "baseline gates")
    require(set(cases) == set(runs) == set(gates) == {(case,) for case in IDS}, "Incomplete case keys")
    case_tsv = keyed(read_tsv(src / "raw_regression_audit/case.tsv"), ("case_id",), "case TSV")
    require(set(case_tsv) == set(cases), "Case TSV keys")
    gpu_instance_ids = {version: set() for version in ("baseline", "main")}
    for case_id in IDS:
        case, run, gate = cases[(case_id,)], runs[(case_id,)], gates[(case_id,)]
        require(case["state"] == "zero_voxel_equivalence_passed" and case["error"] is None,
                f"Non-exact/failed planned case {case_id}")
        require(case["development_seen"] == (case_id == "sub-01"), "Wrong case development flag")
        raw_input = next(row["raw_t1"] for row in main_manifest["cases"] if row["id"] == case_id)
        require(case["input"] == raw_input and run["inputs"] == [raw_input], "Actual input file identity")
        require(run["status"] == "success" and run["state"] == "completed" and
                run["phase"] == "verified_full_run" and run["exit_code"] == 0, "Non-complete run")
        require(json_sha(run) == case["queue_run_sha256"]["main"], "Final raw record SHA")
        expected = {key: case[key] for key in ("case_id", "development_seen", "state", "error")}
        expected.update(case["counts"])
        require(case["counts"] == {"compared_maps": 5, "different_maps": 0, "different_voxels": 0,
                                    "sum_defined_same_index_voxel_differences": 0, "maps_without_comparable_voxel_count": 0,
                                    "compared_volume_entries": 110, "different_volume_entries": 0,
                                    "compared_context_entries": 16, "different_context_entries": 0}, "Case planned/difference count")
        for version in ("baseline", "main"):
            row = case[version]
            require(row["input_sha256"] == {raw_input["path"]: raw_input["sha256"]} and
                    row["source_manifest_sha256"] == source_ids[version]["sha256"], "Input/source observed binding")
            require(row["configuration"] == {"optimization": "fast", "torch_version": "2.5.1", "cuda_version": "11.8"},
                    "Actual version/runtime configuration")
            require(all(value is True for key, value in row["numeric_gates"].items() if key != "jacobian_scope"),
                    "Failed numeric gate")
            require(len(row["fit_min_jacobians"]) == 4 and
                    all(math.isfinite(value) and value > 0 for value in row["fit_min_jacobians"].values()), "Fitted mesh Jacobian gate")
            gpu = row["gpu"]
            require(gpu["sampling_errors"] == 0 and gpu["identity"]["predeclared_identity_verified"] is True and
                    gpu["identity"]["samples"] > 0 and gpu["identity"]["sampled_peak_own_memory_mib"] == gpu["own_sampled_peak_mib"] and
                    gpu["own_limit_mib"] == 19073 and gpu["own_sampled_peak_mib"] <= 19073, "GPU monitor peak gate")
            require(gpu["physical_gpu"]["uuid"] == "GPU-e25cac06-0ce8-a833-abf9-09ab18c9c9ba", "Different raw GPU")
            gpu_instance_ids[version].add((gpu["own_pid"], gpu["launch_unix"]))
            expected.update({f"{version}_{key}": row["timings"][key] for key in TIME_FIELDS})
            expected[f"{version}_own_sampled_peak_mib"] = gpu["own_sampled_peak_mib"]
            if version == "baseline":
                require(row["input_sha256"] == gate["raw_input_sha256"] and row["configuration"] == gate["configuration"] and
                        row["numeric_gates"] == gate["numeric_gates"] and row["fit_min_jacobians"] == gate["fit_min_jacobians"],
                        "Baseline independent gate changed")
                require(all(row["timings"][key] == gate["timings"][key] for key in TIME_FIELDS), "Baseline observed timing binding")
                require(gpu["identity"] == gate["own_gpu_monitor"] and gpu["own_pid"] == gate["own_pid"], "Baseline monitor binding")
                require(gate["source_observer"]["runtime_before_map_sha256"] == gate["source_observer"]["runtime_after_map_sha256"] ==
                        summary["runtime_map_sha256"][version], "Baseline before/after source map")
            else:
                require(gpu["identity"]["sha256"] == run["gpu_monitor"]["sha256"] and gpu["own_pid"] == run["pid"] and
                        gpu["launch_unix"] == run["started_unix"] and row["observer"]["sha256"] == run["context_identity"]["sha256"],
                        "Main observed monitor/process/observer identity")
                aliases = {"api_compute_seconds": "compute_seconds", **{key: key for key in TIME_FIELDS if key != "api_compute_seconds"}}
                require(all(row["timings"][key] == run[name] for key, name in aliases.items()), "Main queue timing binding")
                require(run["input_sha256"] == row["input_sha256"] and
                        run["source_manifest_sha256"] == row["source_manifest_sha256"], "Main run identity binding")
        require(case["numeric_gates"] == {version: case[version]["numeric_gates"] for version in ("baseline", "main")},
                "Case numeric gate copies")
        for choices in case["preprocessing_choices"]:
            require(choices["exact_without_recorded_timers"] is True and choices["baseline"] == choices["main"], "Preprocessing choices")
        equal_tsv(case_tsv[(case_id,)], expected, case_id)
    require(all(len(ids) == 10 for ids in gpu_instance_ids.values()), "Unique ten processes per version")
    map_rows = keyed(read_tsv(src / "raw_regression_audit/map_pairs.tsv"), ("case_id", "map"), "map TSV")
    summary_maps = keyed(summary["map_pairs"], ("case_id", "map"), "summary maps")
    require(set(map_rows) == set(summary_maps) == {(case, name) for case in IDS for name in MAPS}, "All 50 map keys")
    for key, row in summary_maps.items():
        equal_tsv(map_rows[key], row, "/".join(key))
        require(row["measurement_status"] == "compared" and row["different_voxels"] == 0 and
                all(row[field] is True for field in ("shape_exact", "affine_exact", "dtype_exact", "all_values_exact", "exact")),
                "Non-exact map")
        for suffix in ("file_sha256", "array_sha256", "shape", "affine", "array_dtype", "header_dtype"):
            require(row[f"baseline_{suffix}"] == row[f"main_{suffix}"], f"Map {suffix} differs")
        saved = gates[(key[0],)]["native_and_hr_label_arrays"][key[1]]
        require(row["baseline_file_sha256"] == saved["sha256"] and row["baseline_shape"] == saved["geometry"]["shape"] and
                row["baseline_affine"] == saved["geometry"]["affine"], "Baseline independent map identity/geometry")
    volume_rows = keyed(read_tsv(src / "raw_regression_audit/volume_pairs.tsv"), ("case_id", "label"), "volume TSV")
    require(set(volume_rows) == {(case, label) for case in IDS for label in canonical}, "All 1100 ROI keys")
    for key, row in volume_rows.items():
        require(row["measurement_status"] == "compared" and row["exact"] == row["entire_dictionary_exact"] == "True" and
                row["baseline_entry_sha256"] == row["main_entry_sha256"] and
                row["name"] == canonical[key[1]]["name"], "Volume dictionary/name identity")
        for scope in ("hard", "soft"):
            a, b, delta = (float(row[field]) for field in
                           (f"baseline_{scope}_mm3", f"main_{scope}_mm3", f"{scope}_delta_mm3"))
            require(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 and a == b and delta == b - a == 0,
                    "Actual volume delta/value gate")
    context_rows = keyed(read_tsv(src / "raw_regression_audit/context_pairs.tsv"), ("case_id", "phase", "array"), "context TSV")
    summary_context = keyed(summary["context_pairs"], ("case_id", "phase", "array"), "summary context")
    require(set(context_rows) == set(summary_context) == {(case, phase, name) for case in IDS
            for phase in ("prepared_context", "raw_processing_context") for name in CONTEXT_ARRAYS}, "All 160 context keys")
    for key, row in summary_context.items():
        # CSV has the union of row columns; strides only exist for fingerprint rows.
        equal_tsv(context_rows[key], {**row, **({"strides_equal": None} if "strides_equal" not in row else {})}, "/".join(key))
        require(row["measurement_status"] == "compared" and row["exact"] is True and row["baseline"] == row["main"],
                "Observer fingerprint/geometry difference")
        if isinstance(row["baseline"], dict):
            require(row["baseline"]["status"] == "observed" and len(row["baseline"]["sha256"]) == 64 and
                    row["strides_equal"] is True, "Invalid observed fingerprint")
    require(summary["summary_counts"] == {"audited_subjects": 10, "exact_subjects": 10, "fully_compared_maps": 50,
            "fully_compared_volume_entries": 1100, "fully_compared_context_entries": 160,
            "different_maps": 0, "different_volume_entries": 0}, "Aggregate counts")
    distributions_checked = 0
    recalculated = {}
    for group, ids in (("cohort_all", IDS), ("cohort_new_subjects", IDS[1:])):
        declared = summary[group]
        require(declared["case_ids"] == ids and declared["planned_subjects"] == declared["compared_subjects"] == len(ids) and
                declared["failed_or_unavailable_subjects"] == [], "Cohort denominator or unavailable records")
        selected = [cases[(case,)] for case in ids]
        measured = declared["measurements"]
        computed = {"baseline": {}, "main": {}, "paired_main_over_baseline": {}, "recipe_timings": {}}
        for version in ("baseline", "main"):
            require(set(measured[version]) == set(TIME_FIELDS) | {"own_sampled_peak_mib"}, "Cohort timing fields")
            for key in measured[version]:
                values = [case[version]["gpu"][key] if key == "own_sampled_peak_mib" else case[version]["timings"][key]
                          for case in selected]
                computed[version][key] = independent_stats(values, len(ids))
                check_stats(computed[version][key], measured[version][key], f"{group}/{version}/{key}")
                distributions_checked += 1
        for key in TIME_FIELDS:
            values = [case["main"]["timings"][key] / case["baseline"]["timings"][key] for case in selected]
            computed["paired_main_over_baseline"][key] = independent_stats(values, len(ids))
            check_stats(computed["paired_main_over_baseline"][key], measured["paired_main_over_baseline"][key], f"{group}/paired/{key}")
            distributions_checked += 1
        recipe_keys = {key for case in selected for version in ("baseline", "main") for key in case[version]["recipe_timings"]}
        require(recipe_keys == set(measured["recipe_timings"]), "Recipe timer key inventory")
        for key in sorted(recipe_keys):
            computed["recipe_timings"][key] = {}
            for version in ("baseline", "main"):
                values = [case[version]["recipe_timings"].get(key) for case in selected]
                computed["recipe_timings"][key][version] = independent_stats(values, len(ids))
                check_stats(computed["recipe_timings"][key][version], measured["recipe_timings"][key][version], f"{group}/recipe/{key}/{version}")
                distributions_checked += 1
        recalculated[group] = {"planned_subjects": len(ids), "case_ids": ids, "measurements": computed}
    copied = []
    for name, target in COPIES.items():
        source, curated = src / name, dest / target
        curated.parent.mkdir(parents=True, exist_ok=True)
        if curated.exists():
            require(curated.read_bytes() == source.read_bytes(), f"Refusing to replace changed final file: {curated}")
        else:
            shutil.copyfile(source, curated)
        actual = identity(curated)
        require(all(actual[key] == fetched_ids[name][key] for key in ("bytes", "sha256")) and
                curated.read_bytes() == source.read_bytes(), "Copy bytes changed")
        copied.append({"source": str(source), "curated_file": target, "bytes": actual["bytes"],
                       "sha256": actual["sha256"], "byte_identical": True})
    preserved_after = {name: identity(dest / name) for name in PRESERVED}
    require(preserved_before == preserved_after, "Prior preparation metadata modified")
    receipt = {
        "schema_version": 1, "state": "passed", "reviewed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "review_script": identity(Path(__file__)), "scope": "Independent local pure-text integrity and arithmetic review",
        "no_images_read": True, "no_server_requests": True, "no_jobs_launched": True,
        "server_array_audit_repeated_locally": False,
        "baseline_commit": BASE_COMMIT, "main_commit": MAIN_COMMIT,
        "files": copied, "preserved_preparation_files": preserved_after,
        "baseline_source_manifest": source_ids["baseline"], "main_source_manifest": source_ids["main"],
        "baseline_gate": identity(validation / "raw_all_live_audit.json"),
        "validation": {"unique_cases": 10, "unique_map_pairs": len(map_rows), "unique_volume_pairs": len(volume_rows),
                       "unique_context_pairs": len(context_rows), "complete_planned_keys": True, "all_deltas_zero": True,
                       "all_24_gems_files_identical": True, "baseline_runtime_python_files": 428, "main_runtime_python_files": 435,
                       "all_actual_input_identities_match": True, "all_saved_array_hash_pairs_match": True,
                       "all_default_configurations_match": True, "independently_recomputed_distributions": distributions_checked,
                       "statistics_tolerance": {"relative": 1e-12, "absolute": 1e-10}, "std_ddof": 1,
                       "own_sampled_peak_limit_mib": 19073, "unique_process_instances_per_version": 10,
                       "official_final_metrics_or_ratios_added": False, "expanded_evidence_copied": False},
        "recomputed_statistics": recalculated,
        "context_scope": "Prepared/raw context array SHA/geometry are observed fingerprints; preprocessing arrays were not saved as images.",
        "precision_scope": summary["default_precision_validation_scope"],
    }
    receipt_path = dest / "copy_audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": "passed", "files": len(copied), "map_pairs": len(map_rows), "volume_pairs": len(volume_rows),
                      "context_pairs": len(context_rows), "distributions": distributions_checked,
                      "receipt": identity(receipt_path)}, indent=2))


if __name__ == "__main__":
    main()
