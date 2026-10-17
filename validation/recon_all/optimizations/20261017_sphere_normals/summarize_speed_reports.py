"""只读汇总本轮已知 JSON 契约，按字段白名单生成脱敏公开摘要。

不运行重建或评估，不读取影像，不复制未知字段/错误原文。省略的输入及
未知契约保持 not_assessed；科学执行、严格复现、退化与整体等效分开。
所有私有原报告以文件 SHA 绑定，详细输入输出和具名示例见 PUBLIC_SUMMARY.md。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


STATES = frozenset(("complete", "completed_with_failures", "running", "failed", "failed_quality",
    "failed_before_report", "comparison_failed", "passed", "difference_detected", "requires_diagnosis",
    "not_assessed", "not_run", "not_applicable", "unavailable", "available", "ownership_unresolved",
    "not_observed_in_this_fixed_input_test", "not_detected_under_exact_comparison", "unknown", "measured",
    "missing", "blocked", "error", "invalid_vertex_count", "not_assessed_vertex_correspondence",
    "not_assessed_invalid_geometry_or_space"))
COMPARISON_KEYS = ("same_vertex_shape", "ordered_faces_equal", "index_correspondence", "coordinates_equal",
    "finite_coordinates", "footer_equal", "different_coordinates", "max_coordinate_error_mm",
    "p99_coordinate_error_mm", "max_vertex_displacement_mm", "p99_vertex_displacement_mm",
    "mean_vertex_displacement_mm", "strict_reproduction", "reported_trajectory_equal")
FINAL_MAPS = tuple("smoothwm." + name + ".crv" for name in ("H", "K", "K1", "K2", "BE", "C", "FI", "S"))
EARLY_MAPS = tuple(surface + "." + kind for surface in ("white.preaparc", "inflated") for kind in ("H", "K"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def failed_sample_count(value):
    """兼容失败记录列表和非负整数计数，只公开数量，不公开错误记录。"""
    if isinstance(value, list):
        return len(value)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def boolean(value):
    return value if isinstance(value, bool) else None


def state(value):
    return value if isinstance(value, str) and value in STATES else "unknown"


def strict(value):
    return value if isinstance(value, bool) else state(value) if isinstance(value, str) else "not_assessed"


def commit(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) else None


def scalar_fields(value, names):
    value = value if isinstance(value, dict) else {}
    return {name: value[name] for name in names if name in value and
            (isinstance(value[name], bool) or number(value[name]) is not None or value[name] is None)}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_binding(version, files=None, manifest_sha256=None):
    """只保留科学相关仓库相对文件名；完整原映射用canonical摘要绑定。"""
    files = files if isinstance(files, dict) else {}
    selected = {}
    pattern = re.compile(r"^(?:src/)?fnit/recon_all/([A-Za-z0-9_]+\.(?:py|cpp|cu|cuh|h))$")
    prefixes = ("mris_register_", "sphere_", "place_surface_normals", "topology_vnl_svd",
                "native_free", "surface_roi_curvature", "curvature_stats_", "discrete_curvature_")
    for name, file_digest in files.items():
        match = pattern.fullmatch(name) if isinstance(name, str) else None
        if match and match[1].startswith(prefixes) and digest(file_digest):
            selected["src/fnit/recon_all/" + match[1]] = file_digest
    return {"actual_commit": commit(version),
            "version_label_sha256": hashlib.sha256(version.encode()).hexdigest() if isinstance(version, str) else None,
            "manifest_file_sha256": digest(manifest_sha256),
            "complete_file_mapping_canonical_sha256": canonical_digest(files) if files else None,
            "scientific_file_sha256": selected}


def variation(values):
    values = [float(value) for value in values if number(value) is not None]
    if not values:
        return {"status": "not_assessed", "samples": 0}
    median = statistics.median(values)
    return {"status": "measured", "samples": len(values), "values_seconds": values,
            "minimum_seconds": min(values), "maximum_seconds": max(values), "median_seconds": median,
            "range_over_median_percent": 100 * (max(values) - min(values)) / median if median else None}


def memory(sampler=None, torch_memory=None):
    sampler = sampler if isinstance(sampler, dict) else {}
    torch_memory = torch_memory if isinstance(torch_memory, dict) else {}
    ownership = state(sampler.get("status"))
    owned = number(sampler.get("peak_tree_total_bytes")) if ownership == "available" else None
    return {"ownership_status": ownership, "owned_simultaneous_peak_bytes": owned,
            "sampled_target_device_peak_bytes": number(sampler.get("peak_target_device_used_bytes")),
            "sampled_target_compute_process_sum_peak_bytes": number(sampler.get("peak_target_compute_process_sum_bytes")),
            "sampling_interval_seconds": number(sampler.get("sampling_interval_seconds")),
            "max_observed_interval_seconds": number(sampler.get("max_observed_interval_seconds")),
            "failed_samples": failed_sample_count(sampler.get("failed_samples")),
            "pytorch": scalar_fields(torch_memory, ("allocated_entry_bytes", "reserved_entry_bytes",
                "allocated_current_bytes", "reserved_current_bytes", "allocated_peak_bytes", "reserved_peak_bytes",
                "pytorch_peak_allocated_bytes", "pytorch_peak_reserved_bytes", "independent_cold_process_peak")),
            "scope": "target-device samples can include other jobs; allocator statistics are not whole FNIT process memory; unknown ownership stays null",
            "20GB_owned_memory_acceptance": "not_assessed"}


def child_json(path, expected):
    """仅跟随带有效SHA的JSON子报告，不读取影像或无绑定文件。"""
    if not isinstance(path, str) or not digest(expected) or Path(path).suffix != ".json":
        return {}, "not_assessed"
    try:
        if sha(path) != expected:
            return {}, "hash_mismatch"
        value = json.loads(Path(path).read_text())
        return (value, "verified") if isinstance(value, dict) else ({}, "invalid_json_object")
    except (OSError, ValueError):
        return {}, "unavailable"


def register_summary(value):
    if not (isinstance(value.get("runs"), list) and "baseline_version" in value and "candidate_version" in value):
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    rows = []
    groups = {arm: [] for arm in ("baseline", "candidate")}
    for pair in value["runs"]:
        for arm in groups:
            wrapper = pair.get(arm, {})
            child = wrapper.get("stage_report", {})
            if not isinstance(child, dict):
                child = {}
            row = {"repeat": number(pair.get("repeat")), "arm": arm,
                "status": state(child.get("status")), "exit_code": number(wrapper.get("returncode")),
                "process_wall_seconds": number(wrapper.get("process_wall_seconds_including_imports")),
                "synchronized_API_wrapper_seconds": number(child.get("wall_seconds_including_io_jit_sync_init")),
                "API_seconds_including_io": number(child.get("api_total_seconds_including_io")),
                "report_sha256": digest(wrapper.get("report_sha256")), "API_report_sha256": digest(child.get("api_report_sha256")),
                "output_sha256": digest(child.get("output_sha256")),
                "source": source_binding(child.get("source_version"), child.get("source_sha256")),
                "comparison": scalar_fields(pair.get("comparison"), COMPARISON_KEYS),
                "strict_reproduction": strict(pair.get("strict_reproduction")),
                "allocator_cache_disabled": boolean(child.get("cuda_allocation_cache_disabled")),
                "precision": scalar_fields(child.get("precision"), ("matmul_tf32", "cudnn_tf32", "float16_or_bfloat16", "autocast_cuda")),
                "thread_budget": {name: number(child.get(name)) for name in ("torch_threads", "numba_threads", "torch_interop_threads")},
                "memory": memory(torch_memory={
                    "allocated_peak_bytes": None if child.get("cuda_allocation_cache_disabled") is True else child.get("gpu_peak_allocated_bytes"),
                    "reserved_peak_bytes": None if child.get("cuda_allocation_cache_disabled") is True else child.get("gpu_peak_reserved_bytes")})}
            cache = child.get("source_order_fit_numba_cache", {})
            row["fit_signature_cache_loads"] = {name: sum(v for v in cache.get(name, {}).values() if number(v) is not None)
                                                 if isinstance(cache.get(name), dict) else None for name in ("hits", "misses")}
            rows.append(row)
            groups[arm].append(row)
    repeated = {}
    for arm, selected in groups.items():
        valid = [row for row in selected if row["status"] == "complete" and row["exit_code"] == 0]
        hashes = [row["output_sha256"] for row in valid]
        repeated[arm] = {"API_wrapper_variation": variation(row["synchronized_API_wrapper_seconds"] for row in valid),
                         "process_variation": variation(row["process_wall_seconds"] for row in valid),
                         "repeated_saved_output_bytes_equal": len(set(hashes)) == 1 if len(hashes) >= 2 and all(hashes) else None}
    return {"schema_status": "recognized", "execution_status": state(value.get("status")),
            "hemisphere": value.get("hemi") if value.get("hemi") in ("lh", "rh") else "unknown",
            "input_sha256": {name: digest(value.get("input_sha256", {}).get(name)) for name in ("sphere", "smoothwm", "sulc", "atlas")},
            "strict_reproduction": strict(value.get("strict_reproduction")), "runs": rows,
            "same_arm_repetition": repeated, "whole_speedup": "not_assessed",
            "timing_scope": "same-input stage; complete API/wrapper/process times are nested and must not be summed"}


def map_comparison(value):
    return scalar_fields(value, ("same_bytes", "different_elements", "max_abs", "p99_abs", "mean_abs"))


def stats_comparison(value):
    result = scalar_fields(value, ("same_bytes", "header_lines_equal", "noncomment_text_equal", "ordered_structure_equal",
        "finite", "numeric_fields_exact", "different_numeric_fields", "max_numeric_abs", "numerical_and_semantic_equal"))
    for name in ("noncomment_rows", "numeric_fields"):
        values = value.get(name) if isinstance(value, dict) else None
        if isinstance(values, list) and all(number(v) is not None for v in values):
            result[name] = values
    return result


def curvature_summary(value):
    if value.get("order") != ["legacy", "deferred", "deferred", "legacy"] or not isinstance(value.get("cases"), list):
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    rows, identities = [], {}
    for index, case in enumerate(value["cases"]):
        identifier = case.get("case")
        case_index = identities.setdefault(identifier, len(identities) + 1) if isinstance(identifier, str) else None
        trials = []
        for trial in case.get("trials", []):
            comparisons = trial.get("comparison_to_first_legacy", {})
            trials.append({"iteration": number(trial.get("iteration")),
                "mode": trial.get("mode") if trial.get("mode") in ("legacy", "deferred") else "unknown",
                **scalar_fields(trial, ("input_copy_seconds", "early_seconds_including_io", "native_stats_seconds_including_io",
                                        "complete_api_seconds", "trial_total_seconds")),
                "final_eight_maps": {name: map_comparison(comparisons.get(name, {})) for name in FINAL_MAPS},
                "final_stats": stats_comparison(comparisons.get("curv.stats", {})),
                "early_four_maps": {name: map_comparison(comparisons.get(name, {})) for name in EARLY_MAPS},
                "output_sha256": {name: digest(trial.get("output_sha256", {}).get(name, {}).get("sha256"))
                                  for name in (*FINAL_MAPS, "curv.stats", *EARLY_MAPS)},
                "memory": memory(trial.get("process_tree_memory"), trial.get("torch_memory"))})
        repetitions = {}
        for mode in ("legacy", "deferred"):
            selected = [trial for trial in trials if trial["mode"] == mode]
            repetitions[mode] = {"complete_API_variation": variation(trial.get("complete_api_seconds") for trial in selected)}
            for label, maps in (("final_eight_maps", FINAL_MAPS), ("early_four_maps", EARLY_MAPS)):
                fingerprints = [tuple(trial["output_sha256"][name] for name in maps) for trial in selected]
                repetitions[mode][label + "_repeat_byte_equal"] = (len(set(fingerprints)) == 1
                    if len(fingerprints) >= 2 and all(all(row) for row in fingerprints) else None)
        rows.append({"input_group": index + 1, "case_index": case_index,
                     "input_sha256": {name: digest(case.get("inputs", {}).get(name, {}).get("sha256"))
                                      for name in ("white.preaparc", "inflated", "smoothwm", "curv", "sulc")},
                     "hemisphere": case.get("hemisphere") if case.get("hemisphere") in ("lh", "rh") else "unknown",
                     "all_outputs_byte_identical": boolean(case.get("all_outputs_byte_identical")),
                     "all_output_values_and_semantics_equal": boolean(case.get("all_output_values_and_semantics_equal")),
                     "trials": trials, "same_arm_repetition": repetitions})
    return {"schema_status": "recognized", "execution_status": state(value.get("status")),
            "source": source_binding(value.get("code_version"), value.get("source_sha256"), value.get("manifest", {}).get("sha256")),
            "native_program_sha256": digest(value.get("native_program", {}).get("sha256")),
            "expected_input_groups": 4, "reported_input_groups": len(rows),
            "run_coverage_complete": len(rows) == 4 and all(len(row["trials"]) == 4 for row in rows),
            **scalar_fields(value, ("setup_seconds", "outer_wall_seconds")), "cases": rows,
            "memory_scope": "ABBA in one process with allocator cache enabled; reserved/process occupancy may retain earlier-trial cache; not independent cold peaks",
            "whole_speedup": "not_assessed", "overall_metric_equivalence": "not_assessed"}


def sphere_summary(value):
    if not ("baseline_commit" in value and "candidate_commit" in value and isinstance(value.get("runs"), list)):
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    rows, groups, identities = [], {}, {}
    for run in value["runs"]:
        child, binding = child_json(run.get("worker_report"), run.get("worker_report_sha256"))
        key = (run.get("case"), run.get("hemisphere"), run.get("arm"))
        group_index = list(groups).index(key) + 1 if key in groups else len(groups) + 1
        groups.setdefault(key, [])
        identifier = run.get("case")
        case_index = identities.setdefault(identifier, len(identities) + 1) if isinstance(identifier, str) else None
        phase = child.get("phase")
        row = {"input_arm_group": group_index, "case_index": case_index,
            "phase": phase if phase in ("source_and_input_validation", "complete_sphere_api_running", "complete_sphere_api_returned",
                "saved_surface_quality_running", "saved_surface_quality_returned", "completed_all_validation") else "unknown",
            "arm": run.get("arm") if run.get("arm") in ("A", "B") else "unknown",
            "hemisphere": run.get("hemisphere") if run.get("hemisphere") in ("lh", "rh") else "unknown",
            "mode": run.get("mode") if run.get("mode") in ("timing", "trace") else "unknown",
            "execution_status": state(run.get("status")), "exit_code": number(run.get("exit_code")),
            "child_metadata_binding": binding, "worker_report_sha256": digest(run.get("worker_report_sha256")),
            "parent_full_wall_seconds": number(run.get("parent_full_wall_seconds")),
            "stage_API_seconds": number(child.get("stage_seconds_including_io_and_synchronization")),
            "worker_seconds_including_validation_and_quality": number(child.get("worker_seconds_including_validation_and_quality")),
            "memory": memory(torch_memory=child.get("memory")),
            "quality": scalar_fields(child.get("quality"), ("passed", "vertices", "faces", "fp32_negative_faces", "fp64_negative_faces", "all_finite"))}
        row["performance_sample_eligible"] = (binding == "verified" and row["mode"] == "timing" and
            row["exit_code"] == 0 and row["execution_status"] == "complete" and child.get("timing_qualification") == "normal")
        rows.append(row)
        groups[key].append(row)
    repeats = [{"input_arm_group": selected[0]["input_arm_group"],
                "stage_API_variation": variation(row["stage_API_seconds"] for row in selected if row["performance_sample_eligible"]),
                "process_variation": variation(row["parent_full_wall_seconds"] for row in selected if row["performance_sample_eligible"])}
               for selected in groups.values()]
    comparisons = []
    for index, row in enumerate(value.get("comparisons", [])):
        comparisons.append({"input_group": index + 1, "case_index": identities.get(row.get("case")),
            "hemisphere": row.get("hemisphere") if row.get("hemisphere") in ("lh", "rh") else "unknown",
            **scalar_fields(row, ("observed_stage_reduction_percent", "all_six_saved_files_and_ordered_geometry_equal",
                                  "every_update_and_finish_trace_equal", "all_quality_passed")),
            "strict_reproduction": strict(row.get("strict_reproduction")),
            "same_index_geometry_error": scalar_fields(row.get("same_index_geometry_error"), ("different_vertices", "coordinate_mae_mm",
                 "vertex_distance_mean_mm", "vertex_distance_p99_mm", "vertex_distance_max_mm"))})
    return {"schema_status": "recognized", "execution_status": state(value.get("status")),
            "baseline_commit": commit(value.get("baseline_commit")), "candidate_commit": commit(value.get("candidate_commit")),
            "inputs_manifest_sha256": digest(value.get("inputs_manifest_sha256")),
            "source_manifest_sha256": {arm: digest(value.get("source_manifests", {}).get(arm)) for arm in ("A", "B")},
            "strict_reproduction": strict(value.get("strict_reproduction")), "runs": rows, "comparisons": comparisons,
            "same_arm_repetition": repeats, "whole_speedup": "not_assessed"}


def gate(value):
    if isinstance(value, dict):
        result = {"status": state(value.get("status")), **scalar_fields(value, ("expected", "present", "missing_count", "passed", "checked"))}
        if isinstance(value.get("missing"), list):
            result["missing_count"] = len(value["missing"])
        for hemi in ("lh", "rh"):
            if isinstance(value.get(hemi), dict):
                result[hemi] = {"status": state(value[hemi].get("status")),
                    **scalar_fields(value[hemi], ("vertices", "faces", "euler", "unclosed_edges", "ordered_faces_preserved",
                        "finite_coordinates", "intersecting_faces"))}
        return result
    return {"status": state(value)}


def speed_pair_summary(value):
    if value.get("schema_version") != 1 or value.get("scope") != "two-raw-T1-four-empty-directory-paired-whole-runs":
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    rows = []
    for row in value.get("runs", []):
        summary = row.get("benchmark_summary", {})
        rows.append({"case_index": number(row.get("case_index")), "arm": row.get("arm") if row.get("arm") in ("baseline", "candidate") else "unknown",
            "actual_commit": commit(row.get("code_version")), "input_sha256": digest(row.get("input_sha256")),
            "execution_status": state(row.get("status")), "exit_code": number(row.get("exit_code")),
            **scalar_fields(row, ("outer_supervisor_wall_seconds", "supervisor_case_outer_wall_seconds", "benchmark_cli_wall_seconds", "benchmark_full_harness_seconds")),
            "benchmark_sha256": digest(row.get("benchmark_sha256")), "supervisor_report_sha256": digest(row.get("supervisor_report_sha256")),
            "output_completeness": gate(summary.get("output_completeness")), "grid_quality": gate(summary.get("grid_quality")),
            "memory": memory(summary.get("process_memory"))})
    paired = []
    for case_index in (0, 1):
        selected = {row["arm"]: row for row in rows if row["case_index"] == case_index}
        before, after = selected.get("baseline", {}), selected.get("candidate", {})
        matching = bool(before.get("input_sha256") and before.get("input_sha256") == after.get("input_sha256"))
        a, b = before.get("benchmark_cli_wall_seconds"), after.get("benchmark_cli_wall_seconds")
        valid = matching and all(row.get("execution_status") == "complete" and row.get("exit_code") == 0
                                 for row in (before, after)) and number(a) is not None and number(b) is not None and a > 0
        paired.append({"case_index": case_index, "same_input_verified_from_report": matching,
                       "observed_CLI_reduction_percent": 100 * (1 - b / a) if valid else None,
                       "status": "observed_single_pair" if valid else "not_assessed"})
    return {"schema_status": "recognized", "execution_status": state(value.get("status")),
            "sources": {arm: source_binding(value.get("sources", {}).get(arm, {}).get("version"),
                        value.get("sources", {}).get(arm, {}).get("sha256")) for arm in ("baseline", "candidate")},
            "expected_runs": 4, "reported_runs": len(rows), "run_coverage_complete": len(rows) == 4,
            **scalar_fields(value, ("preflight_wall_seconds", "total_pair_wall_seconds")), "runs": rows,
            "within_case_observations": paired, "same_arm_repetition": "not_assessed_different_subjects",
            "overall_metric_equivalence": "not_assessed", "performance_acceptance": "not_assessed",
            "timing_scope": "same-host/GPU blocked A-B and B-A whole runs; shared-load attribution requires review; never use historical times as current controls"}


ATLAS_NAMES = ("aparc", "aparc.DKTatlas", "aparc.a2009s", "aparc.pial", "BA_exvivo", "BA_exvivo.thresh")
REGION_FIELDS = ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv")
DICE_FILES = ("aseg.mgz", "aparc+aseg.mgz", "aparc.a2009s+aseg.mgz", "aparc.DKTatlas+aseg.mgz", "wmparc.mgz", "ribbon.mgz", "filled.mgz")
SURFACE_NAMES = ("orig.nofix", "orig.premesh", "orig", "white.preaparc", "smoothwm", "inflated", "white", "pial", "sphere", "sphere.reg")
EVALUATION_ROLES = ("strict", "dice", "surface", "geometry", "region", "no_th3", "local", "all_regions", "quality_candidate", "quality_reference")


def strict_metadata(value):
    if not isinstance(value, dict):
        return {"status": strict(value)}
    result = scalar_fields(value, ("checked", "passed", "all_pass"))
    failed = value.get("failed_files")
    if isinstance(failed, list):
        result["failed_file_count"] = len(failed)
    count, passed = number(value.get("checked")), number(value.get("passed"))
    result["status"] = ("passed" if count == 138 and passed == 138 else "difference_detected"
                         if count == 138 and passed is not None and 0 <= passed < 138 else "not_assessed")
    if isinstance(failed, list) and count is not None and passed is not None and len(failed) != count - passed:
        result["status"] = "inconsistent_metadata"
    if isinstance(value.get("all_pass"), bool) and count == 138 and passed is not None and value["all_pass"] != (passed == count):
        result["status"] = "inconsistent_metadata"
    return result


def metric_summary(value):
    """读取已有_metric输出，不重新计算MAE、P90或新的等效门槛。"""
    result = scalar_fields(value, ("matched_regions", "pearson_r", "mae", "median_absolute_relative_error_percent",
        "p90_absolute_relative_error_percent", "maximum_absolute_error"))
    for field in ("missing_in_candidate", "extra_in_candidate", "zero_reference_regions"):
        items = value.get(field) if isinstance(value, dict) else None
        result[field + "_count"] = len(items) if isinstance(items, list) else None
    worst = value.get("worst_regions_by_relative_error", []) if isinstance(value, dict) else []
    regions = value.get("per_region", {}) if isinstance(value, dict) else {}
    result["reported_worst_regions"] = [
        {"region_key_sha256": hashlib.sha256(name.encode()).hexdigest(),
         **scalar_fields(regions.get(name), ("reference", "candidate", "absolute_error", "signed_relative_difference_percent"))}
        for name in worst[:5] if isinstance(name, str)] if isinstance(worst, list) else []
    result["reported_region_count"] = len(regions) if isinstance(regions, dict) else None
    return result


def evaluation_numbers(reports):
    """仅抽取既有比较器的确定字段，独立保留空间、局部质量和覆盖。"""
    output = {"verified_subreports": list(reports), "expected_core_subreports": len(EVALUATION_ROLES),
              "core_subreport_coverage_complete": all(role in reports for role in EVALUATION_ROLES)}
    if "strict" in reports:
        value = reports["strict"]
        output["strict_138"] = strict_metadata(value)
        files = value.get("files", {})
        output["strict_138"]["reported_file_count"] = len(files) if isinstance(files, dict) else None
    if "dice" in reports:
        files, rows = reports["dice"].get("files", {}), {}
        for name in DICE_FILES:
            value = files.get(name)
            if not isinstance(value, dict):
                continue
            row = scalar_fields(value, ("different_voxels", "minimum_dice", "p05_dice", "median_dice"))
            labels = value.get("per_label", {})
            row["label_union_count"] = len(labels) if isinstance(labels, dict) else None
            row["reported_worst_labels"] = [
                {"label_id": int(label), **scalar_fields(labels.get(label),
                    ("reference_voxels", "candidate_voxels", "intersection_voxels", "dice"))}
                for label in value.get("worst_labels", [])[:10] if isinstance(label, str) and re.fullmatch(r"-?[0-9]+", label)]
            rows[name] = row
        output["segmentation_dice"] = {"expected_maps": 7, "reported_maps": len(rows), "maps": rows}
    if "surface" in reports:
        stages, rows = reports["surface"].get("stages", {}), {}
        for hemi in ("lh", "rh"):
            for name in SURFACE_NAMES:
                value = stages.get(hemi, {}).get(name)
                if not isinstance(value, dict):
                    continue
                row = scalar_fields(value, ("reference_vertices", "candidate_vertices", "reference_faces", "candidate_faces", "ordered_faces_equal"))
                for direction in ("candidate_to_reference_triangle", "reference_to_candidate_triangle"):
                    row[direction] = scalar_fields(value.get(direction), ("mean_mm", "p99_mm", "max_mm", "over_0_1_mm"))
                rows[hemi + "." + name] = row
        output["surface_distances"] = {"unit": "mm", "expected_surfaces": 20, "reported_surfaces": len(rows), "surfaces": rows,
            "final_white_pial_directions_with_complete_statistics": sum(
                all(number(rows.get(hemi + "." + name, {}).get(direction, {}).get(key)) is not None for key in ("mean_mm", "p99_mm", "max_mm"))
                for hemi in ("lh", "rh") for name in ("white", "pial")
                for direction in ("candidate_to_reference_triangle", "reference_to_candidate_triangle")),
            "expected_final_directions": 8,
            "definition": "all source vertices to opposite full triangle mesh; sampled maximum is not continuous Hausdorff"}
    if "geometry" in reports:
        rows = reports["geometry"].get("surfaces", {})
        selected = {hemi + "." + name: scalar_fields(rows.get(hemi + "." + name), (
            "surface_ras_mm_comparable", "ordered_faces_and_vertex_counts_equal", "vertex_correspondence"))
            for hemi in ("lh", "rh") for name in SURFACE_NAMES if isinstance(rows.get(hemi + "." + name), dict)}
        output["space_and_correspondence"] = {"expected_surfaces": 20, "reported_surfaces": len(selected),
            "common_conform_and_tkras": boolean(reports["geometry"].get("common_conform_and_tkras")),
            "surface_ras_comparable_passed": sum(row.get("surface_ras_mm_comparable") is True for row in selected.values()),
            "surface_ras_comparable_checked": sum(isinstance(row.get("surface_ras_mm_comparable"), bool) for row in selected.values()),
            "surfaces": selected}
    if "region" in reports:
        value = reports["region"]
        output["aparc_68_metrics"] = {name: metric_summary(value.get("aparc_68", {}).get(name, {})) for name in REGION_FIELDS}
        output["aseg_metrics"], output["wmparc_metrics"] = metric_summary(value.get("aseg", {})), metric_summary(value.get("wmparc", {}))
    if "all_regions" in reports:
        atlases = reports["all_regions"].get("atlases", {})
        output["all_cortical_statistics"] = {"expected_atlases": 6,
            "reported_atlases": sum(name in atlases for name in ATLAS_NAMES),
            "atlases": {name: {field: metric_summary(atlases[name].get(field, {})) for field in REGION_FIELDS}
                        for name in ATLAS_NAMES if name in atlases}}
    if "no_th3" in reports:
        atlases = reports["no_th3"].get("atlases", {})
        output["common_no_th3_statistics"] = {name: metric_summary(atlases[name]) for name in ATLAS_NAMES[:3] if name in atlases}
        output["no_th3_definition"] = "same existing no-th3 algorithm on each own mesh; distinct from raw GrayVol or TH3 vertex maps"
    if "local" in reports:
        coverage = {}
        for name in ("maps", "annotations"):
            values = reports["local"].get(name, {})
            counts = {}
            if isinstance(values, dict):
                for row in values.values():
                    status = state(row.get("status")) if isinstance(row, dict) else "unknown"
                    counts[status] = counts.get(status, 0) + 1
            coverage[name] = {"reported_count": len(values) if isinstance(values, dict) else None, "status_counts": counts}
        output["local_comparison_coverage"] = coverage
        maps = reports["local"].get("maps", {})
        names = ("thickness", "area", "area.pial", "area.mid", "volume", "curv", "curv.pial", "avg_curv", "sulc", "jacobian_white", *FINAL_MAPS, *EARLY_MAPS)
        output["same_index_local_map_diagnostics"] = {
            hemi + "." + name: {"status": state(maps["surf/" + hemi + "." + name].get("status")),
                **scalar_fields(maps["surf/" + hemi + "." + name], ("count", "outlier_count", "outlier_fraction", "nonfinite_count",
                    "bias", "mae", "rmse", "p95_abs_error", "p99_abs_error", "max_abs_error"))}
            for hemi in ("lh", "rh") for name in names if "surf/" + hemi + "." + name in maps}
        annotations = reports["local"].get("annotations", {})
        annotation_names = ("aparc", "aparc.a2009s", "aparc.DKTatlas", "BA_exvivo", "BA_exvivo.thresh", "mpm.vpnl")
        output["same_index_annotation_diagnostics"] = {
            hemi + "." + name: {"status": state(annotations["label/" + hemi + "." + name + ".annot"].get("status")),
                "configured_min_dice": number(annotations["label/" + hemi + "." + name + ".annot"].get("min_dice")),
                **scalar_fields(annotations["label/" + hemi + "." + name + ".annot"], ("mismatch_count", "agreement", "color_table_equal"))}
            for hemi in ("lh", "rh") for name in annotation_names if "label/" + hemi + "." + name + ".annot" in annotations}
    for role in ("quality_candidate", "quality_reference"):
        if role not in reports:
            continue
        value, hemispheres = reports[role], {}
        for hemi in ("lh", "rh"):
            row = value.get("hemispheres", {}).get(hemi, {})
            hemispheres[hemi] = {"topology": scalar_fields(row.get("topology"), ("vertices", "faces", "euler", "components",
                    "boundary_edges", "nonmanifold_edges", "degenerate_index_faces", "duplicate_faces", "valid")),
                "vertex_links": scalar_fields(row.get("vertex_links"), ("checked_vertices", "noncycle_links", "degree_not_two", "disconnected_cycles", "isolated_vertices")),
                "sphere_orientation": {name: scalar_fields(row.get("sphere_orientation", {}).get(name),
                    ("negative_faces", "zero_area_faces", "nonfinite_areas", "minimum_signed_area_mm2")) for name in ("sphere", "sphere.reg")},
                "white_pial_crossings": {"status": state(row.get("white_pial_crossings", {}).get("status")),
                    **scalar_fields(row.get("white_pial_crossings"), ("proper_transverse_pairs", "proper_pairs_with_any_cortex_vertex",
                        "proper_pairs_with_both_faces_fully_in_cortex", "proper_pairs_with_no_cortex_vertex", "coincident_same_index_faces",
                        "white_faces_examined", "white_faces_total"))}}
        output[role] = {"execution_status": state(value.get("status")), "hemispheres": hemispheres,
                        "independent_white_pial_self_intersections": "not_assessed_by_this_report"}
    return output


def bound_evaluation_reports(path, bindings):
    """新输入清单绑定main与子JSON的实际SHA；完成状态和同目录均先核对。"""
    primary = Path(path).resolve()
    raw = json.loads(primary.read_text())
    if raw.get("status") != "complete":
        return {}, {"status": "main_not_complete"}
    matches = []
    for binding_path in bindings:
        try:
            value = json.loads(Path(binding_path).read_text())
            record = value.get("evaluation_report", {})
            if (value.get("schema") == "fnit-evaluation-summary-bindings-v1" and
                    Path(record.get("path", "")).resolve() == primary and record.get("sha256") == sha(primary)):
                matches.append((binding_path, value))
        except (OSError, ValueError, TypeError):
            continue
    if len(matches) != 1:
        return {}, {"status": "not_assessed", "matching_bindings": len(matches)}
    binding_path, manifest = matches[0]
    label = manifest.get("comparison_label")
    if label not in ("candidate_vs_control", "candidate_vs_official"):
        return {}, {"status": "unsupported_comparison_label"}
    reference_role = "control" if label == "candidate_vs_control" else "official"
    expected = {role: role + "_" + label + ".json" for role in EVALUATION_ROLES if not role.startswith("quality_")}
    expected.update(quality_candidate="quality_candidate/report.json", quality_reference="quality_" + reference_role + "/report.json")
    verified, statuses, hashes = {}, {}, {}
    for role, filename in expected.items():
        record = manifest.get("reports", {}).get(role, {})
        if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not digest(record.get("sha256")):
            statuses[role] = "not_assessed"
            continue
        child_path = Path(record["path"])
        if child_path.resolve() != (primary.parent / filename).resolve():
            statuses[role] = "unexpected_path"
            continue
        payload, status = child_json(str(child_path), record["sha256"])
        statuses[role] = status
        if status == "verified":
            verified[role], hashes[role] = payload, record["sha256"]
    return verified, {"status": "complete" if all(v == "verified" for v in statuses.values()) else "partial_or_failed_binding",
                      "binding_manifest_sha256": sha(binding_path), "subreport_binding_status": statuses,
                      "subreport_sha256": hashes, "comparison_kind": "FNIT_candidate_vs_control" if reference_role == "control" else "candidate_vs_official"}


def evaluation_summary(value, reports=None, binding=None):
    if value.get("candidate_profile") != "mixed-native-surfaces":
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    paired = isinstance(value.get("code_versions"), dict)
    if not paired and not commit(value.get("tested_code_version")):
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    result = {"schema_status": "recognized_pair_metadata_only" if paired else "recognized_official_evaluation",
            "execution_status": state(value.get("status")),
            "actual_commits": {role: commit(value.get("code_versions", {}).get(role)) for role in ("control", "candidate")} if paired else
                              {"candidate": commit(value.get("tested_code_version"))},
            "evaluator_sha256": digest(value.get("evaluator_sha256")), "adapter_sha256": digest(value.get("adapter_sha256")),
            **scalar_fields(value, ("comparison_wall_seconds", "threads", "official_reference_read", "cli_wall_seconds", "full_harness_seconds", "execution_complete")),
            "strict_reproduction": strict_metadata(value.get("strict_reproduction")),
            "numeric_error_metrics": evaluation_numbers(reports) if reports else "not_assessed",
            "subreport_binding": binding or {"status": "not_assessed"},
            "optimization_regression": "not_assessed", "overall_metric_equivalence": "not_assessed"}
    if not paired:
        result.update(output_validation=gate(value.get("output_validation")), production_mesh_validation=gate(value.get("production_mesh_validation")),
            memory=memory(value.get("process_memory")), native_program_manifest_sha256=digest(value.get("native_program_manifest_sha256")),
            reference_manifest_sha256=digest(value.get("reference_manifest_sha256")),
            generator_source=source_binding(value.get("tested_code_version"),
                {"fnit/" + key: item for key, item in value.get("generator_source_sha256", {}).items() if isinstance(key, str)}))
    else:
        for name in ("output_validation", "production_mesh_validation"):
            result[name] = {role: gate(value.get(name, {}).get(role)) for role in ("control", "candidate")}
    return result


MONITOR_SCOPE = "independent outer read-only process-tree monitor; original science reports remain unchanged"
MONITOR_STATES = frozenset(("preflight", "monitoring", "complete_monitoring", "science_command_failed",
                            "monitor_failed", "monitor_source_binding_changed"))
MONITOR_FILE_ROLES = ("monitor", "diagnose_tool", "profiling", "nvml", "command_script", "source_manifest")


def monitor_state(value):
    return value if isinstance(value, str) and value in MONITOR_STATES else "unknown"


def monitor_frame_totals(value, *, completed_and_bound):
    """仅读取原监测器汇总；unknown/failed帧不成为完整own峰值或20GB通过。"""
    value = value if isinstance(value, dict) else {}
    names = ("sampled_frames", "resolved_frames", "unknown_or_failed_frames")
    counts = {name: value.get(name) if type(value.get(name)) is int and value[name] >= 0 else None for name in names}
    valid_counts = all(item is not None for item in counts.values()) and counts["sampled_frames"] == counts["resolved_frames"] + counts["unknown_or_failed_frames"]
    own = number(value.get("owned_simultaneous_sampled_peak_bytes"))
    resolved = bool(completed_and_bound and valid_counts and counts["sampled_frames"] > 0 and
                    counts["unknown_or_failed_frames"] == 0 and own is not None and own >= 0)
    return {**counts, "reported_counts_consistent": valid_counts,
            "maximum_observed_interval_seconds": number(value.get("maximum_observed_interval_seconds")),
            "owned_simultaneous_sampled_peak_bytes": own if resolved else None,
            "owned_peak_assessment": "resolved_observed_samples_only" if resolved else "not_assessed",
            **{name: number(value.get(name)) for name in ("partial_attributed_same_snapshot_peak_bytes",
                "target_device_sampled_peak_bytes", "target_compute_process_sum_sampled_peak_bytes")},
            "partial_attribution_scope": "lower-bound observation from verified processes in one sampler snapshot; never full own peak",
            "target_device_scope": "samples can include other jobs; no hardware identifiers exported",
            "20GB_owned_memory_acceptance": "not_assessed"}


def outer_monitor_summary(value):
    """读取outer monitor主JSON白名单；不读取样本JSONL或推断精确CLI边界。"""
    if value.get("scope") != MONITOR_SCOPE or not isinstance(value.get("wrapper_totals"), dict):
        return {"schema_status": "unsupported", "assessment": "not_assessed"}
    status = monitor_state(value.get("status"))
    files = value.get("files_sha256", {})
    files = files if isinstance(files, dict) else {}
    unchanged = value.get("files_unchanged_after_monitoring", {})
    unchanged = unchanged if isinstance(unchanged, dict) else {}
    source_bound = (all(digest(files.get(name)) and unchanged.get(name) is True for name in MONITOR_FILE_ROLES) and
                    commit(value.get("source_commit")) is not None and digest(value.get("source_manifest_sha256")) is not None and
                    files.get("source_manifest") == value["source_manifest_sha256"])
    completed = (status == "complete_monitoring" and value.get("child_started") is True and
                 type(value.get("command_exit_code")) is int and value["command_exit_code"] == 0 and
                 value.get("cuda_initialized_at_entry") is False and value.get("cuda_initialized_after") is False and source_bound)
    metadata, totals = value.get("run_metadata_observed", {}), value.get("per_run_observed_samples", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    totals = totals if isinstance(totals, dict) else {}
    runs = []
    for private_key, row in metadata.items():
        if not isinstance(row, dict) or row.get("run_id") != private_key or private_key not in totals:
            continue
        if (row.get("arm") not in ("baseline", "candidate") or not commit(row.get("code_version")) or
                not digest(row.get("input_sha256")) or not digest(row.get("benchmark_report_sha256_observed")) or
                not digest(row.get("pair_report_sha256_observed"))):
            continue
        runs.append({"run_index": len(runs), "arm": row["arm"], "actual_commit": commit(row["code_version"]),
            "input_sha256": row["input_sha256"], "benchmark_report_sha256_observed": row["benchmark_report_sha256_observed"],
            "pair_report_sha256_observed": row["pair_report_sha256_observed"],
            "reported_run_status": state(row.get("status")), "reported_run_exit_code": number(row.get("exit_code")),
            "benchmark_execution_status": state(row.get("benchmark_execution")),
            "observed_frames": monitor_frame_totals(totals[private_key], completed_and_bound=completed)})
    return {"schema_status": "recognized_outer_monitor_metadata", "monitoring_status": status,
        "child_started": boolean(value.get("child_started")), "child_exit_code": number(value.get("command_exit_code")),
        "monitor_error_present": bool(value.get("error_type") or value.get("traceback") or value.get("error")),
        "source": {"actual_commit": commit(value.get("source_commit")),
            "source_manifest_sha256": digest(value.get("source_manifest_sha256")),
            "source_manifest_declared_files": number(value.get("source_manifest_declared_files")),
            "file_sha256": {name: digest(files.get(name)) for name in MONITOR_FILE_ROLES},
            "files_unchanged_after_monitoring": {name: boolean(unchanged.get(name)) for name in MONITOR_FILE_ROLES},
            "declared_sources_bound_and_unchanged": bool(source_bound)},
        "declared_samples_jsonl_sha256": digest(value.get("samples_jsonl_sha256")),
        "samples_jsonl_read_or_verified_by_summary": False,
        "sampling_interval_seconds": number(value.get("sampling_interval_seconds")),
        "query_backend": value.get("query_backend") if value.get("query_backend") in ("nvml", "nvidia-smi") else "unknown",
        "parent_cuda_initialized_at_entry": boolean(value.get("cuda_initialized_at_entry")),
        "parent_cuda_initialized_after": boolean(value.get("cuda_initialized_after")),
        "wrapper_observed_frames": monitor_frame_totals(value.get("wrapper_totals"), completed_and_bound=completed),
        "reported_per_run_blocks": len(totals), "metadata_bound_run_blocks": len(runs),
        "unbound_per_run_blocks": len(totals) - len(runs), "runs": runs,
        "run_index_scope": "index within metadata-bound entries of this summary; private run IDs and case names are not parsed or exported",
        "observation_scope": "stable live CLI identity observations only; no exact CLI monotonic start/end, wrapper gaps not assigned to individual runs",
        "timing_scope": "monitor clocks and frame intervals only; scientific CLI/harness times remain in original speed-pair reports",
        "scientific_E2E_completion": "not_assessed_by_monitor", "20GB_owned_memory_acceptance": "not_assessed",
        "overall_metric_equivalence": "not_assessed"}


ADAPTERS = {"register_v2": register_summary, "register_v3": register_summary,
            "curvature": curvature_summary, "standard_sphere": sphere_summary,
            "speed_pair": speed_pair_summary, "evaluation": evaluation_summary,
            "outer_monitor": outer_monitor_summary}


def summarize(inputs):
    """显式输入{类型:[JSON路径]}→不含原路径、host、UUID的公开字典。"""
    result = {"schema_version": 1, "summary_tool_sha256": sha(__file__),
              "scope": "allowlisted metadata from current stage/whole reports; no reconstruction or numeric evaluation performed",
              "overall_metric_equivalence": "not_assessed", "reports": {}}
    for kind, adapter in ADAPTERS.items():
        records = []
        for path in inputs.get(kind, []):
            binding = {"input_report_sha256": sha(path)}
            try:
                value = json.loads(Path(path).read_text())
                if not isinstance(value, dict):
                    raise ValueError("JSON object required")
                binding["original_status"] = monitor_state(value.get("status")) if kind == "outer_monitor" else state(value.get("status"))
                binding["original_script_sha256"] = digest(value.get("script_sha256"))
                binding["original_error_present"] = bool(value.get("error") or value.get("exception") or value.get("traceback"))
                if kind == "evaluation":
                    reports, proof = bound_evaluation_reports(path, inputs.get("evaluation_bindings", []))
                    binding["summary"] = evaluation_summary(value, reports, proof)
                else:
                    binding["summary"] = adapter(value)
            except (ValueError, TypeError, KeyError, AttributeError):
                binding["summary"] = {"schema_status": "invalid_or_unsupported", "assessment": "not_assessed"}
            records.append(binding)
        result["reports"][kind] = records if records else {"status": "not_assessed", "input_supplied": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in ADAPTERS:
        parser.add_argument("--" + kind.replace("_", "-"), type=Path, action="append", default=[])
    parser.add_argument("--evaluation-bindings", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    options = vars(parser.parse_args())
    output = options.pop("output")
    if output.exists():
        raise FileExistsError("public output must be new")
    result = summarize(options)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"public_report_sha256": sha(output), "overall_metric_equivalence": "not_assessed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
