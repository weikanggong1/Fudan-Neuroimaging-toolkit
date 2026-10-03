"""CPU-only tables derived from actual immutable cohort comparison reports.

Missing cases remain pending/null. This tool never starts a pipeline, queries a
GPU, changes original data/reports, or substitutes inferred stage durations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

try:
    from . import benchmark_connectome_cohort_compare as compare
except ImportError:
    import benchmark_connectome_cohort_compare as compare

MATRIX_FIELDS = ("numeric_neq", "raw_scalar_bits_neq", "rmse", "max_abs_error", "relative_l2", "support_neq",
                 "correlation", "common_edge_correlation", "strict_count_equal")
TIME_FIELDS = ("raw_dwi_cli_total_runtime_seconds", "gpu_command_wall_seconds", "gpu_lock_queue_seconds", "worker_wall_seconds")


def bounded_json(path):
    path = Path(path)
    compare.check(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32_000_000,
                  "missing, linked or unexpected oversized observation JSON")
    contents = path.read_bytes()
    compare.check(len(contents) <= 32_000_000, "oversized observation JSON")
    return json.loads(contents), {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def report_from_record(identity, namespaces):
    path = Path(identity["path"])
    compare.check(path.is_absolute() and any(path.resolve().parent == root.resolve() for root in namespaces),
                  "comparison report is outside its declared actual namespace")
    report, actual = bounded_json(path)
    compare.check(actual == {key: identity[key] for key in ("path", "sha256")}, "immutable scientific comparison report changed")
    if "all_requested_scientific_data_equal" in identity:
        compare.check(report.get("all_requested_scientific_data_equal") == identity["all_requested_scientific_data_equal"],
                      "anatomy status metadata differs from its actual report")
    return report


def checked_number(value):
    compare.check(value is None or isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                  "nonfinite or invalid table statistic")
    return value


def configuration_GPU_origins(configuration, cases):
    path = configuration.get("gpu_origin_bindings")
    if not path: return {}
    options = argparse.Namespace(**{key: Path(configuration[key]) for key in
        ("baseline_root", "candidate_root", "baseline_anatomy_root", "baseline_driver", "candidate_driver")})
    bindings, _ = compare.gpu_origins.load_bindings(path, cases, options)
    return bindings


def actual_pair_ledger(result, case, configuration, source_cache, GPU_bindings=None):
    """Tie table metrics to the actual completed GPU/wall and output bytes."""
    ledgers = {}
    for arm in ("baseline", "candidate"):
        options = argparse.Namespace(**{key: Path(configuration[key]) for key in ("baseline_root", "candidate_root", "baseline_driver", "candidate_driver")})
        root, driver, binding = compare.gpu_origins.selected_origin(options, arm, case["case_id"], GPU_bindings)
        compare.check(result[arm].get("actual_GPU_root", configuration[f"{arm}_root"]) == str(root) and
                      result[arm].get("actual_GPU_driver", configuration[f"{arm}_driver"]) == str(driver), "table actual GPU root/driver differs from explicit origin")
        job = root / arm / case["case_id"]
        GPU, gpu_identity = bounded_json(job / "gpu_report.json")
        wall, wall_identity = bounded_json(job / "raw_bids_wall.json")
        source = result[arm]["actual_source"]
        compare.check(result[arm]["gpu_report"] == gpu_identity and result[arm]["wall_report"] == wall_identity,
                      "completed comparison GPU/wall evidence changed")
        compare.check(GPU.get("status") == "completed" and GPU.get("exit_code") == 0 and GPU.get("case_id") == case["case_id"] and
                      GPU.get("version") == arm and GPU.get("raw_input_provenance") == case["input_files"],
                      "actual completed GPU case/source provenance does not match table")
        compare.check(wall.get("status") == "completed" and wall.get("exit_code") == 0 and wall.get("outputs", {}).get("status") == "complete",
                      "actual CLI outputs are incomplete")
        compare.check(GPU["source_before"]["source_fingerprint"] == GPU["source_after"]["source_fingerprint"] == source["source_fingerprint"],
                      "actual frozen source changed")
        if binding:
            actual_binding = compare.gpu_origins.verify_replacement(binding, GPU, wall, case, wall["selected_inputs"]["freesurfer_subject_dir"])
            compare.check(result[arm].get("GPU_origin_binding") == actual_binding, "actual eligible replacement differs from scientific comparison")
        else:
            compare.check(result[arm].get("GPU_origin_binding") is None, "implicit replacement GPU origin rejected")
        if source["source_fingerprint"] not in source_cache:
            source_cache[source["source_fingerprint"]] = compare.verify_source(source)
        output = job / "connectome"
        for name, identity in wall["outputs"]["files"].items():
            path = output / name
            compare.check(path.resolve().is_relative_to(output.resolve()) and identity.get("path") == str(path) and
                          identity.get("exists") is True and compare.anatomy.sha(path) == identity["sha256"],
                          "actual output bytes changed since scientific comparison")
        ledgers[arm] = {"GPU_report_sha256": gpu_identity["sha256"], "wall_report_sha256": wall_identity["sha256"],
                        "actual_GPU_root": str(root), "actual_GPU_driver": str(driver), "GPU_origin_binding": result[arm].get("GPU_origin_binding"),
                        "source_fingerprint": source["source_fingerprint"], "actual_source_file_count": source["file_count"],
                        "actual_git_metadata": source.get("git_commit"), "outputs": wall["outputs"]["files"],
                        "actual_eddy_gp_seeds": wall.get("actual_eddy_gp_seeds"), "memory_budget": result[arm]["memory_budget"]}
    return ledgers


def observation(root, driver_path, arm, case_id):
    """Saved stage evidence is not a live-process observation or completion."""
    driver_status = None
    driver = Path(driver_path)
    if driver.exists():
        value, _ = bounded_json(driver)
        driver_status = value.get("cases", {}).get(f"{arm}/{case_id}", {}).get("status")
    job = Path(root) / arm / case_id
    gpu = job / "gpu_report.json"
    wall_path = job / "raw_bids_wall.json"
    result = {"driver_status": driver_status, "GPU_report_status": None, "CLI_status": None,
              "saved_stage_evidence": {}, "live_process_state": "not_queried", "GPU_used_by_this_tool": False,
              "completed_time_observations": {},
              "interpretation": "gpu_queued is a coordinator status and can include an already running worker; missing final report is not proof that computation has not started"}
    if gpu.exists():
        GPU_value, identity = bounded_json(gpu)
        compare.check(GPU_value.get("case_id") == case_id and GPU_value.get("version") == arm, "observed GPU report has wrong case/arm")
        result.update(GPU_report_status=GPU_value.get("status"), GPU_report_sha256=identity["sha256"], GPU_error=GPU_value.get("error"))
    if wall_path.exists():
        value, identity = bounded_json(wall_path)
        result.update(CLI_status=value.get("status"), CLI_report_sha256=identity["sha256"], CLI_error=value.get("error"))
        if result["GPU_report_status"] == "completed" and GPU_value.get("exit_code") == 0 and value.get("status") == "completed" and \
                value.get("exit_code") == 0 and value.get("outputs", {}).get("status") == "complete":
            result["completed_time_observations"] = {"raw_dwi_cli_total_runtime_seconds": checked_number(value.get("total_runtime_seconds")),
                **{key: checked_number(GPU_value.get(key)) for key in TIME_FIELDS if key != "raw_dwi_cli_total_runtime_seconds"}}
    for stage, relative in (("raw_selection", "preproc/raw/state.json"), ("topup", "preproc/topup/state.json"),
                            ("eddy", "preproc/eddy/state.json"), ("eddy_QC", "preproc/eddy/data.eddy_qc.json")):
        path = job / "connectome" / relative
        if path.exists():
            value, identity = bounded_json(path)
            result["saved_stage_evidence"][stage] = {"relative_path": relative, "sha256": identity["sha256"],
                "serialized_status": value.get("status"), "actual_elapsed_seconds": checked_number(value.get("elapsed_seconds")),
                "scope": "actual saved file observation; no process phase or end-to-end duration inferred from file mtime"}
    return result


def build_summary(comparison_root, *, candidate_source_label=None, verified_pairs=None):
    comparison_root = Path(comparison_root)
    state, status_identity = bounded_json(comparison_root / "status.json")
    compare.check(state.get("requested_cases") == 10 and len(state.get("cases", {})) == 10, "actual ten-case comparison status required")
    compare.check(state.get("tool_sha256") == compare.anatomy.sha(compare.__file__) and
                  state.get("anatomy_tool_sha256") == compare.anatomy.sha(compare.anatomy.__file__),
                  "table reader must use exactly the frozen comparison implementation")
    configuration = state["configuration"]
    manifest, manifest_identity = bounded_json(configuration["manifest"])
    cases = compare.manifest_cases(manifest)
    compare.check(manifest_identity == state["manifest"], "actual canonical raw manifest changed")
    GPU_bindings = configuration_GPU_origins(configuration, cases)
    if state.get("GPU_origin_tool_sha256"):
        compare.check(state["GPU_origin_tool_sha256"] == compare.anatomy.sha(compare.gpu_origins.__file__), "frozen GPU origin reader changed")
    baseline_config, _ = bounded_json(Path(configuration["baseline_root"]) / "cohort_config.json")
    atlases = baseline_config["atlases"]
    compare.check(len(atlases) == 8 and len(set(atlases)) == 8, "this formal table requires eight distinct declared atlases")
    namespaces = [comparison_root]
    if state.get("prior_comparison_binding"):
        original = state["prior_comparison_binding"]["original_status"]
        _, actual = bounded_json(original["path"])
        compare.check(actual == original, "preserved prior comparison failure changed")
        namespaces.append(Path(original["path"]).parent)
    source_cache, verified_pairs = {}, {} if verified_pairs is None else verified_pairs
    frozen_sources = {"baseline": compare.verify_source(baseline_config["frozen_sources"]["baseline"])}
    candidate_config = Path(configuration["candidate_root"]) / "staged_gpu_config.json"
    if candidate_config.exists():
        candidate_config_value, _ = bounded_json(candidate_config)
        frozen_sources["candidate"] = compare.verify_source(candidate_config_value["frozen_sources"]["candidate"])
    case_rows, matrix_rows, anatomy_rows, source_rows = [], [], [], []
    for case in cases:
        case_id = case["case_id"]; record = state["cases"][case_id]
        row = {"case_id": case_id, "anatomy_status": "pending", "anatomy_exact": None,
               "paired_status": record["status"], "count_exact_all_atlases": None, "matrix_numeric_exact_all_atlases": None,
               "waiting_reason": record.get("waiting_reason"), "error": record.get("error"),
               "raw_input_ledger_sha256": hashlib.sha256(json.dumps(case["input_files"], sort_keys=True).encode()).hexdigest()}
        for arm in ("baseline", "candidate"):
            options = argparse.Namespace(**{key: Path(configuration[key]) for key in ("baseline_root", "candidate_root", "baseline_driver", "candidate_driver")})
            root, driver, binding = compare.gpu_origins.selected_origin(options, arm, case_id, GPU_bindings)
            row[arm + "_observation"] = observation(root, driver, arm, case_id)
            for field in TIME_FIELDS: row[arm + "_" + field] = row[arm + "_observation"]["completed_time_observations"].get(field)
            row[arm + "_official_recon_command_seconds"] = None
            row[arm + "_full_timing_scope"] = None
        if record.get("anatomy"):
            fs = report_from_record(record["anatomy"], namespaces)
            compare.check(fs.get("case_id") == case_id and fs.get("status") == "completed" and len(fs["files"]) == 13,
                          "actual official anatomy report is incomplete")
            row.update(anatomy_status="completed", anatomy_exact=fs["all_requested_scientific_data_equal"])
            for arm in ("baseline", "candidate"):
                row[arm + "_official_recon_command_seconds"] = checked_number(fs[arm]["original_execution_timing"]["recon_command_seconds"])
            for name, values in fs["files"].items():
                anatomy_rows.append({"case_id": case_id, "file": name, "status": "completed",
                                     "strict_scientific_equal": values["strict_scientific_equal"],
                                     "file_bytes_equal": values["file_bytes_equal"]})
        else:
            for _, names in compare.anatomy.SCIENTIFIC_GROUPS:
                for name in names:
                    anatomy_rows.append({"case_id": case_id, "file": name, "status": "pending",
                                         "strict_scientific_equal": None, "file_bytes_equal": None})
        result = None
        if record["status"] == "completed_comparison":
            compare.check(record.get("connectome"), "completed case has no actual scientific report")
            result = report_from_record(record["connectome"], [comparison_root])
            compare.check(result.get("status") == "completed" and list(result["atlases"]) == atlases, "actual eight-atlas report is incomplete")
            identity_key = record["connectome"]["sha256"]
            if identity_key not in verified_pairs:
                verified_pairs[identity_key] = actual_pair_ledger(result, case, configuration, source_cache, GPU_bindings)
            ledger = verified_pairs[identity_key]
            row.update(count_exact_all_atlases=result["count_exact_all_atlases"],
                       matrix_numeric_exact_all_atlases=result["matrix_numeric_exact_all_atlases"],
                       scientific_report_sha256=identity_key, images=result["images"], numeric_files=result["numeric_files"])
            for arm in ("baseline", "candidate"):
                for field in TIME_FIELDS: row[arm + "_" + field] = checked_number(result[arm].get(field))
                row[arm + "_full_timing_scope"] = result[arm]["driver_timing"]
        else:
            compare.check(not record.get("connectome"), "pending/failed case cannot carry a completed scientific result")
        for arm in ("baseline", "candidate"):
            actual_source = frozen_sources.get(arm)
            if result:
                compare.check(actual_source and ledger[arm]["source_fingerprint"] == actual_source["source_fingerprint"],
                              "case source differs from the actual frozen runtime configuration")
            source_rows.append({"case_id": case_id, "arm": arm, "status": "completed_case_verified" if result else "case_pending",
                "declared_source_label": candidate_source_label if arm == "candidate" else None,
                "declared_label_scope": "human supplied label; actual executed files are identified by the verified fingerprint, not inferred git metadata",
                "source_fingerprint": actual_source["source_fingerprint"] if actual_source else None,
                "actual_source_file_count": actual_source["file_count"] if actual_source else None,
                "actual_git_metadata": actual_source.get("git_commit") if actual_source else None,
                "GPU_report_sha256": ledger[arm]["GPU_report_sha256"] if result else None,
                "wall_report_sha256": ledger[arm]["wall_report_sha256"] if result else None,
                "actual_GPU_root": ledger[arm].get("actual_GPU_root") if result else None,
                "actual_GPU_driver": ledger[arm].get("actual_GPU_driver") if result else None,
                "GPU_origin_binding": ledger[arm].get("GPU_origin_binding") if result else None,
                "actual_eddy_gp_seeds": ledger[arm]["actual_eddy_gp_seeds"] if result else None,
                "memory_budget": ledger[arm]["memory_budget"] if result else None})
        for atlas in atlases:
            actual = result["atlases"][atlas] if result else None
            node_count = actual["node_count"] if actual else None
            if actual:
                compare.check(isinstance(node_count, int) and node_count == len(actual["node_rows"]) and
                              actual["nodes_semantics_equal"] is True and set(actual["matrices"]) == set(compare.MATRICES),
                              "actual per-case node semantics or matrix set is incomplete")
            for kind in compare.MATRICES:
                values = actual["matrices"][kind] if actual else None
                entry = {"case_id": case_id, "atlas": atlas, "kind": kind, "node_count": node_count,
                         "status": "completed" if values else "pending", "nodes_semantics_equal": actual["nodes_semantics_equal"] if actual else None,
                         "exact_scientific_array_equal": values["exact_scientific_array_equal"] if values else None,
                         **{key: checked_number(values.get(key)) if values and key != "strict_count_equal" else values.get(key) if values else None for key in MATRIX_FIELDS},
                         "ULP_diagnostics": values.get("ULP_diagnostics") if values else None}
                if values:
                    compare.check(values["baseline"]["shape"] == values["candidate"]["shape"] == [node_count, node_count] and
                                  values["baseline"]["finite"] and values["candidate"]["finite"], "actual matrix shape or finite gate changed")
                    for arm in ("baseline", "candidate"):
                        key = f"atlases/{atlas}/connectome_{kind}.csv"
                        entry[arm + "_CSV_sha256"] = ledger[arm]["outputs"][key]["sha256"]
                matrix_rows.append(entry)
        case_rows.append(row)
    complete = sum(row["paired_status"] == "completed_comparison" for row in case_rows)
    finalized = complete == 10 and all(row["anatomy_status"] == "completed" for row in case_rows) and \
                state.get("status") == "completed_actual_ten_case_comparison" and bool(state.get("end_utc"))
    return {"schema_version": 1, "observed_utc": compare.anatomy.utc(), "comparison_status_sha256": status_identity["sha256"],
            "status": "complete_actual_ten_case_tables" if finalized else "failed_actual_comparison_tables" if state.get("failed_cases") else "partial_actual_comparison_tables",
            "requested_cases": 10, "completed_pairs": complete, "requested_atlases_per_case": 8, "requested_matrices_per_case": 32,
            "atlas_names": atlases, "node_policy": "same-case same-atlas ordered nodes.tsv; case-specific K without trimming, padding or cross-subject K equality",
            "case_rows": case_rows, "matrix_rows": matrix_rows, "anatomy_rows": anatomy_rows, "source_rows": source_rows,
            "ready_for_ten_case_render": finalized, "MRtrix_repeat_acceptance": "not_decided_by_this_table_tool",
            "scope": "actual immutable scientific comparison reports and verified current output bytes; no missing value imputation, source/report mutation, GPU or MRI computation; stage walls/queue/gaps retained separately",
            "GPU_used": False}


def write_tables(report_dir, summary):
    report_dir = Path(report_dir)
    compare.anatomy.atomic_json(report_dir / "summary.json", summary)
    columns = {
        "cases": ("case_id", "anatomy_status", "anatomy_exact", "paired_status", "count_exact_all_atlases", "matrix_numeric_exact_all_atlases",
                  *(arm + "_" + field for arm in ("baseline", "candidate") for field in ("official_recon_command_seconds", *TIME_FIELDS)), "waiting_reason"),
        "matrices": ("case_id", "atlas", "kind", "node_count", "status", "nodes_semantics_equal", "exact_scientific_array_equal", *MATRIX_FIELDS,
                     "baseline_CSV_sha256", "candidate_CSV_sha256"),
        "anatomy": ("case_id", "file", "status", "strict_scientific_equal", "file_bytes_equal"),
        "sources": ("case_id", "arm", "status", "declared_source_label", "source_fingerprint", "actual_source_file_count", "actual_git_metadata", "GPU_report_sha256", "wall_report_sha256")}
    for name, fields in columns.items():
        temporary = report_dir / (name + ".csv.partial")
        with temporary.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
            writer.writerows(summary[{"cases": "case_rows", "matrices": "matrix_rows", "sources": "source_rows", "anatomy": "anatomy_rows"}[name]])
        temporary.replace(report_dir / (name + ".csv"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True, help="fresh summary namespace; existing summaries are never resumed/overwritten")
    parser.add_argument("--candidate-source-label", help="declared human-readable frozen source label; cannot substitute for file fingerprint")
    parser.add_argument("--watch", action="store_true", help="wait for actual case comparisons; no MRI computation or live-process query")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=72)
    options = parser.parse_args(argv)
    compare.check(options.comparison_root.is_absolute() and options.report_dir.is_absolute() and
                  1 <= options.poll_seconds <= 60 and 0 < options.timeout_hours <= 168, "invalid explicit paths or observation interval")
    state, _ = bounded_json(options.comparison_root / "status.json")
    protected = [options.comparison_root, *[state["configuration"][key] for key in ("baseline_anatomy_root", "baseline_root", "candidate_root")]]
    manifest, manifest_identity = bounded_json(state["configuration"]["manifest"])
    compare.check(manifest_identity == state["manifest"], "canonical manifest changed before summary namespace creation")
    origins, _, binding_identity = compare.load_origins(state["configuration"]["prep_bindings"], compare.manifest_cases(manifest))
    compare.check(binding_identity == state["preparation_bindings"], "original official preparation binding changed")
    protected += [origin[key] for origin in origins for key in ("root", "driver_dir")]
    protected += [Path(state["configuration"][key]).parent for key in ("manifest", "prep_bindings", "baseline_driver", "candidate_driver")]
    GPU_bindings = configuration_GPU_origins(state["configuration"], compare.manifest_cases(manifest))
    for binding in GPU_bindings.values():
        declaration = binding["declaration"]
        protected += [declaration["replacement"]["root"], Path(declaration["replacement"]["driver_status"]).parent,
                      Path(declaration["replacement"]["runtime_preflight"]["path"]).parent, Path(declaration["original"]["driver_snapshot"]["path"]).parent]
    if state.get("prior_comparison_binding"): protected.append(Path(state["prior_comparison_binding"]["original_status"]["path"]).parent)
    for arm, filename in (("baseline", "cohort_config.json"), ("candidate", "staged_gpu_config.json")):
        config_path = Path(state["configuration"][arm + "_root"]) / filename
        if config_path.exists():
            config, _ = bounded_json(config_path)
            protected += [source["directory"] for source in config["frozen_sources"].values()]
    compare.check_report_namespace(options.report_dir, protected)
    options.report_dir.parent.mkdir(parents=True, exist_ok=True); options.report_dir.mkdir(exist_ok=False)
    verified = {}; start = time.perf_counter()
    try:
        while True:
            summary = build_summary(options.comparison_root, candidate_source_label=options.candidate_source_label, verified_pairs=verified)
            if summary["status"] == "complete_actual_ten_case_tables":
                summary = build_summary(options.comparison_root, candidate_source_label=options.candidate_source_label)
            write_tables(options.report_dir, summary)
            print(json.dumps({key: summary[key] for key in ("status", "completed_pairs", "ready_for_ten_case_render")}), flush=True)
            if not options.watch or summary["status"] != "partial_actual_comparison_tables": break
            if (options.report_dir / "STOP_OBSERVATION").exists() or time.perf_counter() - start >= options.timeout_hours * 3600: break
            time.sleep(options.poll_seconds)
    except Exception as error:
        compare.anatomy.atomic_json(options.report_dir / "summary.json", {"schema_version": 1, "status": "failed_summary",
            "observed_utc": compare.anatomy.utc(), "error": {"type": type(error).__name__, "message": str(error)},
            "ready_for_ten_case_render": False, "GPU_used": False, "original_reports_modified": False})
        raise
    return 1 if summary["status"] == "failed_actual_comparison_tables" else 0


if __name__ == "__main__":
    raise SystemExit(main())
