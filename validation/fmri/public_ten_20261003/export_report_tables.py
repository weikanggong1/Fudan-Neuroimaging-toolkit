#!/usr/bin/env python3
"""从具名匿名报告导出全域、全部21结构与独立计时 CSV；不读取 MRI。

--cohort-report PUBLIC.json --comparisons-root DIRECTORY --output-root NEW_DIRECTORY
[--reference-stages PUBLIC.json]
Python: export_tables(cohort_report, comparisons_root, output_root, *, reference_stages=None) -> dict
十例/21结构固定；尚未配对病例仍保留状态行。空单元格表示未测/未定义。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CASES = ("CON01", "CON03", "CON04", "CON05", "CON06", "CON07", "CON08", "CON09", "CON10", "CON11")
STRUCTURES = tuple("CIFTI_STRUCTURE_" + name for name in (
    "CORTEX_LEFT", "CORTEX_RIGHT", "ACCUMBENS_LEFT", "ACCUMBENS_RIGHT", "AMYGDALA_LEFT",
    "AMYGDALA_RIGHT", "BRAIN_STEM", "CAUDATE_LEFT", "CAUDATE_RIGHT", "CEREBELLUM_LEFT",
    "CEREBELLUM_RIGHT", "DIENCEPHALON_VENTRAL_LEFT", "DIENCEPHALON_VENTRAL_RIGHT",
    "HIPPOCAMPUS_LEFT", "HIPPOCAMPUS_RIGHT", "PALLIDUM_LEFT", "PALLIDUM_RIGHT",
    "PUTAMEN_LEFT", "PUTAMEN_RIGHT", "THALAMUS_LEFT", "THALAMUS_RIGHT"))
COMPARE_SHA = "cb04445f6acf4252bf067500e70d53ec2981ee3ce9a24d9f47365c6ec12737d1"
COLLECTOR_SHA = "b508af760ee79f8a7b228904cc39cbff9ffdebe80a206cedc4ad7e4d627b26d5"
REVISION = "1128bc52c7a0233266e5b8a8d7dc0b382994e676"
REFERENCE_STAGE_SOURCES = {
    "0dd60bbae3e15a6ca07d5989e6ee14c10cde3f2d1871e86fc4e946fdc22d4193",
    "703a27edbd6c29fc897228b6056f36233873dac004e37d05ded03090826a4026"}
COUNTS = ("points", "frames", "values", "nonfinite_candidate_values", "nonfinite_reference_values",
          "constant_candidate", "constant_reference", "both_constant", "one_side_constant",
          "zero_candidate", "zero_reference", "both_zero", "one_side_zero")
SCALARS = ("rmse", "reference_rms", "normalized_rmse")
METRIC_COLUMNS = (*COUNTS, "defined_temporal_r", "undefined_temporal_r", "r_mean", "r_median",
                  "r_p05", "r_p95", "r_minimum", "r_maximum", *SCALARS,
                  "absolute_difference_mean", "absolute_difference_p99", "absolute_difference_maximum",
                  "temporal_mean_bias_mean", "temporal_mean_bias_median", "temporal_mean_bias_p05",
                  "temporal_mean_bias_p95", "temporal_mean_bias_minimum", "temporal_mean_bias_maximum",
                  "temporal_mean_bias_mean_absolute", "tsnr_valid_pairs", "tsnr_undefined_pairs",
                  "tsnr_candidate_mean", "tsnr_candidate_median", "tsnr_reference_mean",
                  "tsnr_reference_median", "tsnr_bias_mean", "tsnr_bias_median", "tsnr_bias_p05", "tsnr_bias_p95")


def read_json(path):
    payload = Path(path).read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("table inputs must be named report JSON objects")
    return value, hashlib.sha256(payload).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten_metrics(value):
    row = {key: value[key] for key in (*COUNTS, *SCALARS)}
    row["defined_temporal_r"] = value["temporal_r"]["count"]
    row["undefined_temporal_r"] = value["points"] - row["defined_temporal_r"]
    row.update({"r_" + key: value["temporal_r"][key] for key in ("mean", "median", "p05", "p95", "minimum", "maximum")})
    row.update({"absolute_difference_" + key: value["absolute_difference"][key] for key in ("mean", "p99", "maximum")})
    row.update({"temporal_mean_bias_" + key: value["temporal_mean_bias"]["signed"][key]
                for key in ("mean", "median", "p05", "p95", "minimum", "maximum")})
    row["temporal_mean_bias_mean_absolute"] = value["temporal_mean_bias"]["mean_absolute"]
    row.update({"tsnr_" + key: value["tsnr"][key] for key in ("valid_pairs", "undefined_pairs")})
    row.update({"tsnr_" + side + "_" + key: value["tsnr"][side][key]
                for side in ("candidate", "reference") for key in ("mean", "median")})
    row.update({"tsnr_bias_" + key: value["tsnr"]["bias"][key] for key in ("mean", "median", "p05", "p95")})
    if any(v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)) for v in row.values()):
        raise ValueError("measured table metrics must be finite numeric values or explicitly undefined")
    if row["frames"] != 180 or row["nonfinite_candidate_values"] or row["nonfinite_reference_values"]:
        raise ValueError("nonfinite or incomplete-frame results cannot be exported as completed metrics")
    return row


def write_csv(path, columns, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(cohort_report, comparisons_root, output_root, *, reference_stages=None):
    producer_digest = sha256(__file__)
    cohort_report, comparisons_root, output_root = (Path(p).resolve() for p in (cohort_report, comparisons_root, output_root))
    cohort, cohort_sha = read_json(cohort_report)
    if (cohort.get("cohort_id") != "formal-v4" or cohort.get("source_revision_declared") != REVISION
            or cohort.get("collector_script_sha256") != COLLECTOR_SHA or cohort.get("data_license") != "CC0"
            or cohort.get("analysis_tools_unchanged") is not True):
        raise ValueError("table input must be the guarded formal-v4 fixed-source CC0 collection")
    cases = {row["case_id"]: row for row in cohort["cases"]}
    if set(cases) != set(CASES) or len(cohort["cases"]) != len(CASES):
        raise ValueError("cohort table must retain every predeclared case exactly once")
    for protected in (cohort_report, comparisons_root, Path(__file__).resolve()):
        if output_root.is_relative_to(protected) or protected.is_relative_to(output_root):
            raise ValueError("tables require an independent new output directory")
    endpoint_rows, structure_rows, timing_rows, provenance, watched = [], [], [], {}, {cohort_report: cohort_sha}
    completed = []
    for case in CASES:
        record = cases[case]
        comparison = record.get("comparison", {})
        state = "complete" if comparison.get("status") == "complete" and comparison.get("exit_code") == 0 else "failed" if comparison.get("status") == "failed" else "pending"
        provenance[case] = {"comparison_status": state,
                            "candidate_status": record["candidate"]["status"], "reference_status": record["reference"]["status"]}
        metrics = None
        if state == "complete":
            path = comparisons_root / (case + ".public.json")
            if not path.is_file():
                path = comparisons_root / case / "comparison.public.json"
            report, digest = read_json(path)
            if (digest != comparison["report_sha256"] or report.get("case_id") != case
                    or report.get("cohort_id") != cohort["cohort_id"] or report.get("status") != "complete"
                    or report.get("comparison_script_sha256") != COMPARE_SHA
                    or report.get("candidate_source_revision_bound") != REVISION
                    or report.get("sources_unchanged_during_comparison") is not True
                    or report.get("frames") != 180 or report.get("tr_seconds") != record["tr_seconds"]
                    or set(report["cifti"]["per_structure"]) != set(STRUCTURES)
                    or any(report["cifti"].get(key) is not True for key in
                           ("brain_axis_exactly_equal", "series_axis_exactly_equal", "fixed_original_assets_axis_exactly_equal"))):
                raise ValueError("completed comparison table differs from its bound cohort report or fixed full-frame axis")
            watched[path] = digest
            metrics = report
            completed.append(case)
            provenance[case].update(comparison_sha256=digest,
                                    candidate_report_sha256=record["candidate"]["report_sha256"],
                                    reference_report_sha256=record["reference"]["report_sha256"])
        common = {"case_id": case, "comparison_status": state, "frames": record["frames"], "tr_seconds": record["tr_seconds"]}
        for domain in ("volume", "cifti"):
            endpoint_rows.append({**common, "domain": domain, **(flatten_metrics(metrics[domain]) if metrics else {})})
        for structure in STRUCTURES:
            structure_rows.append({**common, "structure": structure,
                                   **(flatten_metrics(metrics["cifti"]["per_structure"][structure]) if metrics else {})})
        for side in ("candidate", "reference"):
            run = record[side]
            qc = run.get("reference_qc_recovery", {})
            timing_base = {"case_id": case, "side": side, "run_status": run["status"],
                           "original_reference_wrapper_status": qc.get("original_harness_status", ""),
                           "original_reference_wrapper_boundary": qc.get("original_continuous_boundary", "")}
            clocks = run.get("timing_seconds", {})
            if side == "candidate" and run.get("queue_process_wall_seconds") is not None:
                timing_rows.append({**timing_base, "category": "process", "field": "queue_process_wall_seconds",
                                    "seconds": run["queue_process_wall_seconds"], "boundary": run["queue_process_wall_boundary"]})
            for field, value in clocks.items():
                timing_rows.append({**timing_base, "category": "independent_clock", "field": field, "seconds": value,
                                    "boundary": run.get("timing_boundaries", {}).get(field, "unavailable boundary")})
            for field, value in run.get("stage_seconds", {}).items():
                timing_rows.append({**timing_base, "category": "nested_stage", "field": field, "seconds": value,
                                    "boundary": run["stage_scope"]})
            for field, value in run.get("reconstruction_execution", {}).get("reconstruction_timing_seconds", {}).items():
                timing_rows.append({**timing_base, "category": "nested_reconstruction_adapter", "field": field,
                                    "seconds": value, "boundary": "adapter-owned interval; nested within pipeline; not summed with outer or parallel clocks"})
            if not clocks and not run.get("stage_seconds"):
                timing_rows.append({**timing_base, "category": "unmeasured", "field": "", "seconds": None,
                                    "boundary": "complete run clock not yet available; retained predeclared case"})
    if completed != cohort["completed_comparisons"]:
        raise ValueError("cohort completed list contradicts explicitly bound comparison rows")
    stage_rows, stage_digest = [], None
    if reference_stages is not None:
        stage_path = Path(reference_stages).resolve()
        stages, stage_digest = read_json(stage_path)
        if (stages.get("collector_sha256") not in REFERENCE_STAGE_SOURCES
                or not set(stages.get("cases", {})).issubset(CASES)):
            raise ValueError("reference stage records must come from the fixed real workflow extractor")
        watched[stage_path] = stage_digest
        for case in CASES:
            reference = cases[case]["reference"]
            value = stages["cases"].get(case)
            if value is None:
                stage_rows.append({"case_id": case, "status": "unmeasured", "workflow_group": ""})
                continue
            if (reference.get("status") != "complete" or value["report_sha256"] != reference.get("report_sha256")
                    or value.get("runtime_extraction_errors")):
                raise ValueError("reference stages differ from the explicitly bound strict completed report")
            for name, item in value["stages"].items():
                record = {key: item[key] for key in ("recorded_node_active_interval_union_seconds", "elapsed_span_seconds",
                          "unique_execution_records", "execution_records_with_start_end", "maximum_single_node_duration_seconds",
                          "maximum_observed_node_mem_peak_gb", "node_records_with_mem_peak",
                          "maximum_observed_node_cpu_percent", "node_records_with_cpu_percent")}
                if any(v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)) for v in record.values()):
                    raise ValueError("reference node durations/resources must be finite or explicitly unmeasured")
                stage_rows.append({"case_id": case, "status": "measured", "workflow_group": name, **record,
                                   "boundary": "union merges overlapping recorded node activity in this group; span includes dependencies/wait; neither groups nor intervals are summed into whole wall",
                                   "resource_scope": "only actually monitored nodes; external recon-all/MSM/MCFLIRT resource peaks may be unavailable"})
        if output_root.is_relative_to(stage_path) or stage_path.is_relative_to(output_root):
            raise ValueError("tables must not replace the source stage report")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_csv(output_root / "endpoints.csv", ("case_id", "comparison_status", "tr_seconds", "domain", *METRIC_COLUMNS), endpoint_rows)
    write_csv(output_root / "structures.csv", ("case_id", "comparison_status", "tr_seconds", "structure", *METRIC_COLUMNS), structure_rows)
    write_csv(output_root / "timings.csv", ("case_id", "side", "run_status", "original_reference_wrapper_status",
              "original_reference_wrapper_boundary", "category", "field", "seconds", "boundary"), timing_rows)
    output_names = ["endpoints.csv", "structures.csv", "timings.csv"]
    if reference_stages is not None:
        write_csv(output_root / "reference_stages.csv", ("case_id", "status", "workflow_group",
                  "recorded_node_active_interval_union_seconds", "elapsed_span_seconds", "unique_execution_records",
                  "execution_records_with_start_end", "maximum_single_node_duration_seconds",
                  "maximum_observed_node_mem_peak_gb", "node_records_with_mem_peak",
                  "maximum_observed_node_cpu_percent", "node_records_with_cpu_percent", "boundary", "resource_scope"), stage_rows)
        output_names.append("reference_stages.csv")
    producer_unchanged = sha256(__file__) == producer_digest
    guards = all(sha256(path) == digest for path, digest in watched.items())
    result = {"schema_version": 1, "status": "complete_cohort" if completed == list(CASES) else "partial_cohort",
              "cohort_id": cohort["cohort_id"], "source_revision": REVISION, "data_license": "CC0",
              "input_cohort_sha256": cohort_sha, "producer_script_sha256": producer_digest,
              "producer_unchanged": producer_unchanged, "inputs_unchanged": guards,
              "cases": provenance, "completed_cases": completed, "pending_or_failed_cases": [c for c in CASES if c not in completed],
              "expected_structures_per_case": 21, "structure_rows": len(structure_rows),
              "reference_stages_input_sha256": stage_digest,
              "csv_sha256": {name: sha256(output_root / name) for name in output_names},
              "method": {"operation": "copy numeric fields from SHA-bound published comparison/cohort reports; no MRI read or recomputation",
                         "missing_values": "blank CSV cells mean unmeasured or undefined; pending/failed rows remain",
                         "normalized_rmse": "RMSE / sqrt(mean(reference**2)); full point/frame domain, including constants",
                         "timing": "independent clocks and nested stages exported separately with boundaries; no summed whole, ratio or cross-host UTC inference"}}
    if not guards or not producer_unchanged:
        result["status"] = "failed_input_changed"
    with (output_root / "tables_provenance.public.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not guards or not producer_unchanged:
        raise RuntimeError("published table input/producer changed during export")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-report", required=True, type=Path)
    parser.add_argument("--comparisons-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference-stages", type=Path, help="可选真实节点阶段匿名报告，严格绑定同例 reference report SHA")
    args = parser.parse_args()
    report = export_tables(args.cohort_report, args.comparisons_root, args.output_root, reference_stages=args.reference_stages)
    print(json.dumps({"status": report["status"], "completed_cases": report["completed_cases"], "structure_rows": report["structure_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
