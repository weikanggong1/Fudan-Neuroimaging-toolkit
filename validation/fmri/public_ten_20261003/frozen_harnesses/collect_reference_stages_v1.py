#!/usr/bin/env python3
"""从官方参考的真实节点监控记录提取分步骤时间；不运行MRI，不相加冒充整例时间。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

GROUPS = {
    "anatomical_fit_including_registration": {"anat_fit_wf"},
    "freesurfer_reconstruction": {"surface_recon_wf"},
    "anatomical_template_registration": {"register_template_wf"},
    "bold_fit_including_coregistration": {"bold_fit_wf"},
    "head_motion": {"bold_hmc_wf"},
    "mni6_volume_resampling": {"bold_MNI6_wf"},
    "t1w_volume_resampling": {"bold_anat_wf"},
    "native_surface_sampling": {"bold_surf_wf"},
    "msmsulc_registration": {"msm_sulc_wf"},
    "fslr_metric_resampling": {"bold_fsLR_resampling_wf"},
    "cifti_creation": {"bold_grayords_wf"},
    "confounds": {"bold_confounds_wf"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_key(row):
    # Nipype MapNode aggregate and its mapflow children record the same run.
    return (row.get("start_time"), row.get("end_time"), row.get("duration_seconds"),
            row.get("command_sha256"), row.get("command_executable"))


def summarize(rows):
    unique = {}
    for row in rows:
        key = execution_key(row)
        previous = unique.get(key)
        if previous is None or len(row["relative_result_path"]) > len(previous["relative_result_path"]):
            unique[key] = row
    valid = [row for row in unique.values() if row.get("start_time") and row.get("end_time")]
    starts = [datetime.fromisoformat(row["start_time"]) for row in valid]
    ends = [datetime.fromisoformat(row["end_time"]) for row in valid]
    active_seconds = 0.0
    merged = []
    for start, end in sorted(zip(starts, ends)):
        if end < start:
            raise ValueError("Recorded node end precedes start")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    active_seconds = sum((end - start).total_seconds() for start, end in merged)
    mem = [row["mem_peak_gb"] for row in valid if row.get("mem_peak_gb") is not None]
    cpu = [row["cpu_percent"] for row in valid if row.get("cpu_percent") is not None]
    return {
        "raw_runtime_records": len(rows), "unique_execution_records": len(unique),
        "duplicate_aggregate_records_removed": len(rows) - len(unique),
        "execution_records_with_start_end": len(valid),
        "first_node_start_utc": min(starts).isoformat() if starts else None,
        "last_node_end_utc": max(ends).isoformat() if ends else None,
        "elapsed_span_seconds": (max(ends) - min(starts)).total_seconds() if starts else None,
        "recorded_node_active_interval_union_seconds": active_seconds if valid else None,
        "maximum_single_node_duration_seconds": max((row["duration_seconds"] for row in valid if row.get("duration_seconds") is not None), default=None),
        "maximum_observed_node_mem_peak_gb": max(mem, default=None),
        "node_records_with_mem_peak": len(mem),
        "maximum_observed_node_cpu_percent": max(cpu, default=None),
        "node_records_with_cpu_percent": len(cpu),
        "longest_nodes": [{k: v for k, v in row.items() if k != "interface"}
                          for row in sorted(valid, key=lambda x: x.get("duration_seconds") or 0, reverse=True)[:10]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-name", default="report.corrected.public.json")
    args = parser.parse_args()
    result = {"status": "partial_cohort", "created_utc": datetime.now(timezone.utc).isoformat(),
              "collector_sha256": sha(Path(__file__)), "cases": {},
              "timing_boundary": "Stage elapsed span is earliest recorded node start to latest node end, including internal wait and gaps. Groups overlap and are never summed into whole time. Production process/whole clocks are read directly from each original pipeline report.",
              "active_union_boundary": "Union of the recorded node execution intervals within each selected workflow, with overlaps counted once and intervals between node executions excluded. Includes node-internal I/O and waits; does not recover unsampled activity or GPU kernel time. Workflow active unions also overlap and must not be summed into whole time.",
              "resources_boundary": "Nipype per-node observed samples, not whole-process hard memory or CPU ceilings; absent samples remain unmeasured.",
              "deduplication": "Exact start/end/duration/command SHA/executable tuple collapses MapNode aggregate and mapflow duplicates. No runtime sums are used."}
    for report_path in sorted((args.reference_root / "cases").glob("*/attempt-*/" + args.report_name)):
        report = json.loads(report_path.read_text())
        if report.get("status") != "complete" or report.get("command_exit_code") != 0:
            continue
        nodes_path = report_path.parent / "node_runtime.public.json"
        runtime = json.loads(nodes_path.read_text())
        row = {"case_id": report["subject"], "attempt": report_path.parent.name,
               "report_sha256": sha(report_path), "node_runtime_sha256": sha(nodes_path),
               "container_process_wall_seconds": report["container_process_wall_seconds"],
               "original_continuous_wall_through_saved_QC_seconds": report["continuous_wall_through_saved_QC_seconds"],
               "original_QC_boundary_status": report.get("continuous_wall_through_saved_QC_boundary_status", "original QC passed"),
               "recovered_QC_seconds": report.get("recovered_QC_seconds"),
               "runtime_extraction_errors": runtime["extraction_errors"],
               "all_records": summarize(runtime["nodes"]), "stages": {}}
        for label, workflows in GROUPS.items():
            selected = [node for node in runtime["nodes"] if workflows.intersection(node["relative_result_path"].split("/"))]
            row["stages"][label] = {"workflow_components": sorted(workflows), **summarize(selected)}
        if report["subject"] in result["cases"]:
            raise ValueError("More than one completed fresh attempt per subject")
        result["cases"][report["subject"]] = row
    result["complete_case_count"] = len(result["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": list(result["cases"]), "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
