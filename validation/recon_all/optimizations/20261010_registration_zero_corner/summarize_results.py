"""校验公开/原始收据绑定，汇总真实完整注册；不重跑或读取私有 MRI。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary(root: Path) -> dict:
    versions = {}
    manifest_index = {}
    for version in ("v1", "v2"):
        directory = root / "reports" / version
        manifest = json.loads((directory / "PUBLIC_EXPORT_MANIFEST.json").read_text())
        assert manifest["overall_equivalence"] == "not_assessed"
        entries = {}
        for entry in manifest["files"]:
            path = directory / entry["published_path"]
            assert sha256(path) == entry["published_sha256"], path
            assert entry["numeric_leaves_preserved"] is True, path
            assert path.stat().st_size == entry["published_bytes"], path
            entries[entry["published_path"]] = entry
        manifest_index[version] = entries
        versions[version] = {
            "manifest_sha256": sha256(directory / "PUBLIC_EXPORT_MANIFEST.json"),
            "verified_reports": len(entries),
            "numeric_bool_null_leaves_preserved": sum(e["numeric_leaf_count"] for e in entries.values()),
        }

    def load(version: str, name: str) -> dict:
        return json.loads((root / "reports" / version / (name + ".json")).read_text())

    rows = []
    for version, name in (("v1", "79-old"), ("v1", "79-new"), ("v1", "v4-new"),
                          ("v2", "v4-new2"), ("v2", "79-new2")):
        report = load(version, name)
        source_name = "source-old" if name == "79-old" else "source-new2" if version == "v2" else "source-new"
        binding = load(version, source_name)
        assert report["source_manifest_sha256"] == manifest_index[version][source_name + ".json"]["original_sha256"]
        assert len(binding["source_files"]) == 601
        assert binding["source_files"]["src/fnit/recon_all/mris_register_nonlinear.py"] == report["selected_module_sha256"]
        for module, digest in report["module_sha256"].items():
            assert binding["source_files"]["src/fnit/recon_all/" + module] == digest
        case = next(c for c in load(version, "driver")["cases"] if c["name"] == name)
        assert (case["exit_code"] == 0) == (report["status"] == "complete")
        qc = report.get("qc")
        if report["status"] == "complete":
            assert report["input_sha256_before"] == report["input_sha256_after"]
            assert report["output_sha256"] == report["registration"]["output_sha256"]
            quality_pass = bool(qc["finite_coordinates"] and qc["euler"] == 2
                                and qc["boundary_edges"] == qc["nonmanifold_edges"]
                                == qc["zero_area_faces_double"] == qc["negative_radial_faces_double"] == 0)
            sampler = report["gpu_process_sampler"]
            device_values = [s["target_device_used_bytes"] for s in sampler["samples"]
                             if s["target_device_used_bytes"] is not None]
            tree_values = [s["tree_total_bytes"] for s in sampler["samples"]
                           if s["tree_total_bytes"] is not None]
        else:
            quality_pass, sampler, device_values, tree_values = None, report["gpu_process_sampler"], [], []
        row = {
            "version": version, "case": name,
            "source_module_sha256": report["selected_module_sha256"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "execution_status": report["status"], "exit_code": case["exit_code"],
            "mesh_quality_status": "pass_checked_properties" if quality_pass is True else
                                   "failed" if quality_pass is False else "not_evaluable",
            "mesh_quality_scope": "finite/Euler/boundary/nonmanifold/zero_area/radial_orientation; selfintersection not evaluated",
            "API_wall_seconds_including_IO_sync": report.get("complete_stage_API_wall_seconds"),
            "harness_wall_seconds_including_imports_checks_compute_IO_QC":
                report["harness_wall_seconds_including_imports_checks_compute_IO_QC"],
            "full_cold_process_wall_seconds": case["full_cold_process_wall_seconds"],
            "vertices": None if qc is None else qc["vertices"],
            "faces": None if qc is None else qc["faces"],
            "negative_radial_faces_double": None if qc is None else qc["negative_radial_faces_double"],
            "allocated_peak_bytes": report["allocated_peak_bytes"],
            "reserved_peak_bytes": report["reserved_peak_bytes"],
            "tree_peak_bytes": max(tree_values) if tree_values else None,
            "ownership_status": sampler["status"],
            "shared_device_peak_upper_bound_bytes": max(device_values) if device_values else None,
            "sampling_interval_seconds": sampler["sampling_interval_seconds"],
            "max_observed_interval_seconds": sampler["max_observed_interval_seconds"],
            "error": report.get("error"), "overall_equivalence": "not_assessed",
        }
        rows.append(row)

    comparisons = []
    for version, candidate_version, candidate in (("v1", "v1", "79-new"), ("v2", "v2", "79-new2")):
        comparison = load(version, "79-comparison")
        # 比较器原始 report SHA 只能与导出清单 original_sha256 比；公开副本的 SHA 不同。
        assert comparison["control_report_sha256"] == manifest_index["v1"]["79-old.json"]["original_sha256"]
        assert comparison["candidate_report_sha256"] == manifest_index[candidate_version][candidate + ".json"]["original_sha256"]
        control_report, candidate_report = load("v1", "79-old"), load(candidate_version, candidate)
        assert comparison["input_sha256"] == control_report["input_sha256_before"] == candidate_report["input_sha256_before"]
        assert comparison["control_source_module_sha256"] == control_report["selected_module_sha256"]
        assert comparison["candidate_source_module_sha256"] == candidate_report["selected_module_sha256"]
        assert comparison["control_output_sha256"] == control_report["output_sha256"]
        assert comparison["candidate_output_sha256"] == candidate_report["output_sha256"]
        assert sha256(root / "compare_frozen_outputs.py") == comparison["comparison_script_sha256"]
        comparisons.append(comparison)
    assert rows[3]["mesh_quality_status"] == "failed" and rows[3]["negative_radial_faces_double"] == 7507
    return {
        "scope": "actual frozen same-input registration stage; not original-T1 whole pipeline",
        "publication_manifest_checks": versions,
        "full_stage_runs": rows,
        "normal_input_strict_comparisons": comparisons,
        "checkpoint_comparisons": {"corner_new1": load("v1", "checkpoint"),
                                   "corner_new2": load("v2", "corner-checkpoint"),
                                   "face_new2": load("v2", "face-checkpoint")},
        "performance_claim": "single shared-load observations; stable stage acceleration not established",
        "overall_equivalence": "not_assessed",
        "original_FS_same_abnormal_input_repeat": "not_run; declared mris_register and mris_sphere unavailable",
        "whole_pipeline_after_repair": "not_run",
        "20GB_process_tree_memory_gate": "not_verified; ownership unresolved and shared-card upper bound is not task memory",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path(__file__).resolve().parent,
                        help="已公开、带原始/公开 SHA 清单的本报告目录")
    args = parser.parse_args()
    summary = build_summary(args.report_root)
    (args.report_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (args.report_root / "stage_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["full_stage_runs"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["full_stage_runs"])
    print(json.dumps(summary["publication_manifest_checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
