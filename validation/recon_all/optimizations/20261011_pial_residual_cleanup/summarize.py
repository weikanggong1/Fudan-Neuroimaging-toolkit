"""从本轮公开聚合收据生成面对分类与严格诊断，不读取影像或参考几何。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(*, report_directory: Path, output: Path) -> dict:
    """输入四份 JSON 聚合收据；输出分类、版本哈希与完成/质量分离状态。

    report_directory 必须含 cleanup_report.json、pairs_report.json 和
    figure_receipt.json、completion_receipt.json。核对实际退出码、执行
    状态、相同末态和 marker 哈希；缺文件、
    分类计数矛盾或不同末态会报错。只消费报告，不运行 MRI/表面算子。
    """
    names = ("cleanup_report.json", "pairs_report.json", "figure_receipt.json", "completion_receipt.json")
    loaded = {name: json.loads((report_directory / name).read_text()) for name in names}
    cleanup, pairs, figure, completion = (loaded[name] for name in names)
    if cleanup["status"] != "complete_diagnosis" or pairs["status"] != "complete_readonly_pairs":
        raise ValueError("complete diagnostic receipts required")
    exit_codes = completion["observed_exit_codes"]
    if set(exit_codes) != {"cleanup", "pairs", "figure"} or any(value != 0 for value in exit_codes.values()):
        raise ValueError("all three observed exit sentinels must be zero")
    terminal = cleanup["snapshot_sha256"]["terminal"]
    if terminal != pairs["terminal_file_sha256"] or terminal != figure["terminal_sha256"]:
        raise ValueError("different terminal inputs")
    if cleanup["active_source_sha256"]["place_surface_intersection_marking"] != pairs["source_marker_sha256"]:
        raise ValueError("different source marker")
    total = pairs["valid_positive_pairs"]
    both = pairs["pairs_with_both_faces_all_ripped"]
    any_fixed = pairs["pairs_with_any_face_all_ripped"]
    if not 0 <= both <= any_fixed <= total:
        raise ValueError("inconsistent pair categories")
    first, extra = cleanup["source_replays"]
    references = cleanup["reference_diagnostics"]
    same_geometry = all(row["same_ordered_faces"] and row["terminal_vs_reference"]["coordinate_bytes_equal"]
                        for row in references)
    result = {
        "schema_version": 1,
        "scope": "frozen single-hemisphere pial cleanup diagnosis; no production algorithm change",
        "execution_complete": True,
        "cleanup_child_exit_code": exit_codes["cleanup"],
        "pairs_child_exit_code": exit_codes["pairs"],
        "figure_child_exit_code": exit_codes["figure"],
        "cleanup_outputs_equal_original_terminal_bits": all(
            row["vs_original_terminal"]["coordinate_bytes_equal"] for row in cleanup["source_replays"]),
        "same_seven_stage_input_sha256_verified_for_both_references": cleanup["same_input_reference_repeats_prevalidated"],
        "same_ordered_reference_geometry": same_geometry,
        "reference_repeats_coordinate_bits_equal": cleanup["reference_repeats_coordinate_bits_equal"],
        "reference_surface_file_bytes_equal": references[0]["file_sha256"] == references[1]["file_sha256"],
        "reference_file_comment_difference_evidence": "existing v3 reference receipt; current tool compares decoded geometry",
        "producer_mesh_quality": cleanup["producer_mesh_quality"],
        "source_residual_faces": pairs["source_intersecting_faces"],
        "source_residual_vertices": pairs["source_marked_vertices"],
        "positive_pairs": {
            "total": total,
            "both_faces_all_vertices_ripped": both,
            "exactly_one_face_all_vertices_ripped": any_fixed - both,
            "each_face_has_at_least_one_movable_vertex": total - any_fixed,
            "both_directions_positive": pairs["pairs_both_directions_positive"],
            "classification_exhaustive_and_disjoint": both + (any_fixed - both) + (total - any_fixed) == total,
            "movable_vertex_does_not_establish_a_safe_repair_or_guaranteed_convergence": True,
        },
        "same_input_white_source_qc": {
            "intersecting_faces": pairs["white_source_intersecting_faces"],
            "marked_vertices": pairs["white_source_marked_vertices"],
            "scope": "white only; does not certify all upstream stages",
        },
        "complete_source_cleanup": {"before": first["diagnostics"]["intersecting_faces_before"],
                                    "after": first["diagnostics"]["intersecting_faces_after"],
                                    "seconds": first["complete_cleanup_seconds"]},
        "additional_terminal_cleanup": {"before": extra["diagnostics"]["intersecting_faces_before"],
                                       "after": extra["diagnostics"]["intersecting_faces_after"],
                                       "seconds": extra["complete_cleanup_seconds"],
                                       "coordinate_elements_changed": extra["vs_replay_input"]["different_coordinate_elements"]},
        "diagnostic_wall_seconds_including_validation_io_and_reference_read": cleanup["diagnostic_wall_seconds"],
        "pairs_and_white_qc_wall_seconds": pairs["diagnostic_wall_seconds"],
        "gpu_memory_bytes": None,
        "peak_cpu_memory_bytes": None,
        "overall_metric_equivalence": "not_assessed",
        "raw_t1_workflow_completed_by_this_diagnosis": False,
        "v3_terminal_equals_v4_terminal_sha_verified": False,
        "whole_pipeline_speedup": None,
        "report_sha256": {name: hashlib.sha256((report_directory / name).read_bytes()).hexdigest() for name in names},
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(report_directory=arguments.report_directory, output=arguments.output)
