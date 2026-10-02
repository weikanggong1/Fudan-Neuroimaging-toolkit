#!/usr/bin/env python3
"""Render all 110 ROI Dice rows from the final authenticated 880-row TSV.

Requires completed F8 publication and ROI-flattener evidence. Copies recorded
statistics without scoring images or fitting. Keeps original files unchanged
and refuses to overwrite its independent Markdown and artifact receipt.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

IDS = tuple(f"sub-{index:02d}" for index in range(1, 11))
COHORTS = {"cohort_all": IDS, "cohort_new_subjects": IDS[1:]}
SPACES = {f"{mode}_{resolution}": (mode, resolution)
          for mode in ("raw", "stage") for resolution in ("native", "hr")}
FAMILIES = ("brainstem", "thalamus", "hippocampus_left", "hippocampus_right",
            "amygdala_left", "amygdala_right")
METRICS = ("dice", "jaccard", "hard_volume_difference_mm3",
           "hard_volume_relative_difference", "soft_volume_difference_mm3",
           "soft_volume_relative_difference")
FLATTENER_SHA = "47df4961b6494e8add8e6d76b8efbf3bf994c64f3cc83e21993f835801c1278e"
F8_SHA = "f8c433c1451c3c45c3a1afee43738bd874788199434cdf24bc9803f028b1b011"
BASE_COMMIT = "ac692bb4f7868a24ea4bd67180162e81726de9b4"
MAIN_COMMIT = "f436de588647a0de80735e4a98d53df5d88e502d"
BASE_SOURCE_SHA = "d8c8f7aecdae521751f30fdd09bea6acbffd2c0a50c9fb70e1dc97ad74717b07"
MAIN_SOURCE_SHA = "4fb86d811b92c9db6be002f2c58a46b8f07f624b42671d034bc996294ce0b602"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path, data=None):
    data = path.read_bytes() if data is None else data
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha(data)}


def read_json(data):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, parse_constant=nonfinite)


def verify(path, data, expected):
    if expected.get("sha256") != sha(data) or expected.get("bytes") != len(data):
        raise ValueError(f"Actual size/SHA differs from authenticated evidence: {path}")


def integer(cell):
    value = int(cell)
    if str(value) != cell or value < 0:
        raise ValueError("Expected a canonical nonnegative integer cell")
    return value


def number(cell):
    if cell == "":
        return None
    value = float(cell)
    if not math.isfinite(value):
        raise ValueError("Nonfinite metric in authenticated TSV")
    return value


def validate_grid(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
    required = {"cohort", "space", "mode", "resolution", "label", "name", "source", "family",
                "planned_subjects", "completed_subjects", "defined_subjects", "missing_or_na",
                "hard_status_counts"}
    required |= {metric + "_" + field for metric in METRICS for field in (
        "defined_subjects", "missing_or_na_subjects", "mean", "std_sample", "min", "max", "median", "std_ddof")}
    if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not required.issubset(reader.fieldnames)):
        raise ValueError("ROI-flattener column schema differs")
    rows = list(reader)
    if len(rows) != 880:
        raise ValueError("Exactly 880 cohort-space-ROI rows are required")
    grid, labels = {}, {}
    for row in rows:
        cohort, space = row["cohort"], row["space"]
        if cohort not in COHORTS or space not in SPACES:
            raise ValueError("Unknown cohort or scoring space")
        if (row["mode"], row["resolution"]) != SPACES[space]:
            raise ValueError("Mode/resolution disagree with scoring space")
        label = integer(row["label"])
        if label == 0 or row["family"] not in FAMILIES or not row["name"]:
            raise ValueError("Invalid ROI identifier/family/name")
        key = (cohort, space, label)
        if key in grid:
            raise ValueError("Duplicate aggregate ROI row")
        meta = {name: row[name] for name in ("name", "source", "family")}
        if label in labels and labels[label] != meta:
            raise ValueError("ROI metadata differ between cohorts or spaces")
        labels[label] = meta
        planned = len(COHORTS[cohort])
        if integer(row["planned_subjects"]) != planned:
            raise ValueError("Planned denominator must remain ten/nine")
        if integer(row["completed_subjects"]) > planned:
            raise ValueError("More completed subjects than planned")
        counts = read_json(row["hard_status_counts"])
        allowed = {"evaluated", "both_empty_hard_label", "not_evaluated",
                   "official_present_fnit_hard_label_absent", "official_hard_label_absent_fnit_present"}
        if (not isinstance(counts, dict) or set(counts) - allowed
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values())
                or sum(counts.values()) != planned):
            raise ValueError("Invalid hard-label status counts")
        for metric in METRICS:
            defined = integer(row[metric + "_defined_subjects"])
            missing = integer(row[metric + "_missing_or_na_subjects"])
            if defined + missing != planned or integer(row[metric + "_std_ddof"]) != 1:
                raise ValueError("Metric count or sample-SD convention differs")
            values = {name: number(row[metric + "_" + name]) for name in ("mean", "std_sample", "min", "max", "median")}
            for name in ("mean", "min", "max", "median"):
                if (values[name] is not None) != bool(defined):
                    raise ValueError("Undefined/defined distribution value differs")
            if (values["std_sample"] is not None) != (defined >= 2):
                raise ValueError("Sample SD must remain NA with fewer than two defined subjects")
            if values["std_sample"] is not None and values["std_sample"] < 0:
                raise ValueError("Negative sample SD")
            if defined:
                tolerance = 1e-12 * max(1, abs(values["min"]), abs(values["max"]))
                if not values["min"] - tolerance <= values["mean"] <= values["max"] + tolerance:
                    raise ValueError("Recorded mean lies outside its range")
                if metric in ("dice", "jaccard") and not 0 <= values["min"] <= values["max"] <= 1:
                    raise ValueError("Overlap metric outside [0, 1]")
        if (integer(row["defined_subjects"]) != integer(row["dice_defined_subjects"])
                or integer(row["missing_or_na"]) != integer(row["dice_missing_or_na_subjects"])
                or integer(row["defined_subjects"]) != planned - counts.get("both_empty_hard_label", 0) - counts.get("not_evaluated", 0)
                or integer(row["completed_subjects"]) != planned - counts.get("not_evaluated", 0)):
            raise ValueError("Dice counts differ from original recorded statuses")
        grid[key] = row
    expected = {(cohort, space, label) for cohort in COHORTS for space in SPACES for label in labels}
    if len(labels) != 110 or set(grid) != expected or {meta["family"] for meta in labels.values()} != set(FAMILIES):
        raise ValueError("The full 110 ROI x 2 cohort x 4 space grid is required")
    return grid, labels


def validate_main_audit(audit, baseline_source, baseline_cohort_sha):
    if (audit.get("state") != "zero_voxel_equivalence_passed"
            or audit.get("final_main_regression_outcomes_ready") is not True
            or audit.get("zero_voxel_equivalence_passed") is not True
            or audit.get("final_queue_state") != "completed"
            or audit.get("final_source_asset_metadata_error") is not None
            or audit.get("baseline_commit") != BASE_COMMIT or audit.get("main_commit") != MAIN_COMMIT):
        raise ValueError("Existing main audit is not a completed exact-equivalence result")
    if (audit["source_audits"]["baseline"]["source_manifest"]["sha256"] != baseline_source
            or audit["source_audits"]["main"]["source_manifest"]["sha256"] != MAIN_SOURCE_SHA
            or audit["manifests"]["baseline"]["sha256"] != baseline_cohort_sha):
        raise ValueError("Main audit uses another baseline/main source or cohort")
    cases = audit["cases"]
    if (len(cases) != 10 or {case["case_id"] for case in cases} != set(IDS)
            or any(case["state"] != "zero_voxel_equivalence_passed" or case["error"] is not None
                   or case["development_seen"] is not (case["case_id"] == "sub-01") for case in cases)):
        raise ValueError("Every predetermined main case must have passed")
    maps = audit["map_pairs"]
    map_keys = ("labels", *("highres/" + name for name in (
        "brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")))
    if len(maps) != 50 or {(row["case_id"], row["map"]) for row in maps} != {(case, name) for case in IDS for name in map_keys}:
        raise ValueError("The main audit must cover all fifty raw/native+HR maps")
    for row in maps:
        if (row["measurement_status"] != "compared" or row["different_voxels"] != 0
                or not all(row[name] is True for name in ("exact", "shape_exact", "affine_exact", "dtype_exact", "all_values_exact"))):
            raise ValueError("Main raw label-map values or geometry differ")
        for name in ("array_sha256", "shape", "affine", "array_dtype", "header_dtype"):
            if row["baseline_" + name] != row["main_" + name]:
                raise ValueError("Main raw map identity differs")
    return {"raw_main_dice_reuse_supported": True, "compared_maps": 50,
            "different_voxels": 0, "current_main_commit": MAIN_COMMIT,
            "stage_scope": "Actual ac692bb stage results retained; no f436 stage rerun inferred."}


def dice_cell(row):
    mean, sd = number(row["dice_mean"]), number(row["dice_std_sample"])
    value = "NA" if mean is None else f"{mean:.4f}"
    deviation = "NA" if sd is None else f"{sd:.4f}"
    return f"{value} ± {deviation}; {row['defined_subjects']}/{row['planned_subjects']}, NA {row['missing_or_na']}"


def markdown(grid, labels, main_reuse):
    lines = ["# 全部 110 分区 Dice", "",
             "本表直接展示已认证 ROI 汇总的既有统计值；全部十例和排除开发用 sub-01 的新九例分别汇总。"
             "raw-native 使用公开 T1 的原网格；stage-native 使用该病例本轮 fresh 官方 norm 网格。"
             "两种输入各自与对应网格的官方细标签比较，分组保留。", "",
             f"原计量版本为 `{BASE_COMMIT}`。"]
    if main_reuse is not None:
        lines.append(f"当前 main `{MAIN_COMMIT}` 的十例独立 raw 回归中，50 张原网格/高分辨率标签图体素、"
                     "shape、affine 和 dtype 完全相同，因此本表 raw Dice 可用于当前 main。"
                     "该判断来自[实际 main 审计](../latest_main_regression/audit/summary.json)，不是重复拟合或重新评分。"
                     "stage 仍为 ac692bb 的实际条件测试，不称为当前 main 新跑。")
    else:
        lines.append("本次未绑定已完成的 main 零差异审计，因此表内保留原计量版本，不据此认定其他版本精度。")
    lines.extend(["", "每格为 **均值 ± 样本 SD；有效数/计划数，NA 数**，SD 为病例间差异（ddof=1）。"
                  "单方硬标签缺失的 0 Dice 保留；双方均空及失败为 NA。仅一个有效病例时均值可定义、SD 为 NA。"
                  "名称和 ID 直接使用原表，不另译亚区名称。显示保留四位小数，原精确数值及范围见完整 TSV。", "",
                  "[880 行完整 ROI 汇总](cohort_roi_summary.tsv)、[逐例 110 ROI](cohort_roi.tsv)、"
                  "[统计来源及 SHA](roi_summary_artifacts.json)、[本表文件清单](per_region_dice_artifacts.json)、"
                  "[六层真实脑图](../benchmark_results.md#真实-t1-脑图)。", ""])
    columns = [("cohort_all", "raw_native"), ("cohort_all", "stage_native"),
               ("cohort_new_subjects", "raw_native"), ("cohort_new_subjects", "stage_native")]
    for family in FAMILIES:
        ids = sorted(label for label, metadata in labels.items() if metadata["family"] == family)
        lines.extend(["## " + family, "", f"{len(ids)} 个分区。", "",
                      "| 标签 ID | 原分区名称 | 全部 10 例 raw-native | 全部 10 例 stage-native | 新 9 例 raw-native | 新 9 例 stage-native |",
                      "|---|---|---|---|---|---|"])
        for label in ids:
            name = labels[label]["name"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            values = [dice_cell(grid[(cohort, space, label)]) for cohort, space in columns]
            lines.append("| " + " | ".join([str(label), name, *values]) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def export(root, main_audit_path):
    root = root.resolve()
    cache = {}
    def read(path):
        path = path.resolve()
        if path not in cache:
            cache[path] = path.read_bytes()
        return cache[path]
    tsv_path = root / "analysis/cohort_roi_summary.tsv"
    receipt_path = root / "analysis/roi_summary_artifacts.json"
    receipt = read_json(read(receipt_path))
    if (receipt.get("state") != "completed" or receipt.get("final_outcome_ready") is not True
            or receipt.get("script", {}).get("sha256") != FLATTENER_SHA
            or receipt.get("rows") != 880 or receipt.get("labels_per_space") != 110
            or receipt.get("existing_inputs_unchanged") is not True):
        raise ValueError("Final authenticated ROI flattener result is not ready")
    for cohort, ids in COHORTS.items():
        if receipt["cohorts"][cohort] != {"case_ids": list(ids), "planned_subjects": len(ids)}:
            raise ValueError("ROI artifact cohort denominators differ")
    if set(receipt["spaces"]) != set(SPACES) or set(receipt["metrics"]) != set(METRICS):
        raise ValueError("ROI artifact scope differs")
    verify(tsv_path, read(tsv_path), receipt["outputs"]["cohort_roi_summary.tsv"])
    evidence_path = root / "benchmark_evidence.json"
    verify(evidence_path, read(evidence_path), receipt["f8_benchmark_evidence"])
    evidence = read_json(read(evidence_path))
    if (evidence.get("state") != "completed" or evidence.get("final_outcome_ready") is not True
            or evidence.get("script", {}).get("sha256") != F8_SHA
            or evidence["source_audit"]["source_manifest"]["sha256"] != BASE_SOURCE_SHA
            or receipt["source_manifest"]["sha256"] != BASE_SOURCE_SHA
            or receipt["cohort_manifest_sha256"] != evidence["manifest"]["sha256"]):
        raise ValueError("F8 publication/source identity differs")
    summary_path = root / "analysis/cohort_summary.json"
    verify(summary_path, read(summary_path), receipt["summary"])
    verify(summary_path, read(summary_path), evidence["curated_artifacts"]["analysis/cohort_summary.json"]["publication"])
    summary = read_json(read(summary_path))
    if summary.get("final_outcome_ready") is not True or summary.get("not_completed_attempts"):
        raise ValueError("Original cohort summary is not final")
    grid, labels = validate_grid(read(tsv_path))
    inputs = {"ROI summary TSV": identity(tsv_path, read(tsv_path)),
              "ROI flattener artifacts": identity(receipt_path, read(receipt_path)),
              "F8 benchmark evidence": identity(evidence_path, read(evidence_path)),
              "F8 cohort summary": identity(summary_path, read(summary_path))}
    main_reuse = None
    if main_audit_path is not None:
        main_audit_path = main_audit_path.resolve()
        audit = read_json(read(main_audit_path))
        main_reuse = validate_main_audit(audit, BASE_SOURCE_SHA, evidence["manifest"]["sha256"])
        inputs["completed main audit"] = identity(main_audit_path, read(main_audit_path))
        # The link names the standard curated path, so the linked file must have
        # the same bytes even when --main-audit reads a relocated text mirror.
        linked_audit = root / "latest_main_regression/audit/summary.json"
        verify(linked_audit, read(linked_audit), inputs["completed main audit"])
    output_path = root / "analysis/per_region_dice.md"
    artifact_path = root / "analysis/per_region_dice_artifacts.json"
    if output_path.exists() or artifact_path.exists():
        raise ValueError("Preserve existing region report; refusing overwrite")
    markdown_data = markdown(grid, labels, main_reuse).encode("utf-8")
    artifact = {"schema_version": 1, "state": "completed", "final_outcome_ready": True,
                "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "script": identity(Path(__file__)), "inputs": inputs,
                "rendered_ROI_rows": 110, "validated_aggregate_rows": 880,
                "family_rows": {family: sum(metadata["family"] == family for metadata in labels.values()) for family in FAMILIES},
                "label_ids": sorted(labels), "statistics_recomputed": False,
                "original_measured_commit": BASE_COMMIT, "main_raw_reuse": main_reuse,
                "scoring_spaces": {"raw_native": "Original published public T1 grid",
                                   "stage_native": "This subject's fresh official norm grid"},
                "rules": {"display": "Four decimals only; exact original numbers remain in authenticated TSV.",
                          "SD": "Between-subject sample SD ddof=1; NA below two defined subjects.",
                          "NA": "Both-empty and failed remain NA; single-sided absence zeros retained.",
                          "names": "Original TSV label ID/name, no inferred region-name translation.",
                          "runtime": "Metadata only, no image/software runtime/fit/GPU operation."},
                "output": identity(output_path, markdown_data), "existing_inputs_unchanged": True}
    artifact_data = (json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for path, original in cache.items():
        if path.read_bytes() != original:
            raise ValueError("Input changed during verification: " + str(path))
    for path, data in ((output_path, markdown_data), (artifact_path, artifact_data)):
        with path.open("xb") as handle:
            handle.write(data)
    return {"state": "completed", "ROI_rows": 110, "validated_aggregate_rows": 880,
            "output": identity(output_path, markdown_data), "artifacts": identity(artifact_path, artifact_data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Final F8 curated root after ROI flattener completed")
    parser.add_argument("--main-audit", type=Path, help="Optional completed main audit; default binds curated latest_main_regression/audit/summary.json when present")
    args = parser.parse_args()
    root = args.root.resolve()
    default_audit = root / "latest_main_regression/audit/summary.json"
    main_audit = args.main_audit or (default_audit if default_audit.exists() else None)
    print(json.dumps(export(root, main_audit), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
