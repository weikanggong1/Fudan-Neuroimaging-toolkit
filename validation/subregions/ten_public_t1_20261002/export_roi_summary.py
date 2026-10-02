#!/usr/bin/env python3
"""Flatten final, F8-published ROI distributions without recalculating metrics.

Only curated JSON/TSV evidence is read. No image, fit, or software runtime is
loaded. Original publication files are preserved; existing outputs are refused.
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
METRICS = ("dice", "jaccard", "hard_volume_difference_mm3",
           "hard_volume_relative_difference", "soft_volume_difference_mm3",
           "soft_volume_relative_difference")
STAT_FIELDS = ("defined_subjects", "missing_or_na_subjects", "mean", "std_sample",
               "min", "max", "median", "std_ddof")
HARD_STATUS = {"evaluated", "both_empty_hard_label", "not_evaluated",
               "official_present_fnit_hard_label_absent",
               "official_hard_label_absent_fnit_present"}
FAMILIES = {"brainstem", "thalamus", "hippocampus_left", "hippocampus_right",
            "amygdala_left", "amygdala_right"}
SOURCES = {"brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right"}
PUBLISHER_F8_SHA = "f8c433c1451c3c45c3a1afee43738bd874788199434cdf24bc9803f028b1b011"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(path: Path, data: bytes | None = None) -> dict:
    data = path.read_bytes() if data is None else data
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha256(data)}


def read_json(data: bytes, name: str) -> dict:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON scalar in {name}: {value}")
    value = json.loads(data, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {name}")
    return value


def verify_publication_file(path: Path, data: bytes, declared: dict) -> None:
    # Paths in the original F8 identity may refer to the server or full mirror.
    # Exact bytes, not path equality, establish the relocated curated identity.
    if (declared.get("sha256") != sha256(data)
            or declared.get("bytes") != len(data)):
        raise ValueError(f"F8 publication identity differs: {path}")


def integer(value, label: str, lower=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise ValueError(f"Expected integer >= {lower}: {label}")
    return value


def numeric(value, label: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"Expected finite scalar: {label}")


def validate_distribution(stats: dict, planned: int, metric: str) -> None:
    if (integer(stats["planned_subjects"], metric + ".planned", 1) != planned
            or integer(stats["std_ddof"], metric + ".std_ddof", 1) != 1):
        raise ValueError(f"Distribution denominator or sample SD differs: {metric}")
    defined = integer(stats["defined_subjects"], metric + ".defined")
    missing = integer(stats["missing_or_na_subjects"], metric + ".missing")
    if defined + missing != planned:
        raise ValueError(f"Distribution counts differ: {metric}")
    for field in ("mean", "min", "max", "median"):
        if defined:
            numeric(stats[field], metric + "." + field)
        elif stats[field] is not None:
            raise ValueError(f"An undefined distribution must remain NA: {metric}.{field}")
    sd = stats["std_sample"]
    if defined >= 2:
        numeric(sd, metric + ".std_sample")
        if sd < 0:
            raise ValueError(f"Negative sample SD: {metric}")
    elif sd is not None:
        raise ValueError(f"Sample SD requires at least two subjects: {metric}")
    if "std" in stats and stats["std"] != sd:
        raise ValueError(f"Sample SD aliases differ: {metric}")
    if defined:
        # A valid arithmetic mean can lie one ULP beyond identical min/max.
        tolerance = 1e-12 * max(1, abs(stats["min"]), abs(stats["max"]))
        if not stats["min"] - tolerance <= stats["mean"] <= stats["max"] + tolerance:
            raise ValueError(f"Mean outside declared range: {metric}")
        if not stats["min"] <= stats["median"] <= stats["max"]:
            raise ValueError(f"Median outside declared range: {metric}")
        if metric in ("dice", "jaccard") and not 0 <= stats["min"] <= stats["max"] <= 1:
            raise ValueError(f"Overlap metric outside [0, 1]: {metric}")


def roi_metadata(data: bytes) -> tuple[dict, dict]:
    records = list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))
    if len(records) != 4400:
        raise ValueError("The F8-published per-subject table must retain 4,400 rows")
    metadata, keys, statuses = {}, set(), {}
    for record in records:
        label = int(record["label"])
        if label <= 0 or str(label) != record["label"]:
            raise ValueError("Noncanonical ROI ID")
        case, space = record["case_id"], record["space"]
        if case not in IDS or space not in SPACES:
            raise ValueError("Unexpected subject or scoring space")
        if (record["mode"], record["resolution"]) != SPACES[space]:
            raise ValueError("Space/mode/resolution differ")
        if record["hard_status"] not in HARD_STATUS:
            raise ValueError("Unknown hard-label status")
        key = (case, space, label)
        if key in keys:
            raise ValueError("Duplicate per-subject ROI row")
        keys.add(key)
        meta = {field: record[field] for field in ("name", "source", "family")}
        if (not meta["name"] or meta["family"] not in FAMILIES
                or meta["source"] not in SOURCES):
            raise ValueError("Invalid ROI metadata")
        if label in metadata and metadata[label] != meta:
            raise ValueError("ROI metadata differs across subjects or spaces")
        metadata[label] = meta
        statuses[key] = record["hard_status"]
    expected = {(case, space, label) for case in IDS for space in SPACES for label in metadata}
    if len(metadata) != 110 or keys != expected:
        raise ValueError("Incomplete canonical 110 ROI x 10 subject x 4 space grid")
    return metadata, statuses


def flatten(summary: dict, metadata: dict, statuses: dict) -> list[dict]:
    flattened = []
    for cohort, ids in COHORTS.items():
        section = summary[cohort]
        planned = len(ids)
        if (section["cohort"] != cohort or section["case_ids"] != list(ids)
                or section["planned_subjects"] != planned or len(section["roi"]) != 440):
            raise ValueError(f"Unexpected cohort grid or denominator: {cohort}")
        keys = set()
        for item in section["roi"]:
            space, label = item["space"], integer(item["label"], "label", 1)
            if space not in SPACES or label not in metadata or (space, label) in keys:
                raise ValueError("Unknown or duplicate aggregate ROI")
            keys.add((space, label))
            meta = metadata[label]
            if item["name"] != meta["name"] or item["family"] != meta["family"]:
                raise ValueError("Aggregate and per-subject ROI metadata differ")
            completed = integer(item["completed_subjects"], "completed_subjects")
            if item["planned_subjects"] != planned or completed > planned:
                raise ValueError("Aggregate ROI denominator differs")
            counts = item["hard_status_counts"]
            if set(counts) - HARD_STATUS:
                raise ValueError("Unknown aggregate hard-label status")
            for status, count in counts.items():
                integer(count, "hard_status_counts." + status)
            observed_counts = {}
            for case in ids:
                status = statuses[(case, space, label)]
                observed_counts[status] = observed_counts.get(status, 0) + 1
            if counts != observed_counts or sum(counts.values()) != planned:
                raise ValueError("Aggregate and per-subject hard-label counts differ")
            if completed != planned - counts.get("not_evaluated", 0):
                raise ValueError("Completed ROI count differs from recorded hard status")
            mode, resolution = SPACES[space]
            dice = item["measurements"]["dice"]
            row = {"cohort": cohort, "space": space, "mode": mode, "resolution": resolution,
                   "label": label, **meta, "planned_subjects": planned,
                   "completed_subjects": completed, "defined_subjects": dice["defined_subjects"],
                   "missing_or_na": dice["missing_or_na_subjects"],
                   "hard_status_counts": json.dumps(counts, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":"))}
            if set(item["measurements"]) != set(METRICS):
                raise ValueError("Unexpected metric schema")
            for metric in METRICS:
                stats = item["measurements"][metric]
                validate_distribution(stats, planned, metric)
                for field in STAT_FIELDS:
                    row[metric + "_" + field] = stats[field]
            expected_defined = planned - counts.get("not_evaluated", 0) - counts.get("both_empty_hard_label", 0)
            if dice["defined_subjects"] != expected_defined:
                raise ValueError("Dice defined count differs from original hard-label status")
            flattened.append(row)
        if keys != {(space, label) for space in SPACES for label in metadata}:
            raise ValueError("Incomplete aggregate ROI-space grid")
    if len(flattened) != 880:
        raise ValueError("Exactly 880 aggregate rows are required")
    return sorted(flattened, key=lambda row: (list(COHORTS).index(row["cohort"]),
                                             row["space"], row["label"]))


def markdown_number(value) -> str:
    return "NA" if value is None else f"{value:.4f}"


def lowest_dice_markdown(rows: list[dict]) -> str:
    lines = ["# Raw 原网格平均 Dice 最低的 20 个分区", "",
             "按各分区已有受试者平均 Dice 升序排列，同分按标签 ID。全部计划病例均保留，"
             "不据此筛选整体结果。单方标签缺失的 0 分参与已有统计，双方均空或失败仍为 NA；"
             "各行给出有效数和缺失数，样本 SD 为受试者间差异（ddof=1）。", "",
             "完整 880 行：[ROI 汇总表](cohort_roi_summary.tsv)。", ""]
    titles = {"cohort_all": "全部 10 例", "cohort_new_subjects": "新 9 例（排除开发用 sub-01）"}
    for cohort in COHORTS:
        current = [row for row in rows if row["cohort"] == cohort and row["space"] == "raw_native"]
        defined = [row for row in current if row["dice_mean"] is not None]
        selected = sorted(defined, key=lambda row: (row["dice_mean"], row["label"]))[:20]
        lines.extend(["## " + titles[cohort], "",
                      f"110 个分区中，平均 Dice 有定义 {len(defined)} 个、全 NA {110 - len(defined)} 个；列出最低 {len(selected)} 个。", "",
                      "| 标签 | 分区 | 家族 | Dice 均值 ± 样本 SD | 范围 | 有效/计划 | NA |",
                      "|---|---|---|---|---|---|---|"])
        for row in selected:
            name = row["name"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['label']} | {name} | {row['family']} | "
                         f"{markdown_number(row['dice_mean'])} ± {markdown_number(row['dice_std_sample'])} | "
                         f"[{markdown_number(row['dice_min'])}, {markdown_number(row['dice_max'])}] | "
                         f"{row['defined_subjects']}/{row['planned_subjects']} | {row['missing_or_na']} |")
        lines.append("")
    return "\n".join(lines)


def export(root: Path) -> dict:
    root = root.resolve()
    evidence_path = root / "benchmark_evidence.json"
    summary_path = root / "analysis/cohort_summary.json"
    per_case_path = root / "analysis/cohort_roi.tsv"
    paths = [evidence_path, summary_path, per_case_path]
    original = {path: path.read_bytes() for path in paths}
    evidence = read_json(original[evidence_path], str(evidence_path))
    summary = read_json(original[summary_path], str(summary_path))
    if evidence.get("state") != "completed" or evidence.get("final_outcome_ready") is not True:
        raise ValueError("F8 final publication is not complete; refusing ROI export")
    if evidence.get("script", {}).get("sha256") != PUBLISHER_F8_SHA:
        raise ValueError("Expected the frozen F8 publisher identity")
    if (summary.get("final_outcome_ready") is not True or summary.get("not_completed_attempts")
            or summary.get("planned_subjects") != 10 or not summary.get("queue_sources")
            or any(queue.get("state") not in ("completed", "completed_with_failures", "failed")
                   for queue in summary["queue_sources"])):
        raise ValueError("Final cohort analysis is not terminal; refusing ROI export")
    for name, path in (("analysis/cohort_summary.json", summary_path), ("analysis/cohort_roi.tsv", per_case_path)):
        declaration = evidence["curated_artifacts"][name]
        verify_publication_file(path, original[path], declaration["publication"])
        if declaration["input"]["sha256"] != declaration["publication"]["sha256"]:
            raise ValueError("F8 input and publication identities differ")
    if (summary["manifest_sha256"] != evidence["manifest"]["sha256"]
            or summary["source_audit"] != evidence["source_audit"]):
        raise ValueError("Summary and F8 source/manifest differ")
    metadata, statuses = roi_metadata(original[per_case_path])
    rows = flatten(summary, metadata, statuses)
    output = root / "analysis"
    tsv_path = output / "cohort_roi_summary.tsv"
    markdown_path = output / "lowest_raw_native_roi.md"
    artifacts_path = output / "roi_summary_artifacts.json"
    for path in (tsv_path, markdown_path, artifacts_path):
        if path.exists():
            raise ValueError(f"Preserve existing derived output; refusing overwrite: {path}")
    # All numerical, final-state and SHA checks precede writing any output.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)  # None becomes an empty field; zeros retain literal 0.
    tsv_data = buffer.getvalue().encode("utf-8")
    markdown_data = lowest_dice_markdown(rows).encode("utf-8")
    script_path = Path(__file__).resolve()
    script_data = script_path.read_bytes()
    receipt = {"schema_version": 1, "state": "completed", "final_outcome_ready": True,
               "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "script": identity(script_path, script_data),
               "summary": identity(summary_path, original[summary_path]),
               "f8_benchmark_evidence": identity(evidence_path, original[evidence_path]),
               "per_subject_roi": identity(per_case_path, original[per_case_path]),
               "source_manifest": summary["source_audit"]["source_manifest"],
               "cohort_manifest_sha256": summary["manifest_sha256"],
               "rows": len(rows), "labels_per_space": 110,
               "cohorts": {cohort: {"case_ids": list(ids), "planned_subjects": len(ids)}
                           for cohort, ids in COHORTS.items()},
               "spaces": list(SPACES), "metrics": list(METRICS),
               "outputs": {"cohort_roi_summary.tsv": identity(tsv_path, tsv_data),
                           "lowest_raw_native_roi.md": identity(markdown_path, markdown_data)},
               "rules": {"statistics": "Exact existing distributions copied; no metrics recomputed.",
                         "counts": "Dice effective/NA counts retained; each metric also has its original counts.",
                         "na": "Undefined values stay empty TSV fields; one-sided absence zeros stay zero.",
                         "std": "Subject sample SD, ddof=1, NA below two defined subjects.",
                         "source": "Fixed recipe metadata from the F8-authenticated per-subject TSV.",
                         "ranking": "Raw/native mean Dice ascending, label ID tie-break; no outcome filtering.",
                         "runtime": "Only published text evidence read; no images, external software or fitting."},
               "existing_inputs_unchanged": True}
    receipt_data = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for path, data in original.items():
        if path.read_bytes() != data:
            raise ValueError(f"Input changed during verification: {path}")
    # Exclusive creation protects prior outputs even if another process races us.
    for path, data in ((tsv_path, tsv_data), (markdown_path, markdown_data), (artifacts_path, receipt_data)):
        with path.open("xb") as handle:
            handle.write(data)
    return {"state": "completed", "rows": len(rows),
            "outputs": receipt["outputs"], "artifacts": identity(artifacts_path, receipt_data)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Final F8 curated directory containing benchmark_evidence.json and analysis/")
    args = parser.parse_args()
    print(json.dumps(export(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
