#!/usr/bin/env python3
"""Publish verified final cohort text/figures without loading MRI or running fits.

Full audited text evidence may live outside the repository. --output can point
at the curated repository directory; --copy-evidence copies only an explicit
allowlist of JSON/TSV/PNG. No full analysis, API report, image, weight or source
snapshot is copied. Nonterminal analyses fail before any output is created.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct


IDS = tuple(f"sub-{i:02d}" for i in range(1, 11))
FAMILIES = {
    "brainstem": "脑干四分区", "thalamus": "双侧丘脑细核",
    "hippocampus_left": "左海马亚区", "hippocampus_right": "右海马亚区",
    "amygdala_left": "左杏仁核", "amygdala_right": "右杏仁核",
}
STRUCTURES = {"brainstem": "脑干", "thalamus": "丘脑",
              "hippo-amygdala-left": "左海马及杏仁核", "hippo-amygdala-right": "右海马及杏仁核"}
OFFICIAL_SUBREGIONS = ("official_brainstem", "official_thalamus", "official_hippo_amygdala")
TERMINAL_ANALYSIS = ("completed", "completed_with_failures_or_unavailable")
COHORTS = {"cohort_all": IDS, "cohort_new_subjects": IDS[1:]}
COHORT_TITLES = {"cohort_all": "全部 10 例", "cohort_new_subjects": "未参与开发的 9 例（排除 sub-01）"}


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_file(path, expected):
    """An explicit current path permits relocation; identity must still match."""
    actual = identity(path)
    if actual["sha256"] != expected.get("sha256") or (
            "bytes" in expected and actual["bytes"] != expected["bytes"]):
        raise ValueError(f"Artifact identity differs: {path}")
    return actual


def finite(value, *, positive=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid numeric measurement: {value!r}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid nonnegative/positive measurement: {value!r}")
    return float(value)


def distribution(values, planned):
    observed = [finite(value) for value in values if value is not None]
    if len(observed) > planned:
        raise ValueError("More observations than planned subjects")
    return {"planned_subjects": planned, "defined_subjects": len(observed),
            "missing_or_na_subjects": planned - len(observed),
            "mean": statistics.fmean(observed) if observed else None,
            "median": statistics.median(observed) if observed else None,
            "std_sample": statistics.stdev(observed) if len(observed) > 1 else None,
            "min": min(observed) if observed else None, "max": max(observed) if observed else None,
            "std_ddof": 1}


def check_distribution(actual, expected, missing_key="missing_or_na_subjects"):
    for key in ("planned_subjects", "defined_subjects", "mean", "median", "min", "max", "std_sample"):
        left, right = actual.get(key), expected.get(key)
        if left is None or right is None:
            equal = left is right
        elif isinstance(left, float) or isinstance(right, float):
            equal = math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
        else:
            equal = left == right
        if not equal:
            raise ValueError(f"Recomputed distribution differs at {key}: {left} vs {right}")
    if actual["missing_or_na_subjects"] != expected.get(missing_key):
        raise ValueError("Missing/planned denominators differ")


def unique_rows(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise ValueError(f"Duplicate observation: {key}")
        result[key] = row
    return result


def tsv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def check_tsv_keys(path, rows, keys):
    text = tsv_rows(path)
    actual = unique_rows(text, keys)
    expected = {tuple(str(row[key]) for key in keys) for row in rows}
    if set(actual) != expected or len(text) != len(rows):
        raise ValueError(f"TSV does not cover the same observations: {path}")


def assert_cases(records, field):
    indexed = unique_rows(records, (field,))
    if set(key[0] for key in indexed) != set(IDS):
        raise ValueError("Exactly the predeclared ten case IDs are required")
    for record in records:
        if record.get("development_seen") is not (record[field] == "sub-01"):
            raise ValueError("Development/unseen cohort membership differs")


def preparation_audit(analysis):
    """Count saved preparation case outcomes separately from current attempts."""
    summaries = analysis.get("run_preparation_summary", [])
    if not isinstance(summaries, list):
        raise ValueError("Preparation summary must be a list of archived records")
    records, artifact_hashes, failed_subjects = [], set(), set()
    for entry in summaries:
        source = entry["artifact"]
        if source["sha256"] in artifact_hashes:
            raise ValueError("Duplicate preparation artifact would double-count startup history")
        artifact_hashes.add(source["sha256"])
        outcomes = entry["case_outcomes"]
        unique_rows(outcomes, ("case_id",))
        if any(row["case_id"] not in IDS for row in outcomes):
            raise ValueError("Preparation history contains an undeclared subject")
        failed = [row for row in outcomes if row.get("state") == "failed"
                  or row.get("exit_code") not in (None, 0)]
        failed_ids = sorted({row["case_id"] for row in failed})
        failed_subjects.update(failed_ids)
        states = sorted({str(row.get("state")) for row in outcomes})
        codes = sorted({str(row.get("exit_code")) for row in outcomes})
        records.append({"artifact": source, "scope": entry.get("scope"), "case_outcomes": outcomes,
                        "case_outcome_records": len(outcomes), "failed_case_outcome_records": len(failed),
                        "failed_case_ids": failed_ids,
                        "launched_failed_records": sum(row.get("child_launched") is True for row in failed),
                        "blocked_case_outcome_records": sum(row.get("state") == "blocked" for row in outcomes),
                        "not_launched_records": sum(row.get("child_launched") is False for row in outcomes),
                        "missing_exit_code_records": sum(row.get("exit_code") is None for row in outcomes),
                        "state_counts": {state: sum(str(row.get("state")) == state for row in outcomes) for state in states},
                        "exit_code_counts": {code: sum(str(row.get("exit_code")) == code for row in outcomes) for code in codes}})
    current = {key: len(analysis.get(key, [])) for key in
               ("failed_attempts", "setup_dependency_blocked_attempts", "setup_or_preflight_failures")}
    return {"archived_preparation_records": len(records), "records": records,
            "archived_case_outcome_records": sum(entry["case_outcome_records"] for entry in records),
            "archived_failed_case_outcome_records": sum(entry["failed_case_outcome_records"] for entry in records),
            "archived_launched_failed_records": sum(entry["launched_failed_records"] for entry in records),
            "archived_failed_subjects": sorted(failed_subjects),
            "current_attempt_counts": current,
            "scope": "Archived environment/startup case-outcome records are separate from current merged-queue attempt categories and final per-subject fit outcomes. No summed algorithm-failure count."}


def preparation_text(audit):
    if audit["records"]:
        summary = (f"分析另保留 {audit['archived_preparation_records']} 份环境/启动准备归档："
                   f"共 {audit['archived_case_outcome_records']} 条病例记录，"
                   f"其中 {audit['archived_failed_case_outcome_records']} 条失败记录，"
                   f"涉及 {len(audit['archived_failed_subjects'])} 名预声明受试者；"
                   f"失败记录中 {audit['archived_launched_failed_records']} 条已启动子进程。"
                   "这些数量从归档的 case_outcomes 逐条计算，不是最终拟合失败人数。"
                   "启动错误、环境修复及重新运行过程见 [官方准备记录](official_protocol.md#启动失败及重试)。"
                   "归档的 state、exit_code、failure、日志路径和 SHA 随发布证据保留。")
        table = markdown_table(["准备归档（SHA 前12位）", "病例记录", "失败记录", "已启动的失败记录", "依赖阻塞记录", "失败受试者"],
            [[f"{Path(record['artifact']['path']).name} ({record['artifact']['sha256'][:12]})",
              record["case_outcome_records"], record["failed_case_outcome_records"], record["launched_failed_records"],
              record["blocked_case_outcome_records"], "、".join(record["failed_case_ids"]) or "无"] for record in audit["records"]])
        summary += "\n\n" + table
    else:
        summary = "最终分析未提供准备阶段归档摘要；不把缺少记录解释为零次历史启动失败。"
    current = audit["current_attempt_counts"]
    return (summary + "\n\n" +
            f"当前合并队列另列尝试类别：进程/输出验证失败 {current['failed_attempts']} 次，"
            f"上游依赖阻塞 {current['setup_dependency_blocked_attempts']} 次，"
            f"设置/预检查失败 {current['setup_or_preflight_failures']} 次。"
            "这一组数量不含独立归档的准备阶段记录；因上游官方预处理未就绪而 blocked 的 FNIT stage 属于依赖阻塞，"
            "不另累计为 FNIT 算法失败。修复后的最终结果使用本轮固定源码和预声明输入；"
            "完成状态、精度和耗时以实际最终进程及保存结果为准。")


def load_evidence(root, analysis_dir, steps_dir, figures_dir):
    status_path = analysis_dir / "analysis_status.json"
    status = read_json(status_path)
    if status.get("final_outcome_ready") is not True or status.get("state") not in TERMINAL_ANALYSIS:
        raise ValueError("Final cohort analysis is not ready; refusing publication")
    verified = {}
    for name in ("cohort_analysis.json", "cohort_summary.json", "cohort_roi.tsv", "cohort_family.tsv", "cohort_case.tsv"):
        verified[name] = verify_file(analysis_dir / name, status["artifacts"][name])
    analysis = read_json(analysis_dir / "cohort_analysis.json")
    summary = read_json(analysis_dir / "cohort_summary.json")
    manifest_path = root / "cohort_manifest.json"
    manifest_id = verify_file(manifest_path, {"sha256": analysis["manifest_sha256"]})
    manifest = read_json(manifest_path)
    if analysis.get("final_outcome_ready") is not True or analysis.get("not_completed_attempts"):
        raise ValueError("Unfinished cohort attempts remain")
    if not analysis.get("queue_sources") or any(queue.get("state") not in ("completed", "completed_with_failures", "failed")
                                                for queue in analysis["queue_sources"]):
        raise ValueError("A final queue is not terminal")
    if (analysis["planned_subjects"] != 10 or manifest["planned_subjects"] != 10
            or analysis["source_audit"]["source_manifest"]["sha256"] != manifest["source"]["manifest_sha256"]):
        raise ValueError("Planned cohort or frozen source differs")
    assert_cases(manifest["cases"], "id")
    assert_cases(analysis["cases"], "case_id")
    for field in ("manifest_sha256", "final_outcome_ready", "source_audit", "dataset", "snapshot", "dataset_doi"):
        if summary[field] != analysis[field]:
            raise ValueError(f"Summary and analysis differ: {field}")
    rows = analysis["roi_rows"]
    roi_index = unique_rows(rows, ("case_id", "mode", "resolution", "label"))
    labels = {row["label"] for row in rows}
    expected = {(case, mode, resolution, label) for case in IDS for mode in ("raw", "stage")
                for resolution in ("native", "hr") for label in labels}
    if len(labels) != 110 or len(rows) != 4400 or set(roi_index) != expected:
        raise ValueError("The complete 110 ROI x 10 subjects x 4 spaces grid is required, including failures")
    for row in rows:
        if (row["family"] not in FAMILIES or row["development_seen"] is not (row["case_id"] == "sub-01")
                or row["space"] != row["mode"] + "_" + row["resolution"]):
            raise ValueError("ROI metadata or cohort identity differs")
        score = finite(row.get("dice"))
        if score is not None and score > 1:
            raise ValueError("Dice outside [0,1]")
    groups = unique_rows(analysis["family_rows"], ("case_id", "mode", "resolution", "family"))
    for key, row in groups.items():
        if (key[0] not in IDS or key[1] not in ("raw", "stage") or key[2] not in ("native", "hr")
                or key[3] not in FAMILIES or row["measurement_status"] != "evaluated"):
            raise ValueError("Invalid evaluated family row")
        score = finite(row.get("reference_weighted_dice"))
        if score is not None and score > 1:
            raise ValueError("Family Dice outside [0,1]")
    check_tsv_keys(analysis_dir / "cohort_roi.tsv", rows, ("case_id", "mode", "resolution", "label"))
    check_tsv_keys(analysis_dir / "cohort_family.tsv", analysis["family_rows"], ("case_id", "mode", "resolution", "family"))
    check_tsv_keys(analysis_dir / "cohort_case.tsv", analysis["case_rows"], ("case_id", "space"))
    step_path = steps_dir / "cohort_steps.json"
    steps = read_json(step_path)
    if steps.get("final_outcome_ready") is not True or steps["planned_subjects"] != 10:
        raise ValueError("Actual final step extraction is unavailable")
    verify_file(analysis_dir / "cohort_analysis.json", steps["analysis"])
    verify_file(manifest_path, steps["manifest"])
    assert_cases(steps["cases"], "case_id")
    step_index = unique_rows(steps["rows"], ("case_id", "method", "mode", "structure", "step"))
    for key, row in step_index.items():
        if key[0] not in IDS or row["development_seen"] is not (key[0] == "sub-01"):
            raise ValueError("Step case identity differs")
        finite(row["seconds"])
    case_index = {case["case_id"]: case for case in analysis["cases"]}
    for case_id in IDS:
        case = case_index[case_id]
        for mode in ("raw", "stage"):
            actual = case.get("timings", {}).get("fnit", {}).get(mode, {})
            source = steps["sources"].get(f"fnit:{case_id}:{mode}")
            if actual and not source:
                raise ValueError("Audited FNIT timing has no actual step source")
            if source:
                if source["source_manifest"]["sha256"] != manifest["source"]["manifest_sha256"]:
                    raise ValueError("Step source was fitted by another frozen runtime")
                declared = case.get("fnit", {}).get(mode, {})
                for field in ("api_report", "report"):
                    if declared.get(field) and source[field]["sha256"] != declared[field]["sha256"]:
                        raise ValueError("Step extraction refers to another actual case report")
            for field, step in (("process_wall_seconds", "process_wall"), ("api_compute_seconds", "compute"),
                                ("output_save_seconds", "save"), ("api_total_seconds", "api_total")):
                if actual.get(field) is not None:
                    observed = step_index.get((case_id, "fnit", mode, "all", step), {})
                    if observed.get("measurement_status") != "completed" or observed.get("seconds") != actual[field]:
                        raise ValueError("Step FNIT wall/API timer differs from final audited case")
        official_source = steps["sources"].get(f"official:{case_id}")
        if official_source and case.get("official_report"):
            if official_source["official_report"]["sha256"] != case["official_report"]["sha256"]:
                raise ValueError("Official step extraction refers to another case report")
            if official_source["software"] != case["official_software"]:
                raise ValueError("Official step software identity differs from final audited case")
        for component, structure in (("reconall", "reconall"), ("official_brainstem", "brainstem"),
                                     ("official_thalamus", "thalamus"), ("official_hippo_amygdala", "hippo-amygdala")):
            actual = case.get("timings", {}).get("official", {}).get(component, {})
            if actual.get("process_wall_seconds") is not None:
                observed = step_index.get((case_id, "official", "raw", structure, "command_process_wall"), {})
                if observed.get("measurement_status") != "completed" or observed.get("seconds") != actual["process_wall_seconds"]:
                    raise ValueError("Step official command timer differs from final audited case")
    if read_json(steps_dir / "cohort_steps_summary.json") != steps["summary"]:
        raise ValueError("Extracted step summaries differ")
    check_tsv_keys(steps_dir / "cohort_steps.tsv", steps["rows"], ("case_id", "method", "mode", "structure", "step"))
    plot_path = figures_dir / "plot_manifest.json"
    plot = read_json(plot_path)
    if plot.get("status") != "completed" or plot.get("whole_head_images_exported") is not False:
        raise ValueError("Completed derived brain figures are required")
    verify_file(analysis_dir / "cohort_analysis.json", plot["analysis"])
    verify_file(status_path, plot["analysis_status"])
    verify_file(manifest_path, plot["manifest"])
    if plot["source_audit"] != analysis["source_audit"]:
        raise ValueError("Figures were made from another source audit")
    figure_files = {}
    selected_cases = {case for case in plot["selection"]["roles"].values() if case is not None}
    figure_keys = set(unique_rows(plot["figures"], ("case_id", "group")))
    if figure_keys != {(case, group) for case in selected_cases for group in ("brainstem", "thalamus", "hippocampus", "amygdala")}:
        raise ValueError("Selected real subjects do not have all four brain-figure outcomes")
    for role, field in (("median", "median_subject"), ("worst", "worst_subject")):
        expected_case = analysis["figure_selection"].get(field)
        if plot["selection"]["roles"][role] != (expected_case["case_id"] if expected_case else None):
            raise ValueError("Brain figure selection differs from final analyzer")
    for entry in [plot["heatmap"], *plot["figures"]]:
        if entry.get("status") == "both_empty_family_no_image":
            continue
        if entry is not plot["heatmap"] and entry.get("status") != "completed":
            raise ValueError("A selected brain figure is not completed")
        expected_image = entry["artifact"]
        name = Path(expected_image["path"]).name
        if Path(name).suffix.lower() != ".png" or name in figure_files:
            raise ValueError("Invalid/duplicate derived PNG")
        record = verify_file(figures_dir / name, expected_image)
        with (figures_dir / name).open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError("Derived figure is not an actual PNG")
        width, height = struct.unpack(">II", header[16:24])
        if width == 0 or height == 0:
            raise ValueError("Empty PNG")
        figure_files[name] = {**record, "width": width, "height": height}
    if plot["heatmap"]["rows_in_fixed_manifest_order"] != list(IDS):
        raise ValueError("Brain heatmap excludes a planned subject")
    if plot["heatmap"]["columns"] != list(FAMILIES):
        raise ValueError("Brain heatmap uses different six families")
    expected_heatmap = [[groups.get((case, "raw", "native", family), {}).get("reference_weighted_dice")
                         for family in FAMILIES] for case in IDS]
    if plot["heatmap"]["values"] != expected_heatmap:
        raise ValueError("Brain heatmap values differ from actual family outcomes")
    return {"status": status, "analysis": analysis, "summary": summary, "manifest": manifest,
            "steps": steps, "step_index": step_index, "groups": groups, "plot": plot,
            "figure_files": figure_files, "analysis_artifacts": verified, "manifest_identity": manifest_id}


def aggregate(evidence):
    analysis, groups = evidence["analysis"], evidence["groups"]
    cases = {case["case_id"]: case for case in analysis["cases"]}
    result = {}
    paired = []
    for case_id in IDS:
        timing = cases[case_id].get("timings", {})
        official = timing.get("official", {})
        subregion_values = [official.get(name, {}).get("process_wall_seconds") for name in OFFICIAL_SUBREGIONS]
        for name in OFFICIAL_SUBREGIONS:
            component = official.get(name, {})
            if component.get("process_wall_seconds") is not None and component.get("exit_code") != 0:
                raise ValueError("Audited successful official time has a nonzero/unknown exit code")
        for mode in ("raw", "stage"):
            run = timing.get("fnit", {}).get(mode, {})
            fnit = finite(run.get("process_wall_seconds"), positive=True)
            if run and run.get("exit_code") != 0:
                raise ValueError("Audited successful FNIT time has a nonzero exit code")
            if mode == "raw":
                reference = finite(timing.get("official_end_to_end_seconds"), positive=True)
                reference_scope = timing.get("official_end_to_end_scope")
            else:
                reference = (sum(finite(value, positive=True) for value in subregion_values)
                             if all(value is not None for value in subregion_values) else None)
                reference_scope = "Sum of this subject's three actual official subregion command process walls; recon-all excluded."
            paired.append({"case_id": case_id, "development_seen": case_id == "sub-01", "mode": mode,
                           "official_seconds": reference, "fnit_process_wall_seconds": fnit,
                           "official_over_fnit": reference / fnit if reference is not None and fnit is not None else None,
                           "measurement_status": "paired" if reference is not None and fnit is not None else "unavailable_pair",
                           "official_scope": reference_scope, "fnit_scope": run.get("process_wall_scope"),
                           "official_report": cases[case_id].get("official_report"), "fnit_report": cases[case_id].get("fnit", {}).get(mode, {}).get("report")})
    for name, ids in COHORTS.items():
        planned = len(ids)
        declared = evidence["summary"][name]
        if declared["case_ids"] != list(ids) or declared["planned_subjects"] != planned:
            raise ValueError("Declared cohort denominator differs")
        family_stats = []
        declared_family = unique_rows(declared["family"], ("space", "family"))
        for mode in ("raw", "stage"):
            for family in FAMILIES:
                values = [groups.get((case, mode, "native", family), {}).get("reference_weighted_dice") for case in ids]
                stats = distribution(values, planned)
                check_distribution(stats, declared_family[(mode + "_native", family)]["measurements"]["reference_weighted_dice"])
                family_stats.append({"mode": mode, "family": family, **stats})
        runtimes = []
        for component, field in [("fnit_raw", key) for key in ("process_wall_seconds", "api_compute_seconds", "output_save_seconds", "api_total_seconds")] + [
                ("fnit_stage", key) for key in ("process_wall_seconds", "api_compute_seconds", "output_save_seconds", "api_total_seconds")] + [
                (component, "process_wall_seconds") for component in ("reconall", *OFFICIAL_SUBREGIONS)] + [
                ("official_end_to_end_seconds", "official_end_to_end_seconds")]:
            values = []
            for case_id in ids:
                timing = cases[case_id].get("timings", {})
                if component.startswith("fnit_"):
                    value = timing.get("fnit", {}).get(component.removeprefix("fnit_"), {}).get(field)
                elif component == "official_end_to_end_seconds":
                    value = timing.get(field)
                else:
                    value = timing.get("official", {}).get(component, {}).get(field)
                values.append(value)
            stats = distribution(values, planned)
            declared_runtime = next(entry for entry in declared["runtime"] if entry["component"] == component)
            check_distribution(stats, declared_runtime["measurements"][field])
            runtimes.append({"component": component, "field": field, **stats})
        step_stats = []
        for declared_step in evidence["steps"]["summary"][name]["rows"]:
            key = tuple(declared_step[field] for field in ("method", "mode", "structure", "step"))
            values = []
            for case_id in ids:
                entry = evidence["step_index"].get((case_id, *key), {})
                values.append(entry.get("seconds") if entry.get("measurement_status") == "completed" else None)
            stats = distribution(values, planned)
            check_distribution(stats, declared_step, "missing_or_failed_subjects")
            step_stats.append({**dict(zip(("method", "mode", "structure", "step"), key)), **stats,
                               "scope": declared_step["scope"], "measurement_kinds": declared_step["measurement_kinds"]})
        result[name] = {"case_ids": list(ids), "planned_subjects": planned, "family_native": family_stats,
                        "runtime": runtimes, "steps": step_stats,
                        "paired_speed": [{"mode": mode, **distribution([row["official_over_fnit"] for row in paired
                                          if row["case_id"] in ids and row["mode"] == mode], planned)} for mode in ("raw", "stage")],
                        "case_status": declared["case_status"], "completed_subjects": declared["completed_subjects"]}
    return result, paired


def fmt(value, digits):
    return "NA" if value is None else f"{value:.{digits}f}"


def cell(stats, digits=1):
    return (f"{fmt(stats['mean'], digits)} ± {fmt(stats['std_sample'], digits)} "
            f"[{fmt(stats['min'], digits)}, {fmt(stats['max'], digits)}]；"
            f"{stats['defined_subjects']}/{stats['planned_subjects']}，缺 {stats['missing_or_na_subjects']}")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in rows])


def family_tables(stats):
    parts = []
    for cohort in COHORTS:
        index = unique_rows(stats[cohort]["family_native"], ("mode", "family"))
        parts.extend([f"### {COHORT_TITLES[cohort]}", "",
                      markdown_table(["区域", "raw native Dice", "stage native Dice"],
                          [[label, cell(index[("raw", family)], 4), cell(index[("stage", family)], 4)]
                           for family, label in FAMILIES.items()]), ""])
    return "\n".join(parts)


def speed_table(stats):
    return markdown_table(["受试者集合", "raw 完整流程：官方/FNIT", "stage 细分割：官方/FNIT"],
        [[COHORT_TITLES[cohort], *[cell(next(row for row in stats[cohort]["paired_speed"] if row["mode"] == mode), 2)
          for mode in ("raw", "stage")]] for cohort in COHORTS])


def step_cell(stats, method, mode, structure, step):
    selected = next((entry for entry in stats["steps"] if tuple(entry[key] for key in ("method", "mode", "structure", "step")) ==
                     (method, mode, structure, step)), None)
    return cell(selected or distribution([], stats["planned_subjects"]))


def docs(stats, evidence, artifacts):
    manifest, analysis = evidence["manifest"], evidence["analysis"]
    dataset = manifest["dataset"]
    doi = manifest["dataset_doi"].removeprefix("doi:")
    header = (f"公开数据 [{dataset} {manifest['snapshot']}](https://doi.org/{doi}) 的 `ses-test` T1："
              "按参与者编号预先固定 sub-01–sub-10。全部 10 例与排除开发受试者 sub-01 的 9 例分别报告。"
              "公开快照已有去脸处理，本次未追加去脸。")
    preparation = preparation_audit(analysis)
    preparation_section = preparation_text(preparation)
    convention = ("表格为受试者均值 ± 样本标准差（ddof=1）[最小值, 最大值]；随后为有效数/计划数和缺失数。"
                  "各受试者区域 Dice 先按该区域官方细分区体素数加权，再对受试者等权平均。"
                  "双方均空的细分区为 NA，单方缺失为 0；失败和缺测保留计划分母 10 或 9。"
                  "标准差描述受试者间差异，范围仅取该指标有定义的受试者。")
    design = ("`raw` 从本例原始公开 T1 调用 FNIT 完整四结构流程；官方从同一 T1 重新执行 "
              "`recon-all -all -openmp 4` 及脑干、丘脑、双侧海马/杏仁核分割。"
              "`stage` 从本例新官方 `norm/aseg/wmparc` 运行 FNIT，时间排除官方预处理。"
              "两者均为 `structures=\"all\", optimization=\"fast\", threads=4`。"
              "raw native 为原始 T1 网格，stage native 为本例官方 norm 网格；官方细标签只用于评分。")
    hardware = read_json(artifacts["hardware_identity.json"]["input_path"]) if "hardware_identity.json" in artifacts else {}
    hardware_text = "硬件信息未随文本证据提供。"
    if hardware:
        gpu_names = list(dict.fromkeys(line.split(",")[2].strip() for line in hardware.get("gpu_csv", "").splitlines() if len(line.split(",")) >= 3))
        hardware_text = (f"本轮使用共享 {', '.join(gpu_names) or 'GPU 型号未记录'}；"
                         f"FNIT raw/stage 分别固定物理 GPU {hardware.get('fnit_raw_gpu', 'NA')}/{hardware.get('fnit_stage_gpu', 'NA')}，"
                         f"每进程 {hardware.get('cpu_threads_per_process', 'NA')} 线程；"
                         f"官方 CPU 同时运行 {hardware.get('official_jobs', 'NA')} 例。"
                         "配置与运行共享负载见硬件、协议及分析证据。进程 wall 保留导入、CUDA 初始化、保存和观察回调；"
                         "源码/输入检查、输入就绪等待和 GPU 预算等待单独记录。")
    official_builds = sorted({case["official_software"]["build_stamp"] for case in analysis["cases"]
                             if case.get("official_software")})
    fnit_versions = sorted({(record.get("configuration", {}).get("torch_version"),
                            record.get("configuration", {}).get("cuda_version"))
                           for case in analysis["cases"] for record in case.get("fnit", {}).values()},
                           key=lambda value: tuple(str(part) for part in value))
    versions_text = ("逐例实际软件记录：官方 build `" + "`、`".join(official_builds) + "`。" if official_builds else
                     "官方实际软件版本未随可评分结果提供。")
    if fnit_versions:
        versions_text += " FNIT 的 PyTorch/CUDA 记录：" + "、".join(f"`{torch or 'NA'}/{cuda or 'NA'}`" for torch, cuda in fnit_versions) + "。"
    speed_rule = ("速度倍数逐例计算后汇总：raw = 本例官方实际完整流程 wall / 本例 FNIT raw 进程 wall；"
                  "stage = 本例官方三个细分割命令的实际进程 wall 之和 / 本例 FNIT stage 进程 wall。"
                  "stage 分子排除 recon-all，保留双侧海马/杏仁核同一命令。"
                  "仅两项都实际可用的同一受试者组成配对，缺失对仍占计划分母；"
                  "倍数范围是逐例比值的最小值和最大值。共享 CPU/H100 上的实测流程比值不能推为独占硬件的算法加速。")
    completion = markdown_table(["受试者集合", "两模式及官方均完成", "其余受试者"],
        [[COHORT_TITLES[name], f"{stats[name]['completed_subjects']}/{len(ids)}",
          "、".join(f"{case['case_id']} ({case['status']})" for case in stats[name]["case_status"] if case["status"] != "completed") or "无"]
         for name, ids in COHORTS.items()])
    link_text = ("[110 分区完整 TSV](analysis/cohort_roi.tsv)（110 × 10 × raw/stage × native/hr，共 4,400 行，保留失败）；"
                 "[六区域逐例 TSV](analysis/cohort_family.tsv)、[整例 TSV](analysis/cohort_case.tsv)、"
                 "[分析摘要](analysis/cohort_summary.json)、[逐例配对耗时](paired_runtime.tsv)、"
                 "[分步骤 TSV](analysis/steps/cohort_steps.tsv)、[分步骤汇总](analysis/steps/cohort_steps_summary.tsv)、"
                 "[recon-all 显式 FSTIME 来源](analysis/steps/cohort_reconall_fstime.tsv)、"
                 "[发布证据和 SHA](benchmark_evidence.json)、[脑图清单](brain_figures/plot_manifest.json)。")
    readme = "\n\n".join(["# 10 例公开 T1：四结构分割 benchmark", header, design, completion,
        "## 原网格精度", convention, family_tables(stats), "## 逐例配对运行时间比值", speed_rule, speed_table(stats),
        "## 完整结果与使用方法", "[详细结果、实际分步耗时与真实脑图](benchmark_results.md)。" + link_text,
        "函数输入、输出、参数、Python/CLI 示例及参考文献见 [segment_4_subregions](../../../docs/subregions/README.md)。"
        "本轮设计见 [benchmark 协议](benchmark_protocol.md)、[官方运行协议](official_protocol.md)及 [脑图协议](plot_protocol.md)。",
        "## 准备阶段记录", preparation_section,
        "## 本次记录", f"冻结源码基于 `{manifest['source']['base_commit']}`，清单 SHA-256 `{manifest['source']['manifest_sha256']}`。"
        "本页仅从已终态的本轮文本分析与实际 PNG 生成；历史开发案例仍列于功能文档的版本记录。",
        "复现发布：先完成分析、步骤提取和脑图，再运行 `python publish_cohort_summary.py --root /absolute/path/text_evidence "
        "--output /absolute/path/curated_validation --copy-evidence`。", ""]) 
    results = ["# 10 例公开 T1：精度、耗时与脑图", header, design, completion,
               "## 六区域原网格精度", convention, family_tables(stats), "## 完整流程及 API 计时", versions_text, hardware_text,
               "单位为秒。不同项各自保留有效数；API 计算、保存和 API 总计与独立观察的进程 wall 分列。"
               "队列/输入等待不计入 FNIT 进程 wall。官方完整流程取本例实际保存的完整 wall；不使用历史时间或 recon-all 估计。"]
    runtime_labels = {
        ("fnit_raw", "process_wall_seconds"): "FNIT raw 完整进程 wall",
        ("fnit_stage", "process_wall_seconds"): "FNIT stage 进程 wall（排除官方预处理）",
        ("official_end_to_end_seconds", "official_end_to_end_seconds"): "官方原始 T1 完整流程 wall",
        ("reconall", "process_wall_seconds"): "官方 recon-all 进程 wall",
        ("official_brainstem", "process_wall_seconds"): "官方脑干命令 wall",
        ("official_thalamus", "process_wall_seconds"): "官方丘脑命令 wall",
        ("official_hippo_amygdala", "process_wall_seconds"): "官方双侧海马/杏仁核命令 wall",
    }
    for mode in ("raw", "stage"):
        for field, label in (("api_compute_seconds", "API 计算"), ("output_save_seconds", "输出保存"), ("api_total_seconds", "API 总计")):
            runtime_labels[("fnit_" + mode, field)] = f"FNIT {mode} {label}"
    runtime_index = {name: unique_rows(stats[name]["runtime"], ("component", "field")) for name in COHORTS}
    results.append(markdown_table(["计时范围", *COHORT_TITLES.values()],
        [[label, *[cell(runtime_index[name][key]) for name in COHORTS]] for key, label in runtime_labels.items()]))
    results.extend(["## 同一受试者的配对速度比值", speed_rule, speed_table(stats),
                    "逐例分子、分母、范围和缺测状态均见 [paired_runtime.tsv](paired_runtime.tsv)。",
                    "## 实际分步骤计时", "单位为秒；表格继续采用均值 ± 样本标准差 [最小值, 最大值] 和有效数/计划数、缺失数。"
                    "官方合成/强度列为日志显式整数计时，包含相应阶段准备。FNIT 列使用已保存的独立计时："
                    "脑干强度列为 intensity_fit；丘脑及海马强度列为多层准备与拟合。工作图准备另列，不能混为同一个计时范围。"
                    "子阶段或 subset 计时已包含在其父步骤，不能与父步骤再次相加。"])
    for cohort in COHORTS:
        current = stats[cohort]
        results.append(f"### {COHORT_TITLES[cohort]}")
        results.append(markdown_table(["步骤", "FNIT raw", "FNIT stage"],
            [["共享预处理", *[step_cell(current, "fnit", mode, "shared", "shared_preprocessing") for mode in ("raw", "stage")]]]))
        results.append(markdown_table(["结构", "官方合成准备/拟合", "官方强度准备/拟合", "FNIT raw 合成", "raw 工作图准备", "raw 强度", "FNIT stage 合成", "stage 工作图准备", "stage 强度"],
            [[label, step_cell(current, "official", "raw", structure, "synthetic_prepare_and_fit"),
              step_cell(current, "official", "raw", structure, "intensity_prepare_and_fit"),
              *[step_cell(current, "fnit", mode, structure, step) for mode in ("raw", "stage")
                for step in ("synthetic_fit_and_preparation", "image_preparation", "intensity_fit" if structure == "brainstem" else "intensity_multiscale_prepare_fit")]]
             for structure, label in STRUCTURES.items()]))
        results.append(markdown_table(["结构", "FNIT raw recipe 总计", "FNIT stage recipe 总计"],
            [[label, *[step_cell(current, "fnit", mode, structure, "recipe_total") for mode in ("raw", "stage")]] for structure, label in STRUCTURES.items()]))
    results.extend(["完整 [步骤明细](analysis/steps/cohort_steps.tsv) 和 [步骤汇总](analysis/steps/cohort_steps_summary.tsv) 包含可用的图谱对齐、"
                    "多尺度准备、solver 拟合、拟合后处理及逐层计时。没有独立计时的对齐、后处理保留 NA；"
                    "未分配 recipe 开销不改名为任何未测步骤。中间阶段没有共同保存的可评分标签，阶段 Dice 未测，"
                    "不由 objective 或初始化 mask Dice 代替。最终四结构/六区域精度见本页和完整 ROI 表。",
                    "## 真实 T1 脑图", "展示按 raw native 全 110 分区官方体素加权 Dice 预先定义的中位例、最低例，以及开发用 sub-01。"
                    "每幅分区脑图为 6 个 axial slice；官方、FNIT 和差异三行使用相同裁剪、切面和灰度范围。"
                    "显示重采样只用于画图，Dice 按原始评分网格计算。"])
    heatmap = Path(evidence["plot"]["heatmap"]["artifact"]["path"]).name
    results.append(f"![10 例原网格六区域 Dice 热图](brain_figures/{heatmap})")
    results.append(markdown_table(["展示角色", "受试者"], [[{"continuity": "开发用连续性例", "median": "中位例", "worst": "最低例"}.get(role, role), case or "NA（无可评分受试者）"]
                   for role, case in evidence["plot"]["selection"]["roles"].items()]))
    for figure in evidence["plot"]["figures"]:
        title = f"{figure['case_id']}：" + {"brainstem": "脑干", "thalamus": "丘脑", "hippocampus": "海马", "amygdala": "杏仁核"}[figure["group"]]
        if figure["status"] == "completed":
            results.extend(["### " + title, f"![{title}，官方、FNIT 与标签差异](brain_figures/{Path(figure['artifact']['path']).name})"])
        else:
            results.append(title + "：双方均空，无可展示分区；不生成假脑图。")
    results.extend(["## 准备阶段记录", preparation_section, "## 完整证据与本次记录", link_text,
                    f"冻结源码：`{manifest['source']['base_commit']}`；清单 SHA-256：`{manifest['source']['manifest_sha256']}`。",
                    "本次发布只汇总固定十例最终结果，未按精度筛选受试者。完整分析及逐例扩展报告保留在服务器或外部文本归档，"
                    "其路径、大小与 SHA 记入发布证据；仓库只保留汇总、完整 ROI TSV、实际步骤计时及衍生脑图。",
                    "数据选择与许可见 [数据记录](data_selection.md)；算法原实现、论文及使用方法见 [功能文档](../../../docs/subregions/README.md#reference)。", ""])
    return readme, "\n\n".join(results)


def write_tsv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Server task root or relocated complete text-evidence mirror")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--steps-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, help="Curated publication directory; default --root")
    parser.add_argument("--copy-evidence", action="store_true", help="Copy only checked curated JSON/TSV/PNG, never MRI/full analysis/source snapshots")
    args = parser.parse_args()
    root = args.root.resolve()
    analysis_dir = (args.analysis_dir or root / "analysis").resolve()
    steps_dir = (args.steps_dir or analysis_dir / "steps").resolve()
    figures_dir = (args.figures_dir or root / "brain_figures").resolve()
    output = (args.output or root).resolve()
    evidence = load_evidence(root, analysis_dir, steps_dir, figures_dir)
    stats, paired = aggregate(evidence)
    curated = {}
    for name in ("analysis_status.json", "cohort_summary.json", "cohort_roi.tsv", "cohort_family.tsv", "cohort_case.tsv"):
        curated["analysis/" + name] = analysis_dir / name
    for name in ("cohort_steps.json", "cohort_steps.tsv", "cohort_steps_summary.json", "cohort_steps_summary.tsv", "cohort_reconall_fstime.tsv"):
        curated["analysis/steps/" + name] = steps_dir / name
    curated["brain_figures/plot_manifest.json"] = figures_dir / "plot_manifest.json"
    for name in evidence["figure_files"]:
        curated["brain_figures/" + name] = figures_dir / name
    for name in ("hardware_identity.json", "benchmark_protocol.md", "official_protocol.md", "plot_protocol.md", "data_selection.md"):
        if (root / name).exists():
            curated[name] = root / name
        elif name != "hardware_identity.json":
            raise ValueError(f"Publication protocol/data link requires {root / name}")
    artifacts = {name: {"input_path": str(path), **identity(path)} for name, path in curated.items()}
    generated = ("README.md", "benchmark_results.md", "benchmark_evidence.json", "paired_runtime.tsv")
    if any((output / name).exists() for name in generated):
        raise ValueError("Preserve prior publication; use a fresh output directory")
    for name, path in curated.items():
        destination = output / name
        if destination.exists():
            verify_file(destination, artifacts[name])
        elif not args.copy_evidence:
            raise ValueError(f"Curated link is absent: {destination}; copy evidence first or pass --copy-evidence")
    readme, results = docs(stats, evidence, artifacts)
    # All identity, terminal, numeric and destination checks precede mutations.
    output.mkdir(parents=True, exist_ok=True)
    for name, path in curated.items():
        destination = output / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        verify_file(destination, artifacts[name])
    write_tsv(output / "paired_runtime.tsv", paired)
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "benchmark_results.md").write_text(results, encoding="utf-8")
    publication = {"schema_version": 1, "final_outcome_ready": True, "state": "completed",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "script": identity(Path(__file__)),
        "manifest": evidence["manifest_identity"], "analysis": evidence["analysis_artifacts"]["cohort_analysis.json"],
        "analysis_status": identity(analysis_dir / "analysis_status.json"), "source_audit": evidence["analysis"]["source_audit"],
        "input_paths_may_be_relocated": True, "images_read_or_copied": "Only derived PNG; no MRI, weights, atlas, license or source snapshot",
        "statistics": stats, "paired_runtime": paired, "figure_selection": evidence["plot"]["selection"],
        "figures": evidence["figure_files"],
        "run_preparation_summary": evidence["analysis"].get("run_preparation_summary", []),
        "preparation_audit": preparation_audit(evidence["analysis"]),
        "current_attempt_counts": preparation_audit(evidence["analysis"])["current_attempt_counts"],
        "preserved_preparation_history_artifacts": [entry["artifact"] for entry in evidence["analysis"].get("preserved_preparation_history", [])],
        "curated_artifacts": {name: {"input": artifacts[name], "publication": identity(output / name)} for name in curated},
        "generated_artifacts": {name: identity(output / name) for name in ("README.md", "benchmark_results.md", "paired_runtime.tsv")},
        "rules": {"failure_denominators": "10 all / 9 excluding sub-01; failed and NA cases retained",
                  "family_dice": "Official fine-label voxel weighted within each subject, arithmetic mean across subjects",
                  "std": "Sample standard deviation, ddof=1; undefined for fewer than two available subjects",
                  "speed": "Per-subject paired official/FNIT wall ratios, then subject distribution; no ratios of unrelated extrema",
                  "range": "Min/max of available subject measurements (or same-subject ratios)",
                  "stage": "Conditional on fresh official norm/aseg/wmparc, official preprocessing excluded",
                  "step_timers": "Recorded timers only; nested/subset and recipe totals not added together; missing timers and stage Dice not invented"}}
    (output / "benchmark_evidence.json").write_text(json.dumps(publication, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": "completed", "output": str(output), "planned_subjects": 10,
                      "new_subjects": 9, "published_png": len(evidence["figure_files"]),
                      "benchmark_evidence_sha256": sha256(output / "benchmark_evidence.json")}))


if __name__ == "__main__":
    main()
