#!/usr/bin/env python3
"""Pair audited current-main raw times with the final F8 fresh-official cohort.

This independent companion reads only metadata/text evidence. It requires both
final publications and exact raw-output equivalence, preserves the F8 files,
and refuses to overwrite its own outputs. No fit, image, or GPU is loaded.
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
import statistics

IDS = tuple(f"sub-{index:02d}" for index in range(1, 11))
COHORTS = {"cohort_all": IDS, "cohort_new_subjects": IDS[1:]}
BASE_COMMIT = "ac692bb4f7868a24ea4bd67180162e81726de9b4"
MAIN_COMMIT = "f436de588647a0de80735e4a98d53df5d88e502d"
BASE_SOURCE_SHA = "d8c8f7aecdae521751f30fdd09bea6acbffd2c0a50c9fb70e1dc97ad74717b07"
MAIN_SOURCE_SHA = "4fb86d811b92c9db6be002f2c58a46b8f07f624b42671d034bc996294ce0b602"
F8_SHA = "f8c433c1451c3c45c3a1afee43738bd874788199434cdf24bc9803f028b1b011"
STRUCTURES = ("brainstem", "thalamus", "hippo-amygdala-left", "hippo-amygdala-right")
MAPS = ("labels", *("highres/" + name for name in STRUCTURES))
RECIPE_PATHS = {"brainstem": "brainstem/timing_seconds/total",
                **{name: name + "/seconds" for name in STRUCTURES[1:]}}
TIMERS = ("api_compute_seconds", "api_total_seconds", "process_wall_seconds",
          "output_save_seconds", "context_observer_seconds")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path, data=None):
    data = path.read_bytes() if data is None else data
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": digest(data)}


def load_json(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, parse_constant=invalid)


def verify(path, data, expected):
    if expected.get("sha256") != digest(data) or expected.get("bytes") != len(data):
        raise ValueError(f"Declared bytes/SHA differ: {path}")


def same_identity(left, right):
    return all(left[key] == right[key] for key in ("bytes", "sha256"))


def finite(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid actual measurement: {value!r}")
    return value


def distribution(values, planned):
    present = [finite(value) for value in values if value is not None]
    if len(present) > planned:
        raise ValueError("More observations than planned subjects")
    return {"planned_subjects": planned, "defined_subjects": len(present),
            "missing_or_na_subjects": planned - len(present),
            "mean": statistics.fmean(present) if present else None,
            "median": statistics.median(present) if present else None,
            "std_sample": statistics.stdev(present) if len(present) > 1 else None,
            "std_ddof": 1, "min": min(present) if present else None,
            "max": max(present) if present else None}


def tsv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))


def bool_cell(value):
    if value not in ("True", "False", "true", "false"):
        raise ValueError("Invalid TSV Boolean")
    return value.lower() == "true"


def same_number(actual, expected):
    if actual != expected:
        raise ValueError(f"Recorded measurements differ: {actual!r} != {expected!r}")


def validate_pair_table(data, evidence):
    records = tsv_rows(data)
    expected = {(row["case_id"], row["mode"]): row for row in evidence["paired_runtime"]}
    if (len(records) != 20 or len(expected) != 20
            or set(expected) != {(case, mode) for case in IDS for mode in ("raw", "stage")}):
        raise ValueError("F8 must retain all twenty same-subject mode pairs")
    result = {}
    for row in records:
        key = (row["case_id"], row["mode"])
        if key not in expected or key in result:
            raise ValueError("Unknown or duplicate F8 runtime pair")
        for name, value in expected[key].items():
            cell = row[name]
            converted = (None if value is None and cell == "" else
                         bool_cell(cell) if isinstance(value, bool) else
                         float(cell) if isinstance(value, (float, int)) else
                         load_json(cell) if isinstance(value, (dict, list)) else cell)
            if converted != value:
                raise ValueError(f"F8 pair TSV and evidence differ: {key}/{name}")
        actual = expected[key]
        if (actual["measurement_status"] != "paired"
                or actual["development_seen"] is not (key[0] == "sub-01")):
            raise ValueError("Each required F8 pair must have actual successful times")
        official = finite(actual["official_seconds"], positive=True)
        fnit = finite(actual["fnit_process_wall_seconds"], positive=True)
        same_number(actual["official_over_fnit"], official / fnit)
        result[key] = actual
    return result


def validate_equivalence(audit, volumes):
    if (audit.get("state") != "zero_voxel_equivalence_passed"
            or audit.get("zero_voxel_equivalence_passed") is not True
            or audit.get("final_main_regression_outcomes_ready") is not True
            or audit.get("final_queue_state") != "completed"
            or audit.get("final_source_asset_metadata_error") is not None
            or audit.get("baseline_commit") != BASE_COMMIT or audit.get("main_commit") != MAIN_COMMIT):
        raise ValueError("The complete pinned main regression has not passed exact equivalence")
    expected = {"planned_subjects": 10, "planned_new_subjects": 9, "planned_map_pairs": 50,
                "planned_volume_pairs": 1100, "planned_context_pairs": 160}
    if any(audit[key] != value for key, value in expected.items()):
        raise ValueError("Regression planned scope differs")
    counts = audit["summary_counts"]
    if any(counts[key] != value for key, value in {
            "audited_subjects": 10, "exact_subjects": 10, "fully_compared_maps": 50,
            "fully_compared_volume_entries": 1100, "fully_compared_context_entries": 160,
            "different_maps": 0, "different_volume_entries": 0}.items()):
        raise ValueError("Regression aggregate counts are incomplete or nonzero")
    cases = {case["case_id"]: case for case in audit["cases"]}
    if len(audit["cases"]) != 10 or set(cases) != set(IDS):
        raise ValueError("Regression must retain exactly ten predetermined subjects")
    for case in cases.values():
        if (case["state"] != "zero_voxel_equivalence_passed" or case["error"] is not None
                or case["development_seen"] is not (case["case_id"] == "sub-01")):
            raise ValueError("Subject exact-equivalence status differs")
        for key, value in {"compared_maps": 5, "different_maps": 0, "different_voxels": 0,
                           "sum_defined_same_index_voxel_differences": 0,
                           "maps_without_comparable_voxel_count": 0, "compared_volume_entries": 110,
                           "different_volume_entries": 0, "compared_context_entries": 16,
                           "different_context_entries": 0}.items():
            if case["counts"][key] != value:
                raise ValueError("Subject regression scope or differences differ")
        if len(case["preprocessing_choices"]) != 2 or not all(
                choice["exact_without_recorded_timers"] is True for choice in case["preprocessing_choices"]):
            raise ValueError("Preprocessing/model choices differ")
    maps = audit["map_pairs"]
    if (len(maps) != 50 or {(r["case_id"], r["map"]) for r in maps}
            != {(case, name) for case in IDS for name in MAPS}):
        raise ValueError("Incomplete fifty-map grid")
    for row in maps:
        if (row["measurement_status"] != "compared" or row["different_voxels"] != 0
                or not all(row[key] is True for key in ("shape_exact", "affine_exact", "dtype_exact", "all_values_exact", "exact"))):
            raise ValueError("Label-map values or geometry differ")
        for name in ("array_sha256", "shape", "array_dtype", "header_dtype", "affine"):
            if row["baseline_" + name] != row["main_" + name]:
                raise ValueError("Actual map identity differs")
    contexts = audit["context_pairs"]
    if len(contexts) != 160 or len({(r["case_id"], r["phase"], r["array"]) for r in contexts}) != 160:
        raise ValueError("Incomplete context comparison grid")
    for case in IDS:
        if sum(row["case_id"] == case for row in contexts) != 16:
            raise ValueError("Context count differs by subject")
    if any(row["measurement_status"] != "compared" or row["exact"] is not True for row in contexts):
        raise ValueError("Preprocessing observation differs")
    records = tsv_rows(volumes)
    labels = {int(row["label"]) for row in records}
    if (len(records) != 1100 or len(labels) != 110
            or {(row["case_id"], int(row["label"])) for row in records}
            != {(case, label) for case in IDS for label in labels}):
        raise ValueError("Incomplete 1,100-volume comparison grid")
    for row in records:
        if (row["measurement_status"] != "compared" or not bool_cell(row["exact"])
                or not bool_cell(row["entire_dictionary_exact"])
                or row["baseline_entry_sha256"] != row["main_entry_sha256"]
                or float(row["soft_delta_mm3"]) != 0 or float(row["hard_delta_mm3"]) != 0
                or row["baseline_soft_mm3"] != row["main_soft_mm3"]
                or row["baseline_hard_mm3"] != row["main_hard_mm3"]):
            raise ValueError("Audited hard/soft volume dictionary differs")
    return cases


def validate_sources(audit, regression, original_source, main_source, evidence):
    if original_source["base_commit"] != BASE_COMMIT or main_source["base_commit"] != MAIN_COMMIT:
        raise ValueError("Actual source manifest commits differ")
    old_gems = {row["path"]: row for row in original_source["files"] if row["path"].startswith("src/fnit/gems/")}
    new_gems = {row["path"]: row for row in main_source["files"] if row["path"].startswith("src/fnit/gems/")}
    if len(old_gems) != 24 or old_gems != new_gems:
        raise ValueError("All twenty-four GEMS runtime/lookup files must be identical")
    reference = regression["regression_against"]
    if (reference["cohort_manifest_sha256"] != audit["manifests"]["baseline"]["sha256"]
            or reference["cohort_manifest_sha256"] != evidence["manifest"]["sha256"]
            or reference["frozen_source_manifest_sha256"] != BASE_SOURCE_SHA
            or not same_identity(audit["manifests"]["baseline"], evidence["manifest"])
            or not same_identity(audit["source_audits"]["baseline"]["source_manifest"], evidence["source_audit"]["source_manifest"])):
        raise ValueError("Regression and F8 do not use the same baseline cohort/source")
    if (regression["source"]["base_commit"] != MAIN_COMMIT
            or regression["source"]["manifest_sha256"] != MAIN_SOURCE_SHA
            or regression.get("reference_used_for_raw_fnit_fitting") is not False
            or regression["planned_subjects"] != 10
            or [case["id"] for case in regression["cases"]] != list(IDS)):
        raise ValueError("Regression cohort/source/raw independence differs")
    for case in regression["cases"]:
        audited = next(row for row in audit["cases"] if row["case_id"] == case["id"])
        expected_input = {case["raw_t1"]["path"]: case["raw_t1"]["sha256"]}
        if (case["development_seen"] is not (case["id"] == "sub-01")
                or case["raw_t1"] != audited["input"]
                or audited["baseline"]["input_sha256"] != audited["main"]["input_sha256"]
                or audited["main"]["input_sha256"] != expected_input):
            raise ValueError("Raw input/development identity differs")
    return {"matched_files": 24, "paths": sorted(old_gems), "source_files_sha_verified": True}


def official_commands(step_data, paired):
    rows = tsv_rows(step_data)
    result = {}
    for row in rows:
        if (row["method"], row["mode"], row["step"]) != ("official", "raw", "command_process_wall"):
            continue
        key = (row["case_id"], row["structure"])
        if key in result or row["measurement_status"] != "completed":
            raise ValueError("Official command timer is duplicate or unavailable")
        result[key] = finite(float(row["seconds"]), positive=True)
    expected = {(case, structure) for case in IDS for structure in ("reconall", "brainstem", "thalamus", "hippo-amygdala")}
    if set(result) != expected:
        raise ValueError("All forty actual fresh-official command timers are required")
    for case in IDS:
        sub = sum(result[(case, structure)] for structure in ("brainstem", "thalamus", "hippo-amygdala"))
        same_number(sub, paired[(case, "stage")]["official_seconds"])
        if result[(case, "reconall")] + sub > paired[(case, "raw")]["official_seconds"] + 1e-9:
            raise ValueError("Fresh official full pipeline wall is below its sequential component sum")
    return result


def aggregate(cases, paired, commands):
    rows = []
    for case_id in IDS:
        case = cases[case_id]
        old_pair = paired[(case_id, "raw")]
        same_number(old_pair["fnit_process_wall_seconds"], case["baseline"]["timings"]["process_wall_seconds"])
        if not same_identity(old_pair["fnit_report"], case["baseline"]["driver_report"]):
            raise ValueError("F8 baseline raw report differs from the exact-regression baseline")
        for version, sha in (("baseline", BASE_SOURCE_SHA), ("main", MAIN_SOURCE_SHA)):
            run = case[version]
            if run["source_manifest_sha256"] != sha or run["timings"]["exit_code"] != 0:
                raise ValueError("Successful timing/source identity differs")
            for timer in TIMERS:
                finite(run["timings"][timer], positive=timer in ("api_compute_seconds", "api_total_seconds", "process_wall_seconds"))
            for structure, path in RECIPE_PATHS.items():
                finite(run["recipe_timings"][path], positive=True)
        current = case["main"]
        wall = current["timings"]["process_wall_seconds"]
        official = old_pair["official_seconds"]
        rows.append({"case_id": case_id, "development_seen": case_id == "sub-01", "mode": "raw",
                     "fnit_source_commit": MAIN_COMMIT, "fnit_source_manifest_sha256": MAIN_SOURCE_SHA,
                     "fnit_run_generation": "current_main_independent_raw_regression",
                     "official_seconds": official, "fnit_process_wall_seconds": wall,
                     "official_over_fnit": official / wall, "measurement_status": "paired",
                     "baseline_raw_process_wall_seconds": old_pair["fnit_process_wall_seconds"],
                     "main_over_baseline_process_wall": wall / old_pair["fnit_process_wall_seconds"],
                     **{key: current["timings"][key] for key in TIMERS if key != "process_wall_seconds"},
                     "own_sampled_peak_mib": current["gpu"]["own_sampled_peak_mib"],
                     "official_scope": old_pair["official_scope"], "fnit_scope": current["timings"]["process_wall_scope"],
                     "official_report": old_pair["official_report"], "fnit_report": current["driver_report"]})
        stage = paired[(case_id, "stage")]
        if not same_identity(stage["official_report"], old_pair["official_report"]):
            raise ValueError("Raw and stage pairing must share the same fresh official report")
        rows.append({**stage, "fnit_source_commit": BASE_COMMIT,
                     "fnit_source_manifest_sha256": BASE_SOURCE_SHA,
                     "fnit_run_generation": "frozen_stage_actual_run_not_current_main_rerun"})
    statistics_by_cohort = {}
    for cohort, ids in COHORTS.items():
        selected = [row for row in rows if row["case_id"] in ids]
        current_cases = [cases[case_id] for case_id in ids]
        stats = {"case_ids": list(ids), "planned_subjects": len(ids),
                 "paired_official_over_fnit": {mode: distribution([row["official_over_fnit"] for row in selected if row["mode"] == mode], len(ids)) for mode in ("raw", "stage")},
                 "current_main_raw": {key: distribution([case["main"]["timings"][key] for case in current_cases], len(ids)) for key in TIMERS},
                 "current_main_raw_gpu_peak_mib": distribution([finite(case["main"]["gpu"]["own_sampled_peak_mib"]) for case in current_cases], len(ids)),
                 "main_over_baseline_raw_process_wall": distribution([row["main_over_baseline_process_wall"] for row in selected if row["mode"] == "raw"], len(ids)),
                 "official_fresh_complete_wall": distribution([paired[(case, "raw")]["official_seconds"] for case in ids], len(ids)),
                 "official_commands": {structure: distribution([commands[(case, structure)] for case in ids], len(ids)) for structure in ("reconall", "brainstem", "thalamus", "hippo-amygdala")},
                 "current_main_raw_recipe_total": {structure: distribution([case["main"]["recipe_timings"][path] for case in current_cases], len(ids)) for structure, path in RECIPE_PATHS.items()}}
        statistics_by_cohort[cohort] = stats
    return rows, statistics_by_cohort


def cell(stats):
    def number(value):
        return "NA" if value is None else f"{value:.3f}"
    return (f"{number(stats['mean'])} ± {number(stats['std_sample'])}; "
            f"中位 {number(stats['median'])}; [{number(stats['min'])}, {number(stats['max'])}]; "
            f"{stats['defined_subjects']}/{stats['planned_subjects']}，NA {stats['missing_or_na_subjects']}")


def markdown(stats, inputs, gems):
    lines = ["# 当前 main 与本轮官方完整流程的配对对照", "",
             f"当前 raw 实测源码 `{MAIN_COMMIT}`；固定同一公开 T1 的十例再次独立执行 `structures=all, optimization=fast`。"
             "逐例采用本轮同病例 fresh 官方 `recon-all -all` 加三项细分割的实际整例 wall，并先计算官方/FNIT 比值再对病例汇总。", "",
             "stage 列仍为原冻结 ac692bb 的实际运行，从该病例本轮 fresh norm/aseg/wmparc 开始；"
             "对照为三项官方细分割 command wall 之和，不包括 recon-all，不称为 f436de5 新跑。"
             "这项条件测试的官方 norm 输入不会进入 raw-native 生产拟合。", "",
             "## 精度及脑图的继承依据", "",
             f"十例 50 张标签图的体素、shape、affine 和 dtype 完全相同；1,100 条硬/软体积字典和 160 条上下文记录相同，"
             f"拟合配置与模型选择相同。两版 {gems['matched_files']} 个 GEMS 文件逐字节一致。"
             "因此原 F8 raw Dice 和由这些相同标签图生成的脑图可复用，精度未另作近似或重新评分。", "",
             "[本轮精度与真实脑图](../benchmark_results.md)、[完整 110 ROI 逐例表](../analysis/cohort_roi.tsv)、"
             "[880 行 ROI 汇总](../analysis/cohort_roi_summary.tsv)、[脑图清单](../brain_figures/plot_manifest.json)。", "",
             "## 实际耗时与同病例速度", "",
             "单位：秒（显存为 MiB）；均值 ± 样本 SD（ddof=1）、中位数、最小–最大值、有效/计划及 NA。"
             "SD 描述病例间差异。raw 的 process wall 包含观察器和进程开销，输入及 GPU 预算等待计时另列；"
             "API compute/save/total 是各自实际计时，不相互代替。", ""]
    titles = {"cohort_all": "全部 10 例", "cohort_new_subjects": "新 9 例（排除开发用 sub-01）"}
    for cohort in COHORTS:
        s = stats[cohort]
        lines.extend(["### " + titles[cohort], "", "| 指标 | 实测统计 |", "|---|---|"])
        for name, value in [("当前 main raw process wall", s["current_main_raw"]["process_wall_seconds"]),
                            ("当前 main raw API compute", s["current_main_raw"]["api_compute_seconds"]),
                            ("当前 main raw API total", s["current_main_raw"]["api_total_seconds"]),
                            ("当前 main raw save", s["current_main_raw"]["output_save_seconds"]),
                            ("当前 main raw 自身 PID 采样峰值显存 MiB", s["current_main_raw_gpu_peak_mib"]),
                            ("官方 fresh 完整流程 wall", s["official_fresh_complete_wall"]),
                            ("官方完整流程 / 当前 main raw（同病例）", s["paired_official_over_fnit"]["raw"]),
                            ("官方三项细分割 / 原冻结 stage（同病例）", s["paired_official_over_fnit"]["stage"]),
                            ("当前 main / 原冻结 raw process wall（同病例）", s["main_over_baseline_raw_process_wall"])]:
            lines.append("| " + name + " | " + cell(value) + " |")
        lines.extend(["", "| 实际步骤 | 耗时统计 |", "|---|---|"])
        for name, value in s["official_commands"].items():
            lines.append(f"| 官方 {name} command wall | {cell(value)} |")
        for name, value in s["current_main_raw_recipe_total"].items():
            lines.append(f"| 当前 main raw {name} recipe 总计 | {cell(value)} |")
        lines.append("")
    lines.extend(["recipe 总计和内部子步骤存在包含关系，不重复相加。官方海马/杏仁核双侧只计一个 command wall。"
                  "[原冻结流程及官方实际步骤](../analysis/steps/cohort_steps_summary.tsv)；"
                  "当前 main raw 的全部实际 recipe timer 路径与值保存在本页 [机器可读对照](official_comparison.json)。", "",
                  "main 与原冻结 raw 的速度变化是在共享 GPU 上分时测得；源码、负载和时间点同时变化，"
                  "这项观测不能将变化量归因于某个 buffer 优化。自身 PID 显存采样和整卡其他进程占用分开记录。", "",
                  "[20 行逐例配对表](official_paired_runtime.tsv)中，十行 raw 为当前 main，十行 stage 明确保留 ac692bb 的版本和实际运行来源。"
                  "本次 companion 不改 F8 的文件、数字或 SHA；[原 F8 发布证据](../benchmark_evidence.json)与"
                  "[main 审计](audit/summary.json)由新 JSON 的输入清单绑定。", "",
                  "## 文件身份", "", "| 输入 | SHA-256 |", "|---|---|"])
    for name, record in inputs.items():
        lines.append(f"| {name} | `{record['sha256']}` |")
    return "\n".join(lines) + "\n"


def publish(root, regression_dir, regression_manifest, baseline_source_manifest):
    root, regression_dir = root.resolve(), regression_dir.resolve()
    cache = {}
    def read(path):
        path = path.resolve()
        if path not in cache:
            cache[path] = path.read_bytes()
        return cache[path]
    evidence_path = root / "benchmark_evidence.json"
    evidence = load_json(read(evidence_path))
    if (evidence.get("state") != "completed" or evidence.get("final_outcome_ready") is not True
            or evidence.get("script", {}).get("sha256") != F8_SHA):
        raise ValueError("Final frozen F8 publication is not ready")
    paired_path = root / "paired_runtime.tsv"
    verify(paired_path, read(paired_path), evidence["generated_artifacts"]["paired_runtime.tsv"])
    def curated(name):
        path = root / name
        declaration = evidence["curated_artifacts"][name]
        verify(path, read(path), declaration["publication"])
        if not same_identity(declaration["input"], declaration["publication"]):
            raise ValueError("F8 input/curated identities differ")
        return path, load_json(read(path)) if path.suffix == ".json" else read(path)
    cohort_summary_path, original_summary = curated("analysis/cohort_summary.json")
    steps_path, steps = curated("analysis/steps/cohort_steps_summary.json")
    step_rows_path, step_rows_data = curated("analysis/steps/cohort_steps.tsv")
    if (original_summary.get("final_outcome_ready") is not True
            or original_summary.get("not_completed_attempts")
            or original_summary["manifest_sha256"] != evidence["manifest"]["sha256"]
            or original_summary["source_audit"] != evidence["source_audit"]):
        raise ValueError("F8 summary is not the same final cohort")
    paired = validate_pair_table(read(paired_path), evidence)
    audit_path = regression_dir / "audit/summary.json"
    volume_path = regression_dir / "audit/volume_pairs.tsv"
    receipt_path = regression_dir / "copy_audit_receipt.json"
    receipt = load_json(read(receipt_path))
    receipt_by_path = {row["curated_file"]: row for row in receipt["files"]}
    for path in (audit_path, volume_path, regression_manifest):
        relative = path.resolve().relative_to(regression_dir).as_posix()
        declaration = receipt_by_path[relative]
        if declaration["byte_identical"] is not True:
            raise ValueError("Main curated copy is not byte-identical")
        verify(path, read(path), declaration)
    audit = load_json(read(audit_path))
    cases = validate_equivalence(audit, read(volume_path))
    regression = load_json(read(regression_manifest))
    verify(regression_manifest, read(regression_manifest), audit["manifests"]["main"])
    old_source = load_json(read(baseline_source_manifest))
    new_source_path = regression_dir / "source_manifest.json"
    new_source = load_json(read(new_source_path))
    verify(baseline_source_manifest, read(baseline_source_manifest), audit["source_audits"]["baseline"]["source_manifest"])
    verify(new_source_path, read(new_source_path), audit["source_audits"]["main"]["source_manifest"])
    if (digest(read(baseline_source_manifest)) != BASE_SOURCE_SHA
            or digest(read(new_source_path)) != MAIN_SOURCE_SHA):
        raise ValueError("Actual prescribed source manifest bytes differ")
    gems = validate_sources(audit, regression, old_source, new_source, evidence)
    commands = official_commands(step_rows_data, paired)
    rows, stats = aggregate(cases, paired, commands)
    for cohort, ids in COHORTS.items():
        if steps[cohort]["case_ids"] != list(ids) or steps[cohort]["planned_subjects"] != len(ids):
            raise ValueError("Official step-summary denominators differ")
        for structure in ("reconall", "brainstem", "thalamus", "hippo-amygdala"):
            declared = [row for row in steps[cohort]["rows"] if (row["method"], row["mode"], row["structure"], row["step"])
                        == ("official", "raw", structure, "command_process_wall")]
            actual = stats[cohort]["official_commands"][structure]
            if len(declared) != 1:
                raise ValueError("Unique official command step summary required")
            for key in ("planned_subjects", "defined_subjects", "mean", "median", "std_sample", "min", "max"):
                same_number(actual[key], declared[0][key])
            same_number(actual["missing_or_na_subjects"], declared[0]["missing_or_failed_subjects"])
    inputs = {name: identity(path, read(path)) for name, path in {
        "F8 benchmark_evidence": evidence_path, "F8 paired_runtime": paired_path,
        "F8 cohort_summary": cohort_summary_path, "F8 steps_summary": steps_path,
        "F8 steps_per_case": step_rows_path, "main regression audit": audit_path,
        "main volume comparisons": volume_path, "main cohort_manifest": regression_manifest,
        "main copy receipt": receipt_path, "original source_manifest": baseline_source_manifest,
        "main source_manifest": new_source_path}.items()}
    tsv_path = regression_dir / "official_paired_runtime.tsv"
    json_path = regression_dir / "official_comparison.json"
    md_path = regression_dir / "official_comparison.md"
    for path in (tsv_path, json_path, md_path):
        if path.exists():
            raise ValueError(f"Preserve existing comparison; refusing overwrite: {path}")
    buffer = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                     for key, value in row.items()} for row in rows)
    tsv_data = buffer.getvalue().encode("utf-8")
    md_data = markdown(stats, inputs, gems).encode("utf-8")
    report = {"schema_version": 1, "state": "completed", "final_outcome_ready": True,
              "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script": identity(Path(__file__)), "inputs": inputs,
              "current_raw_commit": MAIN_COMMIT, "frozen_stage_commit": BASE_COMMIT,
              "current_raw_source_manifest_sha256": MAIN_SOURCE_SHA,
              "frozen_stage_source_manifest_sha256": BASE_SOURCE_SHA,
              "exact_equivalence_counts": audit["summary_counts"], "gems_identity": gems,
              "statistics": stats, "paired_rows": rows,
              "current_main_cases": {case: {key: cases[case]["main"][key] for key in ("timings", "recipe_timings", "gpu")}
                                     for case in IDS},
              "official_step_summary": {cohort: [row for row in steps[cohort]["rows"] if row["method"] == "official"] for cohort in COHORTS},
              "frozen_stage_step_summary": {cohort: [row for row in steps[cohort]["rows"] if (row["method"], row["mode"]) == ("fnit", "stage")] for cohort in COHORTS},
              "accuracy_and_figure_reuse": {"basis": "Fifty raw/native+HR label maps have exact values and geometry; all volume/context records and model choices equal.",
                  "benchmark_results": "../benchmark_results.md", "roi": "../analysis/cohort_roi.tsv",
                  "roi_summary": "../analysis/cohort_roi_summary.tsv", "figure_manifest": "../brain_figures/plot_manifest.json",
                  "no_new_accuracy_scoring_or_plotting": True},
              "rules": {"paired_speed": "Same subject actual official wall / FNIT process wall, then mean/sample SD/median/min/max.",
                        "stage_scope": "Frozen ac692bb actual stage, no f436 stage rerun; fresh norm supplied conditionally. Official preprocessing excluded.",
                        "raw_independence": "Raw published T1 pipeline uses no official norm/finelabel input.",
                        "timer_scope": "Recipe/nested timers not added twice; official bilateral hippo/amygdala command counted once.",
                        "na": "No missing value converted to zero; planned denominators remain ten/nine.",
                        "causality": "Shared-GPU separate runs cannot isolate buffer/source effects from load/time changes.",
                        "privacy": "Only text evidence read, no image, asset, fit or GPU operation."},
              "outputs": {"official_paired_runtime.tsv": identity(tsv_path, tsv_data),
                          "official_comparison.md": identity(md_path, md_data)},
              "F8_inputs_unchanged": True}
    json_data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for path, original in cache.items():
        if path.read_bytes() != original:
            raise ValueError("Input changed during verification: " + str(path))
    for path, data in ((tsv_path, tsv_data), (md_path, md_data), (json_path, json_data)):
        with path.open("xb") as handle:
            handle.write(data)
    return {"state": "completed", "raw_rows": 10, "frozen_stage_rows": 10,
            "outputs": {path.name: identity(path) for path in (tsv_path, md_path, json_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Final F8 curated publication root")
    parser.add_argument("--regression-dir", type=Path, help="Default: --root/latest_main_regression")
    parser.add_argument("--regression-manifest", type=Path, help="Default: regression-dir/cohort_manifest.json; must be the receipt-verified curated copy")
    parser.add_argument("--baseline-source-manifest", type=Path, help="Default: --root/expected_source_manifest.json")
    args = parser.parse_args()
    root = args.root.resolve()
    regression_dir = (args.regression_dir or root / "latest_main_regression").resolve()
    manifest = (args.regression_manifest or regression_dir / "cohort_manifest.json").resolve()
    old_source = (args.baseline_source_manifest or root / "expected_source_manifest.json").resolve()
    print(json.dumps(publish(root, regression_dir, manifest, old_source), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
