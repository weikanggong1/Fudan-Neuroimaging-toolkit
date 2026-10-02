#!/usr/bin/env python3
"""Render sanitized public10 aggregate JSON using only the Python standard library.

Usage:
  python render_report.py --aggregate aggregate.progress_00.json \
      --output-dir summary --dataset-manifest dataset_manifest.json

Writes RESULTS.md, cases.csv, map_metrics.csv and binding.json. Missing pairs
and maps retain their fixed denominators. Exit 0 means the complete aggregate
was rendered; exit 2 means a progress report was rendered; exit 1 is an error.
No images are loaded, metrics fitted, source paths copied, or GPU jobs started.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


SCHEMA = "fnit_public10_aggregate_v1"
BACKENDS = ("tbss", "mmorf")
MAP_NAMES = ("FA", "MD", "L1", "L2", "L3", "MO", "ICVF", "OD", "ISOVF")
SPACES = {"tbss": ("native", "standard", "skeleton"),
          "mmorf": ("native", "standard")}
METRICS = ("pearson_r", "mae", "rmse", "rmse_over_reference_rms",
           "p95_absdiff", "max_absdiff", "mean_signed_difference")
MEMORY_KEYS = (
    "candidate_peak_allocated_bytes", "candidate_peak_reserved_bytes",
    "candidate_max_rss_kib", "reference_max_rss_kib",
    "reference_mmorf_gpu_sampled_peak_mib",
    "candidate_own_gpu_sampled_peak_mib", "reference_own_gpu_sampled_peak_mib",
    "candidate_other_gpu_sampled_peak_mib", "reference_other_gpu_sampled_peak_mib",
)
TIMING_KEYS = ("candidate_processing", "reference_processing",
               "candidate_full_process", "reference_full_process")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def number(value):
    try:
        return value if type(value) in (int, float) and math.isfinite(value) else None
    except OverflowError:
        return None


def label(value, default="unavailable"):
    return value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else default


def obj(value):
    return value if isinstance(value, dict) else {}


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values):
    values = sorted(value for value in (number(v) for v in values) if value is not None)
    if not values:
        return {key: 0 if key == "n" else None
                for key in ("n", "median", "q25", "q75", "iqr", "min", "max")}
    q25, median, q75 = (percentile(values, q) for q in (0.25, 0.5, 0.75))
    return {"n": len(values), "median": median, "q25": q25, "q75": q75,
            "iqr": q75 - q25, "min": values[0], "max": values[-1]}


def fmt(value):
    value = number(value)
    return "—" if value is None else f"{value:.6g}"


def summary_cell(values, denominator):
    s = summarize(values)
    if not s["n"]:
        return f"n=0/{denominator}；—"
    return (f"n={s['n']}/{denominator}；{fmt(s['median'])} "
            f"[{fmt(s['q25'])}, {fmt(s['q75'])}]；IQR {fmt(s['iqr'])}；"
            f"范围 {fmt(s['min'])}–{fmt(s['max'])}")


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|"] + [
                "| " + " | ".join(str(v) for v in row) + " |" for row in rows]


def numeric_dict(value, keys):
    value = obj(value)
    return {key: number(value.get(key)) for key in keys}


def stage_dict(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("stage_values_must_be_object")
    result = {}
    for name, seconds in value.items():
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
            raise ValueError("stage_name_not_public_identifier")
        result[name] = number(seconds)
    return result


def normalize_aggregate(aggregate):
    if not isinstance(aggregate, dict) or aggregate.get("schema") != SCHEMA:
        raise ValueError("unsupported_aggregate_schema")
    if aggregate.get("expected_subjects") != 10 or aggregate.get("expected_branch_pairs") != 20:
        raise ValueError("public10_requires_ten_subjects_twenty_pairs")
    if not isinstance(aggregate.get("cases"), list):
        raise ValueError("aggregate_cases_must_be_list")
    if aggregate.get("status") not in ("complete", "incomplete"):
        raise ValueError("unknown_aggregate_status")
    expected = {(f"case{i:02d}", backend) for i in range(1, 11) for backend in BACKENDS}
    indexed = {}
    for item in aggregate["cases"]:
        if not isinstance(item, dict):
            raise ValueError("aggregate_case_must_be_object")
        key = (item.get("case_id"), item.get("backend"))
        if key not in expected or key in indexed:
            raise ValueError("unexpected_or_duplicate_case")
        indexed[key] = item
    rows = []
    for case_id in (f"case{i:02d}" for i in range(1, 11)):
        for backend in BACKENDS:
            raw = indexed.get((case_id, backend), {})
            status = {side: label(obj(raw.get("run_status")).get(side))
                      for side in ("candidate", "reference", "comparison")}
            binding = label(obj(raw.get("paired_input_and_resource_binding")).get("status"))
            external = obj(raw.get("external_process_metrics"))
            complete = (raw.get("paired_runs_complete") is True
                        and status["candidate"] == status["reference"] == "complete"
                        and binding == "matched" and not raw.get("binding_errors")
                        and all(obj(external.get(side)).get("exit_status") in (None, 0)
                                for side in ("candidate", "reference")))
            times = numeric_dict(raw.get("timing_seconds"), TIMING_KEYS)
            ratios = {}
            for scope, candidate, reference in (
                    ("processing", "candidate_processing", "reference_processing"),
                    ("full_process", "candidate_full_process", "reference_full_process")):
                a, b = times[candidate], times[reference]
                ratios[scope] = b / a if complete and a is not None and b is not None and a > 0 and b > 0 else None
            maps = obj(raw.get("maps"))
            if any(space not in SPACES[backend] for space in maps):
                raise ValueError("unknown_map_space")
            for space, items in maps.items():
                if not isinstance(items, dict) or any(name not in MAP_NAMES for name in items):
                    raise ValueError("unknown_map_name_or_shape")
                if any(not isinstance(item, dict) for item in items.values()):
                    raise ValueError("map_metrics_must_be_object")
            gate_complete = all(obj(obj(maps.get(space)).get(name)).get("geometry_and_finite_passed") is True
                                for space in SPACES[backend] for name in MAP_NAMES)
            retained = raw.get("retained_initial_reference_attempt")
            if retained is not None and not isinstance(retained, dict):
                raise ValueError("retained_reference_attempt_must_be_object")
            retained_candidate = raw.get("retained_initial_candidate_attempt")
            if retained_candidate is not None and not isinstance(retained_candidate, dict):
                raise ValueError("retained_candidate_attempt_must_be_object")
            rows.append({
                "case_id": case_id, "backend": backend,
                "present": raw["plan_row_present"] if type(raw.get("plan_row_present")) is bool else bool(raw),
                "status": status, "binding": binding, "complete": complete,
                "comparison_complete": raw.get("paired_comparison_complete") is True and gate_complete
                    and status["comparison"] == "complete" and not raw.get("binding_errors"),
                "times": times, "ratios": ratios, "memory": numeric_dict(raw.get("memory"), MEMORY_KEYS),
                "maps": maps, "retained": retained, "retained_candidate": retained_candidate,
                "failure_types": {side: label(obj(raw.get("failure_types")).get(side), "")
                                  for side in ("candidate", "reference")},
                "reports": obj(raw.get("reports")), "external": external,
                "source_commit": raw.get("source_commit") if COMMIT.fullmatch(str(raw.get("source_commit", ""))) else None,
                "source_python_hash_count": number(raw.get("source_python_hash_count")),
                "native_mask_dice": number(raw.get("native_mask_dice")),
                "T1_mask_dice": number(raw.get("T1_mask_dice")),
                "stages": {"candidate": stage_dict(raw.get("candidate_stages_seconds")),
                           "reference": stage_dict(raw.get("reference_stages_seconds")),
                           "candidate_nested": stage_dict(raw.get("candidate_nested_events_summed_by_name_seconds")),
                           "reference_detail": stage_dict(raw.get("reference_registration_detail_seconds"))},
                "startup_attempts": raw.get("reference_mmorf_startup_attempts"),
            })
    return rows


def map_rows(rows):
    result = []
    for row in rows:
        for space in SPACES[row["backend"]]:
            for name in MAP_NAMES:
                value = obj(obj(row["maps"].get(space)).get(name))
                stats, support = obj(value.get("fixed_roi")), obj(value.get("support"))
                gate = value.get("geometry_and_finite_passed")
                entry = {"case_id": row["case_id"], "backend": row["backend"],
                         "space": space, "map": name,
                         "geometry_and_finite_passed": gate if type(gate) is bool else None,
                         "metric_status": "available" if stats else "missing",
                         "fixed_roi_elements": number(stats.get("elements")),
                         "different_elements": number(stats.get("different_elements")),
                         "candidate_mean": number(stats.get("candidate_mean")),
                         "reference_mean": number(stats.get("reference_mean"))}
                entry.update(numeric_dict(stats, METRICS))
                entry.update({"support_" + key: number(support.get(key)) for key in
                              ("dice", "jaccard", "candidate_voxels", "reference_voxels",
                               "intersection_voxels", "union_voxels")})
                result.append(entry)
    return result


def attempt_state(status, report, metrics, failure):
    if not (obj(report).get("available") or obj(metrics).get("status") == "loaded"):
        return None
    if (failure or obj(metrics).get("exit_status") not in (None, 0)
            or status in ("failed", "nonfinite_outputs", "output_contract_failed")):
        return "failed"
    return "successful" if status == "complete" else "pending_or_unknown"


def reference_attempt_counts(rows):
    groups = {name: [] for name in ("initial", "rerun", "selected_primary", "all")}
    for row in rows:
        selected = attempt_state(row["status"]["reference"], row["reports"].get("reference"),
                                 row["external"].get("reference"), row["failure_types"]["reference"])
        original = row["retained"]
        if original is not None:
            initial = attempt_state(label(original.get("status")), original.get("report"),
                                    original.get("external_process_metrics"), original.get("failure_type"))
            if initial is not None:
                groups["initial"].append(initial)
            if selected is not None:
                groups["rerun"].append(selected)
        elif selected is not None:
            groups["initial"].append(selected)
        if selected is not None:
            groups["selected_primary"].append(selected)
    groups["all"] = groups["initial"] + groups["rerun"]
    return {key: {"observed": len(values), "successful": values.count("successful"),
                  "failed": values.count("failed"), "pending": values.count("pending_or_unknown")}
            for key, values in groups.items()}


def candidate_attempt_counts(rows):
    adapted = [{
        "status": {"reference": row["status"]["candidate"]},
        "reports": {"reference": row["reports"].get("candidate")},
        "external": {"reference": row["external"].get("candidate")},
        "failure_types": {"reference": row["failure_types"]["candidate"]},
        "retained": row["retained_candidate"],
    } for row in rows]
    return reference_attempt_counts(adapted)


def startup_records(row):
    """Read optional anonymous native-start observations; absence stays None."""
    value = row["startup_attempts"]
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("available") is False:
            return None
        value = value.get("attempts")
    if not isinstance(value, list):
        raise ValueError("unknown_mmorf_startup_attempts_shape")
    if not value:
        return None
    records = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise ValueError("mmorf_startup_attempt_must_be_object")
        allowed = item.get("retry_allowed", obj(item.get("startup_retry")).get("retry_allowed"))
        records.append({"record_index": index, "attempt": number(item.get("attempt")),
                        "name": label(item.get("name"), ""),
                        "seconds": number(item.get("seconds")), "exit_code": number(item.get("exit_code")),
                        "retry_allowed": allowed if type(allowed) is bool else None})
    return records


def case_rows(rows):
    result = []
    for row in rows:
        retained = obj(row["retained"])
        old_times = obj(retained.get("timing_seconds"))
        old_mem = obj(retained.get("memory"))
        old_candidate = obj(row["retained_candidate"])
        old_candidate_times = obj(old_candidate.get("timing_seconds"))
        old_candidate_memory = obj(old_candidate.get("memory"))
        entry = {"case_id": row["case_id"], "backend": row["backend"],
                 "plan_row_present": row["present"],
                 "candidate_status": row["status"]["candidate"], "reference_status": row["status"]["reference"],
                 "comparison_status": row["status"]["comparison"], "input_binding_status": row["binding"],
                 "paired_runs_complete": row["complete"], "paired_comparison_complete": row["comparison_complete"],
                 "candidate_failure_type": row["failure_types"]["candidate"],
                 "reference_failure_type": row["failure_types"]["reference"],
                 "source_commit": row["source_commit"], "source_python_hash_count": row["source_python_hash_count"],
                 "native_mask_dice": row["native_mask_dice"], "T1_mask_dice": row["T1_mask_dice"],
                 "retained_initial_reference_status": label(retained.get("status")) if retained else None,
                 "retained_initial_reference_failure_type": label(retained.get("failure_type"), "") if retained else None,
                 "retained_initial_reference_processing_seconds": number(old_times.get("processing")),
                 "retained_initial_reference_full_process_seconds": number(old_times.get("full_process")),
                 "retained_initial_reference_max_rss_kib": number(old_mem.get("max_rss_kib")),
                 "retained_initial_reference_own_gpu_sampled_peak_mib": number(old_mem.get("own_gpu_sampled_peak_mib")),
                 "retained_initial_reference_other_gpu_sampled_peak_mib": number(old_mem.get("other_gpu_sampled_peak_mib")),
                 "retained_initial_failure_used_for_ratio": False}
        entry.update({
            "retained_initial_candidate_status": label(old_candidate.get("status")) if old_candidate else None,
            "retained_initial_candidate_failure_type": label(old_candidate.get("failure_type"), "") if old_candidate else None,
            "retained_initial_candidate_processing_seconds": number(old_candidate_times.get("processing")),
            "retained_initial_candidate_full_process_seconds": number(old_candidate_times.get("full_process")),
            "retained_initial_candidate_max_rss_kib": number(old_candidate_memory.get("max_rss_kib")),
            "retained_initial_candidate_peak_allocated_bytes": number(old_candidate_memory.get("peak_allocated_bytes")),
            "retained_initial_candidate_peak_reserved_bytes": number(old_candidate_memory.get("peak_reserved_bytes")),
            "retained_initial_candidate_own_gpu_sampled_peak_mib": number(old_candidate_memory.get("own_gpu_sampled_peak_mib")),
            "retained_initial_candidate_other_gpu_sampled_peak_mib": number(old_candidate_memory.get("other_gpu_sampled_peak_mib")),
            "retained_initial_candidate_source_commit": old_candidate.get("source_commit") if COMMIT.fullmatch(str(old_candidate.get("source_commit", ""))) else None,
            "retained_initial_candidate_source_python_hash_count": number(old_candidate.get("source_python_hash_count")),
            "retained_initial_candidate_report_sha256": obj(old_candidate.get("report")).get("sha256") if HASH.fullmatch(str(obj(old_candidate.get("report")).get("sha256", ""))) else None,
            "retained_initial_candidate_failure_used_for_ratio": False,
        })
        entry.update({key + "_seconds": value for key, value in row["times"].items()})
        entry.update({key + "_reference_over_candidate": value for key, value in row["ratios"].items()})
        entry.update(row["memory"])
        attempts = startup_records(row) if row["backend"] == "mmorf" else None
        entry.update({"reference_mmorf_startup_records": len(attempts) if attempts is not None else None,
                      "reference_mmorf_startup_successful": sum(v["exit_code"] == 0 for v in attempts) if attempts is not None else None,
                      "reference_mmorf_startup_failed": sum(v["exit_code"] is not None and v["exit_code"] != 0 for v in attempts) if attempts is not None else None,
                      "reference_mmorf_startup_unknown": sum(v["exit_code"] is None for v in attempts) if attempts is not None else None})
        for side in ("candidate", "reference"):
            metrics = obj(row["external"].get(side))
            entry[side + "_full_process_clock_source"] = label(metrics.get("wall_time_source"))
            entry[side + "_exit_status"] = number(metrics.get("exit_status"))
        result.append(entry)
    return result


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def dataset_description(manifest):
    if manifest is None:
        return "未提供数据清单；本报告的输入为已有匿名 aggregate JSON。"
    if not isinstance(manifest, dict) or manifest.get("dataset") != "ds003138" or manifest.get("version") != "1.0.1":
        raise ValueError("unexpected_public10_dataset_manifest")
    if manifest.get("license") != "CC0" or len(manifest.get("subjects", [])) != 10:
        raise ValueError("unexpected_public10_dataset_license_or_count")
    return "OpenNeuro ds003138 v1.0.1，CC0；固定升序前十人、ses-1，完整 117 帧 AP、shell 1 PA b0 与 T1。"


def markdown(aggregate, rows, maps, manifest, figures):
    complete = (aggregate["status"] == "complete"
                and aggregate.get("all_planned_results_complete", True) is True
                and all(row["complete"] and row["comparison_complete"] for row in rows))
    run_count = sum(row["complete"] for row in rows)
    comparison_count = sum(row["comparison_complete"] for row in rows)
    map_count = sum(row["geometry_and_finite_passed"] is True for row in maps)
    lines = ["# 公开十人 DMRIPipeline 对照" + ("结果" if complete else "进度报告"), "",
             "状态：**" + ("完整结果已汇总" if complete else "尚未完成，以下仅为当前进度") + "**。", "",
             dataset_description(manifest), "",
             f"计划 **10 人 × 2 分支 = 20 配对**；整链完成 {run_count}/20，逐图比较完成 {comparison_count}/20，"
             f"指标图格式检查通过 {map_count}/450。所有缺失和失败被试保留在分母中。", "",
             "结果报告测得的差异和耗时，不声明数值等价，也不把时间差称为等价重建的加速。", "",
             "## 逐人状态", ""]
    lines += table(["case", "分支", "FNIT", "原软件", "比较", "输入绑定", "完整整链", "完整比较"],
                   [[r["case_id"], r["backend"], *r["status"].values(), r["binding"],
                     "是" if r["complete"] else "否", "是" if r["comparison_complete"] else "否"] for r in rows])
    lines += ["", "## 完整耗时观测", "",
              "单位为秒。原/FNIT 比值逐人计算，只使用两侧完整运行且输入绑定匹配的配对；失败或未完成运行的时钟单列，比值为空。", ""]
    lines += table(["case", "分支", "FNIT API", "原软件处理", "FNIT 完整命令", "原软件完整命令", "处理原/FNIT", "命令原/FNIT"],
                   [[r["case_id"], r["backend"], *(fmt(r["times"][k]) for k in TIMING_KEYS),
                     fmt(r["ratios"]["processing"]), fmt(r["ratios"]["full_process"])] for r in rows])
    lines += ["", "完整命令优先采用 GNU time；`cases.csv` 保存具体时钟来源。API 与完整命令时钟边界分别保留，不将嵌套阶段相加。", "",
              "### 分支配对汇总", "", "各项为 n/计划人数；中位数 [Q25,Q75]；IQR；范围。", ""]
    for backend in BACKENDS:
        selected = [r for r in rows if r["backend"] == backend and r["complete"]]
        lines += [f"#### {backend.upper()}", ""]
        lines += table(["项目", "汇总"], [[key + "（秒）", summary_cell([r["times"][key] for r in selected], 10)]
                                         for key in TIMING_KEYS] + [
                         [scope + " 原/FNIT", summary_cell([r["ratios"][scope] for r in selected], 10)]
                         for scope in ("processing", "full_process")])
        lines += [""]
    lines += ["## FNIT 完整运行尝试与保留失败", "",
              "主对照采用固定修正版的全新运行。case01 的两个旧版成功运行属于回归诊断，不计入下面的主运行尝试或配对提速；清单中明确保留的旧版失败独立计数，其时钟不加入修正版耗时。", ""]
    candidate_counts = candidate_attempt_counts(rows)
    candidate_names = {"initial": "初次或选定完整运行", "rerun": "失败后新主运行", "selected_primary": "当前选定 FNIT", "all": "主运行与明确保留失败"}
    lines += table(["类别", "已观察次数", "成功", "失败", "待定"],
                   [[candidate_names[k], v["observed"], v["successful"], v["failed"], v["pending"]]
                    for k, v in candidate_counts.items()])
    retained_candidates = [r for r in rows if r["retained_candidate"] is not None]
    if retained_candidates:
        lines += ["", "### 最初保留的 FNIT 失败运行", ""]
        lines += table(["case", "分支", "状态", "异常类型", "API 秒", "完整命令秒", "用于比值"],
                       [[r["case_id"], r["backend"], label(r["retained_candidate"].get("status")),
                         label(r["retained_candidate"].get("failure_type")),
                         fmt(obj(r["retained_candidate"].get("timing_seconds")).get("processing")),
                         fmt(obj(r["retained_candidate"].get("timing_seconds")).get("full_process")), "否"]
                        for r in retained_candidates])
    lines += ["", "## 原软件完整运行尝试与保留失败", "",
              "这里统计完整原软件流程的首次运行和整链重跑；内部 MMORF 启动次数另列。选定运行用于配对，保留失败的时钟不替代恢复运行。", ""]
    counts = reference_attempt_counts(rows)
    names = {"initial": "首次完整运行", "rerun": "恢复整链运行", "selected_primary": "当前选定参考", "all": "所有已观察完整运行"}
    lines += table(["类别", "已观察次数", "成功", "失败", "待定"],
                   [[names[k], v["observed"], v["successful"], v["failed"], v["pending"]] for k, v in counts.items()])
    retained = [r for r in rows if r["retained"] is not None]
    if retained:
        lines += ["", "### 最初保留的参考运行", ""]
        lines += table(["case", "分支", "状态", "异常类型", "处理秒", "完整命令秒", "用于比值"],
                       [[r["case_id"], r["backend"], label(r["retained"].get("status")),
                         label(r["retained"].get("failure_type")),
                         fmt(obj(r["retained"].get("timing_seconds")).get("processing")),
                         fmt(obj(r["retained"].get("timing_seconds")).get("full_process")), "否"] for r in retained])
    lines += ["", "### 原 MMORF 内部启动尝试", "",
              "缺少对应匿名记录时标为未记录，不记为零。启动重试及等待时间已包含在父阶段与完整时钟内。", ""]
    startup_rows = []
    for r in rows:
        if r["backend"] == "mmorf":
            attempts = startup_records(r)
            if attempts is None:
                startup_rows.append([r["case_id"], "未记录", "—", "—", "—"])
            else:
                for item in attempts:
                    startup_rows.append([r["case_id"], fmt(item["attempt"] if item["attempt"] is not None else item["record_index"]),
                                         fmt(item["seconds"]), fmt(item["exit_code"]),
                                         "是" if item["retry_allowed"] is True else "否" if item["retry_allowed"] is False else "未记录"])
    lines += table(["case", "启动记录序号", "子进程秒", "退出码", "满足重试条件"], startup_rows)
    lines += ["", "## 内存与显存", "",
              "allocated/reserved 是 FNIT PyTorch allocator 的全流程峰值；GiB=2³⁰ bytes。RSS 使用 GNU time 最大驻留量（KiB）；"
              "GPU 采样峰值单位 MiB，包含本作业 CUDA context。other 是选定卡上同期其他进程，5 秒采样可能漏过瞬时峰值。未记录值保持为空。", ""]
    lines += table(["case", "分支", "allocated GiB", "reserved GiB", "FNIT RSS KiB", "原 RSS KiB", "FNIT 自身 MiB", "原自身 MiB", "FNIT同期其他 MiB", "原同期其他 MiB"],
                   [[r["case_id"], r["backend"],
                     *(fmt(r["memory"][k] / 2**30 if r["memory"][k] is not None else None)
                       for k in ("candidate_peak_allocated_bytes", "candidate_peak_reserved_bytes")),
                     *(fmt(r["memory"][k]) for k in ("candidate_max_rss_kib", "reference_max_rss_kib",
                           "candidate_own_gpu_sampled_peak_mib", "reference_own_gpu_sampled_peak_mib",
                           "candidate_other_gpu_sampled_peak_mib", "reference_other_gpu_sampled_peak_mib"))] for r in rows])
    lines += ["", "原 MMORF 独立 PID 的采样值另存于 `cases.csv`；不同进程、不同时刻的峰值不相加。", "",
              "## 逐图差异：45 组", "",
              "TBSS 为 native/standard/skeleton 各九图，MMORF 为 native/standard 各九图。主要 ROI 包含零值；"
              "NRMSE=RMSE/原软件 ROI RMS，signed bias=FNIT−原软件，support Dice 比较非零支持。"
              "MAE、RMSE、p95、bias 的单位随影像：MD/L1/L2/L3 为 mm²/s，其余为无量纲。每个统计项保留自己的有效 n/10。", "",
              "450 条逐人逐图记录（包括缺失占位）见 `map_metrics.csv`。", ""]
    for backend in BACKENDS:
        for space in SPACES[backend]:
            lines += [f"### {backend.upper()} / {space}", ""]
            summary_rows = []
            for name in MAP_NAMES:
                chosen = [m for m in maps if m["backend"] == backend and m["space"] == space and m["map"] == name]
                summary_rows.append([name, f"{sum(m['geometry_and_finite_passed'] is True for m in chosen)}/10",
                                     *(summary_cell([m[key] for m in chosen], 10) for key in
                                       ("pearson_r", "mae", "rmse_over_reference_rms", "p95_absdiff", "mean_signed_difference", "support_dice"))])
            lines += table(["图", "格式通过", "Pearson r", "MAE", "NRMSE", "p95 绝对差", "signed bias", "support Dice"], summary_rows)
            lines += [""]
    lines += ["## 分阶段耗时", "",
              "两侧阶段边界可能不同，分别列出；嵌套事件仅作分解，不能再加到父阶段或完整命令上。"
              "下面按对应侧已完成全流程中的可用阶段汇总，各侧保留自己的有效 n/10；另一侧尚未完成时仍可显示已完成侧的进度。"
              "逐人值保留在 aggregate JSON，完整配对耗时汇总另见前表。", ""]
    for backend in BACKENDS:
        for side, title in (("candidate", "FNIT 父阶段"), ("reference", "原软件父阶段"),
                            ("candidate_nested", "FNIT 嵌套事件（不相加）"),
                            ("reference_detail", "原软件配准细分（不相加）")):
            source_side = "candidate" if side.startswith("candidate") else "reference"
            selected = [r for r in rows if r["backend"] == backend and r["status"][source_side] == "complete"]
            names = sorted({name for r in selected for name in r["stages"][side]})
            lines += [f"### {backend.upper()} / {title}", ""]
            if names:
                lines += table(["阶段", "耗时秒"], [[name, summary_cell([r["stages"][side].get(name) for r in selected], 10)] for name in names])
            else:
                lines += ["该侧未有完成全流程的可用阶段记录（n=0/10）。"]
            lines += [""]
    lines += ["## 固定 case01 脑图", ""]
    if figures:
        for name in figures:
            lines += [f"![{name}](../figures/{name})", ""]
    else:
        lines += ["对应两侧完整输出及固定切片脑图尚未齐备，未生成图像链接。", ""]
    commits = sorted({r["source_commit"] for r in rows if r["source_commit"]})
    lines += ["## 来源与文件绑定", "",
              "FNIT 源码提交：" + ("、".join(f"`{c}`" for c in commits) if commits else "未记录") + "。", "",
              "输入 aggregate、可选数据清单、本生成器与三个输出文件的 SHA-256 见 `binding.json`。"
              "报告仅读取匿名 aggregate 中的白名单数字和标识，不复制原始影像、命令或私密文件系统路径。", ""]
    return ("\n".join(lines)).encode("utf-8"), complete


def atomic_write(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render(aggregate_path, output_dir, dataset_manifest=None):
    aggregate_path, output_dir = Path(aggregate_path), Path(output_dir)
    raw = aggregate_path.read_bytes()
    aggregate = json.loads(raw)
    rows = normalize_aggregate(aggregate)
    maps = map_rows(rows)
    manifest_bytes = Path(dataset_manifest).read_bytes() if dataset_manifest is not None else None
    manifest = json.loads(manifest_bytes) if manifest_bytes is not None else None
    figures = {}
    for backend in BACKENDS:
        name = f"case01_{backend}.png"
        path = output_dir.parent / "figures" / name
        if path.is_file():
            figures[name] = {"sha256": digest_bytes(path.read_bytes()), "bytes": path.stat().st_size}
    md, complete = markdown(aggregate, rows, maps, manifest, figures)
    outputs = {"RESULTS.md": md, "cases.csv": csv_bytes(case_rows(rows)),
               "map_metrics.csv": csv_bytes(maps)}
    script = Path(__file__).read_bytes()
    binding = {
        "schema": "fnit_public10_render_binding_v1", "aggregate_schema": SCHEMA,
        "report_status": "complete" if complete else "progress",
        "expected_subjects": 10, "expected_branch_pairs": 20, "expected_map_pairs": 450,
        "case_rows": len(rows), "map_metric_rows": len(maps),
        "numerical_equivalence_claimed": False,
        "input_aggregate": {"bytes": len(raw), "sha256": digest_bytes(raw)},
        "input_dataset_manifest": {"bytes": len(manifest_bytes), "sha256": digest_bytes(manifest_bytes)} if manifest_bytes is not None else None,
        "renderer": {"bytes": len(script), "sha256": digest_bytes(script)},
        "aggregate_script_sha256": aggregate.get("script_sha256") if HASH.fullmatch(str(aggregate.get("script_sha256", ""))) else None,
        "reference_full_run_attempts": reference_attempt_counts(rows),
        "candidate_full_run_attempts": candidate_attempt_counts(rows),
        "reference_mmorf_startup_observations": {row["case_id"]: startup_records(row)
                                                  for row in rows if row["backend"] == "mmorf"},
        "linked_figures": figures,
        "outputs": {name: {"bytes": len(data), "sha256": digest_bytes(data)} for name, data in outputs.items()},
    }
    outputs["binding.json"] = (json.dumps(binding, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        atomic_write(output_dir / name, data)
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True, help="匿名 fnit_public10_aggregate_v1 JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="写入 RESULTS.md、两个 CSV 和 binding.json 的目录")
    parser.add_argument("--dataset-manifest", type=Path, help="可选的固定 ds003138 v1.0.1 十人数据清单")
    args = parser.parse_args(argv)
    try:
        binding = render(args.aggregate, args.output_dir, args.dataset_manifest)
    except (OSError, ValueError, TypeError) as error:
        safe = str(error) if isinstance(error, ValueError) and re.fullmatch(r"[a-z0-9_]+", str(error)) else type(error).__name__
        print("report_render_error: " + safe, file=sys.stderr)
        return 1
    print(json.dumps({"status": binding["report_status"], "case_rows": binding["case_rows"],
                      "map_metric_rows": binding["map_metric_rows"], "renderer_sha256": binding["renderer"]["sha256"]}))
    return 0 if binding["report_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
