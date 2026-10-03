#!/usr/bin/env python3
r"""Plot fixed public10 GNU whole-process timings, without running a benchmark.

CLI:
  python plot_timing.py --aggregate aggregate.final.json --output timing.png \
      --caption-json timing.json
Python:
  make_timing_figure(aggregate=Path("aggregate.final.json"),
                     output=Path("timing.png"), caption_json=Path("timing.json"))

输入是 compare_public10.py 生成的固定20行匿名汇总；输出为两分支耗时PNG和
中文caption JSON。只画已完成、绑定匹配且有GNU来源的配对；不读取影像，
不计算精度，不把失败时钟或缺失值当成成功耗时。依赖已有matplotlib和numpy。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CASE_IDS = tuple(f"case{number:02d}" for number in range(1, 11))
BACKENDS = ("tbss", "mmorf")
RATIO_KEY = "full_process_reference_over_candidate"


def file_record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def display_status(value):
    return value if isinstance(value, str) and value in {"complete", "failed", "pending", "running",
                             "missing", "unavailable", "incomplete"} else "pending"


def plotted_row(case):
    """Consume the aggregate's existing gate and ratio; never recalculate it."""
    statuses = case.get("run_status", {})
    candidate_status = display_status(statuses.get("candidate"))
    reference_status = display_status(statuses.get("reference"))
    binding_matched = (case.get("paired_input_and_resource_binding", {}).get("status")
                       == "matched" and not case.get("binding_errors"))
    ratio = case.get("paired_ratios", {}).get(RATIO_KEY)
    timing = case.get("timing_seconds", {})
    provenance = case.get("external_process_metrics", {})
    checks = {
        "both_runs_complete": candidate_status == reference_status == "complete",
        "binding_matched": binding_matched,
        "aggregate_pair_gate": case.get("paired_runs_complete") is True,
        "aggregate_ratio_available": positive_number(ratio),
    }
    clocks = {}
    sources = {}
    for role in ("candidate", "reference"):
        clock = timing.get(f"{role}_full_process")
        metrics = provenance.get(role, {})
        gnu_valid = (metrics.get("status") == "loaded"
                     and metrics.get("wall_time_source") == "gnu_time_v"
                     and metrics.get("exit_status") == 0
                     and valid_sha(metrics.get("gnu_time_sha256"))
                     and positive_number(clock)
                     and clock == metrics.get("wall_seconds"))
        checks[f"{role}_GNU_clock_valid"] = gnu_valid
        sources[role] = {"wall_time_source": metrics.get("wall_time_source"),
                         "GNU_time_sha256": metrics.get("gnu_time_sha256")}
        clocks[role] = clock
    eligible = all(checks.values())
    label = f"FNIT {candidate_status}\nRef {reference_status}"
    if checks["both_runs_complete"] and not binding_matched:
        label = "Binding unmatched"
    elif checks["both_runs_complete"] and not eligible:
        label = "GNU pair pending"
    return {"case_id": case["case_id"], "backend": case["backend"],
            "candidate_status": candidate_status, "reference_status": reference_status,
            "paired_bars_plotted": eligible, "gate_checks": checks,
            "unpaired_display_label": None if eligible else label,
            "GNU_seconds": clocks if eligible else {"candidate": None, "reference": None},
            "reference_over_FNIT_ratio": ratio if eligible else None,
            "timing_provenance": sources}


def make_timing_figure(*, aggregate, output, caption_json=None):
    """Draw 2 panels and write provenance; paths are aggregate JSON / PNG / JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    aggregate, output = Path(aggregate), Path(output)
    caption_json = output.with_suffix(".json") if caption_json is None else Path(caption_json)
    if output.suffix.lower() != ".png":
        raise ValueError("--output must be a PNG path")
    if len({path.resolve() for path in (aggregate, output, caption_json)}) != 3:
        raise ValueError("aggregate, PNG and caption JSON must have distinct paths")
    data = json.loads(aggregate.read_text())
    expected = {(case_id, backend) for backend in BACKENDS for case_id in CASE_IDS}
    cases = data.get("cases", [])
    identities = [(case.get("case_id"), case.get("backend")) for case in cases]
    if (data.get("schema") != "fnit_public10_aggregate_v1"
            or data.get("expected_subjects") != 10 or data.get("expected_branch_pairs") != 20
            or len(identities) != 20 or set(identities) != expected):
        raise ValueError("aggregate must contain exactly the fixed 10 subjects x 2 branches")
    source_commits = data.get("source_commits", [])
    if (len(source_commits) != 1 or not isinstance(source_commits[0], str)
            or len(source_commits[0]) != 40 or any(character not in "0123456789abcdef"
                                                  for character in source_commits[0])
            or any(case.get("source_commit") != source_commits[0] for case in cases)):
        raise ValueError("all 20 candidate rows must bind one frozen source commit")
    by_id = {(case["case_id"], case["backend"]): plotted_row(case) for case in cases}
    rows = [by_id[(case_id, backend)] for backend in BACKENDS for case_id in CASE_IDS]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.7), sharey=True)
    colours = {"candidate": "#2575A6", "reference": "#D88B36"}
    maximum_minutes = max((max(row["GNU_seconds"].values()) / 60 for row in rows
                           if row["paired_bars_plotted"]), default=1.0)
    y_limit = maximum_minutes * 1.24
    counts = {}
    positions = np.arange(10)
    for axis, backend in zip(axes, BACKENDS):
        branch_rows = [by_id[(case_id, backend)] for case_id in CASE_IDS]
        paired_count = sum(row["paired_bars_plotted"] for row in branch_rows)
        counts[backend] = {
            "planned_pairs": 10, "plotted_pairs": paired_count,
            "candidate_status_counts": dict(Counter(row["candidate_status"] for row in branch_rows)),
            "reference_status_counts": dict(Counter(row["reference_status"] for row in branch_rows)),
            "unpaired_case_ids": [row["case_id"] for row in branch_rows if not row["paired_bars_plotted"]],
        }
        for index, row in enumerate(branch_rows):
            if row["paired_bars_plotted"]:
                times = row["GNU_seconds"]
                for role, offset in (("candidate", -0.19), ("reference", 0.19)):
                    axis.bar(index + offset, times[role] / 60, width=0.36,
                             color=colours[role], edgecolor="white", linewidth=0.5)
                axis.text(index, max(times.values()) / 60 + y_limit * 0.022,
                          f"{row['reference_over_FNIT_ratio']:.2f}x",
                          ha="center", va="bottom", fontsize=8, color="#263238")
            else:
                # 缺失配对只有状态文字，没有失败耗时柱或零值填充。
                axis.axvspan(index - 0.44, index + 0.44, color="#F3F3F3", zorder=0)
                axis.text(index, 0.045, row["unpaired_display_label"],
                          transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                          rotation=90, fontsize=8, color="#9B3636")
        axis.set_title(f"{backend.upper()} | paired n = {paired_count}/10", fontsize=12)
        axis.set_xticks(positions, CASE_IDS, rotation=45, ha="right", fontsize=9)
        axis.set_xlim(-0.65, 9.65)
        axis.set_ylim(0, y_limit)
        axis.set_xlabel("Fixed subject order", fontsize=10)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Whole-process GNU wall time (min)", fontsize=11)
    figure.suptitle("Independent DMRIPipeline runs | whole-process timings", fontsize=14, y=0.985)
    figure.legend([Patch(color=colours["candidate"]), Patch(color=colours["reference"])],
                  ["FNIT", "Original software"], loc="upper center", bbox_to_anchor=(0.5, 0.93),
                  ncol=2, frameon=False, fontsize=10)
    figure.text(0.5, 0.016,
                f"Shared H100 | frozen FNIT {source_commits[0][:7]} | lock wait excluded | "
                "labels: Original/FNIT | numerical equivalence not established",
                ha="center", fontsize=8, color="#505050")
    figure.subplots_adjust(left=0.067, right=0.986, bottom=0.205, top=0.805, wspace=0.11)
    output.parent.mkdir(parents=True, exist_ok=True)
    caption_json.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    paired_total = sum(row["paired_bars_plotted"] for row in rows)
    caption = {
        "schema_version": 1, "figure_type": "public10_paired_GNU_whole_process_timing",
        "caption": f"固定十人、TBSS与MMORF两个分支的独立完整运行耗时，单位为分钟。仅画两侧执行完成、输入及资源绑定匹配且GNU时钟可核验的配对，共{paired_total}/20；其余位置仅显示失败或待完成状态。柱上比值直接采用aggregate中通过运行与绑定门槛的原软件/FNIT逐人比值，没有重新计算影像指标。共享H100的外部负载可能影响耗时，数值等价尚未确立。",
        "source_commit": source_commits[0],
        "source_manifest_hashes": data.get("source_manifest_hashes", []),
        "aggregate_status": data.get("status"), "planned_pairs": 20,
        "plotted_pairs": paired_total, "branch_counts": counts,
        "total_candidate_status_counts": dict(Counter(row["candidate_status"] for row in rows)),
        "total_reference_status_counts": dict(Counter(row["reference_status"] for row in rows)),
        "timing_provenance": {
            "clock": "GNU /usr/bin/time -v whole subprocess wall time",
            "unit": "minutes", "conversion": "aggregate GNU seconds / 60",
            "includes": "Python startup/imports, complete pipeline, audits and report writing",
            "excludes": "shared GPU-lock wait and queue admission; common download/input preparation",
            "failed_or_unpaired_clock_policy": "no timing bars; no zero imputation",
            "ratio": "aggregate.paired_ratios.full_process_reference_over_candidate, not recomputed",
            "shared_device": "H100; independent jobs serialized by one GPU lock; other users are not controlled",
        },
        "numerical_equivalence_claimed": False,
        "source_script": file_record(__file__), "aggregate": file_record(aggregate),
        "output_PNG": file_record(output), "cases": rows,
    }
    caption_json.write_text(json.dumps(caption,ensure_ascii=False,indent=2,allow_nan=False) + "\n")
    return caption


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate", type=Path, required=True, help="固定20行匿名aggregate JSON")
    parser.add_argument("--output", type=Path, required=True, help="输出PNG；只画完成且绑定匹配的GNU配对时钟")
    parser.add_argument("--caption-json", type=Path, help="中文caption与哈希JSON，默认与PNG同名.json")
    arguments = parser.parse_args(argv)
    report = make_timing_figure(**vars(arguments))
    print(json.dumps({"planned_pairs":20,"plotted_pairs":report["plotted_pairs"]}))


if __name__ == "__main__":
    main()
