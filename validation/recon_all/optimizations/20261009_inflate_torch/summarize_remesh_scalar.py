"""汇总真实 remesh 标量存储回归；不运行重建、不修改原始报告。

输入是已有 baseline、候选四网格 ABBA 目录和生产被试目录，读取 surface
RAS/mm 的三角表面。必须已有全部 11 个候选子进程成功记录。复用现有
geometry_compare，对坐标、有序面和 volume-info 原始尾部严格比较；全文件
哈希另记，创建记录与几何数据分开。逐 heap/拆缩边流与全部 pass 的 float64
坐标/有序面 SHA 须匹配冻结 baseline trace。任何缺失、失败或不一致写 failed
并退出非零。耗时是阶段/驱动/新解释器三个层次，不能相加。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

from benchmark_cpu_geometry_pair import digest, geometry_compare


def read_json(path: Path):
    # Earlier private orchestration wrote an intentional literal escape after
    # each JSON value. Preserve that raw receipt and decode only its whitespace.
    return json.loads(path.read_text().rstrip().removesuffix("\\n"))


def decoded_payload(path: Path) -> dict:
    """返回两行创建头及其后完整原始二进制 SHA，不重写 surface 文件。"""
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xff\xfe":
            raise ValueError("expected FreeSurfer triangle surface")
        stamp = [stream.readline(), stream.readline()]
        payload = stream.read()
    # Creation stamps may contain the runtime account or hostname. Preserve
    # only their hashes/lengths; no plaintext or reversible hex in public data.
    return {"creation_lines_sha256": [hashlib.sha256(row).hexdigest() for row in stamp],
            "creation_lines_bytes": [len(row) for row in stamp],
            "count_geometry_footer_payload_sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True,
                        help="含 sub06-candidate/sub07-candidate 的只读整例产物目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new summary path required")
    report = {"status": "running", "scope": "real same-input remesh stage; not GPU or new whole pipeline",
              "summary_script_sha256": digest(Path(__file__)),
              "official_same_input_current": "not_run", "whole_pipeline_current": "not_run",
              "overall_metric_equivalence": "not_assessed"}
    try:
        receipt = read_json(args.candidate / "process_receipts.json")
        report["raw_process_receipts_sha256"] = digest(args.candidate / "process_receipts.json")
        if len(receipt) != 11 or any(row["status"] != 0 for row in receipt):
            raise ValueError("all 11 planned real child runs must complete successfully")
        rows = []
        for process in receipt:
            run = args.candidate / process["name"]
            detail, api = read_json(run / "detail.json"), read_json(run / "report.json")
            if detail["status"] != "complete" or api["status"] != "complete":
                raise ValueError("incomplete stage: " + process["name"])
            row = dict(process)
            row.update(stage_seconds_including_io_jit=api["wall_seconds_including_io_jit"],
                       driver_seconds_including_setup_report=detail["driver_seconds_including_setup_and_report_io"],
                       source_sha256=detail["source_sha256"], input_sha256=detail["input_sha256"],
                       driver_sha256=detail["driver_sha256"], threads=detail["threads"],
                       cpu_affinity=detail["cpu_affinity"], gpu_used=detail["gpu_used"],
                       scalar_storage=detail["scalar_storage"], gc_policy=detail["gc_policy"],
                       gc_seconds=sum(item["seconds"] for item in detail["gc_observer"]),
                       max_rss_kib=api["max_rss_kib"], max_rss_bytes=api["max_rss_kib"] * 1024,
                       split_seconds=sum(item["seconds"] for item in api["substeps"] if item["step"] == "split"),
                       collapse_seconds=sum(item["seconds"] for item in api["substeps"] if item["step"] == "collapse"),
                       smooth_seconds=sum(item["seconds"] for item in api["substeps"] if item["step"] == "smooth"))
            # Nested rebuild/compact/GC observations are already inside these
            # complete stages. They are diagnostics and never added to totals.
            row["nested_rebuild_seconds"] = sum(item["seconds"] for item in detail["nested_steps"] if item["name"] == "rebuild")
            row["nested_compact_seconds"] = sum(item["seconds"] for item in detail["nested_steps"] if item["name"] == "compact")
            rows.append(row)
        report["runs"] = rows
        report["source_sha256"] = sorted({row["source_sha256"] for row in rows})
        report["cpu_affinities"] = [list(value) for value in sorted({tuple(row["cpu_affinity"]) for row in rows})]
        if len(report["source_sha256"]) != 1 or any(row["gc_policy"] != "inherit" for row in rows):
            raise ValueError("ABBA must use one frozen candidate source and original GC policy")
        abba = [row for row in rows if row["name"].startswith("abba_")]
        medians = {}
        for key in ("stage_seconds_including_io_jit", "driver_seconds_including_setup_report", "outer_seconds_including_child_startup_exit"):
            a = statistics.median(row[key] for row in abba if row["storage"] == "numpy")
            b = statistics.median(row[key] for row in abba if row["storage"] == "python")
            medians[key] = {"numpy_median_seconds": a, "python_median_seconds": b,
                            "reduction_percent": (a-b)/a*100, "ratio_numpy_over_python": a/b}
        report["warm_abba_medians"] = medians
        comparisons = []
        ref_lh = args.baseline / "sub07_lh_hot" / "surface"
        for row in rows:
            sub, hemisphere = row["mesh"].split("_")
            reference = (ref_lh if row["mesh"] == "sub07_lh" else
                         args.candidate / (row["mesh"] + "_A") / "surface")
            candidate = args.candidate / row["name"] / "surface"
            comparison = geometry_compare(reference=reference, candidate=candidate)
            comparison.update(name=row["name"], mesh=row["mesh"], storage=row["storage"],
                              reference_raw_payload=decoded_payload(reference),
                              candidate_raw_payload=decoded_payload(candidate))
            comparisons.append(comparison)
        report["same_input_geometry"] = comparisons
        production = []
        for mesh in ("sub06_lh", "sub06_rh", "sub07_lh", "sub07_rh"):
            sub, hemisphere = mesh.split("_")
            candidate = args.candidate / ("abba_B2" if mesh == "sub07_lh" else mesh + "_B") / "surface"
            reference = args.production / (sub + "-candidate") / "subject/surf" / (hemisphere + ".orig")
            production.append(dict(mesh=mesh, **geometry_compare(reference=reference, candidate=candidate)))
        report["existing_raw589_production_geometry"] = production
        pass_counts = []
        for mesh in ("sub06_lh", "sub06_rh", "sub07_lh", "sub07_rh"):
            a_stem = "abba_A1" if mesh == "sub07_lh" else mesh + "_A"
            b_stem = "abba_B2" if mesh == "sub07_lh" else mesh + "_B"
            def phase_decisions(stem):
                api = read_json(args.candidate / stem / "report.json")
                return [{key: value for key, value in row.items() if key != "seconds"}
                        for row in api["substeps"]]
            a_steps, b_steps = phase_decisions(a_stem), phase_decisions(b_stem)
            pass_counts.append(dict(mesh=mesh, equal=a_steps == b_steps,
                                    baseline_complete_pass_decisions=a_steps,
                                    candidate_complete_pass_decisions=b_steps))
        report["per_mesh_full_pass_counts"] = pass_counts
        baseline_trace = read_json(args.baseline / "sub07_lh_trace/detail.json")
        candidate_trace = read_json(args.candidate / "python_decision_trace/detail.json")
        keys = ("per_edge_decisions", "per_edge_decision_stream_sha256", "complete_pass_snapshots")
        report["complete_decision_replay"] = {key + "_equal": baseline_trace[key] == candidate_trace[key] for key in keys}
        report["complete_decision_replay"].update(baseline_source_sha256=baseline_trace["source_sha256"],
                  candidate_source_sha256=candidate_trace["source_sha256"],
                  input_sha256=candidate_trace["input_sha256"],
                  decisions=candidate_trace["per_edge_decisions"],
                  stream_sha256=candidate_trace["per_edge_decision_stream_sha256"],
                  pass_snapshot_count=len(candidate_trace["complete_pass_snapshots"]))
        strict = (all(row["strict_pass"] for row in comparisons + production)
                  and all(row["equal"] for row in pass_counts))
        trace_equal = all(report["complete_decision_replay"][key + "_equal"] for key in keys)
        report["strict_same_input_geometry"] = "passed" if strict else "failed"
        report["strict_same_input_decisions"] = "passed" if trace_equal else "failed"
        report["optimization_degradation_observed"] = not (strict and trace_equal)
        report["status"] = "complete" if strict and trace_equal else "failed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(report["status"], args.output, flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
