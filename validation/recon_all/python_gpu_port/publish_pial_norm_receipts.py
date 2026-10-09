"""汇总已保存的同输入pial限幅回归；公开副本去除主机、路径和逐PID记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


OMIT = {"hostname", "platform", "command", "compiler_command", "link_command", "traceback",
        "samples", "gpu_uuid", "benchmark_pid", "processes", "compute_apps", "pid_namespace_mapping"}


def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(k): sanitize(v) for k, v in value.items() if k not in OMIT}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and (value.startswith("/") or "/cpfs" in value or "/cwStorage" in value):
        return "[private-artifact]/" + value.rsplit("/", 1)[-1]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"schema_version": 1, "scope": "same_input_complete_pial_norm_rounding_regression",
        "hardware": "same A100-SXM4-80GB host; 4 threads per stage; private receipts retain host/affinity",
        "reference_kind": "same-host independently Conda source-built d932c45b7941662ea380a05efef580568b98d41a",
        "official_distribution_cross_environment": "not inferred from Conda same-input reference",
        "overall_metric_equivalence": "not_assessed", "production_default": "unchanged native white/pial",
        "full_recon_all": "not_run_with_this_step_patch", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "receipts": {}, "runs": {}}
    for path in sorted(args.run_root.glob("*v16*/report.json")):
        raw = json.loads(path.read_text())
        # 尚未完成的诊断仍保留其实际状态，不能改写为通过。
        row = {"private_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "report": sanitize(raw)}
        memory = args.run_root / (path.parent.name + ".memory.json")
        if memory.exists():
            memory_raw = json.loads(memory.read_text())
            row["memory"] = sanitize(memory_raw)
            row["memory"]["private_raw_report_sha256"] = hashlib.sha256(memory.read_bytes()).hexdigest()
            row["memory"]["number_of_samples"] = len(memory_raw.get("samples", []))
        report["runs"][path.parent.name] = row
    for name in ("norm_expression_receipt.json", "norm_first_trial_v16.json",
                 "norm_fix_v16_native39_receipt.json", "first_difference_corrected_receipt_v3.json",
                 "white_preaparc_norm_comparison_v16.json"):
        path = args.run_root / name
        if path.exists():
            report["receipts"][name] = {"private_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "report": sanitize(json.loads(path.read_text()))}
    controls, candidates = [], []
    for name, row in report["runs"].items():
        raw = row["report"]
        if not name.startswith("pial_norm_fix_v16_lh_") or raw.get("status") != "complete":
            continue
        (controls if raw["variant"] == "control" else candidates).append(raw)
    if len(controls) == len(candidates) == 2:
        old = statistics.median(r["API_wall_seconds"] for r in controls)
        new = statistics.median(r["API_wall_seconds"] for r in candidates)
        report["LH_BAAB"] = {"order": ["candidate", "control", "control", "candidate"],
            "control_seconds": [r["API_wall_seconds"] for r in controls],
            "candidate_seconds": [r["API_wall_seconds"] for r in candidates],
            "control_median_seconds": old, "candidate_median_seconds": new,
            "candidate_increase_percent": (new/old-1)*100,
            "same_control_coordinate_trace": [r["coordinate_sha256"] for r in controls[0]["trace"]]
                                            == [r["coordinate_sha256"] for r in controls[1]["trace"]],
            "same_candidate_coordinate_trace": [r["coordinate_sha256"] for r in candidates[0]["trace"]]
                                              == [r["coordinate_sha256"] for r in candidates[1]["trace"]],
            "performance_scope": "complete API read/write; shared-card occupancy changed, not an isolated stable-speed gate"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
