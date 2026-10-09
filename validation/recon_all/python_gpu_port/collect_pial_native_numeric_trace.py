"""从两份私有原生日志提取四轮数值轨迹；不发布日志、路径或许可证内容。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(path):
    starts, iterations, cleanup = [], [], []
    outer = -1
    for line in Path(path).read_text(errors="replace").splitlines():
        start = re.match(r"^n_averages=(\d+), current_sigma=([\d.eE+-]+)", line)
        if start:
            outer += 1
            starts.append({"pass": outer, "n_averages": int(start[1]), "sigma": float(start[2])})
        step = re.match(r"^\s*(\d+):\s+dt:\s+([\d.eE+-]+),\s*sse=([\d.eE+-]+),\s*rms=([\d.eE+-]+)", line)
        if step:
            iterations.append({"pass": outer, "step": int(step[1]), "dt": float(step[2]),
                               "sse_printed": float(step[3]), "rms_printed": float(step[4])})
        marker = re.match(r"^\s*(\d+):\s*(\d+)\s+intersecting", line)
        if marker:
            cleanup.append({"iteration": int(marker[1]), "intersecting_faces": int(marker[2])})
    if len(starts) != 4 or not cleanup:
        raise ValueError("expected four explicit native passes and final cleanup trace")
    return {"private_log_sha256": sha256(path), "pass_starts": starts,
            "iteration_rows": iterations, "cleanup_rows": cleanup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-logs", nargs=2, type=Path, required=True)
    parser.add_argument("--python-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [parse(path) for path in args.native_logs]
    python = json.loads(args.python_report.read_text())
    traces = python["stages"]["torch"]["trace"]
    lookup = {(r["pass"], r["step"]): r for r in rows[0]["iteration_rows"] if r["step"]}
    comparisons = []
    for row in traces:
        reference = lookup.get((row["pass"], row["step"]))
        diagnostics = row["diagnostics"]
        comparisons.append({"pass": row["pass"], "step": row["step"], "native_row_found": reference is not None,
            "native_printed": reference, "python_diagnostics": diagnostics,
            "abs_sse_difference_from_printed_native": None if reference is None else
                abs(diagnostics["sse"] - reference["sse_printed"]),
            "abs_rms_difference_from_printed_native": None if reference is None else
                abs(diagnostics["rms"] - reference["rms_printed"])})
    compared_keys = ("pass_starts", "iteration_rows", "cleanup_rows")
    report = {"scope": "same_input_same_host_current_Conda_pial_printed_native_scalars_not_coordinate_snapshots",
        "script_sha256": sha256(__file__), "python_report_sha256": sha256(args.python_report),
        "native_program_sha256": python["native_program_sha256"], "input_sha256": python["input_sha256"],
        "native_repetitions": rows, "same_native_numeric_trace_twice": all(rows[0][k] == rows[1][k] for k in compared_keys),
        "all_four_passes_recorded": True, "python_vs_printed_native": comparisons,
        "comparison_limits": "native CLI prints rounded SSE/RMS and lacks full per-step coordinates/all rejected trials; scalar differences do not prove full geometric agreement or locate the first coordinate error",
        "overall_metric_equivalence": "not_assessed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"numeric_repeat_exact": report["same_native_numeric_trace_twice"],
        "passes": rows[0]["pass_starts"], "native_cleanup": rows[0]["cleanup_rows"],
        "found_python_steps": sum(r["native_row_found"] for r in comparisons), "python_steps": len(comparisons)}))


if __name__ == "__main__":
    main()
