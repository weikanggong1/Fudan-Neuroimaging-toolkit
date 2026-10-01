"""汇总四半球冻结输入配对；严格门控有序面与全部坐标，非整例。"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = []
for subject in ("sub01", "sub02"):
    for hemi in ("lh", "rh"):
        pair = {}
        for variant in ("old", "new"):
            name = f"stage3_{subject}_{hemi}_{variant}"
            path = args.root / name / "report.json"
            report = json.loads(path.read_text())
            monitor_path = args.root / (name + "_monitor") / "monitor.json"
            monitor = json.loads(monitor_path.read_text())
            if monitor["exit_code"] != 0:
                raise RuntimeError(f"failed execution: {name}")
            comparison = report["comparison"]
            if not comparison["correspondence"] or comparison["different_coordinates"] != 0:
                raise RuntimeError(f"ordered geometry regression: {name}")
            pair[variant] = {"report": report, "monitor": monitor,
                             "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if pair["old"]["report"]["input_sha256"] != pair["new"]["report"]["input_sha256"]:
            raise RuntimeError("different frozen inputs")
        def trace(variant):
            return [{k: row[k] for k in ("index", "stage", "weight", "averages", "dt")}
                    for row in pair[variant]["report"]["stage"]["updates"]]
        if trace("old") != trace("new"):
            raise RuntimeError("different ordered sphere update trace")
        rows.append({"subject": subject, "hemisphere": hemi,
                     "update_trace_exact": True,
                     "wall_reduction_percent": 100 * (1 - pair["new"]["report"]["seconds"] / pair["old"]["report"]["seconds"]),
                     **pair})
args.output.write_text(json.dumps({"scope": "same_input_full_stage_including_io_jit_transfer",
                                  "strict_stage_regression": "passed",
                                  "whole_case_pending": True,
                                  "overall_metric_equivalence": "not_assessed",
                                  "pairs": rows}, indent=2) + "\n")
