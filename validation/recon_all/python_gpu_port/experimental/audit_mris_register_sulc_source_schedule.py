"""Audit FreeSurfer 8.2 sulc stopping against saved independent Python SSE."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from fnit.recon_all.mris_register_schedule import next_sulc_scale


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(reports_dir: Path, hemisphere: str) -> dict:
    reports = []
    rows = []
    for path in sorted(reports_dir.glob(
            f"mris_register_{hemisphere}_default_epoch*_control-host.json")):
        report = json.loads(path.read_text())
        if "epochs" not in report:
            continue
        reports.append({"file": path.name, "sha256": digest(path)})
        rows.extend(report["epochs"])
    rows.sort(key=lambda row: row["epoch"])
    state: tuple[str, int, int, int] | None = ("big", 0, 16384, 2)
    result = []
    for index, row in enumerate(rows):
        if state is None:
            raise ValueError(f"unexpected extra sulc update {row['epoch']}")
        phase, sigma_index, averages, steps = state
        observed_scale = (row["sigma"], row["navgs"])
        expected_scale = ((4.0, 2.0, 1.0, 0.5)[sigma_index], averages)
        selected = min(row["line_samples"], key=lambda sample: abs(sample[0] - row["dt"]))[1]
        starting = row["line_samples"][0][1]
        next_state = next_sulc_scale(*state, starting, selected, row["dt"])
        observed_next = (None if index + 1 == len(rows) else
                         (rows[index + 1]["sigma"], rows[index + 1]["navgs"]))
        expected_next = (None if next_state is None else
                         ((4.0, 2.0, 1.0, 0.5)[next_state[1]], next_state[2]))
        predicted_projection = (next_state is not None and next_state[:3] != state[:3])
        observed_projection = (None if index + 1 == len(rows) else
                               rows[index + 1]["starts_integration_call"])
        entry = {"epoch": row["epoch"], "phase": phase, "sigma": row["sigma"],
                 "averages": averages, "steps_at_scale_before": steps,
                 "starting_sse": starting, "selected_sse": selected,
                 "selected_dt": row["dt"],
                 "tolerance_pct": 0.5 * math.sqrt((averages + 1) / 1024),
                 "expected_next_state": next_state,
                 "scale_match": expected_scale == observed_scale,
                 "next_scale_match": expected_next == observed_next,
                 "next_projection_match": (observed_projection is None or
                                           predicted_projection == observed_projection),
                 "exact_vertices": row["comparison"]["exact_vertices"],
                 "total_vertices": row["comparison"]["total_vertices"]}
        result.append(entry)
        state = next_state
    return {"hemisphere": hemisphere, "input_reports": reports,
            "starting_state_at_epoch_4": ["big", 0, 16384, 2],
            "epochs_checked": len(result), "first_epoch": result[0]["epoch"],
            "last_epoch": result[-1]["epoch"],
            "all_source_decisions_match": all(
                row["scale_match"] and row["next_scale_match"] and
                row["next_projection_match"] and
                row["exact_vertices"] == row["total_vertices"] for row in result),
            "terminal_state": state, "rows": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {hemisphere: audit(args.reports_dir, hemisphere)
              for hemisphere in ("lh", "rh")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {"epochs": value["epochs_checked"],
                            "all_match": value["all_source_decisions_match"],
                            "terminal": value["terminal_state"]}
                      for key, value in report.items()}))
    if not all(item["all_source_decisions_match"] and item["terminal_state"] is None
               for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
