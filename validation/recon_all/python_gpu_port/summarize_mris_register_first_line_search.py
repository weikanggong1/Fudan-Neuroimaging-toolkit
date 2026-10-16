"""Pair frozen first-line trials with native FreeSurfer SSE and matrix captures."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
TERMS = ("sse_area", "sse_nl_area", "sse_dist", "sse_corr")
PATTERN = re.compile(
    r"logSSE:(\d+).*?new sse_area : ([\d.]+).*?"
    r"new sse_nl_area : ([\d.]+).*?new sse_dist : ([\d.]+).*?"
    r"new sse_corr : ([\d.]+).*?new sum = ([\d.]+)", re.S)


def summarize(hemisphere: str) -> dict:
    torch_report = json.loads((HERE / f"mris_register_{hemisphere}_first_line_search_control-host.json").read_text())
    native_log = (HERE / f"mris_register_{hemisphere}_native_first_line_terms_control-host.log").read_text()
    fit_log = (HERE / f"mris_register_{hemisphere}_native_first_line_fit_control-host.log").read_text()
    native_records = PATTERN.findall(native_log)
    assert len(native_records) == 12
    assert [int(row[0]) for row in native_records] == list(range(1, 13))
    native_trials = native_records[3:]
    samples = torch_report["samples"]
    terms = torch_report["sample_terms"]
    assert len(samples) == len(terms) == len(native_trials) == 9
    fit = json.loads(re.search(r"FIRST_LINE_FIT (\{.*\})", fit_log).group(1))
    assert samples[6][0] == fit["dt_in"][0]
    assert samples[5][0] == fit["dt_in"][1]
    assert samples[7][0] == fit["dt_in"][2]
    assert samples[8][0] == fit["predicted_float32_dt"]
    native_candidates = [
        (fit["dt_in"][index], fit["sse_out"][index]) for index in range(3)
    ] + [(0.0, float(native_trials[0][5])),
         (fit["predicted_float32_dt"], float(native_trials[8][5]))]
    native_selected_dt = min(native_candidates, key=lambda pair: pair[1])[0]
    assert torch_report["selected_dt"] == native_selected_dt
    assert torch_report["native_dt_reference"] == native_selected_dt
    assert torch_report["exact_ordered_vertices"] == torch_report["vertices"]
    assert torch_report["max_abs_coordinate_error_mm"] == 0
    selected_index = next(i for i, (dt, _) in enumerate(samples) if dt == native_selected_dt)
    maximum_error = {
        name: max(abs(terms[i][name] - float(row[j + 1]))
                  for i, row in enumerate(native_trials))
        for j, name in enumerate(TERMS)
    }
    maximum_error["total"] = max(
        abs(samples[i][1] - float(row[5]))
        for i, row in enumerate(native_trials))
    return {
        "hemisphere": hemisphere,
        "native_selected_dt": native_selected_dt,
        "torch_selected_dt": torch_report["selected_dt"],
        "exact_ordered_vertices": torch_report["exact_ordered_vertices"],
        "vertices": torch_report["vertices"],
        "max_abs_coordinate_error_mm": torch_report["max_abs_coordinate_error_mm"],
        "native_sse_print_precision": "six decimal places",
        "paired_line_search_samples": len(samples),
        "max_abs_sse_difference": maximum_error,
        "selected_trial_native_sse": {
            **{name: float(native_trials[selected_index][j + 1])
               for j, name in enumerate(TERMS)},
            "total": float(native_trials[selected_index][5]),
        },
        "selected_trial_torch_sse": {
            **terms[selected_index], "total": samples[selected_index][1],
        },
        "torch_seconds": torch_report["seconds"],
        "native_fit_coefficients_float32": {
            name: fit[name] for name in ("a", "b", "c")
        },
    }


if __name__ == "__main__":
    for side in ("lh", "rh"):
        report = summarize(side)
        path = HERE / f"mris_register_{side}_first_line_parity_control-host.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(side, report["native_selected_dt"],
              report["exact_ordered_vertices"],
              report["max_abs_sse_difference"])
