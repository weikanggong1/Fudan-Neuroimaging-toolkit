"""Compare the first five white-placement iterations to pinned FreeSurfer RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from validate_white_preaparc_first_step import STATE, _compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--probe-prefix", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    candidate = np.load(args.diagnostics)
    native = {}
    pattern = re.compile(r"PY_OBJ_REF step([1-5]) rms=([^ ]+) sse=([^ ]+)")
    for line in args.native_log.read_text().splitlines():
        match = pattern.search(line)
        if match and int(match.group(1)) not in native:
            native[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))
    records = []
    for step in range(1, 6):
        prefix = f"{args.probe_prefix}.step{step:02d}"
        stages = {}
        for name, source_stage, columns in (
            ("initial", "clear", slice(0, 3)),
            ("tangential_spring", "tangential_spring", slice(6, 9)),
            ("after_collision", "after_collision", slice(0, 3)),
        ):
            reference = np.fromfile(f"{prefix}.{source_stage}", dtype=STATE)
            stages[name] = _compare(candidate[f"step{step}_{name}"], reference["floats"][:, columns])
        rms, sse = native[step]
        records.append({
            "step": step, "stages": stages,
            "native_sse": sse, "python_sse": float(candidate[f"step{step}_sse"]),
            "sse_absolute": abs(sse - float(candidate[f"step{step}_sse"])),
            "native_rms": rms, "python_rms": float(candidate[f"step{step}_rms"]),
            "rms_absolute": abs(rms - float(candidate[f"step{step}_rms"])),
        })
    first_difference = next(
        (f"step{record['step']}.{name}" for record in records
         for name, stage in record["stages"].items()
         if stage["exact_elements"] != stage["elements"]), None,
    )
    report = {"scope": "first five iterations of first white.preaparc pass on frozen real T1",
              "reference": "instrumented pinned FreeSurfer 8.2 source",
              "first_difference": first_difference, "steps": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
