"""Summarize matched FreeSurfer aparc and aseg ROI rows for a fixed T1 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path: Path, key: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    columns = None
    regions = {}
    measures = {}
    for line in path.read_text().splitlines():
        if line.startswith("# ColHeaders "):
            columns = line.split()[2:]
        elif line.startswith("# Measure Cortex,"):
            fields = [field.strip() for field in line[2:].split(",")]
            if len(fields) >= 4:
                try:
                    measures[fields[1]] = float(fields[3])
                except ValueError:
                    pass
        elif line and not line.startswith("#"):
            if columns is None:
                raise ValueError(f"missing ColHeaders: {path}")
            fields = line.split()
            if len(fields) != len(columns):
                raise ValueError(f"unexpected column count: {path}: {line}")
            values = dict(zip(columns, fields))
            regions[values[key]] = {name: float(value) for name, value in values.items()
                                    if name != key}
    return regions, measures


def compare(reference: Path, candidate: Path, key: str,
            fields: tuple[str, ...]) -> dict:
    a, am = rows(reference, key)
    b, bm = rows(candidate, key)
    if set(a) != set(b):
        raise ValueError(f"ROI names differ: {sorted(set(a) ^ set(b))}")
    per_region = {}
    for name in sorted(a):
        per_region[name] = {field: {"reference": a[name][field],
                                    "candidate": b[name][field],
                                    "signed": b[name][field] - a[name][field]}
                            for field in fields}
    summary = {}
    for field in fields:
        av = np.array([a[name][field] for name in sorted(a)], dtype=float)
        bv = np.array([b[name][field] for name in sorted(a)], dtype=float)
        diff = np.abs(bv - av)
        valid = av != 0
        summary[field] = {"mae": float(diff.mean()),
                          "median_abs_relative_pct": float(np.median(diff[valid] / np.abs(av[valid])) * 100)
                          if np.any(valid) else None,
                          "max_abs": float(diff.max(initial=0)),
                          "exact_regions": int(np.count_nonzero(av == bv))}
    globals_ = {name: {"reference": am[name], "candidate": bm[name],
                       "signed": bm[name] - am[name],
                       "signed_relative_pct": (bm[name] - am[name]) / am[name] * 100
                       if am[name] else None}
                for name in sorted(set(am) & set(bm))}
    return {"regions": len(a), "summary": summary,
            "per_region": per_region, "global": globals_}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = {hemi: compare(args.official / "stats" / f"{hemi}.aparc.stats",
                            args.candidate / "stats" / f"{hemi}.aparc.stats",
                            "StructName", ("ThickAvg", "SurfArea", "GrayVol", "MeanCurv"))
              for hemi in ("lh", "rh")}
    result["aseg"] = compare(args.official / "stats/aseg.stats",
                             args.candidate / "stats/aseg.stats",
                             "StructName", ("Volume_mm3",))
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    for key, value in result.items():
        print(key, value["regions"],
              {field: round(row["mae"], 4) for field, row in value["summary"].items()})


if __name__ == "__main__":
    main()
