"""按同名脑区比较真实 recon-all 的 aparc、aseg 与 wmparc 统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FILES = ("stats/lh.aparc.stats", "stats/rh.aparc.stats",
         "stats/aseg.stats", "stats/wmparc.stats", "stats/brainvol.stats")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, float]]:
    """读取 ColHeaders 行和具名全局 Measure；缺字段时直接失败。"""
    headers = None
    rows: dict[str, dict[str, str]] = {}
    measures: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if line.startswith("# ColHeaders "):
            headers = line.split()[2:]
        elif line.startswith("# Measure "):
            fields = [part.strip() for part in line[len("# Measure "):].split(",")]
            if len(fields) >= 4:
                try:
                    measures[fields[1]] = float(fields[3])
                except ValueError:
                    pass
        elif line and not line.startswith("#"):
            if headers is None:
                raise ValueError(f"missing ColHeaders: {path}")
            fields = line.split()
            if len(fields) < len(headers):
                raise ValueError(f"short row in {path}: {line}")
            row = dict(zip(headers, fields))
            name = row.get("StructName")
            if not name or name in rows:
                raise ValueError(f"missing or duplicate StructName in {path}: {line}")
            rows[name] = row
    return rows, measures


def _metric(reference: dict[str, dict[str, str]],
            candidate: dict[str, dict[str, str]], field: str) -> dict:
    """按名称配对给出相关性及绝对、相对误差；不配对网格顶点。"""
    names = sorted(reference.keys() & candidate.keys())
    if not names:
        raise ValueError(f"no matched regions for {field}")
    a = np.array([float(reference[name][field]) for name in names], np.float64)
    b = np.array([float(candidate[name][field]) for name in names], np.float64)
    error = np.abs(a - b)
    nonzero = a != 0
    relative = 100 * error[nonzero] / np.abs(a[nonzero])
    regions = {name: {"reference": float(first), "candidate": float(second),
                      "absolute_error": float(abs(second - first)),
                      "signed_relative_difference_percent":
                      float(100 * (second - first) / first) if first else None}
               for name, first, second in zip(names, a, b)}
    correlation = (float(np.corrcoef(a, b)[0, 1]) if len(names) > 1 and
                   np.std(a) > 0 and np.std(b) > 0 else None)
    return {"matched_regions": len(names),
            "missing_in_candidate": sorted(reference.keys() - candidate.keys()),
            "extra_in_candidate": sorted(candidate.keys() - reference.keys()),
            "pearson_r": correlation, "mae": float(error.mean()),
            "median_absolute_relative_error_percent": float(np.median(relative))
            if relative.size else None,
            "p90_absolute_relative_error_percent": float(np.quantile(relative, .9))
            if relative.size else None,
            "maximum_absolute_error": float(error.max()),
            "zero_reference_regions": [name for name, value in zip(names, a) if value == 0],
            "worst_regions_by_relative_error": sorted(
                (name for name in names if regions[name]["reference"] != 0),
                key=lambda name: abs(regions[name]["signed_relative_difference_percent"]),
                reverse=True)[:10],
            "per_region": regions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    read = {}
    file_hashes = {}
    for relative in FILES:
        first, second = args.reference / relative, args.candidate / relative
        read[relative] = (_rows(first), _rows(second))
        file_hashes[relative] = {"reference": _sha256(first),
                                 "candidate": _sha256(second)}
    aparc_ref = {}
    aparc_got = {}
    for hemi in ("lh", "rh"):
        (ref, _), (got, _) = read[f"stats/{hemi}.aparc.stats"]
        aparc_ref.update({f"{hemi}/{name}": row for name, row in ref.items()})
        aparc_got.update({f"{hemi}/{name}": row for name, row in got.items()})
    aseg_ref, aseg_got = (part[0] for part in read["stats/aseg.stats"])
    wm_ref, wm_got = (part[0] for part in read["stats/wmparc.stats"])
    globals_ref, globals_got = (part[1] for part in read["stats/brainvol.stats"])
    global_rows = {}
    for key in sorted(globals_ref.keys() & globals_got.keys()):
        a, b = globals_ref[key], globals_got[key]
        global_rows[key] = {"reference": a, "candidate": b,
                            "relative_difference_percent":
                            100 * (b - a) / a if a else None}
    report = {"code_commit": args.code_commit,
              "reference": str(args.reference), "candidate": str(args.candidate),
              "comparator_sha256": _sha256(Path(__file__)),
              "input_sha256": file_hashes,
              "aparc_68": {field: _metric(aparc_ref, aparc_got, field)
                           for field in ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv")},
              "aseg": _metric(aseg_ref, aseg_got, "Volume_mm3"),
              "wmparc": _metric(wm_ref, wm_got, "Volume_mm3"),
              "missing_global_measures": sorted(globals_ref.keys() - globals_got.keys()),
              "global_brainvol_measures": global_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"aparc matched: {report['aparc_68']['SurfArea']['matched_regions']}; "
          f"aseg matched: {report['aseg']['matched_regions']}; "
          f"wmparc matched: {report['wmparc']['matched_regions']}")


if __name__ == "__main__":
    main()
