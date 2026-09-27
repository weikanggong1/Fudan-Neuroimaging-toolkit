"""Compare numeric venous-sinus statistics and their Talairach eTIV sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from fnit.recon_all.sclimbic import _etiv_from_lta


FIELDS = ("Index", "SegId", "NVoxels", "Volume_mm3", "StructName",
          "Mean", "StdDev", "Min", "Max", "Range")


def parsed(path: Path) -> tuple[float, dict[int, dict]]:
    text = path.read_text()
    etiv = float(re.search(r"EstimatedTotalIntraCranialVol.*?, ([0-9.]+), mm", text).group(1))
    rows = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        row = dict(zip(FIELDS, parts, strict=True))
        for key in FIELDS:
            if key not in ("StructName",):
                row[key] = float(row[key])
        rows[int(row["SegId"])] = row
    return etiv, rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    a_stats, b_stats = (root / "stats/vsinus.stats" for root in (args.candidate, args.official))
    a_lta, b_lta = (root / "mri/transforms/talairach.xfm.lta"
                    for root in (args.candidate, args.official))
    a_etiv, a_rows = parsed(a_stats)
    b_etiv, b_rows = parsed(b_stats)
    numeric = ("Index", "SegId", "NVoxels", "Volume_mm3", "Mean",
               "StdDev", "Min", "Max", "Range")
    report = {
        "stats_sha256_candidate": sha256(a_stats),
        "stats_sha256_official": sha256(b_stats),
        "numeric_rows": len(a_rows),
        "segids_equal": sorted(a_rows) == sorted(b_rows),
        "struct_names_equal": all(a_rows[k]["StructName"] == b_rows[k]["StructName"]
                                  for k in a_rows),
        "numeric_row_max_abs": {
            key: max(abs(a_rows[k][key] - b_rows[k][key]) for k in a_rows)
            for key in numeric
        },
        "etiv_candidate_mm3": a_etiv,
        "etiv_official_mm3": b_etiv,
        "etiv_delta_mm3": a_etiv - b_etiv,
        "etiv_relative_percent": 100 * (a_etiv - b_etiv) / b_etiv,
        "talairach_lta_sha256_candidate": sha256(a_lta),
        "talairach_lta_sha256_official": sha256(b_lta),
        "python_formula_from_candidate_lta_mm3": _etiv_from_lta(a_lta),
        "python_formula_from_official_lta_mm3": _etiv_from_lta(b_lta),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
