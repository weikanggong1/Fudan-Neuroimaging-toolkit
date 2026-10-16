"""Run Python GCA normalization on the connected nu, mask, and LTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fnit.recon_all.ca_normalize_python import run_ca_normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    mri = args.subject_dir / "mri"
    report = run_ca_normalize(
        mri / "nu.mgz", mri / "brainmask.mgz", args.atlas,
        mri / "transforms/talairach.lta", mri / "norm.mgz", mri / "ctrl_pts.mgz")
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"norm": str(mri / "norm.mgz"),
                      "ctrl_pts": str(mri / "ctrl_pts.mgz"),
                      "total_seconds": report["total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
