"""Replay corrected cortical/WM label IDs on the September 2026 benchmark subject.

The completed reconstruction began before the label-ID correction was added.
This one-time replay is included in its reported duration and provenance.
Fresh runs use native_free.py directly and do not need this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.native_free import _project_parcels, _project_wmparc
from fnit.recon_all.segstats_wmparc_python import write_wmparc_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--launch-source-sha256", required=True)
    args = parser.parse_args()
    report_file = args.subject / "fnit-native-free-run.json"
    report = json.loads(report_file.read_text())
    if report["status"] != "complete" or "projection_replay" in report:
        raise ValueError("expected one complete, not yet replayed subject")
    aseg = np.asarray(nib.load(str(args.subject / "mri/aseg.mgz")).dataobj).astype(np.int16)
    steps = (
        ("corrected_aparc_ids", _project_parcels, (args.subject, aseg)),
        ("corrected_wmparc_ids", _project_wmparc, (args.subject, aseg)),
        ("corrected_wmparc_stats", write_wmparc_stats,
         (args.subject, args.assets / "WMParcStatsLUT.txt", args.subject / "stats/wmparc.stats")),
    )
    original = float(report["total_seconds"])
    for name, operation, inputs in steps:
        started = time.perf_counter()
        operation(*inputs)
        report["stages"].append({"name": name, "seconds": time.perf_counter() - started})
    source = args.source
    report["projection_replay"] = {
        "original_run_seconds": original,
        "source_sha256_at_launch": args.launch_source_sha256,
        "corrected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    report["total_seconds"] = original + sum(row["seconds"] for row in report["stages"][-3:])
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"replayed label IDs; total measured stage time {report['total_seconds']:.1f} s")


if __name__ == "__main__":
    main()
