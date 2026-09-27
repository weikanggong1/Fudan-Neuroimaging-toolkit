"""Compare installed FreeSurfer first-pass RAM mesh with pinned source and Python."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from validate_white_preaparc_first_step import STATE, _compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-surface", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--python-diagnostics", type=Path, required=True)
    parser.add_argument("--installed-capture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.installed_capture.read_text())
    if not (capture["niterations_patch_seen"] and capture["first_pass_stop_seen"]
            and capture["output_exists"] and capture["gdb_status"] == 0):
        raise RuntimeError("installed FreeSurfer first-pass RAM capture did not complete")
    installed, _ = nib.freesurfer.read_geometry(str(args.installed_surface))
    source = np.fromfile(args.source_state, dtype=STATE)["floats"][:, :3]
    with np.load(args.python_diagnostics) as diagnostics:
        candidate = diagnostics["step17_after_collision"]
    if installed.shape != source.shape or candidate.shape != source.shape:
        raise ValueError("first-pass meshes have different vertex shapes")
    installed32 = np.asarray(installed, dtype=np.float32)
    paths = (args.installed_surface, args.source_state, args.python_diagnostics)
    report = {
        "scope": "same frozen real T1, left white first-pass step 17 RAM mesh",
        "vertices": len(installed),
        "files_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
        "installed_binary_sha256": capture["binary_sha256"],
        "source_vs_installed": _compare(source, installed32),
        "python_vs_installed": _compare(candidate, installed32),
        "python_vs_source": _compare(candidate, source),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
