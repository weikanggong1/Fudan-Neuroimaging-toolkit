"""Compare the candidate MRI inputs with archived FreeSurfer on one real T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


FILES = ("orig.mgz", "nu.mgz", "synthseg.rca.mgz", "brain.mgz",
         "brainmask.mgz", "entowm.mgz", "aseg.presurf.mgz")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {}
    for name in FILES:
        a_path, b_path = args.candidate / "mri" / name, args.official / "mri" / name
        a, b = nib.load(str(a_path)), nib.load(str(b_path))
        report[name] = {
            "shape": [int(v) for v in a.shape],
            "mismatched_voxels": int(np.count_nonzero(
                np.asarray(a.dataobj) != np.asarray(b.dataobj))),
            "affine_max_abs": float(np.max(np.abs(a.affine - b.affine))),
            "candidate_sha256": sha256(a_path),
            "official_sha256": sha256(b_path),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({name: row["mismatched_voxels"] for name, row in report.items()})


if __name__ == "__main__":
    main()
