"""Read-only MNI305 Talairach first-difference audit on archived real T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import surfa as sf

from fnit.recon_all.sclimbic import _etiv_from_lta


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    a, b = args.candidate / "mri", args.official / "mri"
    x, y = nib.load(str(a / "synthstrip.mgz")), nib.load(str(b / "synthstrip.mgz"))
    report = {
        "synthstrip": {
            "shape": [int(v) for v in x.shape],
            "voxel_mismatches": int(np.count_nonzero(
                np.asarray(x.dataobj) != np.asarray(y.dataobj))),
            "affine_max_abs": float(np.max(np.abs(x.affine - y.affine))),
            "candidate_sha256": sha256(a / "synthstrip.mgz"),
            "official_sha256": sha256(b / "synthstrip.mgz"),
        }
    }
    for name, file in (("aff", "transforms/synthmorph.mni305/aff.lta"),
                       ("voxel", "transforms/talairach.xfm.lta")):
        c, o = sf.load_affine(str(a / file)), sf.load_affine(str(b / file))
        report[name] = {
            "candidate_sha256": sha256(a / file),
            "official_sha256": sha256(b / file),
            "matrix_max_abs": float(np.max(np.abs(c.matrix - o.matrix))),
            "etiv_candidate_mm3": float(_etiv_from_lta(a / file)),
            "etiv_official_mm3": float(_etiv_from_lta(b / file)),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
