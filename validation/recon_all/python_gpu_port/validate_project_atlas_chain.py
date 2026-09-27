"""Check the runner's four atlas-volume outputs on frozen real recon-all inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.native_free import _project_parcels, _project_wmparc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-subject", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    source, target = args.reference.resolve(), args.output_subject.resolve()
    if target.exists():
        raise FileExistsError(target)
    inputs = ["mri/aseg.mgz"]
    for hemi in ("lh", "rh"):
        inputs += [f"surf/{hemi}.white", f"surf/{hemi}.pial",
                   f"label/{hemi}.cortex.label"]
        inputs += [f"label/{hemi}.{atlas}.annot" for atlas in
                   ("aparc", "aparc.a2009s", "aparc.DKTatlas")]
    for name in inputs:
        original = source / name
        if not original.is_file():
            raise FileNotFoundError(original)
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(original)
    started = time.perf_counter()
    _project_parcels(target)
    cortex_seconds = time.perf_counter() - started
    started = time.perf_counter()
    _project_wmparc(target)
    wm_seconds = time.perf_counter() - started
    files = ("aparc+aseg.mgz", "aparc.a2009s+aseg.mgz",
             "aparc.DKTatlas+aseg.mgz", "wmparc.mgz")
    comparisons = {}
    for name in files:
        candidate, reference = target / "mri" / name, source / "mri" / name
        a, b = nib.load(str(candidate)), nib.load(str(reference))
        aa, bb = np.asarray(a.dataobj), np.asarray(b.dataobj)
        if aa.shape != bb.shape:
            raise ValueError(f"different shape for {name}")
        comparisons[name] = {"voxels": int(aa.size),
                             "different_voxels": int(np.count_nonzero(aa != bb)),
                             "max_absolute_difference": int(np.max(np.abs(aa.astype(np.int32)-bb))),
                             "affine_max_difference": float(np.max(np.abs(a.affine-b.affine))),
                             "candidate_sha256": _sha(candidate),
                             "reference_sha256": _sha(reference)}
    report = {"scope": "four atlas volumes on frozen real subject inputs",
              "reference": str(source), "output_subject": str(target),
              "inputs_sha256": {name: _sha(source / name) for name in inputs},
              "cortex_three_seconds": cortex_seconds, "wmparc_seconds": wm_seconds,
              "comparison": comparisons}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report),
                      "differences": {name: item["different_voxels"] for name, item in comparisons.items()},
                      "seconds": cortex_seconds + wm_seconds}, indent=2))


if __name__ == "__main__":
    main()
