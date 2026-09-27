"""Check the runner's Python hypointensity/ribbon/aseg chain on real frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from fnit.recon_all.relabel_hypointensities_python import relabel_volume
from fnit.recon_all.surf2volseg_fix_python import fix_presurf_volume
from fnit.recon_all.volmask_python import write_ribbon


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-subject", type=Path, required=True)
    parser.add_argument("--color-lut", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, target = args.reference.resolve(), args.output_subject.resolve()
    if target.exists():
        raise FileExistsError(target)
    inputs = ["mri/aseg.presurf.mgz"]
    for hemi in ("lh", "rh"):
        inputs += [f"surf/{hemi}.white", f"surf/{hemi}.pial",
                   f"label/{hemi}.cortex.label"]
    for name in inputs:
        original = source / name
        if not original.is_file():
            raise FileNotFoundError(original)
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(original)
    mri, surf, labels = (target / name for name in ("mri", "surf", "label"))
    timings = {}
    started = time.perf_counter()
    write_ribbon(mri / "aseg.presurf.mgz", surf, mri, args.color_lut)
    timings["ribbon"] = time.perf_counter() - started
    started = time.perf_counter()
    relabel_volume(mri / "aseg.presurf.mgz", surf, mri / "aseg.presurf.hypos.mgz")
    timings["relabel_hypointensities"] = time.perf_counter() - started
    started = time.perf_counter()
    fix_presurf_volume(mri / "aseg.presurf.hypos.mgz", mri / "ribbon.mgz",
                       surf, labels, mri / "aseg.mgz")
    timings["aseg_ribbon_fix"] = time.perf_counter() - started
    names = ("aseg.presurf.hypos.mgz", "ribbon.mgz", "lh.ribbon.mgz",
             "rh.ribbon.mgz", "aseg.mgz")
    comparisons = {}
    for name in names:
        candidate, reference = mri / name, source / "mri" / name
        a, b = nib.load(str(candidate)), nib.load(str(reference))
        aa, bb = np.asarray(a.dataobj), np.asarray(b.dataobj)
        if aa.shape != bb.shape:
            raise ValueError(f"different shape: {name}")
        comparisons[name] = {"voxels": int(aa.size),
                             "different_voxels": int(np.count_nonzero(aa != bb)),
                             "max_absolute_difference": int(np.max(np.abs(aa.astype(np.int32)-bb))),
                             "affine_max_difference": float(np.max(np.abs(a.affine-b.affine))),
                             "candidate_sha256": _sha(candidate),
                             "reference_sha256": _sha(reference)}
    result = {"scope": "frozen real T1 aseg/ribbon chain",
              "reference": str(source), "output_subject": str(target),
              "inputs_sha256": {name: _sha(source / name) for name in inputs},
              "color_lut_sha256": _sha(args.color_lut),
              "timings_seconds": timings, "comparison": comparisons}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"timings_seconds": timings,
                      "differences": {name: data["different_voxels"]
                                      for name, data in comparisons.items()}}, indent=2))


if __name__ == "__main__":
    main()
