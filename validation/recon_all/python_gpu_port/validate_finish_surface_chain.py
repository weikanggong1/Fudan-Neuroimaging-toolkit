"""Run final white, pial and vertex maps on a real frozen upstream subject."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.native_free import _finish_cortical_surface


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(candidate: Path, reference: Path, *, surface: bool) -> dict:
    if surface:
        values, faces = fs.read_geometry(str(candidate))
        truth, truth_faces = fs.read_geometry(str(reference))
        if not np.array_equal(faces, truth_faces):
            raise ValueError(f"ordered faces differ: {candidate.name}")
        distance = np.linalg.norm(values.astype(np.float64) - truth, axis=1)
        exact = int(np.count_nonzero(np.all(values == truth, axis=1)))
        unit = "mm"
    else:
        values = fs.read_morph_data(str(candidate))
        truth = fs.read_morph_data(str(reference))
        distance = np.abs(values.astype(np.float64) - truth)
        exact = int(np.count_nonzero(values == truth))
        unit = "map unit"
    if distance.shape != truth.shape[:1]:
        raise ValueError(f"vertex count differs: {candidate.name}")
    return {"vertices": len(distance), "exact_vertices": exact,
            "mean": float(np.mean(distance)), "p99": float(np.quantile(distance, .99)),
            "max": float(np.max(distance)), "unit": unit,
            "candidate_sha256": _sha(candidate), "reference_sha256": _sha(reference)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Completed FreeSurfer subject supplying frozen real inputs and reference outputs")
    parser.add_argument("--output-subject", type=Path, required=True,
                        help="New isolated output subject directory")
    parser.add_argument("--native-binary", type=Path, required=True,
                        help="Conda-built mris_place_surface for final white")
    parser.add_argument("--assets", type=Path, required=True,
                        help="External FreeSurfer template/data directory")
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source, target, hemi = args.reference.resolve(), args.output_subject.resolve(), args.hemi
    if target.exists():
        raise FileExistsError(target)
    inputs = ["mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz",
              f"surf/{hemi}.white.preaparc", f"surf/autodet.gw.stats.{hemi}.dat",
              f"label/{hemi}.cortex.label", f"label/{hemi}.cortex+hipamyg.label",
              f"label/{hemi}.aparc.annot"]
    for name in inputs:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    for name in inputs:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source / name)
    result = _finish_cortical_surface(target, hemi, args.native_binary.resolve(),
                                      args.assets.resolve(), device=args.device,
                                      threads=args.threads)
    surfaces = ("white", "pial.T1", "pial")
    maps = ("thickness", "area", "area.pial", "area.mid", "curv", "curv.pial", "volume")
    comparison = {}
    for name in (*surfaces, *maps):
        candidate = target / "surf" / f"{hemi}.{name}"
        reference = source / "surf" / f"{hemi}.{name}"
        comparison[name] = _compare(candidate, reference, surface=name in surfaces)
    report = {"scope": "single-hemisphere frozen-real-input finishing chain",
              "reference": str(source), "output_subject": str(target), "hemisphere": hemi,
              "inputs_sha256": {name: _sha(source / name) for name in inputs},
              "result": result, "comparison": comparison}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "steps": result["pial_report"]["steps"],
                      "white_max_mm": comparison["white"]["max"],
                      "pial_max_mm": comparison["pial"]["max"]}, indent=2))


if __name__ == "__main__":
    main()
