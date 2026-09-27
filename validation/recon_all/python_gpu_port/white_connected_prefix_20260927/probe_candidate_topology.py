"""Replay LH topology on candidate MRI and saved candidate nofix surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry

from fnit.recon_all.mris_remesh_python import remesh_surface
from fnit.recon_all.mris_remove_intersection_python import remove_intersection_surface
from fnit.recon_all.topology_conda_ga import run_topology_ga_conda


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_mri(candidate: Path, official: Path) -> dict:
    a, b = nib.load(str(candidate)), nib.load(str(official))
    av, bv = np.asarray(a.dataobj), np.asarray(b.dataobj)
    return {
        "candidate_realpath": str(candidate.resolve()),
        "candidate_sha256": sha256(candidate),
        "official_sha256": sha256(official),
        "different_voxels": int(np.count_nonzero(av != bv)),
        "affine_max_abs_mm": float(np.max(np.abs(a.affine - b.affine))),
    }


def paired_surface(candidate: Path, official: Path) -> dict:
    a, af = read_geometry(str(candidate))
    b, bf = read_geometry(str(official))
    if a.shape != b.shape or af.shape != bf.shape:
        return {"candidate_vertices": len(a), "official_vertices": len(b),
                "candidate_faces": len(af), "official_faces": len(bf),
                "shape_equal": False}
    distance = np.linalg.norm(a.astype(np.float64) - b, axis=1)
    return {"vertices": len(a), "faces": len(af),
            "candidate_realpath": str(candidate.resolve()),
            "candidate_sha256": sha256(candidate),
            "official_sha256": sha256(official),
            "ordered_faces_equal": bool(np.array_equal(af, bf)),
            "exact_coordinate_components": int(np.count_nonzero(a == b)),
            "coordinate_components": int(a.size),
            "mean_euclidean_mm": float(np.mean(distance)),
            "p99_euclidean_mm": float(np.percentile(distance, 99)),
            "max_euclidean_mm": float(np.max(distance))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "candidate_mri", "candidate_aux", "initial_surface",
                 "official", "binary", "assets", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    subject = args.subject.resolve()
    if subject.exists():
        raise FileExistsError(subject)
    for subdir in ("mri", "surf", "scripts"):
        (subject / subdir).mkdir(parents=True, exist_ok=True)
    mri_sources = {name: args.candidate_mri / name for name in
                   ("brain.mgz", "wm.mgz", "filled.mgz", "norm.mgz",
                    "aseg.presurf.mgz")}
    mri_sources.update({name: args.candidate_aux / "mri" / name for name in
                        ("mca-dura.mgz", "vsinus.mgz", "brain.finalsurfs.mgz")})
    for name, source in mri_sources.items():
        (subject / "mri" / name).symlink_to(source.resolve())
    for name in ("orig.nofix", "smoothwm.nofix", "inflated.nofix", "qsphere.nofix"):
        (subject / "surf" / f"lh.{name}").symlink_to(
            (args.initial_surface / "surf" / f"lh.{name}").resolve())
    official = args.official.resolve()
    report = {"subject": str(subject), "official": str(official),
              "candidate_mri": {}, "initial_surfaces": {}}
    for name in ("brain.mgz", "wm.mgz", "filled.mgz", "norm.mgz"):
        candidate = subject / "mri" / name
        if candidate.resolve().is_relative_to(official):
            raise ValueError(f"official input leaked into candidate: {candidate}")
        result = paired_mri(candidate, official / "mri" / name)
        report["candidate_mri"][name] = result
        if result["different_voxels"] or result["affine_max_abs_mm"]:
            raise ValueError(f"nonmatching candidate MRI: {name}")
    for name in ("orig.nofix", "inflated.nofix", "qsphere.nofix"):
        candidate = subject / "surf" / f"lh.{name}"
        if candidate.resolve().is_relative_to(official):
            raise ValueError(f"official input leaked into candidate: {candidate}")
        report["initial_surfaces"][name] = paired_surface(
            candidate, official / "surf" / f"lh.{name}")
    report["topology"] = run_topology_ga_conda(
        subject, "lh", args.binary, args.assets)
    report["orig_premesh"] = paired_surface(
        subject / "surf/lh.orig.premesh", official / "surf/lh.orig.premesh")
    if (report["orig_premesh"].get("exact_coordinate_components") !=
            report["orig_premesh"].get("coordinate_components") or
            not report["orig_premesh"].get("ordered_faces_equal")):
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        return
    start = time.perf_counter()
    remesh_surface(subject / "surf/lh.orig.premesh", subject / "surf/lh.orig",
                   iterations=3)
    report["remesh_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    report["intersection_counts"] = remove_intersection_surface(
        subject / "surf/lh.orig", subject / "surf/lh.orig")
    report["intersection_seconds"] = time.perf_counter() - start
    report["orig"] = paired_surface(subject / "surf/lh.orig",
                                    official / "surf/lh.orig")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.report)


if __name__ == "__main__":
    main()
