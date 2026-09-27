"""Compare the connected same-T1 white-preaparc prefix with FreeSurfer 8.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_files(candidate: Path, official: Path) -> dict:
    return {"candidate_path": str(candidate.resolve()),
            "official_path": str(official.resolve()),
            "candidate_sha256": sha256(candidate),
            "official_sha256": sha256(official)}


def surface(candidate: Path, official: Path) -> dict:
    xyz, faces = read_geometry(str(candidate))
    ref_xyz, ref_faces = read_geometry(str(official))
    if xyz.shape != ref_xyz.shape or faces.shape != ref_faces.shape:
        raise ValueError(f"surface shape mismatch: {candidate}")
    distances = np.linalg.norm(xyz.astype(np.float64) - ref_xyz, axis=1)
    return {
        "vertices": len(xyz), "faces": len(faces),
        "ordered_faces_equal": bool(np.array_equal(faces, ref_faces)),
        "exact_coordinate_components": int(np.count_nonzero(xyz == ref_xyz)),
        "coordinate_components": int(xyz.size),
        "mean_euclidean_mm": float(np.mean(distances)),
        "p99_euclidean_mm": float(np.percentile(distances, 99)),
        "max_euclidean_mm": float(np.max(distances)),
        "vertices_over_0p1_mm": int(np.count_nonzero(distances > 0.1)),
    }


def volume(candidate: Path, official: Path) -> dict:
    image, reference = nib.load(str(candidate)), nib.load(str(official))
    a, b = np.asarray(image.dataobj), np.asarray(reference.dataobj)
    if a.shape != b.shape:
        raise ValueError(f"volume shape mismatch: {candidate}")
    return {
        "shape": list(a.shape),
        "mismatched_voxels": int(np.count_nonzero(a != b)),
        "affine_max_abs_mm": float(np.max(np.abs(image.affine - reference.affine))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("upstream_subject", type=Path)
    parser.add_argument("upstream_report", type=Path)
    parser.add_argument("topology_report", type=Path)
    parser.add_argument("placement_time", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidate, official = args.candidate, args.official
    upstream = json.loads(args.upstream_report.read_text())
    topology = json.loads(args.topology_report.read_text())
    place = json.loads(args.placement_time.read_text())
    topology_subject = Path(topology["hemispheres"]["lh"]["outputs"]["orig"]).parents[1]
    report = {
        "scope": "saved stages from one real T1; LH placement only; see topology MRI provenance",
        "candidate": str(candidate), "official": str(official),
        "upstream_timings_seconds": upstream["timings_seconds"],
        "topology_initial_lh_stages": topology["hemispheres"]["lh"]["stages"],
        "placement_timings_seconds": {
            "thresholds": place["stats_seconds"],
            "white_preaparc": place["place_seconds"],
            "smoothwm_three_passes": place["smoothwm_seconds"],
        },
        "volumes": {
            name: volume(args.upstream_subject / "mri" / name,
                         official / "mri" / name)
            for name in ("mca-dura.mgz", "vsinus.mgz", "brain.finalsurfs.mgz")
        } | {"mrisps.wpa.mgz": volume(candidate / "mri/mrisps.wpa.mgz",
                                      official / "mri/mrisps.wpa.mgz")},
        "surfaces": {name: surface(candidate / "surf" / f"lh.{name}",
                                   official / "surf" / f"lh.{name}")
                     for name in ("orig", "orig.premesh", "white.preaparc",
                                  "smoothwm")},
        "file_provenance": {
            name: paired_files(candidate / subdir / name,
                               official / subdir / name)
            for subdir, names in (
                ("mri", ("wm.mgz", "aseg.presurf.mgz", "brain.finalsurfs.mgz",
                         "mrisps.wpa.mgz")),
                ("surf", ("lh.orig", "lh.orig.premesh", "lh.white.preaparc",
                          "lh.smoothwm", "autodet.gw.stats.lh.dat")))
            for name in names
        },
        "topology_mri_provenance": {
            name: paired_files(topology_subject / "mri" / name,
                               official / "mri" / name)
            for name in ("brain.mgz", "wm.mgz", "filled.mgz", "norm.mgz")
        },
        "auxiliary_file_provenance": {
            name: paired_files(args.upstream_subject / "mri" / name,
                               official / "mri" / name)
            for name in ("mca-dura.mgz", "vsinus.mgz")
        },
        "threshold_stats_text_equal": (
            (candidate / "surf/autodet.gw.stats.lh.dat").read_bytes() ==
            (official / "surf/autodet.gw.stats.lh.dat").read_bytes()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
