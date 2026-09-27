"""Compare a final-white stage with the archived FreeSurfer real-T1 subject."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fs
import numpy as np


INPUTS = (
    "mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz",
    "surf/lh.white.preaparc", "surf/autodet.gw.stats.lh.dat",
    "label/lh.cortex.label", "label/lh.aparc.annot",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(candidate: Path, official: Path, log: Path) -> dict:
    candidate, official = candidate.resolve(), official.resolve()
    files = {}
    for name in INPUTS:
        left, right = candidate / name, official / name
        files[name] = {
            "candidate_resolved": str(left.resolve()),
            "official_resolved": str(right.resolve()),
            "borrowed_official_path": left.resolve() == right.resolve(),
            "candidate_sha256": sha256(left),
            "official_sha256": sha256(right),
        }
        files[name]["bytes_equal"] = (
            files[name]["candidate_sha256"] == files[name]["official_sha256"]
        )
    surface, reference = candidate / "surf/lh.white", official / "surf/lh.white"
    xyz, faces, meta = fs.read_geometry(str(surface), read_metadata=True)
    ref_xyz, ref_faces, ref_meta = fs.read_geometry(
        str(reference), read_metadata=True)
    same_vertices = xyz.shape == ref_xyz.shape
    distance = np.linalg.norm(xyz - ref_xyz, axis=1) if same_vertices else None
    geometry = {
        "candidate_sha256": sha256(surface),
        "official_sha256": sha256(reference),
        "vertices": len(xyz),
        "reference_vertices": len(ref_xyz),
        "faces": len(faces),
        "reference_faces": len(ref_faces),
        "ordered_faces_equal": bool(np.array_equal(faces, ref_faces)),
        "volume_info_equal": bool(
            meta.keys() == ref_meta.keys() and all(
                np.array_equal(meta[key], ref_meta[key]) for key in meta)),
    }
    if distance is not None:
        geometry.update({
            "exact_coordinate_components": int(np.count_nonzero(xyz == ref_xyz)),
            "total_coordinate_components": int(xyz.size),
            "mean_euclidean_mm": float(np.mean(distance)),
            "p50_euclidean_mm": float(np.quantile(distance, .5)),
            "p95_euclidean_mm": float(np.quantile(distance, .95)),
            "p99_euclidean_mm": float(np.quantile(distance, .99)),
            "max_euclidean_mm": float(np.max(distance)),
            "vertices_over_0_1_mm": int(np.count_nonzero(distance > .1)),
        })
    volume, ref_volume = (nib.load(str(root / "mri/mrisps.white.mgz"))
                          for root in (candidate, official))
    values, ref_values = (np.asanyarray(img.dataobj)
                          for img in (volume, ref_volume))
    diagnostic = {
        "candidate_sha256": sha256(candidate / "mri/mrisps.white.mgz"),
        "official_sha256": sha256(official / "mri/mrisps.white.mgz"),
        "shape_equal": values.shape == ref_values.shape,
        "affine_max_abs_mm": float(np.max(np.abs(
            volume.affine - ref_volume.affine))),
        "differing_voxels": int(np.count_nonzero(values != ref_values)),
        "voxels": int(values.size),
    }
    stage = None
    for line in log.read_text().splitlines():
        if line.startswith("{'output':"):
            stage = ast.literal_eval(line)
    if stage is None:
        raise ValueError(f"wrapper result not found in {log}")
    return {"candidate": str(candidate), "official": str(official),
            "inputs": files, "white": geometry, "mrisps_white": diagnostic,
            "stage": stage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.candidate, args.official, args.log)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"seconds": result["stage"]["seconds"],
                      "white": result["white"],
                      "mrisps_white": result["mrisps_white"]}, indent=2))


if __name__ == "__main__":
    main()
