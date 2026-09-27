"""Compare frozen-input pial.T1 placement with the archived real-T1 subject."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fs
import numpy as np


INPUTS = (
    "mri/brain.finalsurfs.mgz", "mri/wm.mgz", "mri/aseg.presurf.mgz",
    "surf/lh.white", "surf/autodet.gw.stats.lh.dat",
    "label/lh.cortex.label", "label/lh.cortex+hipamyg.label",
    "label/lh.aparc.annot",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(candidate: Path, official: Path, log: Path) -> dict:
    candidate, official = candidate.resolve(), official.resolve()
    inputs = {}
    for name in INPUTS:
        left, right = candidate / name, official / name
        inputs[name] = {
            "candidate_resolved": str(left.resolve()),
            "official_resolved": str(right.resolve()),
            "borrowed_official_path": left.resolve() == right.resolve(),
            "candidate_sha256": sha256(left),
            "official_sha256": sha256(right),
        }
        inputs[name]["bytes_equal"] = (
            inputs[name]["candidate_sha256"] == inputs[name]["official_sha256"]
        )
    surface, reference = (root / "surf/lh.pial.T1"
                          for root in (candidate, official))
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
    stage = None
    for line in log.read_text().splitlines():
        if line.startswith("{'output':"):
            stage = ast.literal_eval(line)
    if stage is None:
        raise ValueError(f"wrapper result not found in {log}")
    return {"candidate": str(candidate), "official": str(official),
            "inputs": inputs, "pial_t1": geometry, "stage": stage}


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
                      "pial_t1": result["pial_t1"]}, indent=2))


if __name__ == "__main__":
    main()
