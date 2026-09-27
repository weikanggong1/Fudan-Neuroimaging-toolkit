"""Audit saved LH geometry, sulc and aparc annotation against one official T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fs
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header(candidate: Path, official: Path) -> dict:
    return {"candidate_realpath": str(candidate.resolve()),
            "candidate_sha256": sha256(candidate),
            "official_sha256": sha256(official)}


def geometry(candidate: Path, official: Path) -> dict:
    a, af = fs.read_geometry(str(candidate))
    b, bf = fs.read_geometry(str(official))
    result = header(candidate, official) | {
        "candidate_vertices": len(a), "official_vertices": len(b),
        "candidate_faces": len(af), "official_faces": len(bf)}
    if a.shape != b.shape or af.shape != bf.shape:
        return result
    error = np.linalg.norm(a.astype(np.float64) - b, axis=1)
    return result | {
        "ordered_faces_equal": bool(np.array_equal(af, bf)),
        "exact_coordinate_components": int(np.count_nonzero(a == b)),
        "coordinate_components": int(a.size),
        "mean_euclidean_mm": float(np.mean(error)),
        "p99_euclidean_mm": float(np.percentile(error, 99)),
        "max_euclidean_mm": float(np.max(error)),
        "vertices_over_0p1_mm": int(np.count_nonzero(error > 0.1)),
    }


def morph(candidate: Path, official: Path) -> dict:
    a, b = fs.read_morph_data(str(candidate)), fs.read_morph_data(str(official))
    result = header(candidate, official) | {
        "candidate_values": len(a), "official_values": len(b)}
    if a.shape != b.shape:
        return result
    error = np.abs(a.astype(np.float64) - b)
    return result | {"exact_values": int(np.count_nonzero(a == b)),
                     "mean_abs": float(np.mean(error)),
                     "p99_abs": float(np.percentile(error, 99)),
                     "max_abs": float(np.max(error))}


def annotation(candidate: Path, official: Path) -> dict:
    a, ac, an = fs.read_annot(str(candidate), orig_ids=True)
    b, bc, bn = fs.read_annot(str(official), orig_ids=True)
    result = header(candidate, official) | {
        "candidate_vertices": len(a), "official_vertices": len(b),
        "color_table_equal": bool(np.array_equal(ac, bc)),
        "names_equal": an == bn}
    if a.shape != b.shape:
        return result
    per_label = {}
    for label in sorted(set(a.tolist()) | set(b.tolist())):
        ca, rb = a == label, b == label
        intersection = int(np.count_nonzero(ca & rb))
        candidate_count, official_count = int(np.count_nonzero(ca)), int(np.count_nonzero(rb))
        per_label[str(label)] = {
            "candidate_vertices": candidate_count,
            "official_vertices": official_count,
            "intersection": intersection,
            "dice": (2 * intersection / (candidate_count + official_count)
                     if candidate_count + official_count else 1.0),
        }
    return result | {"matching_vertices": int(np.count_nonzero(a == b)),
                     "mismatched_vertices": int(np.count_nonzero(a != b)),
                     "per_label": per_label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {"candidate": str(args.candidate.resolve()),
              "official": str(args.official.resolve()), "outputs": {}}
    for name in ("lh.smoothwm", "lh.inflated", "lh.sphere", "lh.sphere.reg"):
        candidate = args.candidate / "surf" / name
        report["outputs"][name] = (
            geometry(candidate, args.official / "surf" / name)
            if candidate.is_file() else {"present": False})
    for name, subdir, function in (("lh.sulc", "surf", morph),
                                   ("lh.aparc.annot", "label", annotation)):
        candidate = args.candidate / subdir / name
        report["outputs"][name] = (
            function(candidate, args.official / subdir / name)
            if candidate.is_file() else {"present": False})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
