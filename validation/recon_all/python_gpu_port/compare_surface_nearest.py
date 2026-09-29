"""Measure geometric proximity when two real-subject surface meshes differ in topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nibabel.freesurfer.io import read_geometry
import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    """Read both subject trees and write bidirectional nearest-vertex distances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = {"reference": str(args.reference), "candidate": str(args.candidate),
              "unit": "surface RAS mm", "interpretation":
              "Nearest-vertex distance diagnoses geometry; it does not pair vertices."}
    for hemi in ("lh", "rh"):
        result[hemi] = {}
        for name in ("orig", "white", "pial"):
            reference, reference_faces = read_geometry(
                str(args.reference / "surf" / f"{hemi}.{name}"))
            candidate, candidate_faces = read_geometry(
                str(args.candidate / "surf" / f"{hemi}.{name}"))
            a = cKDTree(reference).query(candidate, workers=4)[0]
            b = cKDTree(candidate).query(reference, workers=4)[0]
            result[hemi][name] = {
                "reference_vertices": len(reference),
                "candidate_vertices": len(candidate),
                "reference_faces": len(reference_faces),
                "candidate_faces": len(candidate_faces),
                "candidate_to_reference_mean_mm": float(a.mean()),
                "candidate_to_reference_p99_mm": float(np.quantile(a, .99)),
                "reference_to_candidate_mean_mm": float(b.mean()),
                "reference_to_candidate_p99_mm": float(np.quantile(b, .99)),
            }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
