"""Check final pial-derived vertex maps against a frozen official subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.surface_area_gpu import area_map
from fnit.recon_all.surface_curvature_gpu import curvature_map
from fnit.recon_all.surface_roi_gpu import vertex_volume_map
from fnit.recon_all.surface_thickness_gpu import thickness_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("hemisphere", choices=("lh", "rh"))
    parser.add_argument("candidate_pial", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    hemi = args.hemisphere
    white = args.subject / "surf" / f"{hemi}.white"
    label = args.subject / "label" / f"{hemi}.cortex.label"
    area_map(args.candidate_pial, output / f"{hemi}.area.pial", device="cpu")
    curvature_map(args.candidate_pial, output / f"{hemi}.curv.pial", device="cpu")
    thickness_map(white, args.candidate_pial, output / f"{hemi}.thickness", device="cpu")
    vertex_volume_map(white, args.candidate_pial, label,
                      output / f"{hemi}.volume", device="cpu")
    result = {"subject": str(args.subject), "hemisphere": hemi,
              "candidate_pial": str(args.candidate_pial), "maps": {}}
    for suffix in ("area.pial", "curv.pial", "thickness", "volume"):
        reference = fs.read_morph_data(str(args.subject / "surf" / f"{hemi}.{suffix}"))
        candidate = fs.read_morph_data(str(output / f"{hemi}.{suffix}"))
        if reference.shape != candidate.shape:
            raise ValueError(f"{suffix}: vertex count differs")
        left, right = reference.astype(np.float32), candidate.astype(np.float32)
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        tolerance = (0.001 if suffix == "area.pial" else 0.005) + 0.001 * np.abs(left)
        result["maps"][suffix] = {
            "vertices": len(left),
            "over_established_tolerance": int(np.count_nonzero(delta > tolerance)),
            "exact_float32_values": int(np.count_nonzero(left.view(np.uint32)
                                                           == right.view(np.uint32))),
            "maximum_absolute_error": float(delta.max()),
            "p99_absolute_error": float(np.quantile(delta, .99)),
        }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["maps"], indent=2))


if __name__ == "__main__":
    main()
