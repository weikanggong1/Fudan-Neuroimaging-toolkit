"""Separate coordinate drift from objective drift at white third-pass steps 33–34."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from compare_white_second_pass_boundary import VERTEX_DTYPE
from fnit.recon_all.place_surface_curvature import two_ring_neighbors
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.place_surface_normals import initial_vertex_normals
from fnit.recon_all.place_surface_objective import (
    intensity_error, surface_total_area, tangential_spring_energy,
)
from fnit.recon_all.place_surface_self_repulsion import (
    mean_vertex_spacing, self_repulsion_energy, vertex_buckets_current,
)
from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
from fnit.recon_all.place_surface_volume import prepare_placement_volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--boundary-raw", type=Path, required=True)
    parser.add_argument("--boundary-npz", type=Path, required=True)
    parser.add_argument("--official-steps", type=Path, required=True)
    parser.add_argument("--python-steps-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    initial = np.frombuffer(args.boundary_raw.read_bytes(), dtype=VERTEX_DTYPE)
    with np.load(args.boundary_npz) as data:
        ripped = data["ripped"].astype(bool)
        values = data["values"]
    _, faces, metadata = nib.freesurfer.read_geometry(
        str(args.subject / "surf/lh.orig"), read_metadata=True)
    brain = nib.load(str(args.subject / "mri/brain.finalsurfs.mgz"))
    wm = nib.load(str(args.subject / "mri/wm.mgz"))
    stats = dict(line.split()[:2] for line in
                 (args.subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, _ = prepare_placement_volume(
        np.asarray(brain.dataobj), np.asarray(wm.dataobj),
        surface="white", mid_gray=float(stats["MID_GRAY"]))
    affine = surface_ras_to_voxel(brain.header, metadata)
    ordered_indices, ordered_valid, _ = _ordered_neighbors(faces, len(initial))
    ordered = (ordered_indices, ordered_valid)
    two_offsets, two_neighbors = two_ring_neighbors(
        faces, len(initial), ordered_neighbors=ordered)
    original_area = surface_total_area(initial["xyz"], faces)
    prepared_at = time.perf_counter()

    def terms(xyz: np.ndarray) -> dict:
        normals = initial_vertex_normals(xyz, faces)
        intensity_sse, rms, _ = intensity_error(volume, xyz, values, ripped, affine)
        spring = tangential_spring_energy(
            xyz, normals, faces, ripped, ordered_neighbors=ordered)
        area = surface_total_area(xyz, faces)
        spacing = mean_vertex_spacing(xyz, ripped, two_offsets, two_neighbors)
        offsets, members = vertex_buckets_current(xyz, ripped, resolution=spacing)
        repulsion = self_repulsion_energy(
            xyz, ripped, offsets, members, two_offsets, two_neighbors, weight=5.0)
        spring_weighted = float(np.float32(0.3)) * spring * float(np.float32(original_area / area))
        intensity_weighted = float(np.float32(0.2)) * intensity_sse
        return {
            "repulsion_weighted": repulsion,
            "spring_weighted": spring_weighted,
            "intensity_weighted": intensity_weighted,
            "total": repulsion + spring_weighted + intensity_weighted,
            "rms": rms,
            "area": area,
            "spacing": spacing,
        }

    rows = []
    with np.load(args.python_steps_npz) as python_steps:
        for step in (33, 34):
            reference = np.frombuffer(
                (args.official_steps / f"step{step:03d}.raw").read_bytes(),
                dtype=VERTEX_DTYPE)["xyz"]
            candidate = python_steps[f"step{step}"]
            reference_terms = terms(reference)
            candidate_terms = terms(candidate)
            rows.append({
                "step": step,
                "official_coordinates_python_objective": reference_terms,
                "python_coordinates_python_objective": candidate_terms,
                "terms_candidate_minus_reference": {
                    key: candidate_terms[key] - reference_terms[key]
                    for key in ("repulsion_weighted", "spring_weighted",
                                "intensity_weighted", "total")
                },
                "coordinate_component_exact": int(np.count_nonzero(candidate == reference)),
                "coordinate_component_count": int(candidate.size),
            })
            print(json.dumps(rows[-1]), flush=True)
    report = {
        "scope": "real T1 LH white third pass; Python objective evaluated twice on official and Python coordinates",
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.boundary_raw, args.boundary_npz,
                                      args.python_steps_npz,
                                      args.official_steps / "step033.raw",
                                      args.official_steps / "step034.raw")},
        "original_area": original_area,
        "records": rows,
        "seconds": {"prepare": prepared_at - started,
                    "evaluation": time.perf_counter() - prepared_at,
                    "total": time.perf_counter() - started},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
