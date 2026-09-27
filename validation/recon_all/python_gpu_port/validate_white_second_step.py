"""Compare the first white optimizer step of pass two with installed FS 8.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import nibabel as nib
import numpy as np

from compare_white_second_pass_boundary import VERTEX_DTYPE
from fnit.recon_all.place_surface_collision import asynchronous_first_step
from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
from fnit.recon_all.place_surface_geometry import surface_ras_to_voxel
from fnit.recon_all.place_surface_gradient_average import average_signed_gradients
from fnit.recon_all.place_surface_intensity import intensity_gradient
from fnit.recon_all.place_surface_normals import initial_vertex_normals
from fnit.recon_all.place_surface_objective import intensity_error, surface_total_area, tangential_spring_energy
from fnit.recon_all.place_surface_self_repulsion import (
    mean_vertex_spacing, self_repulsion_energy, self_repulsion_gradient, vertex_buckets_current,
)
from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
from fnit.recon_all.place_surface_spring import spring_gradient
from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets
from fnit.recon_all.place_surface_volume import prepare_placement_volume


def _vertex_error(candidate: np.ndarray, reference: np.ndarray) -> dict:
    distances = np.linalg.norm(candidate.astype(np.float64) - reference.astype(np.float64), axis=1)
    largest = np.argsort(distances)[-10:][::-1]
    return {
        "exact_components": int(np.count_nonzero(candidate == reference)),
        "components": int(candidate.size),
        "mean_mm": float(distances.mean()), "p99_mm": float(np.percentile(distances, 99)),
        "max_mm": float(distances.max()),
        "count_gt_1e4_mm": int(np.count_nonzero(distances > 1e-4)),
        "count_gt_01_mm": int(np.count_nonzero(distances > 0.1)),
        "largest": [{"vertex": int(v), "distance_mm": float(distances[v])} for v in largest],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--before-raw", type=Path, required=True)
    parser.add_argument("--after-raw", type=Path, required=True)
    parser.add_argument("--boundary-npz", type=Path, required=True)
    parser.add_argument("--official-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    before = np.frombuffer(args.before_raw.read_bytes(), dtype=VERTEX_DTYPE)
    after = np.frombuffer(args.after_raw.read_bytes(), dtype=VERTEX_DTYPE)
    with np.load(args.boundary_npz) as boundary:
        ripped = boundary["ripped"].astype(bool)
        values = boundary["values"]
        used_sigma = boundary["sigma"]
    _, faces, metadata = nib.freesurfer.read_geometry(str(args.subject / "surf/lh.orig"), read_metadata=True)
    brain = nib.load(str(args.subject / "mri/brain.finalsurfs.mgz"))
    wm = nib.load(str(args.subject / "mri/wm.mgz"))
    stats = dict(line.split()[:2] for line in
                 (args.subject / "surf/autodet.gw.stats.lh.dat").read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, _ = prepare_placement_volume(
        np.asarray(brain.dataobj), np.asarray(wm.dataobj),
        surface="white", mid_gray=float(stats["MID_GRAY"]),
    )
    affine = surface_ras_to_voxel(brain.header, metadata)
    current = before["xyz"].copy()
    normals = initial_vertex_normals(current, faces)
    normal_difference = _vertex_error(normals[~ripped], before["normal"][~ripped])
    ordered_indices, ordered_valid, _ = _ordered_neighbors(faces, len(current))
    ordered = (ordered_indices, ordered_valid)
    two_offsets, two_neighbors = two_ring_neighbors(
        faces, len(current), ordered_neighbors=ordered)
    # MRISpositionSurface stores current metric properties at each outer-pass entry.
    original_area = surface_total_area(current, faces)
    objective_parts = []

    def objective(xyz: np.ndarray) -> tuple[float, float]:
        surface_normals = initial_vertex_normals(xyz, faces)
        intensity_sse, rms, _ = intensity_error(volume, xyz, values, ripped, affine)
        spring = tangential_spring_energy(
            xyz, surface_normals, faces, ripped, ordered_neighbors=ordered)
        area = surface_total_area(xyz, faces)
        spacing = mean_vertex_spacing(xyz, ripped, two_offsets, two_neighbors)
        offsets, members = vertex_buckets_current(xyz, ripped, resolution=spacing)
        repulsion = self_repulsion_energy(
            xyz, ripped, offsets, members, two_offsets, two_neighbors, weight=5.0)
        scale = float(np.float32(original_area / area))
        parts = {"intensity_unweighted": intensity_sse, "intensity_weighted": float(np.float32(0.2)) * intensity_sse,
                 "spring_unweighted": spring, "spring_weighted": float(np.float32(0.3)) * spring * scale,
                 "repulsion_weighted": repulsion, "original_area": float(original_area),
                 "current_area": float(area), "area_scale": scale}
        objective_parts.append(parts)
        return parts["spring_weighted"] + parts["intensity_weighted"] + parts["repulsion_weighted"], rms

    prepared_at = time.perf_counter()
    initial_sse, initial_rms = objective(current)
    objective_at = time.perf_counter()
    intensity = intensity_gradient(
        volume, current, normals, ripped, values, used_sigma, affine,
        brain.header.get_zooms()[:3], weight=0.2, sigma_global=1.0,
    )
    averaged = average_signed_gradients(intensity, faces, ripped, 2, ordered_neighbors=ordered)
    bucket_offsets, bucket_members = vertex_buckets_current(current, ripped)
    repulsion = self_repulsion_gradient(
        current, ripped, bucket_offsets, bucket_members,
        two_offsets, two_neighbors, weight=5.0,
    )
    gradient = np.float32(averaged + repulsion)
    gradient = np.float32(gradient + spring_gradient(
        current, normals, faces, ripped, weight=0.3, direction="normal",
        ordered_neighbors=ordered,
    ))
    curvature = quadratic_curvature(
        current, normals, tangent_basis(normals), ripped, two_offsets, two_neighbors)
    gradient = np.float32(gradient + np.float32(curvature[:, None] * normals))
    gradient = np.float32(gradient + spring_gradient(
        current, normals, faces, ripped, weight=0.3, direction="tangent",
        ordered_neighbors=ordered,
    ))
    gradient_at = time.perf_counter()
    proposed, offsets = unconstrained_step_with_offsets(current, gradient, ripped, dt=0.5)
    placed, _ = asynchronous_first_step(
        current, faces, proposed, ripped, fast=True, offsets=offsets,
        accepted_offsets=gradient, ordered_neighbors=ordered,
    )
    collision_at = time.perf_counter()
    step_sse, step_rms = objective(placed)
    finish = time.perf_counter()
    log = args.official_log.read_text(errors="replace")
    initial_match = re.findall(r"starting sse = ([0-9.]+), rms = ([0-9.]+)", log, re.I)
    step_match = re.findall(r"^018: dt: ([0-9.]+), sse=([0-9.]+), rms=([0-9.]+)", log, re.M)
    report = {
        "subject": str(args.subject),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.before_raw, args.after_raw, args.boundary_npz)},
        "normal_difference": normal_difference,
        "initial_sse": initial_sse, "initial_rms": initial_rms,
        "step_sse": step_sse, "step_rms": step_rms,
        "objective_parts": {"initial": objective_parts[0], "step18": objective_parts[1]},
        "official_initial_lines": initial_match[-2:], "official_step18_lines": step_match,
        "coordinates": _vertex_error(placed, after["xyz"]),
        "seconds": {"prepare": prepared_at - started, "initial_objective": objective_at - prepared_at,
                    "gradient": gradient_at - objective_at, "collision": collision_at - gradient_at,
                    "step_objective": finish - collision_at, "total": finish - started},
    }
    arrays = args.out.with_suffix(".npz")
    np.savez_compressed(arrays, initial=current, normals=normals, intensity=intensity,
                        averaged=averaged, repulsion=repulsion, gradient=gradient,
                        proposed=proposed, placed=placed)
    report["candidate_arrays"] = str(arrays)
    report["candidate_arrays_sha256"] = hashlib.sha256(arrays.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("input_sha256", "candidate_arrays_sha256")}, indent=2))


if __name__ == "__main__":
    main()
