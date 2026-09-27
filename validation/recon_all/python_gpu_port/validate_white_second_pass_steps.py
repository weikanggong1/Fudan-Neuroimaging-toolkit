"""Replay the real T1 white second pass from installed first-pass RAM state."""

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
from validate_white_second_step import _vertex_error
from fnit.recon_all.place_surface_collision import asynchronous_first_step
from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
from fnit.recon_all.place_surface_decision import pial_step_decision
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--boundary-raw", type=Path, required=True)
    parser.add_argument("--boundary-npz", type=Path, required=True)
    parser.add_argument("--official-steps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    initial = np.frombuffer(args.boundary_raw.read_bytes(), dtype=VERTEX_DTYPE)
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
    current = initial["xyz"].copy()
    ordered_indices, ordered_valid, _ = _ordered_neighbors(faces, len(current))
    ordered = (ordered_indices, ordered_valid)
    two_offsets, two_neighbors = two_ring_neighbors(
        faces, len(current), ordered_neighbors=ordered)
    original_area = surface_total_area(current, faces)

    def objective(xyz: np.ndarray) -> tuple[float, float]:
        normals = initial_vertex_normals(xyz, faces)
        intensity_sse, rms, _ = intensity_error(volume, xyz, values, ripped, affine)
        spring = tangential_spring_energy(
            xyz, normals, faces, ripped, ordered_neighbors=ordered)
        area = surface_total_area(xyz, faces)
        spacing = mean_vertex_spacing(xyz, ripped, two_offsets, two_neighbors)
        offsets, members = vertex_buckets_current(xyz, ripped, resolution=spacing)
        repulsion = self_repulsion_energy(
            xyz, ripped, offsets, members, two_offsets, two_neighbors, weight=5.0)
        scale = float(np.float32(original_area / area))
        return float(np.float32(0.3)) * spring * scale + float(np.float32(0.2)) * intensity_sse + repulsion, rms

    prepared_at = time.perf_counter()
    last_sse, last_rms = objective(current)
    initial_sse = last_sse
    initial_objective_at = time.perf_counter()
    dt, reductions = 0.5, 0
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {"initial": current.copy(), "ripped": ripped, "target_values": values}
    stage_time = {"gradient": 0.0, "collision": 0.0, "step_objective": 0.0}
    log = (args.official_steps / "capture.log").read_text(errors="replace")
    official_printed = {int(step): (float(sse), float(rms)) for step, _, sse, rms in
                        re.findall(r"^(\d{3}): dt: ([0-9.]+), sse=([0-9.]+), rms=([0-9.]+)", log, re.M)}

    for step in range(18, 118):
        gradient_start = time.perf_counter()
        normals = initial_vertex_normals(current, faces)
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
        stage_time["gradient"] += time.perf_counter() - gradient_start
        stale_trial = None
        rejected_trials: list[dict] = []
        for trial in range(3):
            collision_start = time.perf_counter()
            proposed, offsets = unconstrained_step_with_offsets(current, gradient, ripped, dt=dt)
            placed, _ = asynchronous_first_step(
                current, faces, proposed, ripped, fast=True, offsets=offsets,
                accepted_offsets=gradient, stale_mht_trial=stale_trial,
                ordered_neighbors=ordered,
            )
            stage_time["collision"] += time.perf_counter() - collision_start
            objective_start = time.perf_counter()
            step_sse, step_rms = objective(placed)
            stage_time["step_objective"] += time.perf_counter() - objective_start
            next_dt, reductions, reduced, rejected, stop = pial_step_decision(
                last_sse, last_rms, step_sse, step_rms, dt, reductions)
            if rejected:
                rejected_trials.append({"dt": dt, "sse": step_sse, "rms": step_rms})
                dt = next_dt
                stale_trial = placed
                if stop:
                    raise RuntimeError(f"second pass rejected step {step} at reduction {reductions}")
                continue
            dt = next_dt
            break
        else:
            raise RuntimeError(f"second pass rejected all trials at step {step}")
        current = placed
        last_sse, last_rms = step_sse, step_rms
        arrays[f"step{step}"] = current.copy()
        native_path = args.official_steps / f"step{step:03d}.raw"
        if not native_path.is_file():
            raise FileNotFoundError(native_path)
        native = np.frombuffer(native_path.read_bytes(), dtype=VERTEX_DTYPE)
        coordinates = _vertex_error(current, native["xyz"])
        printed = official_printed.get(step)
        records.append({
            "step": step, "trials": trial + 1, "rejected_trials": rejected_trials,
            "sse": step_sse, "rms": step_rms, "next_dt": dt,
            "reductions": reductions, "official_printed_sse_rms": printed,
            "sse_error_from_one_decimal_print": abs(step_sse - printed[0]) if printed else None,
            "coordinates": coordinates,
        })
        print(json.dumps({"step": step, "trials": trial + 1,
                          "sse": step_sse, "max_mm": coordinates["max_mm"],
                          "count_gt_1e4_mm": coordinates["count_gt_1e4_mm"]}), flush=True)
        if stop:
            break
    finished_at = time.perf_counter()
    npz = args.out.with_suffix(".npz")
    np.savez_compressed(npz, **arrays)
    report = {
        "subject": str(args.subject),
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.boundary_raw, args.boundary_npz)},
        "official_capture_report_sha256": hashlib.sha256(
            (args.official_steps / "capture_report.json").read_bytes()).hexdigest(),
        "first_step": 18, "last_step": records[-1]["step"],
        "initial_sse": initial_sse,
        "records": records,
        "diagnostic_npz": str(npz),
        "diagnostic_npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "seconds": {"prepare": prepared_at - started,
                    "initial_objective": initial_objective_at - prepared_at,
                    **stage_time, "total": finished_at - started},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"last_step": report["last_step"], "seconds": report["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
