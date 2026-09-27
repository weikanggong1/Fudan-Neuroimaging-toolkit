"""First-pass prefix of FreeSurfer 8.2 white.preaparc from matched inputs.

This first-pass diagnostic does not create a complete white surface. It reuses the
independently validated MRI, ripping, border, and collision operators and adds
the white-specific current-surface self-repulsion force.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from .place_pial_python import _write_vertices_like
from .place_surface_border import compute_border_values_first_pass
from .place_surface_collision import asynchronous_first_step
from .place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
from .place_surface_decision import pial_step_decision
from .place_surface_geometry import surface_ras_to_voxel
from .place_surface_gradient_average import average_signed_gradients
from .place_surface_intensity import intensity_gradient
from .place_surface_normals import initial_vertex_normals
from .place_surface_objective import intensity_error, surface_total_area, tangential_spring_energy
from .place_surface_rip import rip_white_preaparc_pass
from .place_surface_self_repulsion import (
    mean_vertex_spacing, self_repulsion_energy, self_repulsion_gradient, vertex_buckets_current,
)
from .place_surface_smoothing import average_marked_values, average_vertex_positions, _ordered_neighbors
from .place_surface_spring import spring_gradient
from .place_surface_step import unconstrained_step_with_offsets
from .place_surface_volume import prepare_placement_volume


def place_white_preaparc_prefix(
    subject_dir: str | Path, hemi: str, output: str | Path,
    *, steps: int = 1, diagnostics: str | Path | None = None,
) -> dict:
    """Run a prefix of the first pass and write the current diagnostic mesh.

    ``subject_dir`` contains ``surf/H.orig``, the gray/white threshold file,
    and ``mri/{brain.finalsurfs,wm,aseg.presurf}.mgz``. ``output`` is a
    diagnostic FreeSurfer surface, not ``H.white.preaparc``. Optional
    ``diagnostics`` writes a NumPy ``.npz`` of intermediate force and mesh
    arrays to compare against a pinned-source probe. ``steps`` is 1–17
    iterations of the first pass; the output contains coordinates after the
    last requested step and the return dict includes each step's SSE/RMS.
    """
    started = time.perf_counter()
    if not 1 <= steps <= 17:
        raise ValueError("steps must be from 1 to 17")
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    subject = Path(subject_dir)
    orig = subject / f"surf/{hemi}.orig"
    stats_path = subject / f"surf/autodet.gw.stats.{hemi}.dat"
    brain_path = subject / "mri/brain.finalsurfs.mgz"
    wm_path = subject / "mri/wm.mgz"
    seg_path = subject / "mri/aseg.presurf.mgz"
    for source in (orig, stats_path, brain_path, wm_path, seg_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    output = Path(output)
    stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines()
                 if len(line.split()) >= 2)
    vertices, faces, metadata = nib.freesurfer.read_geometry(str(orig), read_metadata=True)
    xyz = average_vertex_positions(vertices, faces, 5)
    normals = initial_vertex_normals(xyz, faces)
    brain = nib.load(str(brain_path))
    seg_image = nib.load(str(seg_path))
    seg = np.asarray(seg_image.dataobj)
    volume, _ = prepare_placement_volume(
        np.asarray(brain.dataobj), np.asarray(nib.load(str(wm_path)).dataobj),
        surface="white", mid_gray=float(stats["MID_GRAY"]),
    )
    rip_affine = surface_ras_to_voxel(seg_image.header, metadata)
    ripped = values = None
    for _ in range(2):
        ripped, values = rip_white_preaparc_pass(
            xyz, normals, faces, seg, volume, rip_affine, hemisphere=hemi,
            ripped=ripped, values=values,
        )
    affine = surface_ras_to_voxel(brain.header, metadata)
    thresholds = np.array([float(stats[f"white_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    border = compute_border_values_first_pass(
        volume, seg, xyz, normals, xyz, ripped, values, affine, thresholds,
        hemisphere=hemi, surface="white", sigma=2.0,
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    ordered_indices, ordered_valid, _ = _ordered_neighbors(faces, len(xyz))
    ordered = (ordered_indices, ordered_valid)
    two_offsets, two_neighbors = two_ring_neighbors(
        faces, len(xyz), ordered_neighbors=ordered)
    original_area = surface_total_area(xyz, faces)

    def objective(current: np.ndarray) -> tuple[float, float]:
        current_normals = initial_vertex_normals(current, faces)
        intensity_sse, rms, _ = intensity_error(volume, current, values, ripped, affine)
        spring = tangential_spring_energy(
            current, current_normals, faces, ripped, ordered_neighbors=ordered)
        area = surface_total_area(current, faces)
        spacing = mean_vertex_spacing(current, ripped, two_offsets, two_neighbors)
        offsets, members = vertex_buckets_current(current, ripped, resolution=spacing)
        repulsion = self_repulsion_energy(
            current, ripped, offsets, members, two_offsets, two_neighbors, weight=5.0)
        area_scale = float(np.float32(original_area / area))
        return float(np.float32(0.3)) * spring * area_scale + float(np.float32(0.2)) * intensity_sse + repulsion, rms

    prepared_at = time.perf_counter()
    initial_sse, initial_rms = objective(xyz)
    initial_objective_at = time.perf_counter()
    current = xyz.copy()
    last_sse, last_rms = initial_sse, initial_rms
    cropped = np.zeros(len(xyz), dtype=np.int32)
    dt, reductions = 0.5, 0
    gradient_seconds = collision_seconds = objective_seconds = 0.0
    records: list[dict] = []
    snapshots: dict[str, np.ndarray | float] = {
        "initial": xyz, "ripped": ripped, "target_values": values,
        "initial_sse": initial_sse, "initial_rms": initial_rms,
    } if diagnostics is not None else {}
    for step in range(1, steps + 1):
        stage_start = time.perf_counter()
        normals = initial_vertex_normals(current, faces)
        intensity = intensity_gradient(
            volume, current, normals, ripped, values, border[5], affine,
            brain.header.get_zooms()[:3], weight=0.2, sigma_global=2.0,
        )
        averaged = average_signed_gradients(
            intensity, faces, ripped, 4, ordered_neighbors=ordered)
        bucket_offsets, bucket_members = vertex_buckets_current(current, ripped)
        self_repulsion = self_repulsion_gradient(
            current, ripped, bucket_offsets, bucket_members,
            two_offsets, two_neighbors, weight=5.0,
        )
        with_repulsion = np.float32(averaged + self_repulsion)
        normal = spring_gradient(
            current, normals, faces, ripped, weight=0.3, direction="normal",
            ordered_neighbors=ordered,
        )
        with_normal = np.float32(with_repulsion + normal)
        curvature = quadratic_curvature(
            current, normals, tangent_basis(normals), ripped, two_offsets, two_neighbors)
        with_curvature = np.float32(with_normal + np.float32(curvature[:, None] * normals))
        tangent = spring_gradient(
            current, normals, faces, ripped, weight=0.3, direction="tangent",
            ordered_neighbors=ordered,
        )
        gradient = np.float32(with_curvature + tangent)
        gradient_seconds += time.perf_counter() - stage_start
        before_collision = gradient.copy() if diagnostics is not None else None
        if diagnostics is not None:
            snapshots[f"step{step}_initial"] = current.copy()
            snapshots[f"step{step}_tangential_spring"] = before_collision
        stale_trial = None
        for trial in range(3):
            trial_start = time.perf_counter()
            proposed, offsets = unconstrained_step_with_offsets(current, gradient, ripped, dt=dt)
            placed, _ = asynchronous_first_step(
                current, faces, proposed, ripped, fast=True, offsets=offsets,
                accepted_offsets=gradient, stale_mht_trial=stale_trial,
                ordered_neighbors=ordered,
            )
            collision_seconds += time.perf_counter() - trial_start
            blocked = np.any(proposed != current, axis=1) & np.all(placed == current, axis=1)
            cropped = np.where(ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
            objective_start = time.perf_counter()
            step_sse, step_rms = objective(placed)
            objective_seconds += time.perf_counter() - objective_start
            next_dt, reductions, reduced, rejected, stop = pial_step_decision(
                last_sse, last_rms, step_sse, step_rms, dt, reductions)
            dt = next_dt
            if rejected:
                stale_trial = placed
                if stop:
                    raise RuntimeError(f"white prefix rejected step {step} after {trial + 1} trials")
                continue
            break
        else:
            raise RuntimeError(f"white prefix rejected all trials at step {step}")
        current = placed
        last_sse, last_rms = step_sse, step_rms
        records.append({
            "step": step, "trials": trial + 1, "sse": step_sse, "rms": step_rms,
            "next_dt": dt, "reductions": reductions,
            "held_vertices": int(np.count_nonzero(blocked)),
        })
        if diagnostics is not None:
            snapshots[f"step{step}_after_collision"] = current.copy()
            snapshots[f"step{step}_sse"] = step_sse
            snapshots[f"step{step}_rms"] = step_rms
            if step == 1:
                snapshots.update(
                    intensity=intensity, averaged=averaged, self_repulsion=self_repulsion,
                    pre_normal_spring=with_repulsion, normal_spring=with_normal,
                    curvature=with_curvature, tangential_spring=before_collision,
                    proposed=proposed, after_collision=current.copy(),
                    step_sse=step_sse, step_rms=step_rms,
                )
        if stop:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_vertices_like(orig, output, current)
    if diagnostics is not None:
        diagnostic_path = Path(diagnostics)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(diagnostic_path, **snapshots)
    finished_at = time.perf_counter()
    return {
        "output": str(output), "hemisphere": hemi, "steps": len(records),
        "vertices": int(len(xyz)), "faces": int(len(faces)),
        "ripped_vertices": int(np.count_nonzero(ripped)),
        "held_vertices": records[-1]["held_vertices"],
        "initial_sse": initial_sse, "initial_rms": initial_rms,
        "step_sse": last_sse, "step_rms": last_rms,
        "per_step": records,
        "seconds": finished_at - started,
        "stage_seconds": {
            "prepare": prepared_at - started,
            "initial_objective": initial_objective_at - prepared_at,
            "gradient": gradient_seconds,
            "collision": collision_seconds,
            "step_objective": objective_seconds,
            "write": finished_at - initial_objective_at - gradient_seconds
                     - collision_seconds - objective_seconds,
        },
    }


def first_white_preaparc_step(
    subject_dir: str | Path, hemi: str, output: str | Path,
    *, diagnostics: str | Path | None = None,
) -> dict:
    """Run one diagnostic white optimizer step without final surface cleanup."""
    return place_white_preaparc_prefix(
        subject_dir=subject_dir, hemi=hemi, output=output,
        steps=1, diagnostics=diagnostics,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--steps", type=int, choices=range(1, 18), default=1)
    args = parser.parse_args()
    print(json.dumps(place_white_preaparc_prefix(
        subject_dir=args.subject_dir, hemi=args.hemi, output=args.output,
        steps=args.steps, diagnostics=args.diagnostics,
    ), indent=2))


if __name__ == "__main__":
    main()
