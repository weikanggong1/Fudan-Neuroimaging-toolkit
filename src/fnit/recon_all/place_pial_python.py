"""FreeSurfer 8.2 pial T1 geometry placement from matched upstream files.

This is the four-pass, CPU NumPy/Numba path of ``mris_place_surface --pial``.
The input surface, MRI, autodetected thresholds, and labels must already have
been produced in the official stage order. This module does not build them.
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np

from .place_surface_border import compute_border_values_first_pass
from .place_surface_collision import asynchronous_first_step
from .place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
from .place_surface_decision import pial_step_decision
from .place_surface_final_cleanup import pin_medial_wall, repair_intersections
from .place_surface_geometry import surface_ras_to_voxel
from .place_surface_gradient_average import average_signed_gradients
from .place_surface_intensity import intensity_gradient
from .place_surface_normals import FaceNormalTopology
from .place_surface_objective import (
    intensity_error, pial_placement_sse, surface_total_area, tangential_spring_energy,
)
from .place_surface_repulsion import (
    original_vertex_normals, surface_repulsion_gradient, OriginalVertexBuckets,
)
from .place_surface_rip import rip_outside_label
from .place_surface_smoothing import average_marked_values, _ordered_neighbors
from .place_surface_spring import spring_gradient
from .place_surface_step import unconstrained_step_with_offsets
from .place_surface_volume import prepare_placement_volume


def _write_vertices_like(source: Path, output: Path, vertices: np.ndarray) -> None:
    """Retain the ordered faces, full volume geometry and extra footer tags."""
    raw = source.read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError(f"expected FreeSurfer triangle surface: {source}")
    first = raw.index(b"\n\n", 3) + 2
    count = int.from_bytes(raw[first:first + 4], "big")
    if len(vertices) != count:
        raise ValueError("pial placement changed the vertex count")
    start = first + 8
    output.write_bytes(raw[:start] + np.asarray(vertices, dtype=">f4").tobytes()
                       + raw[start + 12 * count:])


def place_pial_t1(
    subject: str | Path,
    hemisphere: str,
    output: str | Path | None = None,
    *,
    max_steps: int = 200,
    sampling_backend: str = "cpu",
    device: str | None = None,
    trace_callback=None,
) -> dict:
    """Place and save one hemisphere's pial.T1 using the native four-pass order.

    Requires ``surf/{hemi}.white``, ``surf/autodet.gw.stats.{hemi}.dat``,
    ``label/{hemi}.cortex.label``, ``label/{hemi}.cortex+hipamyg.label``,
    and ``mri/{brain.finalsurfs,wm,aseg.presurf}.mgz``. The return value gives
    the output path, accepted step count, pass boundaries, cleanup, and seconds.
    ``sampling_backend`` defaults to cpu; torch/triton select explicit CUDA MRI
    sampling through ``device``. Ordered updates and objective remain CPU.
    ``trace_callback(step, pass_index, coordinates_copy, diagnostics)`` is an
    optional read-only diagnostic sink called after every accepted step.
    ``max_steps`` guards against non-convergence; hitting it raises an error.
    """
    if sampling_backend not in ("cpu", "torch", "triton"):
        raise ValueError("sampling_backend must be cpu, torch or triton")
    if sampling_backend != "cpu" and device is None:
        raise ValueError("GPU sampling requires an explicit device")
    started = time.perf_counter()
    subject = Path(subject)
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be 'lh' or 'rh'")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    hemi = hemisphere
    white = subject / f"surf/{hemi}.white"
    cortex = subject / f"label/{hemi}.cortex.label"
    label = subject / f"label/{hemi}.cortex+hipamyg.label"
    stats_path = subject / f"surf/autodet.gw.stats.{hemi}.dat"
    brain_path = subject / "mri/brain.finalsurfs.mgz"
    wm_path = subject / "mri/wm.mgz"
    aseg_path = subject / "mri/aseg.presurf.mgz"
    for source in (white, cortex, label, stats_path, brain_path, wm_path, aseg_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    output = Path(output) if output is not None else subject / f"surf/{hemi}.pial.T1"

    xyz, faces, metadata = fs.read_geometry(str(white), read_metadata=True)
    xyz = xyz.astype(np.float32)
    ripped = np.asarray(rip_outside_label(
        len(xyz), fs.read_label(str(label))), dtype=np.bool_)
    brain = nib.load(str(brain_path))
    wm = np.asarray(nib.load(str(wm_path)).dataobj)
    aseg = np.asarray(nib.load(str(aseg_path)).dataobj)
    stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, bright = prepare_placement_volume(
        np.asarray(brain.dataobj), wm, surface="pial",
        mid_gray=float(stats["MID_GRAY"]),
    )
    placement = volume.copy()
    placement[bright == 130] = 0
    affine = surface_ras_to_voxel(brain.header, metadata)
    sampler = None
    if sampling_backend != "cpu":
        from .place_surface_sampling import PlacementSampling
        sampler = PlacementSampling(placement, affine, device=device,
                                    implementation=sampling_backend)
    thresholds = np.array([float(stats[f"pial_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    normal_topology = FaceNormalTopology(faces, len(xyz))
    repulsion_index = OriginalVertexBuckets(xyz, ripped)
    normals = normal_topology.evaluate(xyz)
    border = compute_border_values_first_pass(
        volume, aseg, xyz, normals, xyz, ripped,
        np.full(len(xyz), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere=hemi, surface="pial",
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(xyz))
    ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(
        faces, len(xyz), ordered_neighbors=ordered)
    fixed_normals = original_vertex_normals(xyz, faces, topology=normal_topology)
    original_area = surface_total_area(xyz, faces)
    sigma, n_averages = 2.0, 16

    def objective(current: np.ndarray) -> tuple[float, float]:
        current_normals = normal_topology.evaluate(current)
        intensity_sse, rms, _ = intensity_error(
            placement, current, values, ripped, affine)
        spring = tangential_spring_energy(
            current, current_normals, faces, ripped,
            ordered_neighbors=ordered)
        area = surface_total_area(current, faces)
        return pial_placement_sse(intensity_sse, spring, original_area, area), rms

    def gradient(current: np.ndarray, cropped: np.ndarray) -> np.ndarray:
        current_normals = normal_topology.evaluate(current)
        if sampler is None:
            intensity = intensity_gradient(
                placement, current, current_normals, ripped, values, border[5],
                affine, brain.header.get_zooms()[:3], weight=0.2, sigma_global=sigma,
            )
        else:
            intensity = sampler.gradient(
                current, current_normals, ripped, values, border[5],
                brain.header.get_zooms()[:3], weight=0.2, sigma_global=sigma,
            )
        offsets, candidates = repulsion_index.query(current)
        repulsion = surface_repulsion_gradient(
            current, current_normals, xyz, fixed_normals, ripped,
            offsets, candidates, weight=5.0, cropped=cropped,
        )
        with_repulsion = np.float32(intensity + repulsion)
        averaged = average_signed_gradients(
            with_repulsion, faces, ripped, n_averages, ordered_neighbors=ordered)
        normal = spring_gradient(
            current, current_normals, faces, ripped, weight=0.3,
            direction="normal", ordered_neighbors=ordered)
        basis = tangent_basis(current_normals)
        scalar = quadratic_curvature(
            current, current_normals, basis, ripped, two_offsets, two_candidates)
        with_curvature = np.float32(np.float32(averaged + normal)
                                    + np.float32(scalar[:, None] * current_normals))
        tangent = spring_gradient(
            current, current_normals, faces, ripped, weight=0.3,
            direction="tangent", ordered_neighbors=ordered)
        return np.float32(with_curvature + tangent)

    last_sse, last_rms = objective(xyz)
    current, cropped = xyz.copy(), np.zeros(len(xyz), dtype=np.int32)
    dt, reductions, outer_pass = 0.5, 0, 0
    pass_ends: list[int] = []
    for step in range(1, max_steps + 1):
        momentum = gradient(current, cropped)
        stale_trial = None
        accepted = None
        for _ in range(3):
            proposal, displacement = unconstrained_step_with_offsets(
                current, momentum, ripped, dt=dt)
            candidate, _ = asynchronous_first_step(
                current, faces, proposal, ripped, fast=True,
                offsets=displacement, accepted_offsets=momentum,
                stale_mht_trial=stale_trial, ordered_neighbors=ordered)
            blocked = np.any(proposal != current, axis=1) & np.all(
                candidate == current, axis=1)
            trial_cropped = np.where(
                ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
            sse, rms = objective(candidate)
            dt, reductions, _, rejected, stop = pial_step_decision(
                last_sse, last_rms, sse, rms, dt, reductions)
            cropped = trial_cropped
            if rejected:
                stale_trial = candidate
                if stop:
                    break
                continue
            accepted = candidate
            last_sse, last_rms = sse, rms
            break
        if accepted is None:
            raise RuntimeError(f"pial optimizer rejected every trial at step {step}")
        current = accepted
        if trace_callback is not None:
            trace_callback(step, outer_pass, current.copy(),
                           {"sse": last_sse, "rms": last_rms, "dt": dt,
                            "reductions": reductions, "stop": bool(stop)})
        if stop:
            pass_ends.append(step)
            if outer_pass == 3:
                break
            outer_pass += 1
            sigma = 2.0 / (1 << outer_pass)
            n_averages = 16 >> outer_pass
            current_normals = normal_topology.evaluate(current)
            border = compute_border_values_first_pass(
                volume, aseg, current, current_normals, xyz, ripped,
                values, affine, thresholds, hemisphere=hemi, surface="pial",
                sigma=sigma,
            )
            values = average_marked_values(border[0], border[4], ripped, faces, 5)
            original_area = surface_total_area(current, faces)
            last_sse, last_rms = objective(current)
            dt, reductions = 0.5, 0
    else:
        raise RuntimeError(f"pial optimizer did not finish four passes in {max_steps} steps")

    pinned = pin_medial_wall(current, xyz, fs.read_label(str(cortex)))
    repaired, cleanup = repair_intersections(pinned, faces, ripped)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_vertices_like(white, output, repaired)
    return {"output": str(output), "hemisphere": hemi, "steps": step,
            "pass_ends": pass_ends, "cleanup": cleanup,
            "sampling_backend": sampling_backend, "device": device,
            "seconds": time.perf_counter() - started}
