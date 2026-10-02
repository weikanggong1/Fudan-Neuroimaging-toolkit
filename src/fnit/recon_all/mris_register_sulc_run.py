"""Run FreeSurfer 8.2's sulcal spherical-registration pass in Python."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from .mris_register_average_numba import RegistrationGradientAverager
from .mris_register_atlas import sample_atlas_on_canonical_sphere
from .mris_register_blur import blur_atlas_frame
from .mris_register_kernels import center_sphere, normalize_mean_curvature, project_sphere
from .mris_register_line_search import first_registration_line_search, first_registration_sse
from .mris_register_nonlinear import (
    apply_spherical_gradient, correlation_gradient_add,
    first_area_gradient, first_distance_gradient,
    prepare_registration_force_cache, registration_total_area,
    sphere_arc_distances, sphere_vertex_normals, tangent_basis,
)
from .mris_register_parameterization import parameterize_curvature
from .mris_register_rigid import register_rigid
from .mris_register_schedule import next_sulc_scale
from .sphere_standard_python import write_standard_sphere_surface


@torch.no_grad()
def run_register_sulc(sphere: str | Path, smoothwm: str | Path,
                      sulc: str | Path, atlas_file: str | Path,
                      output: str | Path, *, max_updates: int = 1024,
                      averaging_device: str = "cpu") -> dict:
    """Create the sulc-pass sphere from the ordered conventional sphere.

    This CPU PyTorch/Numba stage produces the seed for ``run_register_smoothwm``.
    It uses the FreeSurfer 8.2 default rigid search and source-derived scale
    schedule; no native registration executable or snapshot is read.
    averaging_device 默认 cpu，可指定 cuda:N，仅搬运静态邻接及各次梯度，
    完整 rounds 与 float32 顺序不变。max_updates 默认 1024 是未收敛时
    的报错上限，不能减少正常轮次来加速。输出路径与逐轮轨迹、时间、
    刚体搜索信息返回为 dict；表面坐标为 surface RAS/mm。
    """
    started = time.perf_counter()
    sphere, smoothwm, sulc = Path(sphere), Path(smoothwm), Path(sulc)
    atlas_file, output = Path(atlas_file), Path(output)
    sphere_xyz, faces = fsio.read_geometry(str(sphere))
    smooth_xyz, smooth_faces = fsio.read_geometry(str(smoothwm))
    if not np.array_equal(faces, smooth_faces):
        raise ValueError("sphere and smoothwm must have identical face order")
    vertices = torch.from_numpy(sphere_xyz.astype(np.float32))
    original = torch.from_numpy(smooth_xyz.astype(np.float32))
    triangles = torch.from_numpy(faces.astype(np.int64))
    sulc_values = torch.from_numpy(fsio.read_morph_data(str(sulc)).astype(np.float32))
    if len(sulc_values) != len(vertices):
        raise ValueError("sulc vertex count differs from sphere")
    atlas = tifffile.imread(atlas_file)
    raw_mean = torch.from_numpy(atlas[3].view(np.float32).copy())
    raw_variance = torch.from_numpy(atlas[4].view(np.float32).copy())
    canonical = project_sphere(center_sphere(vertices))
    rigid_start = time.perf_counter()
    current, angles, rigid_score, evaluations = register_rigid(
        vertices, sulc_values, raw_mean, raw_variance)
    rigid_seconds = time.perf_counter() - rigid_start
    normalized_sulc = normalize_mean_curvature(sulc_values)
    cache = prepare_registration_force_cache(vertices, original, triangles)
    neighbors, degrees = cache.neighbors, cache.degrees
    average = RegistrationGradientAverager(neighbors, degrees, device=averaging_device)
    original_distances, original_areas = cache.original_distances, cache.original_areas
    original_area = cache.orig_area
    total_area = registration_total_area()
    setup_seconds = time.perf_counter() - started

    state: tuple[str, int, int, int] | None = ("big", 0, 16384, 0)
    previous: tuple[str, int, int, int] | None = None
    updates: list[dict] = []
    for index in range(max_updates):
        if state is None:
            break
        step_start = time.perf_counter()
        phase, sigma_index, averages, _ = state
        if previous is None or sigma_index != previous[1]:
            sigma = (4.0, 2.0, 1.0, 0.5)[sigma_index]
            positions = canonical if previous is None else current
            source_grid = parameterize_curvature(positions, normalized_sulc)
            curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                positions, blur_atlas_frame(source_grid, sigma)))
            mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                positions, blur_atlas_frame(raw_mean, sigma)))
            mean_grid = parameterize_curvature(positions, mean_curve)
            variance_grid = blur_atlas_frame(raw_variance, sigma)
        integration_start = previous is None or state[:3] != previous[:3]
        projected = project_sphere(current) if integration_start else current
        normals = sphere_vertex_normals(projected, triangles, incidence=cache.incidence)
        distances = sphere_arc_distances(projected, neighbors, degrees)
        avg_vertex_dist = float(distances.double().sum() / degrees.sum())
        force = first_area_gradient(
            vertices, original, projected, triangles,
            first_distance_gradient(vertices, original, projected, triangles,
                                    project=False, cache=cache),
            project=False, cache=cache)
        e1, e2 = tangent_basis(normals)
        force = correlation_gradient_add(force, projected, curvature, e1, e2,
                                         mean_grid, variance_grid, avg_vertex_dist)
        averaged = average(force, averages)

        def objective(trial: torch.Tensor) -> float:
            return first_registration_sse(
                trial, triangles, neighbors, degrees, original_distances,
                original_areas, curvature, mean_grid, variance_grid,
                original_area, total_area)

        dt, samples = first_registration_line_search(projected, averaged, objective)
        current = apply_spherical_gradient(projected, averaged, dt)
        selected_sse = min(samples, key=lambda item: abs(item[0] - dt))[1]
        next_state = next_sulc_scale(
            *state, samples[0][1], selected_sse, dt)
        updates.append({"iteration": index + 2, "phase": phase, "sigma": sigma,
                        "averages": averages, "dt": dt,
                        "next_state": next_state,
                        "seconds": time.perf_counter() - step_start})
        previous, state = state, next_state
    if state is not None:
        raise RuntimeError("sulc registration did not converge within max_updates")
    write_standard_sphere_surface(
        output, current.cpu().numpy(), faces, sphere,
        create_stamp="created by Python sulcal registration")
    return {"sphere": str(sphere), "smoothwm": str(smoothwm), "sulc": str(sulc),
            "atlas": str(atlas_file), "output": str(output),
            "rigid_angles": angles, "rigid_score": rigid_score,
            "rigid_evaluations": evaluations, "rigid_seconds": rigid_seconds,
            "averaging_device": str(average.device),
            "setup_seconds": setup_seconds, "updates": updates,
            "last_iteration": updates[-1]["iteration"],
            "total_seconds_including_io": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "sulc", "atlas", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--averaging-device", default="cpu")
    args = parser.parse_args(argv)
    report = run_register_sulc(
        args.sphere, args.smoothwm, args.sulc, args.atlas, args.output,
        averaging_device=args.averaging_device)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
