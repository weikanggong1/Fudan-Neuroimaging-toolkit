"""Run the FreeSurfer 8.2 smoothwm spherical-registration pass in Python."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import tifffile
import torch

from .mris_register_average_numba import RegistrationGradientAverager
from .mris_register_atlas import sample_atlas_on_canonical_sphere
from .mris_register_blur import blur_atlas_frame
from .mris_register_kernels import normalize_mean_curvature, project_sphere
from .mris_register_line_search import first_registration_line_search, first_registration_sse
from .mris_register_nonlinear import (
    apply_spherical_gradient, correlation_gradient_add,
    face_area_normals, first_area_gradient, first_distance_gradient,
    prepare_registration_force_cache, registration_total_area,
    sphere_arc_distances, sphere_vertex_normals, spring_gradient_add,
    tangent_basis,
)
from .mris_register_overlap import remove_overlap_sphere
from .mris_register_parameterization import parameterize_curvature
from .mris_register_schedule import next_smoothwm_scale
from .mris_register_smoothwm import smoothwm_mean_curvature
from .sphere_standard_python import write_standard_sphere_surface


@torch.no_grad()
def run_register_smoothwm(sphere: str | Path, smoothwm: str | Path,
                          sulc_seed: str | Path, atlas_file: str | Path,
                          output: str | Path, *, seed_iteration: int,
                          overlap_device: str = "cpu", max_updates: int = 1024,
                          averaging_device: str = "cpu") -> dict:
    """Continue an already registered sulc sphere through smoothwm and repair.

    The optimizer uses source-order PyTorch/Numba CPU arithmetic. The final
    overlap repair may use CUDA. ``seed_iteration`` is the sulc pass update
    count, needed by the native overlap stopping rule.
    averaging_device 默认 cpu，显式 cuda:N 只加速有序梯度平均；目标项、
    步长接受和顺序更新不变。所有输入表面是 surface RAS/mm 的同序网格，
    atlas_file 是半球图谱 TIFF。max_updates 默认 1024，未收敛直接报错。
    返回 dict 包含逐轮状态、负面积清理轨迹、输出路径及嵌套耗时。
    """
    started = time.perf_counter()
    sphere, smoothwm = Path(sphere), Path(smoothwm)
    sulc_seed, atlas_file, output = Path(sulc_seed), Path(atlas_file), Path(output)
    sphere_xyz, faces = fsio.read_geometry(str(sphere))
    smooth_xyz, smooth_faces = fsio.read_geometry(str(smoothwm))
    seed_xyz, seed_faces = fsio.read_geometry(str(sulc_seed))
    if not (np.array_equal(faces, smooth_faces) and np.array_equal(faces, seed_faces)):
        raise ValueError("sphere, smoothwm, and sulc seed must have identical face order")
    vertices = torch.from_numpy(sphere_xyz.astype(np.float32))
    original = torch.from_numpy(smooth_xyz.astype(np.float32))
    current = torch.from_numpy(seed_xyz.astype(np.float32))
    triangles = torch.from_numpy(faces.astype(np.int64))
    raw = smoothwm_mean_curvature(original, triangles)
    normalized = normalize_mean_curvature(raw)
    cache = prepare_registration_force_cache(vertices, original, triangles)
    neighbors, degrees = cache.neighbors, cache.degrees
    average = RegistrationGradientAverager(neighbors, degrees, device=averaging_device)
    original_distances, original_areas = cache.original_distances, cache.original_areas
    original_area = cache.orig_area
    total_area = registration_total_area()
    area_scale = float(np.float32(original_area / total_area))
    dist_scale = torch.tensor(math.sqrt(area_scale), dtype=torch.float32)
    atlas = tifffile.imread(atlas_file)
    raw_mean = torch.from_numpy(atlas[6].view(np.float32).copy())
    raw_variance = torch.from_numpy(atlas[7].view(np.float32).copy())
    active = torch.arange(neighbors.shape[1])[None, :] < degrees[:, None]
    setup_seconds = time.perf_counter() - started

    state: tuple[str, int, int, int] | None = ("smoothwm", 0, 1024, 0)
    previous: tuple[str, int, int, int] | None = None
    updates: list[dict] = []
    for index in range(max_updates):
        if state is None:
            break
        step_start = time.perf_counter()
        stage, sigma_index, averages, _ = state
        if previous is None or sigma_index != previous[1]:
            sigma = (4.0, 2.0, 1.0, 0.5)[sigma_index]
            source_grid = parameterize_curvature(current, normalized)
            curvature = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                current, blur_atlas_frame(source_grid, sigma)))
            mean_curve = normalize_mean_curvature(sample_atlas_on_canonical_sphere(
                current, blur_atlas_frame(raw_mean, sigma)))
            mean_grid = parameterize_curvature(current, mean_curve)
            variance_grid = blur_atlas_frame(raw_variance, sigma)
        fold = stage == "fold_cleanup"
        l_parea = float(np.float32(np.float32(0.2) / 100)) if fold else 0.2
        l_nlarea = 100.0 if fold else 1.0
        l_dist = float(np.float32(np.float32(5.0) / 100)) if fold else 5.0
        l_corr = float(np.float32(np.float32(0.05) / 100)) if fold else float(np.float32(0.05))
        l_spring = float(np.float32(np.float32(0.5) / 100)) if fold else 0.5
        integration_start = previous is None or state[:3] != previous[:3]
        projected = project_sphere(current) if integration_start else current
        normals = sphere_vertex_normals(projected, triangles, incidence=cache.incidence)
        distances = sphere_arc_distances(projected, neighbors, degrees)
        avg_vertex_dist = float(distances.double().sum() / degrees.sum())
        if index == 0:
            force = first_area_gradient(
                vertices, original, current, triangles,
                first_distance_gradient(vertices, original, current, triangles,
                                        cache=cache), cache=cache)
        else:
            force = first_area_gradient(
                vertices, original, projected, triangles,
                first_distance_gradient(vertices, original, projected, triangles,
                                        project=False, weight=l_dist, cache=cache),
                project=False, l_nlarea=l_nlarea, l_parea=l_parea, cache=cache)
        e1, e2 = tangent_basis(normals)
        force = correlation_gradient_add(force, projected, curvature, e1, e2,
                                         mean_grid, variance_grid, avg_vertex_dist,
                                         l_corr=l_corr)
        force = average(force, averages)
        force = spring_gradient_add(force, projected, neighbors, degrees,
                                    dist_scale, l_spring)

        def objective(trial: torch.Tensor) -> float:
            terms = first_registration_sse(
                trial, triangles, neighbors, degrees, original_distances,
                original_areas, curvature, mean_grid, variance_grid,
                original_area, total_area, return_terms=True)
            trial_distances = sphere_arc_distances(trial, neighbors, degrees)
            spring = area_scale * float(((trial_distances.double() ** 2) * active).sum())
            return ((l_parea / 0.2) * terms["sse_area"]
                    + l_nlarea * terms["sse_nl_area"]
                    + (l_dist / 5.0) * terms["sse_dist"]
                    + l_corr * terms["sse_corr"] + l_spring * spring)

        dt, samples = first_registration_line_search(projected, force, objective)
        current = apply_spherical_gradient(projected, force, dt)
        selected_sse = min(samples, key=lambda item: abs(item[0] - dt))[1]
        negative_faces = 0
        if stage == "smoothwm" and sigma_index == 3 and averages == 0:
            signed_area, _ = face_area_normals(current, triangles, signed_sphere=True)
            negative_faces = int((signed_area < 0).sum())
        next_state = next_smoothwm_scale(
            *state, samples[0][1], selected_sse, dt,
            negative_faces=negative_faces)
        updates.append({"iteration": seed_iteration + index + 1,
                        "stage": stage, "sigma": sigma, "averages": averages,
                        "dt": dt, "next_state": next_state,
                        "seconds": time.perf_counter() - step_start})
        previous, state = state, next_state
    if state is not None:
        raise RuntimeError("smoothwm registration did not converge within max_updates")
    integration_seconds = time.perf_counter() - started - setup_seconds
    repair_start = time.perf_counter()
    repaired, negative_counts = remove_overlap_sphere(
        current.to(overlap_device), triangles.to(overlap_device),
        start_iteration=seed_iteration + len(updates))
    repair_seconds = time.perf_counter() - repair_start
    write_standard_sphere_surface(
        output, repaired.cpu().numpy(), faces, sphere,
        create_stamp="created by Python sphere registration")
    return {"sphere": str(sphere), "smoothwm": str(smoothwm),
            "sulc_seed": str(sulc_seed), "atlas": str(atlas_file),
            "output": str(output), "seed_iteration": seed_iteration,
            "overlap_device": overlap_device, "setup_seconds_including_raw_fit": setup_seconds,
            "averaging_device": str(average.device),
            "integration_seconds": integration_seconds, "updates": updates,
            "negative_counts": negative_counts, "repair_seconds": repair_seconds,
            "total_seconds_including_io": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sphere", "smoothwm", "sulc_seed", "atlas", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed-iteration", type=int, required=True)
    parser.add_argument("--overlap-device", default="cpu")
    parser.add_argument("--averaging-device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_register_smoothwm(
        args.sphere, args.smoothwm, args.sulc_seed, args.atlas, args.output,
        seed_iteration=args.seed_iteration, overlap_device=args.overlap_device,
        averaging_device=args.averaging_device)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
