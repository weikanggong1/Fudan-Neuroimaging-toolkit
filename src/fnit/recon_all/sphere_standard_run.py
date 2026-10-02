"""Run FreeSurfer 8.2's conventional sphere stage without native executables."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from .sphere_python import project_radially
from .place_surface_normals import FaceNormalTopology
from .mris_register_average_numba import RegistrationGradientAverager
from .sphere_standard_average import average_standard_gradient
from .sphere_standard_finish import finish_standard_sphere
from .sphere_standard_line_search import first_epoch_line_search, first_epoch_sse
from .sphere_standard_metric import average_standard_metric, sample_standard_metric_matrix
from .sphere_standard_nonlinear import (
    nonlinear_epoch_gradient, nonlinear_epoch_line_search, nonlinear_epoch_sse,
    one_ring_metric,
)
from .sphere_standard_python import project_before_standard_unfold, write_standard_sphere_surface
from .sphere_standard_schedule import next_standard_sphere_scale
from .sphere_standard_unfold import _face_geometry, first_epoch_gradient


def run_standard_sphere(inflated: str | Path, smoothwm: str | Path,
                        output: str | Path, *, finish_device: str = "cpu",
                        averaging_device: str = "cpu") -> dict:
    """Read same-face-order inflated/smoothwm meshes and write an ordered sphere.

    averaging_device默认cpu；显式CUDA时复用RegistrationGradientAverager，
    只迁移有序Jacobi平均，包含必要H2D/D2H，不改TF32或精度。主体SSE仍CPU。
    缓存本次有序面的整数CSR及原smoothwm面积；坐标、法向每次重算。
    Return paths, device, negative-area fraction, setup/finish/total seconds,
    ordered update dicts (index, stage, weight, averages, dt, seconds), and
    final overlap-repair negative-face counts. Only cleanup uses finish_device.
    """
    started = time.perf_counter()
    inflated, smoothwm, output = Path(inflated), Path(smoothwm), Path(output)
    xyz_input, faces = fsio.read_geometry(str(inflated))
    metric_input, metric_faces = fsio.read_geometry(str(smoothwm))
    if not np.array_equal(faces, metric_faces):
        raise ValueError("inflated and smoothwm face order differs")
    faces = np.asarray(faces, np.int32)
    xyz = project_radially(project_before_standard_unfold(xyz_input), already_sphere=True)
    area, _ = _face_geometry(xyz, faces)
    negative_area = float(np.float32(-np.sum(area[area < 0], dtype=np.float64)))
    negative_pct = 100 * negative_area / (4 * math.pi * 10000 + negative_area)
    stage, weight = (("initial_repair", 1e-6) if negative_pct > 0.005
                     else ("unfold_epoch_1", 0.1))
    averages, steps_at_scale, prior_scale = 1024, 0, None
    setup_projection_seconds = time.perf_counter() - started
    t0 = time.perf_counter()
    offsets, neighbors, raw, _ = sample_standard_metric_matrix(metric_input, faces)
    distances, _, _ = average_standard_metric(offsets, neighbors, raw)
    local_offsets, local_neighbors, local_distances, old_avg = one_ring_metric(
        faces, len(xyz), offsets, neighbors, distances)
    original_area, _ = _face_geometry(np.asarray(metric_input, np.float32), faces)
    original_area = np.abs(original_area)
    original_total = np.float32(np.sum(original_area, dtype=np.float64))
    setup_metric_seconds = time.perf_counter() - t0
    topology_tick = time.perf_counter()
    normal_topology = FaceNormalTopology(faces, len(xyz))
    original_metric = (original_area, original_total)
    average = None
    if torch.device(averaging_device).type == "cuda":
        degree = np.diff(local_offsets).astype(np.int64)
        dense = np.zeros((len(xyz), int(degree.max(initial=0))), np.int64)
        for vertex in range(len(xyz)):
            dense[vertex, :degree[vertex]] = local_neighbors[local_offsets[vertex]:local_offsets[vertex+1]]
        average = RegistrationGradientAverager(torch.from_numpy(dense),
            torch.from_numpy(degree), device=averaging_device)
    elif torch.device(averaging_device).type != "cpu":
        raise ValueError("averaging_device must be cpu or cuda")
    topology_seconds = time.perf_counter() - topology_tick
    updates = []
    for index in range(3000):
        scale = (stage, weight, averages)
        if prior_scale is not None and scale != prior_scale:
            xyz = project_radially(xyz, already_sphere=True)
        t0 = time.perf_counter()
        if stage == "nonlinear_repair":
            gradient, _ = nonlinear_epoch_gradient(
                xyz, faces, metric_input, local_offsets, local_neighbors,
                local_distances, old_avg, weight,
                normal_topology=normal_topology, original_metric=original_metric)
        else:
            _, _, gradient, _ = first_epoch_gradient(
                xyz, faces, metric_input, offsets, neighbors, distances, weight,
                normal_topology=normal_topology, original_metric=original_metric)
        if average is None:
            gradient = average_standard_gradient(
                gradient, local_offsets, local_neighbors, averages)
        else:
            gradient = average(torch.from_numpy(gradient), averages).numpy()
        if stage == "nonlinear_repair":
            search = nonlinear_epoch_line_search(
                xyz, gradient, faces, local_offsets, local_neighbors,
                local_distances, original_area, original_total, weight)
        else:
            search = first_epoch_line_search(
                xyz, gradient, faces, offsets, neighbors, distances,
                original_area, original_total, weight)
        shifted = (xyz.astype(np.float64) + search["selected_dt"]
                   * gradient.astype(np.float64)).astype(np.float32)
        xyz = project_radially(shifted, already_sphere=True)
        if stage == "nonlinear_repair":
            ending_sse = nonlinear_epoch_sse(
                xyz, faces, local_offsets, local_neighbors,
                local_distances, original_total, weight)["total"]
        else:
            ending_sse = first_epoch_sse(
                xyz, faces, offsets, neighbors, distances,
                original_area, original_total, weight)["total"]
        next_scale = next_standard_sphere_scale(
            stage, weight, averages, steps_at_scale,
            search["starting_sse"]["total"], ending_sse,
            search["selected_dt"])
        updates.append({"index": index, "stage": stage, "weight": weight,
                        "averages": averages, "dt": search["selected_dt"],
                        "seconds": time.perf_counter() - t0})
        if next_scale is None:
            break
        prior_scale = scale
        stage, weight, averages, steps_at_scale = next_scale
    else:
        raise RuntimeError("conventional sphere integration did not converge")
    t0 = time.perf_counter()
    finished, negative_counts = finish_standard_sphere(
        xyz, faces, start_iteration=len(updates), device=finish_device)
    finish_seconds = time.perf_counter() - t0
    write_standard_sphere_surface(output, finished, faces, inflated)
    return {"inflated": str(inflated), "smoothwm": str(smoothwm),
            "output": str(output), "finish_device": finish_device, "averaging_device": averaging_device,
            "initial_negative_area_pct": negative_pct,
            "projection_seconds": setup_projection_seconds,
            "metric_seconds_including_jit": setup_metric_seconds,
            "topology_and_gpu_prepare_seconds": topology_seconds,
            "updates": updates, "negative_counts": negative_counts,
            "finish_seconds": finish_seconds,
            "total_seconds_including_io": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inflated", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--finish-device", default="cpu")
    parser.add_argument("--averaging-device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = run_standard_sphere(
        args.inflated, args.smoothwm, args.output,
        finish_device=args.finish_device, averaging_device=args.averaging_device)
    content = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
