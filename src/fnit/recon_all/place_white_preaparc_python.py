"""同输入 white.preaparc 的首轮诊断及显式四轮实验接口。

复用已有白质 MRI、rip、目标强度、自斥力和异步碰撞实现。完整实验接口
执行四轮与相交清理；首轮诊断接口保持原行为。两者均不替代生产默认路径。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from .place_surface_final_cleanup import repair_intersections
from .place_surface_gradient_average import average_signed_gradients
from .place_surface_intensity import intensity_gradient
from .place_surface_normals import CoordinateNormalCache, FaceNormalTopology
from .place_surface_objective import intensity_error, surface_total_area, tangential_spring_energy
from .place_surface_rip import rip_white_preaparc_pass
from .place_surface_self_repulsion import (
    mean_vertex_spacing, self_repulsion_energy, self_repulsion_gradient, vertex_buckets_current,
)
from .place_surface_smoothing import average_marked_values, average_vertex_positions, _ordered_neighbors
from .place_surface_spring import spring_gradient
from .place_surface_step import unconstrained_step_with_offsets
from .place_surface_volume import prepare_placement_volume


def _place_white_preaparc(
    subject_dir: str | Path, hemi: str, output: str | Path,
    *, steps: int = 1, diagnostics: str | Path | None = None,
    regularization_backend: str = "cpu", device: str | None = None,
    complete: bool = False, candidate_backend: str = "tree", trace_callback=None,
    output_volume: str | Path | None = None, sampling_backend: str = "cpu",
    cleanup_marking_backend: str = "legacy",
    candidate_grid_cells_per_axis: int = 2,
    collision_profile: bool = False,
    retained_mht_backend: str = "tree",
) -> dict:
    """共享现有白质算子；complete 选择四轮而非首轮诊断调度。"""
    started = time.perf_counter()
    if complete:
        if steps < 1:
            raise ValueError("max_steps must be positive")
    elif not 1 <= steps <= 17:
        raise ValueError("steps must be from 1 to 17")
    if candidate_backend not in ("tree", "snapshot", "torch_snapshot"):
        raise ValueError("invalid candidate_backend")
    if candidate_grid_cells_per_axis not in (2, 3):
        raise ValueError("candidate_grid_cells_per_axis must be 2 or 3")
    if candidate_grid_cells_per_axis != 2 and candidate_backend != "torch_snapshot":
        raise ValueError("nondefault candidate grid requires torch_snapshot")
    if retained_mht_backend not in ("tree", "compiled"):
        raise ValueError("retained_mht_backend must be tree or compiled")
    if retained_mht_backend == "compiled" and candidate_backend == "tree":
        raise ValueError("compiled retained MHT requires snapshot candidates")
    if cleanup_marking_backend not in ("legacy", "source_numba", "source_torch"):
        raise ValueError("invalid cleanup_marking_backend")
    if cleanup_marking_backend == "source_torch" and device is None:
        raise ValueError("source_torch cleanup requires an explicit device")
    if sampling_backend not in ("cpu", "torch", "triton"):
        raise ValueError("sampling_backend must be cpu, torch or triton")
    if sampling_backend != "cpu" and device is None:
        raise ValueError("GPU sampling requires an explicit device")
    if candidate_backend == "torch_snapshot" and device is None:
        raise ValueError("torch_snapshot requires an explicit device")
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be lh or rh")
    if regularization_backend not in ("cpu", "torch"):
        raise ValueError("regularization_backend must be cpu or torch")
    if regularization_backend == "torch" and device is None:
        raise ValueError("Torch regularization requires an explicit device")
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
    surface_loaded_at = time.perf_counter()
    xyz = average_vertex_positions(vertices, faces, 5)
    initial_cleanup = None
    def clean_intersections(current, rip_flags):
        if cleanup_marking_backend == "legacy":
            return repair_intersections(current, faces, rip_flags)
        return repair_intersections(current, faces, rip_flags,
            marking_backend=cleanup_marking_backend, device=device)
    if complete:
        xyz, initial_cleanup = clean_intersections(xyz, np.zeros(len(xyz), dtype=np.bool_))
        # 固定源码的 MRISremoveIntersections 可在非零残余时正常返回，随后
        # placement 继续优化几何。保留初始化诊断；零相交门只用于最终输出。
    initial_cleanup_finished_at = time.perf_counter()
    normal_topology = FaceNormalTopology(faces, len(xyz))
    normal_cache = CoordinateNormalCache(normal_topology)
    normals = normal_cache.evaluate(xyz)
    initial_normals_finished_at = time.perf_counter()
    brain = nib.load(str(brain_path))
    seg_image = nib.load(str(seg_path))
    wm_image = nib.load(str(wm_path))
    if complete:
        for image, path in ((seg_image, seg_path), (wm_image, wm_path)):
            if image.shape != brain.shape or not np.array_equal(image.affine, brain.affine):
                raise ValueError(f"white MRI grids differ: {path}")
    seg = np.asarray(seg_image.dataobj)
    brain_data, wm_data = np.asarray(brain.dataobj), np.asarray(wm_image.dataobj)
    mri_loaded_at = time.perf_counter()
    volume, _ = prepare_placement_volume(
        brain_data, wm_data,
        surface="white", mid_gray=float(stats["MID_GRAY"]),
    )
    del brain_data, wm_data
    volume_prepared_at = time.perf_counter()
    rip_affine = surface_ras_to_voxel(seg_image.header, metadata)
    ripped = values = None
    for _ in range(2):
        ripped, values = rip_white_preaparc_pass(
            xyz, normals, faces, seg, volume, rip_affine, hemisphere=hemi,
            ripped=ripped, values=values,
        )
    initial_ripping_finished_at = time.perf_counter()
    affine = surface_ras_to_voxel(brain.header, metadata)
    sampler = None
    if sampling_backend != "cpu":
        from .place_surface_sampling import PlacementSampling
        sampler = PlacementSampling(volume, affine, device=device, implementation=sampling_backend)
    thresholds = np.array([float(stats[f"white_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    border = compute_border_values_first_pass(
        volume, seg, xyz, normals, xyz, ripped, values, affine, thresholds,
        hemisphere=hemi, surface="white", sigma=2.0,
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    first_border_finished_at = time.perf_counter()
    ordered_indices, ordered_valid, _ = _ordered_neighbors(faces, len(xyz))
    ordered = (ordered_indices, ordered_valid)
    two_offsets, two_neighbors = two_ring_neighbors(
        faces, len(xyz), ordered_neighbors=ordered)
    regularizer = None
    if regularization_backend == "torch":
        from .place_surface_regularization_torch import PlacementRegularizationTorch
        regularizer = PlacementRegularizationTorch(
            neighbors=ordered_indices, valid=ordered_valid, offsets=two_offsets,
            candidates=two_neighbors, ripped=ripped, device=device)
    original_area = surface_total_area(xyz, faces)

    objective_cache_vertices: np.ndarray | None = None
    objective_cache_result: tuple[float, float] | None = None

    def objective(current: np.ndarray) -> tuple[float, float]:
        nonlocal objective_cache_vertices, objective_cache_result
        if objective_cache_vertices is current and objective_cache_result is not None:
            return objective_cache_result
        current_normals = normal_cache.evaluate(current)
        intensity_sse, rms, _ = intensity_error(volume, current, values, ripped, affine)
        spring = tangential_spring_energy(
            current, current_normals, faces, ripped, ordered_neighbors=ordered)
        area = surface_total_area(current, faces)
        spacing = mean_vertex_spacing(current, ripped, two_offsets, two_neighbors)
        offsets, members = vertex_buckets_current(current, ripped, resolution=spacing)
        repulsion = self_repulsion_energy(
            current, ripped, offsets, members, two_offsets, two_neighbors, weight=5.0)
        area_scale = float(np.float32(original_area / area))
        result = (float(np.float32(0.3)) * spring * area_scale
                  + float(np.float32(0.2)) * intensity_sse + repulsion, rms)
        objective_cache_vertices = current
        objective_cache_result = result
        return result

    prepared_at = time.perf_counter()
    prepare_components = {
        "input_validation_stats_and_surface_read": surface_loaded_at - started,
        "surface_smoothing_and_initial_cleanup": initial_cleanup_finished_at - surface_loaded_at,
        "initial_normals": initial_normals_finished_at - initial_cleanup_finished_at,
        "MRI_read_geometry_check_and_materialization": mri_loaded_at - initial_normals_finished_at,
        "placement_volume_preparation": volume_prepared_at - mri_loaded_at,
        "initial_ripping": initial_ripping_finished_at - volume_prepared_at,
        "sampling_setup_first_border_and_target_smoothing": first_border_finished_at - initial_ripping_finished_at,
        "topology_context_and_objective_setup": prepared_at - first_border_finished_at,
    }
    initial_sse, initial_rms = objective(xyz)
    initial_objective_at = time.perf_counter()
    current = xyz.copy()
    last_sse, last_rms = initial_sse, initial_rms
    cropped = np.zeros(len(xyz), dtype=np.int32)
    dt, reductions = 0.5, 0
    outer_pass, pass_iteration, sigma, n_averages = 0, 0, 2.0, 4
    pass_initial_sse, pass_initial_rms = initial_sse, initial_rms
    pass_ends, pass_records = [], []
    border_seconds = cleanup_seconds = 0.0
    gradient_seconds = collision_seconds = objective_seconds = 0.0
    collision_details = []
    records: list[dict] = []
    snapshots: dict[str, np.ndarray | float] = {
        "initial": xyz, "ripped": ripped, "target_values": values,
        "initial_sse": initial_sse, "initial_rms": initial_rms,
    } if diagnostics is not None else {}
    for step in range(1, steps + 1):
        stage_start = time.perf_counter()
        normals = normal_cache.evaluate(current)
        if sampler is None:
            intensity = intensity_gradient(
                volume, current, normals, ripped, values, border[5], affine,
                brain.header.get_zooms()[:3], weight=0.2, sigma_global=sigma,
            )
        else:
            intensity = sampler.gradient(
                current, normals, ripped, values, border[5], brain.header.get_zooms()[:3],
                weight=0.2, sigma_global=sigma,
            )
        bucket_offsets, bucket_members = vertex_buckets_current(current, ripped)
        self_repulsion = self_repulsion_gradient(
            current, ripped, bucket_offsets, bucket_members,
            two_offsets, two_neighbors, weight=5.0,
        )
        # Diagnostics retains the original individual arrays; the optional
        # Torch result is still used for the actual candidate step below.
        if regularizer is None or diagnostics is not None:
            averaged = average_signed_gradients(
                intensity, faces, ripped, n_averages, ordered_neighbors=ordered)
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
        if regularizer is not None:
            gradient = regularizer.regularize(
                vertices=current, normals=normals, gradient=intensity,
                iterations=n_averages, spring_weight=.3, after_average=self_repulsion)
        gradient_seconds += time.perf_counter() - stage_start
        before_collision = gradient.copy() if diagnostics is not None else None
        if diagnostics is not None:
            snapshots[f"step{step}_initial"] = current.copy()
            snapshots[f"step{step}_tangential_spring"] = before_collision
        stale_trial = None
        trial_trace = []
        accepted = None
        for trial in range(3):
            trial_start = time.perf_counter()
            candidate_details = {} if collision_profile else None
            proposed, offsets = unconstrained_step_with_offsets(current, gradient, ripped, dt=dt)
            placed, _ = asynchronous_first_step(
                current, faces, proposed, ripped, fast=True, offsets=offsets,
                accepted_offsets=gradient, stale_mht_trial=stale_trial,
                ordered_neighbors=ordered, candidate_backend=candidate_backend,
                candidate_device=device if candidate_backend == "torch_snapshot" else None,
                candidate_grid_cells_per_axis=candidate_grid_cells_per_axis,
                candidate_diagnostics=candidate_details,
                retained_mht_backend=retained_mht_backend,
            )
            collision_elapsed = time.perf_counter() - trial_start
            collision_seconds += collision_elapsed
            if candidate_details is not None:
                collision_details.append({"step": step, "trial": trial,
                    "total_collision_seconds": collision_elapsed, **candidate_details})
            blocked = np.any(proposed != current, axis=1) & np.all(placed == current, axis=1)
            cropped = np.where(ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
            objective_start = time.perf_counter()
            step_sse, step_rms = objective(placed)
            objective_seconds += time.perf_counter() - objective_start
            next_dt, reductions, reduced, rejected, stop = pial_step_decision(
                last_sse, last_rms, step_sse, step_rms, dt, reductions)
            trial_trace.append({
                "trial": trial, "dt_used": dt, "dt_next": next_dt,
                "sse": step_sse, "rms": step_rms, "reduced": bool(reduced),
                "rejected": bool(rejected), "stop": bool(stop), "reductions": reductions,
            })
            dt = next_dt
            if rejected:
                stale_trial = placed
                if stop:
                    if complete:
                        accepted = current
                        step_sse, step_rms = last_sse, last_rms
                        break
                    raise RuntimeError(f"white prefix rejected step {step} after {trial + 1} trials")
                continue
            accepted = placed
            break
        else:
            raise RuntimeError(f"white optimizer rejected all trials at step {step}")
        current = accepted
        last_sse, last_rms = step_sse, step_rms
        records.append({
            "step": step, "pass_index": outer_pass, "sigma": sigma, "averages": n_averages,
            "trials": trial + 1, "trial_trace": trial_trace, "sse": step_sse, "rms": step_rms,
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
        if trace_callback is not None:
            trace_callback(step, outer_pass, current.copy(), deepcopy(records[-1]))
        pass_iteration += 1
        if stop or (complete and pass_iteration == 100):
            if not complete:
                break
            pass_ends.append(step)
            pass_records.append({
                "pass_index": outer_pass, "iterations": pass_iteration,
                "sigma": sigma, "averages": n_averages,
                "ripped_vertices": int(np.count_nonzero(ripped)),
                "initial_sse": pass_initial_sse, "initial_rms": pass_initial_rms,
                "final_sse": last_sse, "final_rms": last_rms,
                "reason": "max_reductions" if stop else "iteration_limit",
            })
            if outer_pass == 3:
                break
            outer_pass += 1
            pass_iteration, sigma, n_averages = 0, 2.0 / (1 << outer_pass), 4 >> outer_pass
            border_started = time.perf_counter()
            current_normals = normal_cache.evaluate(current)
            ripped, values = rip_white_preaparc_pass(
                current, current_normals, faces, seg, volume, rip_affine,
                hemisphere=hemi, ripped=ripped, values=values,
            )
            border = compute_border_values_first_pass(
                volume, seg, current, current_normals, xyz, ripped, values,
                affine, thresholds, hemisphere=hemi, surface="white", sigma=sigma,
            )
            values = average_marked_values(border[0], border[4], ripped, faces, 5)
            original_area = surface_total_area(current, faces)
            if regularizer is not None:
                regularizer = PlacementRegularizationTorch(
                    neighbors=ordered_indices, valid=ordered_valid, offsets=two_offsets,
                    candidates=two_neighbors, ripped=ripped, device=device,
                )
            objective_cache_vertices = objective_cache_result = None
            border_seconds += time.perf_counter() - border_started
            objective_started = time.perf_counter()
            last_sse, last_rms = objective(current)
            pass_initial_sse, pass_initial_rms = last_sse, last_rms
            objective_seconds += time.perf_counter() - objective_started
            dt, reductions = 0.5, 0
    else:
        if complete:
            raise RuntimeError(f"white optimizer did not complete four passes in {steps} steps")
    cleanup = None
    if complete:
        cleanup_started = time.perf_counter()
        current, cleanup = clean_intersections(current, ripped)
        cleanup_seconds = time.perf_counter() - cleanup_started
        if cleanup["intersecting_faces_after"]:
            error = RuntimeError("white cleanup left unresolved intersections")
            error.intersection_cleanup = {"initial": initial_cleanup, "final": cleanup}
            error.intersection_coordinates = current.copy()
            error.intersection_faces = faces.copy()
            error.partial_stage = {
                "passes": pass_records, "steps": len(records), "per_step": records,
                "cleanup_marking_backend": cleanup_marking_backend,
                "prepare_components": prepare_components,
                "seconds_to_failed_final_gate": time.perf_counter() - started,
                "measured_stage_seconds": {"prepare": prepared_at - started,
                    "initial_objective": initial_objective_at - prepared_at,
                    "gradient": gradient_seconds, "collision": collision_seconds,
                    "step_objective": objective_seconds, "border_updates": border_seconds,
                    "cleanup": cleanup_seconds},
            }
            raise error
    before_write = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_vertices_like(orig, output, current)
    if complete and output_volume is not None:
        volume_output = Path(output_volume)
        volume_output.parent.mkdir(parents=True, exist_ok=True)
        volume_header = brain.header.copy()
        volume_header.set_data_dtype(np.uint8)
        nib.save(nib.MGHImage(volume, brain.affine, header=volume_header), str(volume_output))
    if diagnostics is not None:
        diagnostic_path = Path(diagnostics)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(diagnostic_path, **snapshots)
    finished_at = time.perf_counter()
    return {
        "output": str(output), "hemisphere": hemi, "steps": len(records),
        "regularization_backend": regularization_backend, "device": device,
        "sampling_backend": sampling_backend,
        "complete_four_passes": complete, "candidate_backend": candidate_backend,
        "cleanup_marking_backend": cleanup_marking_backend,
        "pass_ends": pass_ends, "passes": pass_records, "initial_cleanup": initial_cleanup,
        "cleanup": cleanup, "output_volume": str(output_volume) if output_volume is not None else None,
        "vertices": int(len(xyz)), "faces": int(len(faces)),
        "ripped_vertices": int(np.count_nonzero(ripped)),
        "held_vertices": records[-1]["held_vertices"],
        "initial_sse": initial_sse, "initial_rms": initial_rms,
        "step_sse": last_sse, "step_rms": last_rms,
        "per_step": records,
        "candidate_grid_cells_per_axis": candidate_grid_cells_per_axis,
        "retained_mht_backend": retained_mht_backend,
        "collision_profile": bool(collision_profile), "collision_details": collision_details,
        "prepare_components": prepare_components,
        "seconds": finished_at - started,
        "stage_seconds": {
            "prepare": prepared_at - started,
            "initial_objective": initial_objective_at - prepared_at,
            "gradient": gradient_seconds,
            "collision": collision_seconds,
            "step_objective": objective_seconds,
            "border_updates": border_seconds, "cleanup": cleanup_seconds,
            "write": finished_at - before_write,
            "control": before_write - initial_objective_at - gradient_seconds
                       - collision_seconds - objective_seconds - border_seconds - cleanup_seconds,
        },
    }



def place_white_preaparc_prefix(
    subject_dir: str | Path, hemi: str, output: str | Path,
    *, steps: int = 1, diagnostics: str | Path | None = None,
    regularization_backend: str = "cpu", device: str | None = None,
) -> dict:
    """首轮1–17步诊断；输入MRI/conform和orig surface RAS/mm，非完整white。

    参数、输出NPZ及失败行为保持原接口；不执行完整四轮或最终清理。
    """
    return _place_white_preaparc(
        subject_dir=subject_dir, hemi=hemi, output=output, steps=steps,
        diagnostics=diagnostics, regularization_backend=regularization_backend, device=device,
    )


def place_white_preaparc(
    subject_dir: str | Path, hemi: str, output: str | Path, *, max_steps: int = 400,
    output_volume: str | Path | None = None, regularization_backend: str = "cpu",
    candidate_backend: str = "tree", sampling_backend: str = "cpu",
    device: str | None = None, trace_callback=None,
    cleanup_marking_backend: str = "legacy",
    candidate_grid_cells_per_axis: int = 2,
    collision_profile: bool = False,
    retained_mht_backend: str = "tree",
) -> dict:
    """实验性完整preaparc白质四轮；不替代带aparc的最终white或生产默认。

    输入orig、灰白阈值、brain.finalsurfs/wm/aseg.presurf，MRI为同一conform
    网格，表面为surface RAS/mm。output必须独立于输入orig；可选output_volume
    写uint8预处理MRI、保留MRI几何。max_steps默认400，总保护上限；每轮
    最多100步，averages4/2/1/0、sigma2/1/.5/.25。默认CPU；Torch正则/候选
    须明确device。sampling_backend=cpu默认，torch/triton复用既有GPU强度
    采样，MRI仅缓存一次；每步仍传入当前rip/目标/sigma，不复用过期状态。
    trace_callback接收(step,pass_index,坐标副本,试步诊断)。
    cleanup_marking_backend默认legacy；source_numba/source_torch按固定源码
    逐方向及共享1mm桶标记，Torch需device。两次清理使用同一后端，
    初始允许源程序既有残余，最终仍要求零相交；不改已验证pial默认。
    candidate_grid_cells_per_axis默认2，显式3仅Torch候选使用较小完整网格；
    不裁剪候选或改变有序接受。collision_profile默认False，True记录每次
    候选构建/有序接受秒，诊断与接受轨迹分开，不在生产默认增加同步。
    retained_mht_backend默认tree；显式compiled仅用于snapshot候选，将重试的
    有序循环编译，实际三角命中仍调用原FP64桶规则和实时坐标，非纯GPU。
    返回路径、有序网格大小、四轮边界、rip/目标/接受轨迹、完整清理与分项秒。
    输入/参数、未完成四轮、残余相交及CUDA异常传播，不写未完成表面。
    对应mris_place_surface --white --nsmooth 5 --rip-bg-no-annot --rip-bg。
    """
    subject = Path(subject_dir)
    inputs = {
        (subject / f"surf/{hemi}.orig").resolve(),
        (subject / f"surf/autodet.gw.stats.{hemi}.dat").resolve(),
        *((subject / f"mri/{name}.mgz").resolve()
          for name in ("brain.finalsurfs", "wm", "aseg.presurf")),
    }
    destinations = [Path(output).resolve()]
    if output_volume is not None:
        destinations.append(Path(output_volume).resolve())
    if any(path in inputs for path in destinations) or len(set(destinations)) != len(destinations):
        raise ValueError("experimental white outputs cannot overwrite inputs or each other")
    return _place_white_preaparc(
        subject_dir=subject_dir, hemi=hemi, output=output, steps=max_steps,
        complete=True, output_volume=output_volume,
        regularization_backend=regularization_backend, candidate_backend=candidate_backend,
        sampling_backend=sampling_backend, device=device, trace_callback=trace_callback,
        cleanup_marking_backend=cleanup_marking_backend,
        candidate_grid_cells_per_axis=candidate_grid_cells_per_axis,
        collision_profile=collision_profile,
        retained_mht_backend=retained_mht_backend,
    )


def first_white_preaparc_step(
    subject_dir: str | Path, hemi: str, output: str | Path,
    *, diagnostics: str | Path | None = None,
    regularization_backend: str = "cpu", device: str | None = None,
) -> dict:
    """Run one diagnostic white optimizer step without final surface cleanup."""
    return place_white_preaparc_prefix(
        subject_dir=subject_dir, hemi=hemi, output=output,
        steps=1, diagnostics=diagnostics,
        regularization_backend=regularization_backend, device=device,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--steps", type=int, choices=range(1, 18), default=1)
    parser.add_argument("--regularization-backend", choices=("cpu", "torch"), default="cpu")
    parser.add_argument("--device")
    parser.add_argument("--complete", action="store_true", help="实验性完整 white.preaparc 四轮")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-volume", type=Path)
    parser.add_argument("--candidate-backend", choices=("tree", "snapshot", "torch_snapshot"), default="tree")
    parser.add_argument("--candidate-grid-cells-per-axis", type=int, choices=(2, 3), default=2)
    parser.add_argument("--collision-profile", action="store_true")
    parser.add_argument("--retained-mht-backend", choices=("tree", "compiled"), default="tree")
    parser.add_argument("--sampling-backend", choices=("cpu", "torch", "triton"), default="cpu")
    parser.add_argument("--cleanup-marking-backend", choices=("legacy", "source_numba", "source_torch"), default="legacy")
    args = parser.parse_args()
    if args.complete:
        if args.diagnostics is not None or args.steps != 1:
            parser.error("--complete uses --max-steps and does not accept prefix diagnostics/steps")
        print(json.dumps(place_white_preaparc(
            subject_dir=args.subject_dir, hemi=args.hemi, output=args.output,
            max_steps=args.max_steps, output_volume=args.output_volume,
            regularization_backend=args.regularization_backend,
            candidate_backend=args.candidate_backend, sampling_backend=args.sampling_backend,
            cleanup_marking_backend=args.cleanup_marking_backend,
            candidate_grid_cells_per_axis=args.candidate_grid_cells_per_axis,
            collision_profile=args.collision_profile,
            retained_mht_backend=args.retained_mht_backend,
            device=args.device,
        ), indent=2))
        return
    if (args.output_volume is not None or args.max_steps != 400
            or args.candidate_backend != "tree" or args.sampling_backend != "cpu"
            or args.cleanup_marking_backend != "legacy"
            or args.candidate_grid_cells_per_axis != 2 or args.collision_profile
            or args.retained_mht_backend != "tree"):
        parser.error("--output-volume, --max-steps, --sampling-backend and --candidate-backend require --complete")
    print(json.dumps(place_white_preaparc_prefix(
        subject_dir=args.subject_dir, hemi=args.hemi, output=args.output,
        steps=args.steps, diagnostics=args.diagnostics,
        regularization_backend=args.regularization_backend, device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
