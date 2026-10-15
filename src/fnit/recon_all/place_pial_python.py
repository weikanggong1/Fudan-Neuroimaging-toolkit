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
from numba import get_num_threads

from .place_surface_border import compute_border_values_first_pass
from .place_surface_collision import asynchronous_first_step
from .place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
from .place_surface_decision import pial_step_decision
from .place_surface_final_cleanup import pin_medial_wall, repair_intersections, pial_cleanup_mask
from .place_surface_geometry import surface_ras_to_voxel
from .place_surface_gradient_average import average_signed_gradients
from .place_surface_intensity import intensity_gradient
from .place_surface_normals import CoordinateNormalCache, FaceNormalTopology
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


def _validate_pial_volume_geometry(brain, wm, aseg, surface_metadata) -> np.ndarray:
    """校验三张 MGZ 的同一三维网格，返回既有 surface RAS→voxel 变换。

    brain/wm/aseg 是已加载的 nibabel MGHImage；只读取头信息，不复制影像。
    三者须有完全相同的 shape、scanner RAS affine，单位 mm；
    affine/vox2ras_tkr/间距须有限且间距为正。MRI各自的vox2ras_tkr未用于
    共享体素索引，不要求其位相同；MGH方向与间距的等价分解可使它不同。
    surface_metadata 是 white 的完整体积
    几何，允许与 MRI 不同的有效几何，由既有变换处理；不要求重置其头。
    缺失、非有限或退化几何抛 ValueError，失败发生在强度准备和优化前。
    返回 float32 (4,4)，复用 surface_ras_to_voxel 的原顺序运算，无重采样。
    """
    for name, image in (("brain.finalsurfs", brain), ("wm", wm), ("aseg.presurf", aseg)):
        if len(image.shape) != 3:
            raise ValueError(f"pial MRI must be a 3D volume: {name}")
        affine = np.asarray(image.affine)
        spacing = np.asarray(image.header.get_zooms()[:3])
        if affine.shape != (4, 4) or not np.isfinite(affine).all() \
                or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError(f"pial MRI geometry must be finite with positive spacing: {name}")
        try:
            tkr = np.asarray(image.header.get_vox2ras_tkr())
        except AttributeError as error:
            raise ValueError(f"pial MRI requires MGH surface-RAS geometry: {name}") from error
        if tkr.shape != (4, 4) or not np.isfinite(tkr).all():
            raise ValueError(f"pial MRI surface-RAS geometry must be finite: {name}")
        if image.shape != brain.shape or not np.array_equal(affine, brain.affine):
            raise ValueError(f"pial MRI grids differ: {name}")
    try:
        # 原表面格式允许 ``valid = 1  # volume info valid``；整数标记
        # 与行尾注释分开读取，不能把有效注释误当作无效几何。
        if str(surface_metadata["valid"]).split("#", 1)[0].strip() != "1":
            raise ValueError("white volume geometry is invalid")
        for key in ("volume", "voxelsize", "xras", "yras", "zras", "cras"):
            values = np.asarray(surface_metadata[key])
            if values.shape != (3,) or not np.isfinite(values).all() \
                    or (key in {"volume", "voxelsize"} and np.any(values <= 0)):
                raise ValueError(f"white volume geometry is invalid: {key}")
        directions = np.stack([surface_metadata[key] for key in ("xras", "yras", "zras")])
        if np.linalg.det(directions) == 0:
            raise ValueError("white volume geometry has singular directions")
        affine = surface_ras_to_voxel(brain.header, surface_metadata)
    except (KeyError, TypeError, np.linalg.LinAlgError) as error:
        raise ValueError("white requires complete valid surface-RAS volume geometry") from error
    if not np.isfinite(affine).all():
        raise ValueError("pial surface RAS to voxel transform must be finite")
    return affine


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
    regularization_backend: str = "cpu",
    candidate_backend: str = "tree",
    candidate_grid_cells_per_axis: int = 2,
    retained_mht_backend: str = "tree",
    cleanup_marking_backend: str = "legacy",
    cleanup_candidate_grid_cells_per_axis: int = 2,
    cleanup_constraint_mode: str = "source_rip",
    device: str | None = None,
    trace_callback=None,
    profile: bool = False,
    target_search_backend: str = "serial",
) -> dict:
    """Place and save one hemisphere's pial.T1 using the native four-pass order.

    Requires ``surf/{hemi}.white``, ``surf/autodet.gw.stats.{hemi}.dat``,
    ``label/{hemi}.cortex.label``, ``label/{hemi}.cortex+hipamyg.label``,
    and ``mri/{brain.finalsurfs,wm,aseg.presurf}.mgz``. The return value gives
    the output path, accepted step count, pass boundaries, cleanup, and seconds.
    ``sampling_backend`` defaults to cpu; torch/triton select explicit CUDA MRI
    sampling through ``device``. Ordered updates and objective remain CPU.
    ``regularization_backend="torch"`` caches ordered adjacency on ``device``
    and runs signed averaging, spring and quadratic curvature in PyTorch.
    It is opt-in; collision acceptance, objective and final cleanup retain the
    original CPU implementation. Neither backend changes the four-pass order.
    ``candidate_backend="torch_snapshot"`` is an independent opt-in for complete
    GPU spatial candidate construction on ``device``. Live close-neighbor
    projection and triangle acceptance remain ordered CPU operations; retained
    rejected-trial MHT retries use the original tree rules. No Jacobi updates.
    显式candidate_grid_cells_per_axis=3只改变完整GPU候选单元；
    retained_mht_backend="compiled"复用已有实时有序循环，实际命中仍按
    原FP64 MHT桶重放。两者默认2/tree，不改变拒绝状态或接受顺序。
    cleanup_marking_backend默认legacy；source_numba/source_torch显式复用
    源有向标记，source_torch要求device，清理candidate_grid默认为2，
    显式3只允许source_torch。源清理最终残余非零会抛异常而不写表面。
    target_search_backend默认serial；parallel仅按既定Numba线程预算并行
    独立目标搜索，fastmath=False；有序目标平均与碰撞接受不变。
    七项输入、MRI网格、surface RAS/mm、有序输出和全部参数/真实验证见
    docs/recon_all/PYTHON_PIAL_PLACEMENT.md。函数不执行外部程序或启用半精度。
    三张MRI的三维shape和scanner affine须完全相同；头几何须有限，
    white体积几何须有效。错位输入在强度准备/优化前抛ValueError，不重采样。
    cleanup_constraint_mode默认source_rip：原rip既限制移动又排除SOAP
    邻居贡献，保留非rip但cortex外邻居的源贡献。显式cortex_pinned_union
    把cortex外已pin顶点加入该mask，同时改变移动域和邻居贡献；它是
    强化诊断策略，不声称源清理等价。四轮优化rip始终不变。模式规定的
    固定坐标被移动或源相交残余非零时失败，不写表面。
    ``trace_callback(step, pass_index, coordinates_copy, diagnostics)`` is an
    optional read-only diagnostic sink called after every completed step.
    Its diagnostics include all trial decisions; a rejected terminal step
    restores its starting coordinates and ends the pass, matching native.
    ``max_steps`` guards against non-convergence; hitting it raises an error.
    ``profile=True`` adds ``stage_seconds`` for preparation, gradients,
    ordered collision, objectives, border updates, cleanup, output I/O and
    remaining control work. Only this diagnostic mode synchronizes the explicit
    CUDA target at section boundaries; default execution does not add barriers.
    """
    started = time.perf_counter()
    if cleanup_constraint_mode not in ("source_rip", "cortex_pinned_union"):
        raise ValueError("cleanup_constraint_mode must be source_rip or cortex_pinned_union")
    if target_search_backend not in ("serial", "parallel"):
        raise ValueError("target_search_backend must be serial or parallel")
    if candidate_backend not in ("tree", "snapshot", "torch_snapshot"):
        raise ValueError("candidate_backend must be tree, snapshot or torch_snapshot")
    if candidate_backend == "torch_snapshot" and device is None:
        raise ValueError("torch_snapshot requires an explicit device")
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
    if cleanup_candidate_grid_cells_per_axis not in (2, 3):
        raise ValueError("cleanup_candidate_grid_cells_per_axis must be 2 or 3")
    if cleanup_candidate_grid_cells_per_axis != 2 and cleanup_marking_backend != "source_torch":
        raise ValueError("nondefault cleanup grid requires source_torch")
    if cleanup_marking_backend == "source_torch" and device is None:
        raise ValueError("source_torch cleanup requires an explicit device")
    if sampling_backend not in ("cpu", "torch", "triton"):
        raise ValueError("sampling_backend must be cpu, torch or triton")
    if regularization_backend not in ("cpu", "torch"):
        raise ValueError("regularization_backend must be cpu or torch")
    if regularization_backend == "torch" and device is None:
        raise ValueError("Torch regularization requires an explicit device")
    if sampling_backend != "cpu" and device is None:
        raise ValueError("GPU sampling requires an explicit device")
    stage_seconds = dict.fromkeys(("prepare", "gradient", "collision", "objective",
                                 "border_updates", "cleanup", "write"), 0.0) if profile else None
    profile_torch = profile_device = None
    if profile and device is not None and (sampling_backend != "cpu" or regularization_backend == "torch" or candidate_backend == "torch_snapshot" or cleanup_marking_backend == "source_torch"):
        import torch
        profile_torch, profile_device = torch, torch.device(device)
        if profile_device.type == "cuda" and profile_device.index is None:
            raise ValueError("CUDA profiling requires an explicitly indexed device, e.g. cuda:0")

    def profile_boundary() -> float:
        if stage_seconds is None:
            return 0.0
        # Do not initialize CUDA just for timing a CPU section. Context/model
        # initialization is included in preparation when the GPU backend starts.
        if profile_device is not None and profile_device.type == "cuda" and profile_torch.cuda.is_initialized():
            profile_torch.cuda.synchronize(profile_device)
        return time.perf_counter()

    def record_section(name: str, tick: float) -> None:
        if stage_seconds is not None:
            stage_seconds[name] += profile_boundary() - tick

    prepare_started = profile_boundary()
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
    if not np.isfinite(xyz).all():
        raise ValueError("pial white coordinates must be finite")
    brain = nib.load(str(brain_path))
    wm_image = nib.load(str(wm_path))
    aseg_image = nib.load(str(aseg_path))
    affine = _validate_pial_volume_geometry(brain, wm_image, aseg_image, metadata)
    ripped = np.asarray(rip_outside_label(
        len(xyz), fs.read_label(str(label))), dtype=np.bool_)
    wm = np.asarray(wm_image.dataobj)
    aseg = np.asarray(aseg_image.dataobj)
    stats = dict(line.split()[:2] for line in stats_path.read_text().splitlines()
                 if len(line.split()) >= 2)
    volume, bright = prepare_placement_volume(
        np.asarray(brain.dataobj), wm, surface="pial",
        mid_gray=float(stats["MID_GRAY"]),
    )
    placement = volume.copy()
    placement[bright == 130] = 0
    sampler = None
    if sampling_backend != "cpu":
        from .place_surface_sampling import PlacementSampling
        sampler = PlacementSampling(placement, affine, device=device,
                                    implementation=sampling_backend)
    thresholds = np.array([float(stats[f"pial_{name}"]) for name in
                           ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")])
    normal_topology = FaceNormalTopology(faces, len(xyz))
    normal_cache = CoordinateNormalCache(normal_topology)
    repulsion_index = OriginalVertexBuckets(xyz, ripped)
    normals = normal_cache.evaluate(xyz)
    border = compute_border_values_first_pass(
        volume, aseg, xyz, normals, xyz, ripped,
        np.full(len(xyz), -1.0, dtype=np.float32), affine, thresholds,
        hemisphere=hemi, surface="pial", target_search_backend=target_search_backend,
    )
    values = average_marked_values(border[0], border[4], ripped, faces, 5)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(xyz))
    ordered = (neighbor_indices, neighbor_valid)
    two_offsets, two_candidates = two_ring_neighbors(
        faces, len(xyz), ordered_neighbors=ordered)
    regularizer = None
    if regularization_backend == "torch":
        from .place_surface_regularization_torch import PlacementRegularizationTorch
        regularizer = PlacementRegularizationTorch(
            neighbors=neighbor_indices, valid=neighbor_valid, offsets=two_offsets,
            candidates=two_candidates, ripped=ripped, device=device)
    fixed_normals = original_vertex_normals(xyz, faces, topology=normal_topology)
    original_area = surface_total_area(xyz, faces)
    record_section("prepare", prepare_started)
    sigma, n_averages = 2.0, 16
    objective_cache_vertices: np.ndarray | None = None
    objective_cache_result: tuple[float, float] | None = None

    def objective(current: np.ndarray) -> tuple[float, float]:
        nonlocal objective_cache_vertices, objective_cache_result
        if objective_cache_vertices is current and objective_cache_result is not None:
            return objective_cache_result
        current_normals = normal_cache.evaluate(current)
        intensity_sse, rms, _ = intensity_error(
            placement, current, values, ripped, affine)
        spring = tangential_spring_energy(
            current, current_normals, faces, ripped,
            ordered_neighbors=ordered)
        area = surface_total_area(current, faces)
        result = (pial_placement_sse(intensity_sse, spring, original_area, area), rms)
        objective_cache_vertices = current
        objective_cache_result = result
        return result

    def gradient(current: np.ndarray, cropped: np.ndarray) -> np.ndarray:
        current_normals = normal_cache.evaluate(current)
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
        if regularizer is not None:
            return regularizer.regularize(
                vertices=current, normals=current_normals, gradient=with_repulsion,
                iterations=n_averages, spring_weight=0.3)
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

    objective_started = profile_boundary()
    last_sse, last_rms = objective(xyz)
    record_section("objective", objective_started)
    current, cropped = xyz.copy(), np.zeros(len(xyz), dtype=np.int32)
    dt, reductions, outer_pass = 0.5, 0, 0
    pass_ends: list[int] = []
    from .place_surface_topology_cache import PlacementTopologyCache
    topology_cache = PlacementTopologyCache(faces, len(current))
    for step in range(1, max_steps + 1):
        gradient_started = profile_boundary()
        momentum = gradient(current, cropped)
        record_section("gradient", gradient_started)
        stale_trial = None
        accepted = None
        trial_trace = [] if trace_callback is not None else None
        for trial_index in range(3):
            collision_started = profile_boundary()
            proposal, displacement = unconstrained_step_with_offsets(
                current, momentum, ripped, dt=dt)
            candidate, _ = asynchronous_first_step(
                current, faces, proposal, ripped, fast=True,
                offsets=displacement, accepted_offsets=momentum,
                stale_mht_trial=stale_trial, ordered_neighbors=ordered,
                candidate_backend=candidate_backend,
                candidate_grid_cells_per_axis=candidate_grid_cells_per_axis,
                retained_mht_backend=retained_mht_backend,
                candidate_device=device if candidate_backend == "torch_snapshot" else None,
                topology_cache=topology_cache)
            record_section("collision", collision_started)
            blocked = np.any(proposal != current, axis=1) & np.all(
                candidate == current, axis=1)
            trial_cropped = np.where(
                ripped, cropped, np.where(blocked, cropped + 1, 0)).astype(np.int32)
            objective_started = profile_boundary()
            sse, rms = objective(candidate)
            record_section("objective", objective_started)
            trial_dt = dt
            dt, reductions, reduced, rejected, stop = pial_step_decision(
                last_sse, last_rms, sse, rms, dt, reductions)
            cropped = trial_cropped
            if trial_trace is not None:
                trial_trace.append({"trial": trial_index, "dt_used": trial_dt,
                                    "dt_next": dt, "sse": sse, "rms": rms,
                                    "reduced": bool(reduced), "rejected": bool(rejected),
                                    "stop": bool(stop), "reductions": reductions})
            if rejected:
                stale_trial = candidate
                if stop:
                    # Native MRISpositionSurface restores TMP2_VERTICES, then
                    # ends this pass after MAX_REDUCTIONS; it does not fail.
                    accepted = current
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
                            "reductions": reductions, "stop": bool(stop),
                            "trials": trial_trace})
        if stop:
            pass_ends.append(step)
            if outer_pass == 3:
                break
            outer_pass += 1
            sigma = 2.0 / (1 << outer_pass)
            n_averages = 16 >> outer_pass
            border_started = profile_boundary()
            current_normals = normal_cache.evaluate(current)
            border = compute_border_values_first_pass(
                volume, aseg, current, current_normals, xyz, ripped,
                values, affine, thresholds, hemisphere=hemi, surface="pial",
                sigma=sigma, target_search_backend=target_search_backend,
            )
            values = average_marked_values(border[0], border[4], ripped, faces, 5)
            original_area = surface_total_area(current, faces)
            # sigma/values/original_area changed, so a same-coordinate objective
            # from the preceding pass is no longer valid.
            objective_cache_vertices = None
            objective_cache_result = None
            record_section("border_updates", border_started)
            objective_started = profile_boundary()
            last_sse, last_rms = objective(current)
            record_section("objective", objective_started)
            dt, reductions = 0.5, 0
    else:
        raise RuntimeError(f"pial optimizer did not finish four passes in {max_steps} steps")

    cleanup_started = profile_boundary()
    cortex_vertices = fs.read_label(str(cortex))
    cleanup_fixed = pial_cleanup_mask(ripped=ripped, cortex_vertices=cortex_vertices,
                                      constraint_mode=cleanup_constraint_mode)
    pinned = pin_medial_wall(current, xyz, cortex_vertices)
    if cleanup_marking_backend == "legacy":
        repaired, cleanup = repair_intersections(pinned, faces, cleanup_fixed)
    else:
        repaired, cleanup = repair_intersections(
            pinned, faces, cleanup_fixed, marking_backend=cleanup_marking_backend,
            device=device, candidate_grid_cells_per_axis=cleanup_candidate_grid_cells_per_axis)
    if not np.array_equal(repaired[cleanup_fixed], pinned[cleanup_fixed]):
        raise RuntimeError("pial cleanup moved fixed vertices; output not written")
    cleanup["fixed_vertices"] = int(np.count_nonzero(cleanup_fixed))
    cleanup["additional_pinned_vertices"] = int(np.count_nonzero(cleanup_fixed & ~ripped))
    cleanup["constraint_mode"] = cleanup_constraint_mode
    cleanup["soap_neighbor_mask"] = "original_rip" if cleanup_constraint_mode == "source_rip" else "original_rip_union_outside_cortex"
    if cleanup["intersecting_faces_after"]:
        error = RuntimeError(f"pial cleanup ({cleanup_constraint_mode}) has residual intersections; output not written")
        error.cleanup_diagnostics = cleanup
        raise error
    record_section("cleanup", cleanup_started)
    write_started = profile_boundary()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_vertices_like(white, output, repaired)
    record_section("write", write_started)
    seconds = time.perf_counter() - started
    result = {"output": str(output), "hemisphere": hemi, "steps": step,
            "pass_ends": pass_ends, "cleanup": cleanup,
            "sampling_backend": sampling_backend, "target_search_backend": target_search_backend,
            "target_search_threads": get_num_threads() if target_search_backend == "parallel" else 1, "device": device,
            "regularization_backend": regularization_backend,
            "candidate_backend": candidate_backend,
            "candidate_grid_cells_per_axis": candidate_grid_cells_per_axis,
            "retained_mht_backend": retained_mht_backend,
            "cleanup_marking_backend": cleanup_marking_backend,
            "cleanup_candidate_grid_cells_per_axis": cleanup_candidate_grid_cells_per_axis,
            "cleanup_constraint_mode": cleanup_constraint_mode,
            "seconds": seconds, "profile": bool(profile)}
    if stage_seconds is not None:
        stage_seconds["control"] = seconds - sum(stage_seconds.values())
        result["stage_seconds"] = stage_seconds
        result["profile_cuda_target"] = str(profile_device) if profile_device is not None and profile_device.type == "cuda" else None
    return result
