"""FreeSurfer sphere-registration overlap smoothing in PyTorch."""

from __future__ import annotations

import torch

from .mris_register_kernels import project_sphere
from .mris_register_nonlinear import face_area_normals, ordered_neighbors_from_faces


class SphereOverlapQualityError(RuntimeError):
    """源停止规则执行结束但球面仍有翻折；diagnostics 保存实际质量。"""

    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(
            "sphere overlap cleanup stopped with residual folds: "
            f"float32={diagnostics['final_negative_float32']}, "
            f"double={diagnostics['final_negative_double']}; "
            f"reason={diagnostics['stop_reason']}")


def _validate_overlap_inputs(positions, faces, start_iteration, max_iterations):
    """检查有限(N,3)坐标、同设备(F,3)整数面及非负整数计数。"""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")
    if not positions.is_floating_point() or not torch.isfinite(positions).all():
        raise ValueError("positions must contain finite floating coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.device != positions.device:
        raise ValueError("faces must have shape (F,3) on the positions device")
    if faces.dtype not in (torch.int32, torch.int64):
        raise ValueError("faces must contain integer vertex indices")
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= len(positions)):
        raise ValueError("face vertex index is outside positions")
    for name, value in (("start_iteration", start_iteration),
                        ("max_iterations", max_iterations)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")


def _overlap_diagnostics(xyz, faces, history, *, count, iteration,
                         start_iteration, max_iterations, stop_reason,
                         minimum_negative, minimum_iteration, dt, max_neighbors):
    """末次才计算双精度径向质量；不修改坐标/停止阈值或优化轨迹。"""
    import numpy as np

    wide = xyz.detach().cpu().numpy().astype(np.float64)
    triangles = faces.detach().cpu().numpy()
    corners = wide[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    signed = np.sum(corners.sum(axis=1) * cross, axis=1)
    negative = signed < 0
    area = np.linalg.norm(cross, axis=1) * 0.5
    negative_area = area[negative]
    finite = bool(np.isfinite(wide).all() and np.isfinite(area).all())
    return {"source_niterations": max_iterations,
            "source_limit_comparison": "iteration-start_iteration > niterations",
            "start_iteration": start_iteration, "final_iteration": iteration,
            "updates": len(history), "stop_reason": stop_reason,
            "initial_negative_float32": history[0] if history else count,
            "last_pre_step_negative_float32": history[-1] if history else None,
            "final_negative_float32": count,
            "final_negative_double": int(negative.sum()),
            "negative_area_mm2_sum": float(negative_area.sum()),
            "negative_area_mm2_max": float(negative_area.max(initial=0)),
            "negative_area_mm2_p99": float(np.percentile(negative_area, 99))
            if len(negative_area) else 0.0,
            "zero_area_double": int((area == 0).sum()), "finite": finite,
            "minimum_negative_float32": minimum_negative,
            "minimum_iteration": minimum_iteration, "final_dt": dt,
            "marked_expansion_rings": max_neighbors,
            "quality_status": "passed" if finite and count == 0 and not negative.any()
            else "failed_quality"}


def _finish_overlap_report(xyz, faces, history, *, diagnostics, require_unfolded,
                           **state):
    """保持默认二元组接口；显式报告/严格门才计算末次几何摘要。"""
    if diagnostics is not None or require_unfolded:
        result = _overlap_diagnostics(xyz, faces, history, **state)
        if diagnostics is not None:
            diagnostics.update(result)
        if require_unfolded and result["quality_status"] != "passed":
            raise SphereOverlapQualityError(result)
    return xyz, history


@torch.no_grad()
def remove_overlap_sphere(positions: torch.Tensor, faces: torch.Tensor,
                          *, start_iteration: int = 0,
                          max_iterations: int = 1000,
                          diagnostics: dict | None = None,
                          require_unfolded: bool = False) -> tuple[torch.Tensor, list[int]]:
    """按固定源码清理径向翻折，返回同设备FP32坐标及更新前计数。

    positions=(N,3)surface RAS/mm；faces=(F,3)同设备整数有序面。
    start_iteration默认0；max_iterations默认1000是源命令的niterations，
    源严格大于比较允许1001次更新。另保留1000步无进展停止条件。
    diagnostics默认None不计算末次摘要；传字典写入实际停止原因、
    最后更新前/后计数及双精度负面积mm²。require_unfolded默认False，
    True在残余翻折时抛SphereOverlapQualityError（携带diagnostics）。
    非有限坐标、非法形状/索引/计数抛ValueError，不改原输入。
    内部步骤对应mris_sphere/mris_register，无独立官方CLI。
    """
    _validate_overlap_inputs(positions, faces, start_iteration, max_iterations)
    xyz = positions.float().clone()

    def negative_faces() -> torch.Tensor:
        area, _ = face_area_normals(xyz, faces, signed_sphere=True)
        return area < 0

    negative = negative_faces()
    count = int(negative.sum())
    history: list[int] = []
    if count == 0:
        return _finish_overlap_report(
            xyz, faces, history, diagnostics=diagnostics, require_unfolded=require_unfolded,
            count=count, iteration=start_iteration, start_iteration=start_iteration,
            max_iterations=max_iterations, stop_reason="no_negative_faces",
            minimum_negative=0, minimum_iteration=0, dt=0.99, max_neighbors=0)
    neighbors, degrees = ordered_neighbors_from_faces(faces, len(xyz))
    active = torch.arange(neighbors.shape[1], device=xyz.device)[None, :] < degrees[:, None]
    min_negative, min_iteration = count, 0
    iteration, last_expand, same, max_neighbors = start_iteration, 0, 0, 0
    dt = 0.99
    stop_reason = "no_negative_faces"
    while count > 0:
        history.append(count)
        marked = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
        marked[faces[negative].reshape(-1)] = True
        if max_neighbors:
            marked |= (marked[neighbors] & active).any(dim=1)

        xyz_double = xyz.double()
        displacement = torch.zeros_like(xyz_double)
        for index in range(neighbors.shape[1]):
            difference = xyz_double[neighbors[:, index]] - xyz_double
            displacement = torch.where(active[:, index, None],
                                       displacement + difference, displacement)
        displacement = (displacement / degrees[:, None].double()).float()
        moved = (xyz_double + dt * displacement.double()).float()
        xyz = project_sphere(torch.where(marked[:, None], moved, xyz))

        old_count = count
        iteration += 1
        negative = negative_faces()
        count = int(negative.sum())
        if count < min_negative:
            min_negative, min_iteration = count, iteration
        elif (iteration - min_iteration) % 10 == 0 and iteration > min_iteration:
            if dt > 0.01:
                dt *= 0.95
        elif iteration > min_iteration + 50 and iteration > last_expand + 25:
            if max_neighbors < 1:
                dt, max_neighbors = 0.99, 1
            last_expand, same = iteration, 0
        if iteration > min_iteration + 1000:
            stop_reason = "source_lack_of_progress"
            break
        if old_count == count:
            if same > 25 and max_neighbors < 1:
                max_neighbors, dt, last_expand, same = 1, 0.99, iteration, 0
            else:
                same += 1
        else:
            same = 0
        if iteration - start_iteration > max_iterations:
            stop_reason = "source_niterations_limit"
            break
    return _finish_overlap_report(
        xyz, faces, history, diagnostics=diagnostics, require_unfolded=require_unfolded,
        count=count, iteration=iteration, start_iteration=start_iteration,
        max_iterations=max_iterations, stop_reason=stop_reason,
        minimum_negative=min_negative, minimum_iteration=min_iteration,
        dt=dt, max_neighbors=max_neighbors)
