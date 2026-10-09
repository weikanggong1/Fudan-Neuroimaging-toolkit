"""Final projection and overlap cleanup for conventional FreeSurfer spheres."""

from __future__ import annotations

import numpy as np
import torch

from .mris_register_overlap import remove_overlap_sphere
from .sphere_python import project_radially


@torch.no_grad()
def finish_standard_sphere(vertices: np.ndarray, faces: np.ndarray,
                           *, start_iteration: int, device: str = "cpu",
                           overlap_backend: str = "dense"
                           ) -> tuple[np.ndarray, list[int]]:
    """Continue from the last ``MRISunfold`` checkpoint to the final sphere.

    overlap_backend默认dense保留已验证实现；marked为显式候选，仅省略
    未标顶点被丢弃的邻域位移和判负不需要的法向/平方根，完整投影/停止
    顺序不变。输入float32(N,3)surface RAS/mm和有序(F,3)整数面；
    返回CPU NumPy同序坐标与更新前负面计数。非法后端在计算前报错。
    """
    if overlap_backend not in {"dense", "marked"}:
        raise ValueError("overlap_backend must be dense or marked")
    projected = project_radially(vertices, already_sphere=True)
    xyz = torch.from_numpy(np.ascontiguousarray(projected)).to(device)
    triangles = torch.from_numpy(np.asarray(faces, np.int64)).to(device)
    if overlap_backend == "marked":
        from .mris_register_overlap_marked import remove_overlap_sphere_marked
        operation = remove_overlap_sphere_marked
    else:
        operation = remove_overlap_sphere
    result, negative_counts = operation(
        xyz, triangles, start_iteration=start_iteration)
    return result.cpu().numpy(), negative_counts
