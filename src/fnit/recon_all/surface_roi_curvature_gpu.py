"""FreeSurfer anatomical-stats curvature columns on an existing mesh."""

from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from .surface_curvature_gpu import _neighbours
from .surface_thickness_gpu import _normals


@torch.inference_mode()
def _principal_curvatures_tensor(xyz: torch.Tensor, normal: torch.Tensor,
                                 neighbours: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
    """在已准备的几何上拟合主曲率，返回 (N, 2) float32 张量，单位 mm⁻¹。

    xyz 为 surface RAS/mm 坐标；normal 为相同顶点顺序的单位法线；
    neighbours/mask 为两跳邻接索引及有效位置。输入须在同一设备上。
    分块、SVD 截断、病态回退及浮点运算顺序与原实现一致。
    """
    shifted = torch.stack((normal[:, 1], normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.cross(normal, shifted, dim=1)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.where(torch.linalg.vector_norm(e1, dim=1)[:, None] < .001,
                     torch.cross(normal, alternate, dim=1), e1)
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=1), dim=1)
    principal = torch.empty((len(xyz), 2), dtype=torch.float32, device=xyz.device)
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        delta = xyz[neighbours[start:stop]] - xyz[start:stop, None, :]
        u = (delta * e1[start:stop, None, :]).sum(dim=2)
        v = (delta * e2[start:stop, None, :]).sum(dim=2)
        z = (delta * normal[start:stop, None, :]).sum(dim=2)
        rsq = u * u + v * v
        valid = mask[start:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ (z * valid)[:, :, None]
        left, singular, right = torch.linalg.svd(gram)
        inverse = torch.where(singular >= 1e-4 * singular[:, :1],
                              singular.clamp_min(1e-30).reciprocal(), 0)
        coefficient = (right.transpose(1, 2) @
                       (inverse[:, :, None] * (left.transpose(1, 2) @ rhs)))[:, :, 0]
        a, b, c = (coefficient[:, i] for i in range(3))
        root = torch.sqrt((a - c).square() + 4 * b.square())
        high, low = a + c + root, a + c - root
        fitted = torch.where((high.abs() >= low.abs())[:, None],
                             torch.stack((high, low), dim=1),
                             torch.stack((low, high), dim=1))
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(dim=1).values
        smallest = k.masked_fill(~valid, torch.inf).min(dim=1).values
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-30)
        fallback = torch.stack((largest, smallest), dim=1)
        principal[start:stop] = torch.where((condition >= 500000)[:, None],
                                            fallback, fitted)
    return principal


@torch.inference_mode()
def principal_curvatures(vertices: np.ndarray, faces: np.ndarray,
                         device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """拟合 MRIScomputeSecondFundamentalForm 的两跳主曲率。

    vertices 是 (N, 3) surface RAS/mm 坐标，faces 是 (F, 3) 顶点索引；
    device 默认 cuda:0。返回两个 (N,) float32 NumPy 数组，单位 mm⁻¹；
    k1 按绝对值较大者排列。CUDA 使用 TF32 矩阵乘法，不启用半精度。
    无效网格或设备会抛 NumPy/PyTorch 异常。该步骤属于
    mris_anatomical_stats 内部计算，没有独立 CLI。多图谱调用应通过
    SurfaceStatsCache 复用法线、邻接与结果，见 SURFACE_STATS_CACHE.md。
    """
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    tri = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    normal = _normals(xyz, tri)
    _, _, neighbours, mask = _neighbours(faces, len(xyz))
    principal = _principal_curvatures_tensor(
        xyz, normal, torch.as_tensor(neighbours, device=device),
        torch.as_tensor(mask, device=device))
    values = principal.cpu().numpy()
    return values[:, 0], values[:, 1]


def curvature_columns(surface: Path, area_map: Path, annotation: Path,
                      cortex_label: Path | None, device: str = "cuda:0", *,
                      cache=None) -> dict[str, tuple[float, ...]]:
    """按注释汇总 MeanCurv/GausCurv/FoldInd/CurvInd，保留原公式与顶点顺序。

    surface 是 surface RAS/mm 网格；area_map 是对应顶点的 mm² morph；
    annotation 是 .annot；cortex_label=None 时包含全部已注释顶点。
    device 默认 cuda:0；cache 可为同设备 SurfaceStatsCache，默认临时缓存。
    返回 {脑区名: 四列浮点数}，主曲率计算属于 mris_anatomical_stats。
    顶点数不一致、缺失文件或设备不一致会抛异常；不替换全局表头。
    """
    from .surface_stats_cache import SurfaceStatsCache

    if cache is None:
        with SurfaceStatsCache(device=device) as temporary:
            return temporary.curvature_columns(surface, area_map, annotation,
                                                cortex_label)
    cache.check_device(device)
    return cache.curvature_columns(surface, area_map, annotation, cortex_label)
