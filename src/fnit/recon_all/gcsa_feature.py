"""Mean-curvature input for FreeSurfer's one-feature GCS surface atlases."""

from __future__ import annotations

import numpy as np
import torch

from .surface_curvature_gpu import _neighbours
from .surface_thickness_gpu import _normals


@torch.inference_mode()
def mean_curvature_and_principal_directions(
    vertices: np.ndarray, faces: np.ndarray, *, device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """一次共享几何准备和批量 SVD，返回 GCSA 曲率及主方向。

    ``mris_ca_label`` 同时需要五次平滑平均曲率和主曲率方向。旧路径
    分别建立坐标、法向、二阶邻域和局部二次拟合，导致同一表面传输到
    GPU 两次并执行两次 SVD。本函数共享这些只读中间量；拟合公式、
    float32 输入、SVD 阈值和五次平均顺序保持原实现。输出为
    ``(N,) float32`` 曲率和 ``(N, 2, 3) float32`` surface-RAS 方向，
    仅属于 GCSA 内部步骤，没有独立 CLI。
    """
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    normal = _normals(xyz, triangles)
    first_np, first_mask_np, second_np, second_mask_np = _neighbours(faces, len(xyz))
    first = torch.as_tensor(first_np, device=device)
    first_mask = torch.as_tensor(first_mask_np, device=device)
    second = torch.as_tensor(second_np, device=device)
    second_mask = torch.as_tensor(second_mask_np, device=device)

    shifted = torch.stack((normal[:, 1], normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.cross(normal, shifted, dim=1)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.where(torch.linalg.vector_norm(e1, dim=1)[:, None] < .001,
                     torch.cross(normal, alternate, dim=1), e1)
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=1), dim=1)

    mean = torch.empty(len(xyz), dtype=torch.float32, device=device)
    result = torch.empty((len(xyz), 2, 3), dtype=torch.float32, device=device)
    basis = torch.stack((e1, e2), dim=1)
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        indices = second[start:stop]
        delta = xyz[indices] - xyz[start:stop, None, :]
        u = (delta * e1[start:stop, None, :]).sum(dim=2)
        v = (delta * e2[start:stop, None, :]).sum(dim=2)
        z = (delta * normal[start:stop, None, :]).sum(dim=2)
        rsq = u * u + v * v
        valid = second_mask[start:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        heights = z * valid
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ heights[:, :, None]
        left, singular, right = torch.linalg.svd(gram)
        inverse = torch.where(singular >= 1e-4 * singular[:, :1],
                              singular.clamp_min(1e-30).reciprocal(), 0)
        coefficient = right.transpose(1, 2) @ (inverse[:, :, None] *
                                               (left.transpose(1, 2) @ rhs))

        fitted = coefficient[:, 0, 0] + coefficient[:, 2, 0]
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(dim=1).values
        smallest = k.masked_fill(~valid, torch.inf).min(dim=1).values
        fallback = (largest + smallest) / 2
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-30)
        mean[start:stop] = torch.where(condition >= 500000, fallback, fitted)

        a, b, c = (coefficient[:, i, 0] for i in range(3))
        hessian = torch.stack((torch.stack((2 * a, 2 * b), dim=1),
                               torch.stack((2 * b, 2 * c), dim=1)), dim=1)
        values, vectors = torch.linalg.eigh(hessian)
        swap = values[:, 0].abs() >= values[:, 1].abs()
        first_direction = torch.where(swap[:, None], vectors[:, :, 0], vectors[:, :, 1])
        second_direction = torch.where(swap[:, None], vectors[:, :, 1], vectors[:, :, 0])
        transformed = torch.stack((
            first_direction[:, 0, None] * e1[start:stop] +
            first_direction[:, 1, None] * e2[start:stop],
            second_direction[:, 0, None] * e1[start:stop] +
            second_direction[:, 1, None] * e2[start:stop]), dim=1)
        result[start:stop] = torch.where((condition >= 500000)[:, None, None],
                                         basis[start:stop], transformed)

    degree = first_mask.sum(dim=1).to(torch.float32) + 1
    for _ in range(5):
        mean = (mean + (mean[first] * first_mask).sum(dim=1)) / degree
    return mean.cpu().numpy(), result.cpu().numpy()


@torch.inference_mode()
def mean_curvature_five(vertices: np.ndarray, faces: np.ndarray,
                        *, device: str = "cpu") -> np.ndarray:
    """Compute second-form mean curvature and five one-ring averages."""
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    normal = _normals(xyz, triangles)
    first_np, first_mask_np, second_np, second_mask_np = _neighbours(faces, len(xyz))
    first = torch.as_tensor(first_np, device=device)
    first_mask = torch.as_tensor(first_mask_np, device=device)
    second = torch.as_tensor(second_np, device=device)
    second_mask = torch.as_tensor(second_mask_np, device=device)

    shifted = torch.stack((normal[:, 1], normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.cross(normal, shifted, dim=1)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.where(torch.linalg.vector_norm(e1, dim=1)[:, None] < .001,
                     torch.cross(normal, alternate, dim=1), e1)
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=1), dim=1)

    mean = torch.empty(len(xyz), dtype=torch.float32, device=device)
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        index = second[start:stop]
        delta = xyz[index] - xyz[start:stop, None, :]
        u = (delta * e1[start:stop, None, :]).sum(dim=2)
        v = (delta * e2[start:stop, None, :]).sum(dim=2)
        z = (delta * normal[start:stop, None, :]).sum(dim=2)
        rsq = u * u + v * v
        valid = second_mask[start:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        heights = z * valid
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ heights[:, :, None]
        left, singular, right = torch.linalg.svd(gram)
        inverse = torch.where(singular >= 1e-4 * singular[:, :1],
                              singular.clamp_min(1e-30).reciprocal(), 0)
        coefficient = right.transpose(1, 2) @ (inverse[:, :, None] *
                                                 (left.transpose(1, 2) @ rhs))
        fitted = coefficient[:, 0, 0] + coefficient[:, 2, 0]
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(dim=1).values
        smallest = k.masked_fill(~valid, torch.inf).min(dim=1).values
        fallback = (largest + smallest) / 2
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-30)
        mean[start:stop] = torch.where(condition >= 500000, fallback, fitted)

    degree = first_mask.sum(dim=1).to(torch.float32) + 1
    for _ in range(5):
        mean = (mean + (mean[first] * first_mask).sum(dim=1)) / degree
    return mean.cpu().numpy()


@torch.inference_mode()
def principal_directions(vertices: np.ndarray, faces: np.ndarray,
                         *, device: str = "cpu") -> np.ndarray:
    """Return the two second-form directions used by GCSA Gibbs edges."""
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    normal = _normals(xyz, triangles)
    _, _, second_np, second_mask_np = _neighbours(faces, len(xyz))
    second = torch.as_tensor(second_np, device=device)
    mask = torch.as_tensor(second_mask_np, device=device)

    shifted = torch.stack((normal[:, 1], normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.cross(normal, shifted, dim=1)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.where(torch.linalg.vector_norm(e1, dim=1)[:, None] < .001,
                     torch.cross(normal, alternate, dim=1), e1)
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=1), dim=1)
    basis = torch.stack((e1, e2), dim=1)
    result = torch.empty_like(basis)
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        indices = second[start:stop]
        delta = xyz[indices] - xyz[start:stop, None, :]
        u = (delta * e1[start:stop, None, :]).sum(dim=2)
        v = (delta * e2[start:stop, None, :]).sum(dim=2)
        z = (delta * normal[start:stop, None, :]).sum(dim=2)
        valid = mask[start:stop] & ((u * u + v * v) > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        heights = z * valid
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ heights[:, :, None]
        left, singular, right = torch.linalg.svd(gram)
        inverse = torch.where(singular >= 1e-4 * singular[:, :1],
                              singular.clamp_min(1e-30).reciprocal(), 0)
        coefficient = right.transpose(1, 2) @ (inverse[:, :, None] *
                                                 (left.transpose(1, 2) @ rhs))
        a, b, c = (coefficient[:, i, 0] for i in range(3))
        hessian = torch.stack((torch.stack((2 * a, 2 * b), dim=1),
                               torch.stack((2 * b, 2 * c), dim=1)), dim=1)
        values, vectors = torch.linalg.eigh(hessian)
        swap = values[:, 0].abs() >= values[:, 1].abs()
        first = torch.where(swap[:, None], vectors[:, :, 0], vectors[:, :, 1])
        second_direction = torch.where(swap[:, None], vectors[:, :, 1], vectors[:, :, 0])
        transformed = torch.stack((first[:, 0, None] * e1[start:stop]
                                   + first[:, 1, None] * e2[start:stop],
                                   second_direction[:, 0, None] * e1[start:stop]
                                   + second_direction[:, 1, None] * e2[start:stop]), dim=1)
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-30)
        result[start:stop] = torch.where((condition >= 500000)[:, None, None],
                                         basis[start:stop], transformed)
    return result.cpu().numpy()
