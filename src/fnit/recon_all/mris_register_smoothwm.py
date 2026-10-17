"""Mean curvature used by FreeSurfer 8.2's smoothwm registration pass."""

from __future__ import annotations

import numpy as np
import torch
from numba import njit

from .mris_register_nonlinear import (
    ordered_neighbors_from_faces, sphere_vertex_normals, tangent_basis,
)
from .topology_vnl_svd import add, mul, svd_inverse_3, svdc_3


def _three_hop_neighbors(faces: torch.Tensor, vertices: int
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    one, degrees = ordered_neighbors_from_faces(faces, vertices)
    one_rows = [row[:int(degree)] for row, degree in zip(one.cpu().tolist(),
                                                          degrees.cpu().tolist())]
    rows = []
    for vertex in range(vertices):
        row = one_rows[vertex].copy()
        seen = {vertex, *row}
        begin, end = 0, len(row)
        for _ in (2, 3):
            for neighbor in row[begin:end]:
                for candidate in one_rows[neighbor]:
                    if candidate not in seen:
                        seen.add(candidate)
                        row.append(candidate)
            begin, end = end, len(row)
        rows.append(row)
    width = max(map(len, rows))
    indices = np.zeros((vertices, width), dtype=np.int64)
    active = np.zeros((vertices, width), dtype=bool)
    for vertex, row in enumerate(rows):
        indices[vertex, :len(row)] = row
        active[vertex, :len(row)] = True
    return (torch.from_numpy(indices).to(faces.device),
            torch.from_numpy(active).to(faces.device))


@njit(cache=True, fastmath=False)
def _source_order_fit(gram: np.ndarray, rhs: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """按原 VNL 顺序拟合曲率系数，缓存的是机器码而非矩阵或结果。

    gram 为 float32(N,3,3) Gram 矩阵，rhs 为 float32(N,3) 右端；
    返回 float32(N,3) 系数和 float32(N,) 条件数。输入矩阵及邻接
    相关数据不会跨调用缓存；每次仍执行完整双 SVD 和原阈值判断。
    无可调参数或独立原 CLI；缓存目录沿用 Numba 配置，未开启 fastmath。
    """
    coefficients = np.zeros((len(gram), 3), np.float32)
    condition = np.empty(len(gram), np.float32)
    for vertex in range(len(gram)):
        _, singular, _ = svdc_3(gram[vertex])
        wmax = np.max(np.abs(singular))
        wmin = np.min(np.abs(singular))
        condition[vertex] = np.float32(
            1e8 if wmin < 1.1920928955078125e-7 else wmax / wmin)
        inverse = svd_inverse_3(gram[vertex])
        for row in range(3):
            value = np.float32(0)
            for col in range(3):
                value = add(value, mul(inverse[row, col], rhs[vertex, col]))
            coefficients[vertex, row] = value
    return coefficients, condition


@torch.no_grad()
def smoothwm_mean_curvature(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Fit the quadratic form over FreeSurfer's three-hop vertex neighborhood."""
    xyz = vertices.float()
    triangles = faces.long()
    normal = sphere_vertex_normals(xyz, triangles)
    e1, e2 = tangent_basis(normal)
    neighbors, active = _three_hop_neighbors(triangles, len(xyz))
    curvature = torch.empty(len(xyz), dtype=torch.float32, device=xyz.device)
    for first in range(0, len(xyz), 2048):
        stop = min(first + 2048, len(xyz))
        delta = xyz[neighbors[first:stop]] - xyz[first:stop, None]
        u = (delta * e1[first:stop, None]).sum(2)
        v = (delta * e2[first:stop, None]).sum(2)
        z = (delta * normal[first:stop, None]).sum(2)
        rsq = u * u + v * v
        valid = active[first:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        height = z * valid
        gram = torch.zeros((stop - first, 3, 3), dtype=torch.float32, device=xyz.device)
        rhs = torch.zeros((stop - first, 3, 1), dtype=torch.float32, device=xyz.device)
        for neighbor_index in range(design.shape[1]):
            row = design[:, neighbor_index]
            gram += row[:, :, None] * row[:, None, :]
            rhs[:, :, 0] += row * height[:, neighbor_index, None]
        coefficients, condition_array = _source_order_fit(
            gram.cpu().numpy(), rhs[:, :, 0].cpu().numpy())
        h00 = np.float32(np.float32(2) * coefficients[:, 0]).astype(np.float64)
        h01 = np.float32(np.float32(2) * coefficients[:, 1]).astype(np.float64)
        h11 = np.float32(np.float32(2) * coefficients[:, 2]).astype(np.float64)
        center = (h00 + h11) / 2.0
        spread = np.sqrt(((h00 - h11) / 2.0) ** 2 + h01 ** 2)
        low = np.float32(center - spread)
        high = np.float32(center + spread)
        fitted = torch.from_numpy(np.float32(np.float32(low + high) *
                                              np.float32(0.5))).to(xyz.device)
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(1).values
        smallest = k.masked_fill(~valid, torch.inf).min(1).values
        condition = torch.from_numpy(condition_array).to(xyz.device)
        curvature[first:stop] = torch.where(condition >= 500000,
                                             (largest + smallest) / 2, fitted)
    return curvature
