"""固定三角对的批量 PyTorch Möller 谓词；不改变动态顶点接受顺序。

输入为 (P,3,3) float32、surface RAS/mm，输出 (P,) bool Tensor 与诊断字典。
计算按 placement/tritri.cpp 的 FP64、共面和阈值规则；数值边界按已有
Numba 源谓词复核。属于 mris_place_surface 内部步骤，无独立官方 CLI。
修改的 FNIT 实现；固定 FreeSurfer d932c45，许可证见 licenses/FreeSurfer.txt。
"""
from __future__ import annotations

import numpy as np
from numba import njit
import torch

from .place_surface_collision import triangles_intersect


@njit(cache=True)
def _source_pairs(first, second):
    result = np.empty(len(first), np.bool_)
    for pair in range(len(first)):
        result[pair] = triangles_intersect(first[pair], second[pair])
    return result


def _dot3(first, second):
    # 显式逐项运算，不用不同规约次序的sum/dot或可融合的cross内核。
    return (first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1]) + first[:, 2] * second[:, 2]


def _cross3(first, second):
    return torch.stack((first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1],
                        first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2],
                        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]), dim=1)


def _intervals(values, distances):
    d01, d02, d12 = distances[:, 0] * distances[:, 1], distances[:, 0] * distances[:, 2], distances[:, 1] * distances[:, 2]
    first_case = d01 > 0
    second_case = ~first_case & (d02 > 0)
    third_case = ~first_case & ~second_case & ((d12 > 0) | (distances[:, 0] != 0))
    fourth_case = ~first_case & ~second_case & ~third_case & (distances[:, 1] != 0)
    fifth_case = ~first_case & ~second_case & ~third_case & ~fourth_case & (distances[:, 2] != 0)
    coplanar = ~(first_case | second_case | third_case | fourth_case | fifth_case)
    first = torch.where(first_case, 2, torch.where(second_case, 1, torch.where(third_case, 0, torch.where(fourth_case, 1, 2))))
    second = torch.where(first_case, 0, torch.where(second_case, 0, torch.where(third_case, 1, torch.where(fourth_case, 0, 0))))
    third = torch.where(first_case, 1, torch.where(second_case, 2, torch.where(third_case, 2, torch.where(fourth_case, 2, 1))))
    def gather(array, indices):
        return array.gather(1, indices[:, None])[:, 0]
    vf, df = gather(values, first), gather(distances, first)
    denominator_a, denominator_b = df - gather(distances, second), df - gather(distances, third)
    denominator_a = torch.where(coplanar, torch.ones_like(denominator_a), denominator_a)
    denominator_b = torch.where(coplanar, torch.ones_like(denominator_b), denominator_b)
    a = vf + (gather(values, second) - vf) * df / denominator_a
    b = vf + (gather(values, third) - vf) * df / denominator_b
    return torch.minimum(a, b), torch.maximum(a, b), coplanar


def _coplanar_pairs(first, second, normal):
    absolute = normal.abs()
    yz = (absolute[:, 0] > absolute[:, 1]) & (absolute[:, 0] > absolute[:, 2])
    xy = ((absolute[:, 0] > absolute[:, 1]) & ~(absolute[:, 0] > absolute[:, 2])) | (
        ~(absolute[:, 0] > absolute[:, 1]) & (absolute[:, 2] > absolute[:, 1]))
    x, y = torch.where(yz, 1, 0), torch.where(yz, 2, torch.where(xy, 1, 2))
    def project(array):
        xx = array.gather(2, x[:, None, None].expand(-1, 3, 1))[:, :, 0]
        yy = array.gather(2, y[:, None, None].expand(-1, 3, 1))[:, :, 0]
        return torch.stack((xx, yy), dim=2)
    a, b = project(first), project(second)
    result = torch.zeros(len(a), dtype=torch.bool, device=a.device)
    for edge in range(3):
        p, q = a[:, edge], a[:, (edge + 1) % 3]
        ax, ay = q[:, 0] - p[:, 0], q[:, 1] - p[:, 1]
        for other in range(3):
            c, d = b[:, other], b[:, (other + 1) % 3]
            bx, by, cx, cy = c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], p[:, 0] - c[:, 0], p[:, 1] - c[:, 1]
            f, h = ay * bx - ax * by, by * cx - bx * cy
            preliminary = ((f > 0) & (h >= 0) & (h <= f)) | ((f < 0) & (h <= 0) & (h >= f))
            e = ax * cy - ay * cx
            final = torch.where(f > 0, (e >= 0) & (e <= f), (e >= f) & (e <= 0))
            result |= preliminary & final
    def point_inside(point, triangle):
        distances = []
        for edge in range(3):
            p, q = triangle[:, edge], triangle[:, (edge + 1) % 3]
            aa, bb = q[:, 1] - p[:, 1], -(q[:, 0] - p[:, 0])
            cc = -aa * p[:, 0] - bb * p[:, 1]
            distances.append((aa * point[:, 0] + bb * point[:, 1]) + cc)
        return (distances[0] * distances[1] > 0) & (distances[0] * distances[2] > 0)
    return result | point_inside(a[:, 0], b) | point_inside(b[:, 0], a)


def _pair_kernel(first, second):
    a, b = first.to(torch.float64), second.to(torch.float64)
    n1 = _cross3(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    n2 = _cross3(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    d1, d2 = -_dot3(n1, a[:, 0]), -_dot3(n2, b[:, 0])
    du = torch.stack(tuple(_dot3(n1, b[:, corner]) + d1 for corner in range(3)), dim=1)
    dv = torch.stack(tuple(_dot3(n2, a[:, corner]) + d2 for corner in range(3)), dim=1)
    au, av = du.abs(), dv.abs()
    reject = ((au > 1e-6).any(1) & (du[:, 0] * du[:, 1] > 0) & (du[:, 0] * du[:, 2] > 0)) | (
        (av > 1e-6).any(1) & (dv[:, 0] * dv[:, 1] > 0) & (dv[:, 0] * dv[:, 2] > 0))
    # 保留源 tritri.cpp 的不对称 1e-5/1e-6 共面保护。
    du = torch.where(au < 1e-6, torch.zeros_like(du), du)
    dv = torch.where(av < 1e-6, torch.zeros_like(dv), dv)
    du = torch.where(((au[:, 0] < 1e-5) & (au[:, 1] < 1e-5) & (au[:, 2] < 1e-6))[:, None], torch.zeros_like(du), du)
    dv = torch.where(((av[:, 0] < 1e-5) & (av[:, 1] < 1e-5) & (av[:, 2] < 1e-6))[:, None], torch.zeros_like(dv), dv)
    line = _cross3(n1, n2).abs()
    axis = torch.where(line[:, 1] > line[:, 0], 1, 0)
    axis = torch.where(line[:, 2] > line.gather(1, axis[:, None])[:, 0], 2, axis)
    aa = a.gather(2, axis[:, None, None].expand(-1, 3, 1))[:, :, 0]
    bb = b.gather(2, axis[:, None, None].expand(-1, 3, 1))[:, :, 0]
    a0, a1, ac = _intervals(aa, dv)
    b0, b1, bc = _intervals(bb, du)
    coplanar = ac | bc
    ordinary = ~((a1 < b0) | (b1 < a0))
    result = ~reject & torch.where(coplanar, _coplanar_pairs(a, b, n1), ordinary)
    # 边界只决定源FP64复核范围，不放宽/改写相交规则。
    eps = torch.finfo(torch.float64).eps
    scale_u = torch.stack(tuple((n1.abs() * b[:, corner].abs()).sum(1) + d1.abs() for corner in range(3)), dim=1)
    scale_v = torch.stack(tuple((n2.abs() * a[:, corner].abs()).sum(1) + d2.abs() for corner in range(3)), dim=1)
    error_u, error_v = 256 * eps * (scale_u + 1), 256 * eps * (scale_v + 1)
    near_u = (au <= error_u) | ((au - 1e-6).abs() <= error_u) | ((au - 1e-5).abs() <= error_u)
    near_v = (av <= error_v) | ((av - 1e-6).abs() <= error_v) | ((av - 1e-5).abs() <= error_v)
    near_interval = ((a1 - b0).abs() <= 256 * eps * (a1.abs() + b0.abs() + 1)) | (
        (b1 - a0).abs() <= 256 * eps * (b1.abs() + a0.abs() + 1))
    ambiguous = near_u.any(1) | near_v.any(1) | (~reject & (coplanar | near_interval))
    return result, ambiguous


@torch.no_grad()
def triangle_pairs_intersect_torch(first, second, *, device: str = "cuda:0",
                                  chunk_size: int = 65536, source_recheck: bool = True):
    """批量判断固定三角对；返回目标设备bool(P,)和复核数量/精度诊断。

    first/second均为(P,3,3)float32 NumPy或Tensor、surface RAS/mm，顺序一一
    对应；不修改输入。device须明确CUDA编号，cpu仅算法诊断；chunk_size是
    正整数内存分块，不删候选。默认source_recheck以已有Numba FP64源规则
    复核共面/阈值/接触边界；False只用于原始GPU谓词诊断。所有源阈值固定。
    CUDA不可用/OOM、形状/dtype/非有限输入异常传播，不静默改设备或算法。
    不计算任何顶点接受/拒绝，也不能将固定三角对替代动态Gauss-Seidel状态。
    """
    target = torch.device(device)
    if target.type not in ("cpu", "cuda") or target.type == "cuda" and target.index is None:
        raise ValueError("device must be cpu or an explicitly indexed CUDA target")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    def prepare(value):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float32:
                raise TypeError("triangle coordinates must be float32")
            data = value
        else:
            array = np.asarray(value)
            if array.dtype != np.float32:
                raise TypeError("triangle coordinates must be float32")
            data = torch.as_tensor(array)
        if data.ndim != 3 or tuple(data.shape[1:]) != (3, 3):
            raise ValueError("triangles must have shape (P,3,3)")
        return data
    a, b = prepare(first), prepare(second)
    if a.shape != b.shape:
        raise ValueError("triangle pair arrays must have equal shape")
    result = torch.empty(len(a), dtype=torch.bool, device=target)
    rechecked = 0
    for start in range(0, len(a), chunk_size):
        stop = min(start + chunk_size, len(a))
        # 只上传当前块。此前先将完整host面对搬到CUDA，再分块计算，
        # 会让chunk_size失去控制输入驻留显存的作用。
        aa, bb = a[start:stop].to(target), b[start:stop].to(target)
        if not bool(torch.isfinite(aa).all()) or not bool(torch.isfinite(bb).all()):
            raise ValueError("triangle coordinates must be finite")
        hits, ambiguous = _pair_kernel(aa, bb)
        if source_recheck:
            selected = ambiguous.nonzero().flatten()
            rechecked += len(selected)
            if len(selected):
                selected_a, selected_b = aa[selected].cpu().numpy(), bb[selected].cpu().numpy()
                verified = _source_pairs(selected_a, selected_b)
                hits[selected] = torch.as_tensor(verified, dtype=torch.bool, device=target)
        result[start:stop] = hits
    return result, {"pairs": len(a), "source_rechecked_pairs": rechecked,
                    "dtype": "float64 geometry from float32 vertices", "device": str(target),
                    "chunk_size": chunk_size, "source_recheck": bool(source_recheck),
                    "input_transfer_scope": "one_chunk_at_a_time",
                    "changes_vertex_update_order": False}
