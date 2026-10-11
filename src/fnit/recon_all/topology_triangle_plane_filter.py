"""保守F64平面分离筛选；不确定时保留原三角形相交谓词。"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

_ERROR_FACTOR = 64.0 * np.finfo(np.float64).eps


@njit(cache=True, fastmath=False)
def _one_plane_separates(a, b):
    ux, uy, uz = a[1] - a[0]
    vx, vy, vz = a[2] - a[0]
    nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
    # 包含cross抵消前的各乘积，不能仅以最终小法向估计舍入误差。
    mx, my, mz = abs(uy*vz)+abs(uz*vy), abs(uz*vx)+abs(ux*vz), abs(ux*vy)+abs(uy*vx)
    positive, negative, above_plane_tolerance = True, True, False
    for corner in range(3):
        dx, dy, dz = b[corner] - a[0]
        distance = (dx*nx + dy*ny) + dz*nz
        magnitude = abs(dx)*mx + abs(dy)*my + abs(dz)*mz
        error = _ERROR_FACTOR*magnitude + 1e-300
        if not np.isfinite(distance) or not np.isfinite(error):
            return False
        lower = abs(distance)-error
        # 保护原谓词du[0]*du[1]/du[2]的乘积符号及下溢语义。
        if lower <= 1e-140 or abs(distance)+error >= 1e140:
            return False
        positive = positive and distance > error
        negative = negative and distance < -error
        above_plane_tolerance = above_plane_tolerance or lower > 1e-6
    return (positive or negative) and above_plane_tolerance


@njit(cache=True, fastmath=False, parallel=True)
def _numba_separated(a, b):
    result = np.empty(len(a), np.bool_)
    for index in prange(len(a)):
        result[index] = _one_plane_separates(a[index], b[index]) or _one_plane_separates(b[index], a[index])
    return result


def _torch_one_plane(a, b):
    import torch
    u, v = a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]
    first = u[:, [1, 2, 0]]*v[:, [2, 0, 1]]
    second = u[:, [2, 0, 1]]*v[:, [1, 2, 0]]
    normal = first-second
    magnitude = first.abs()+second.abs()
    delta = b-a[:, :1]
    distance = (delta*normal[:, None]).sum(dim=2)
    error = _ERROR_FACTOR*(delta.abs()*magnitude[:, None]).sum(dim=2)+1e-300
    finite = (torch.isfinite(distance)&torch.isfinite(error)).all(dim=1)
    lower = distance.abs()-error
    product_safe = ((lower > 1e-140)&(distance.abs()+error < 1e140)).all(dim=1)
    same_sign = (distance > error).all(dim=1)|(distance < -error).all(dim=1)
    return finite&product_safe&same_sign&(lower > 1e-6).any(dim=1)


def separated_triangle_pairs(*, triangles_a, triangles_b, backend="numba", device=None,
                             chunk_size=65536):
    """返回能安全排除的三角形对bool(P)，True表示无需原相交谓词。

    triangles_a/triangles_b为同长(P,3,3)坐标，使用原检查器F64/mm精度；
    不要求有限：非有限、共面、近零、误差区间或极端乘积均返回False，
    由调用者继续执行原谓词。backend默认numba，使用调用者Numba线程
    预算；torch要求显式CUDA device，按chunk_size（默认65536）分块。
    两者保持F64，无autocast、无矩阵乘法，不修改输入或进程精度策略。
    chunk_size为正整数。返回CPU bool数组；不输出相交结论或修改网格。
    CUDA执行/内存失败显式传播，不自动CPU回退。没有独立原软件命令，
    对应原三角面平面同侧排除分支，余下完整判定仍由原函数执行。
    """
    if backend not in ("numba", "torch"):
        raise ValueError("backend must be numba or torch")
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    a, b = np.asarray(triangles_a, dtype=np.float64), np.asarray(triangles_b, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or b.shape != a.shape:
        raise ValueError("corresponding (P,3,3) triangles required")
    if backend == "torch":
        import torch
        target = torch.device(device) if device is not None else None
        if target is None or target.type != "cuda" or target.index is None:
            raise ValueError("torch requires explicit cuda:N device")
    result = np.empty(len(a), dtype=np.bool_)
    for start in range(0, len(a), chunk_size):
        stop = min(start+chunk_size, len(a))
        if backend == "numba":
            result[start:stop] = _numba_separated(np.ascontiguousarray(a[start:stop]), np.ascontiguousarray(b[start:stop]))
        else:
            with torch.no_grad():
                # 明确F64；上层autocast不改变本元素算子及其F64输入。
                ta = torch.as_tensor(np.ascontiguousarray(a[start:stop]), dtype=torch.float64, device=target)
                tb = torch.as_tensor(np.ascontiguousarray(b[start:stop]), dtype=torch.float64, device=target)
                rejected = _torch_one_plane(ta, tb)|_torch_one_plane(tb, ta)
                result[start:stop] = rejected.cpu().numpy()
                del ta, tb, rejected
    return result
