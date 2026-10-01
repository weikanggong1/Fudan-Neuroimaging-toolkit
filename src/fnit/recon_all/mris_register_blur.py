"""Spherical parameterization blur used by ``mris_register``."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit, prange
import torch


def _blur_atlas_frame_torch(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur one float32 (azimuth, polar) frame in FreeSurfer source order."""
    v_dim, u_dim = frame.shape
    cart_klen = round(6 * sigma) + 1
    if cart_klen % 2 == 0:
        cart_klen += 1
    sigma_sq_inv = float((torch.tensor(1, dtype=torch.float32)
                          / torch.tensor(sigma, dtype=torch.float32).square()).item())
    output = torch.empty_like(frame)
    for u in range(u_dim):
        sin_sq = math.sin(u * math.pi / u_dim) ** 2
        if sin_sq < torch.finfo(torch.float32).eps:
            klen = 4 * cart_klen
        else:
            k = cart_klen * cart_klen
            klen = min(int(math.sqrt(k + k / sin_sq)), 4 * cart_klen)
        khalf = min(klen, u_dim - 1, v_dim - 1) // 2
        weights = {(uk, vk): math.exp(-(uk * uk + sin_sq * vk * vk) * sigma_sq_inv)
                   for uk in range(khalf + 1) for vk in range(khalf + 1)}
        total = torch.zeros(v_dim, dtype=torch.float64, device=frame.device)
        ktotal = 0.0
        for uk in range(-khalf, khalf + 1):
            u1 = u + uk
            voff = 0
            if u1 < 0:
                u1 = -u1
                voff = v_dim // 2
            elif u1 >= u_dim:
                u1 = u_dim - (u1 - u_dim + 1)
                voff = v_dim // 2
            column = frame[:, u1].to(torch.float64)
            for vk in range(-khalf, khalf + 1):
                weight = weights[abs(uk), abs(vk)]
                ktotal += weight
                total += weight * torch.roll(column, -(vk + voff))
        output[:, u] = (total / ktotal).float()
    return output


@lru_cache(maxsize=8)
def _blur_weights(u_dim: int, v_dim: int, sigma: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """缓存同一网格和 sigma 的原顺序 double 权重及分母，不缓存影像。"""
    cart_klen = round(6 * sigma) + 1
    if cart_klen % 2 == 0:
        cart_klen += 1
    sigma_sq_inv = float((torch.tensor(1, dtype=torch.float32)
                          / torch.tensor(sigma, dtype=torch.float32).square()).item())
    halves = np.empty(u_dim, np.int64)
    for u in range(u_dim):
        sin_sq = math.sin(u * math.pi / u_dim) ** 2
        if sin_sq < torch.finfo(torch.float32).eps:
            klen = 4 * cart_klen
        else:
            k = cart_klen * cart_klen
            klen = min(int(math.sqrt(k + k / sin_sq)), 4 * cart_klen)
        halves[u] = min(klen, u_dim - 1, v_dim - 1) // 2
    weights = np.zeros((u_dim, int(halves.max()) + 1, int(halves.max()) + 1), np.float64)
    totals = np.empty(u_dim, np.float64)
    for u in range(u_dim):
        sin_sq = math.sin(u * math.pi / u_dim) ** 2
        khalf = int(halves[u])
        for uk in range(khalf + 1):
            for vk in range(khalf + 1):
                weights[u, uk, vk] = math.exp(-(uk * uk + sin_sq * vk * vk) * sigma_sq_inv)
        total = 0.0
        for uk in range(-khalf, khalf + 1):
            for vk in range(-khalf, khalf + 1):
                total += weights[u, abs(uk), abs(vk)]
        totals[u] = total
    for array in (halves, weights, totals):
        array.flags.writeable = False
    return halves, weights, totals


@njit(parallel=True, cache=True, fastmath=False)
def _blur_cpu(frame_uv: np.ndarray, halves: np.ndarray, weights: np.ndarray,
              totals: np.ndarray) -> np.ndarray:
    """各纬度独立并行；每像素仍按 uk/vk 原顺序执行 double 乘加。"""
    u_dim, v_dim = frame_uv.shape
    result = np.empty((u_dim, v_dim), np.float32)
    for u in prange(u_dim):
        accumulated = np.zeros(v_dim, np.float64)
        khalf = halves[u]
        for uk in range(-khalf, khalf + 1):
            u1 = u + uk
            voff = 0
            if u1 < 0:
                u1 = -u1
                voff = v_dim // 2
            elif u1 >= u_dim:
                u1 = u_dim - (u1 - u_dim + 1)
                voff = v_dim // 2
            for vk in range(-khalf, khalf + 1):
                weight = weights[u, abs(uk), abs(vk)]
                for v in range(v_dim):
                    sample = np.float64(frame_uv[u1, (v + vk + voff) % v_dim])
                    accumulated[v] = accumulated[v] + weight * sample
        for v in range(v_dim):
            result[u, v] = np.float32(accumulated[v] / totals[u])
    return result


def blur_atlas_frame(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    """按 FreeSurfer MRISPblur 的规则平滑一张球面图谱帧。

    输入 frame 为 (方位角, 极角) 二维张量；通常为 float32，值的单位
    由调用者决定。sigma 是正的有限平滑尺度（默认配准使用 4/2/1/0.5）。
    输出与输入 shape/dtype/device 相同，不修改输入，极点反射同时偏移
    半个方位角周期。CPU float32 使用缓存权重和有序 double 累加的
    Numba 内核；其他输入保留 PyTorch 路径。线程数沿用外部 Numba
    预算，不创建额外线程池；没有自动半精度。无效维度/尺度直接报错。
    属于 mris_register 的内部步骤，没有独立官方 CLI。
    """
    sigma = float(sigma)
    if frame.ndim != 2 or any(size == 0 for size in frame.shape):
        raise ValueError("frame must be a nonempty two-dimensional tensor")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if frame.device.type != "cpu" or frame.dtype != torch.float32 or frame.requires_grad:
        return _blur_atlas_frame_torch(frame, sigma)
    v_dim, u_dim = frame.shape
    halves, weights, totals = _blur_weights(u_dim, v_dim, sigma)
    frame_uv = np.ascontiguousarray(frame.detach().numpy().T)
    result = _blur_cpu(frame_uv, halves, weights, totals)
    return torch.from_numpy(result.T.copy())
