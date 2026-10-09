"""Unconstrained first surface-placement step used to isolate collisions."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _step(xyz: np.ndarray, gradient: np.ndarray, ripped: np.ndarray, dt: float, max_mm: float) -> tuple[np.ndarray, np.ndarray]:
    result = xyz.copy()
    offsets = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        dx = np.float32(dt * gradient[vertex, 0])
        dy = np.float32(dt * gradient[vertex, 1])
        dz = np.float32(dt * gradient[vertex, 2])
        magnitude_squared = np.float32(np.float32(dx * dx + dy * dy) + dz * dz)
        # 固定 C++ 路径 sqrt(float) 先产生 float，再赋给 double mag。
        # 在开方前升为 double 会改变限幅位移，并在退化局部改变后续法向。
        magnitude = np.float64(np.float32(np.sqrt(magnitude_squared)))
        if magnitude > max_mm:
            scale = max_mm / magnitude
            dx = np.float32(dx * scale)
            dy = np.float32(dy * scale)
            dz = np.float32(dz * scale)
        offsets[vertex, 0] = dx
        offsets[vertex, 1] = dy
        offsets[vertex, 2] = dz
        result[vertex, 0] = np.float32(xyz[vertex, 0] + dx)
        result[vertex, 1] = np.float32(xyz[vertex, 1] + dy)
        result[vertex, 2] = np.float32(xyz[vertex, 2] + dz)
    return result, offsets


def unconstrained_step(
    vertices: np.ndarray, gradient: np.ndarray, ripped: np.ndarray,
    *, dt: float = 0.5, max_mm: float = 0.3,
) -> np.ndarray:
    """Apply native momentum zero and max-distance clipping without collisions."""
    return unconstrained_step_with_offsets(
        vertices, gradient, ripped, dt=dt, max_mm=max_mm,
    )[0]


def unconstrained_step_with_offsets(
    vertices: np.ndarray, gradient: np.ndarray, ripped: np.ndarray,
    *, dt: float = 0.5, max_mm: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """按固定原生浮点规则返回限幅终点和未经坐标舍入的位移。

    vertices/gradient 为 (N,3) float32，坐标为 surface RAS/mm，梯度乘以
    dt 后为 mm；ripped 为 (N,) bool，固定顶点保留原坐标并返回零位移。
    dt 默认 0.5，max_mm 默认 0.3 mm。返回两个 (N,3) float32 数组。
    范数平方和及 sqrt 为 float32，随后提升 float64 作限幅比例，再将
    位移写回 float32；这与固定 Conda mris_place_surface 源码重载一致。
    输入按上述类型转换，转换失败的 TypeError/ValueError 原样传播。
    本内部函数不检查形状、有限值或步长范围；调用方须保证匹配的 N、
    有限输入及非负 max_mm。Numba 默认不启用越界检查，非法形状不能
    依赖稳定的异常行为。函数不做碰撞判定或影像写出，也没有独立官方
    CLI；它属于 mris_place_surface 的异步试步准备。固定编译表达式
    与完整真实回归收据见 docs/recon_all/PIAL_STEP_NORM_DIAGNOSTIC.md。
    """
    return _step(
        np.asarray(vertices, dtype=np.float32), np.asarray(gradient, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_), float(np.float32(dt)), float(np.float32(max_mm)),
    )
