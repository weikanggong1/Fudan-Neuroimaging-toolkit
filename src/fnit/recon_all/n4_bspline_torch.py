"""固定残差的 N4 单层 cubic B-spline 拟合候选。

这是 ITK 5.4.7 单层拟合公式的张量实现，不是完整 N4。适用于三维、
全 1 掩膜、无 confidence、开放边界、cubic、单位方向和零起始索引。
权重和每点的 64 项平方和按源顺序以 FP32 计算；控制点归约使用
FP64 分段求和，故不承诺复现源实现的逐点 FP32 累加尾差。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch


def _cubic_kernel(u: torch.Tensor) -> torch.Tensor:
    """ITK cardinal cubic 表达式；输入和返回为 FP32，支撑范围 [-2,2]。"""
    a = u.abs()
    squared = a * a
    inner = ((4.0 - 6.0 * squared) + ((3.0 * squared) * a)) / 6.0
    outer = (((8.0 - 12.0 * a) + 6.0 * squared) - squared * a) / 6.0
    return torch.where(a < 1.0, inner, torch.where(a < 2.0, outer, 0.0))


def _triple(values: Sequence, *, name: str, integer: bool) -> tuple:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    if integer:
        if any(int(v) != v for v in values):
            raise ValueError(f"{name} must contain integers")
        return tuple(int(v) for v in values)
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass
class N4DenseBSplineFit:
    """缓存一个 N4 层的固定几何，只在每次调用读取新的残差。

    ``field_shape`` 和 ``control_shape`` 是 (x,y,z)；spacing 为 mm，origin
    是 identity direction 下的物理 mm 坐标。device 显式选择 CPU/CUDA。
    每个轴 field_shape>=2、control_shape>=4；尚不支持掩膜筛选、confidence、
    非 cubic、闭合边界、多层 refine 或非零 region index。构造有一次排序
    成本，不能把已缓存的热调用计时当成完整 N4 时间。
    """

    field_shape: Sequence[int]
    control_shape: Sequence[int]
    spacing: Sequence[float] = (1.0, 1.0, 1.0)
    origin: Sequence[float] = (0.0, 0.0, 0.0)
    device: str | torch.device = "cuda:0"

    def __post_init__(self) -> None:
        self.field_shape = _triple(self.field_shape, name="field_shape", integer=True)
        self.control_shape = _triple(self.control_shape, name="control_shape", integer=True)
        self.spacing = _triple(self.spacing, name="spacing", integer=False)
        self.origin = _triple(self.origin, name="origin", integer=False)
        if min(self.field_shape) < 2 or min(self.control_shape) < 4:
            raise ValueError("field_shape>=2 and cubic control_shape>=4 are required")
        if min(self.spacing) <= 0:
            raise ValueError("spacing must be positive")
        self.device = torch.device(self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._build_geometry()

    def _axis(self, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
        size, count = self.field_shape[axis], self.control_shape[axis]
        spacing, origin = self.spacing[axis], self.origin[axis]
        spans = count - 3
        # ITK PointSet 的 PointType 为 float；r/epsilon 为 RealType float，
        # 而物理 origin/spacing 的运算为 double。
        rate = np.float32(spans / (float(np.float32(size - 1)) * spacing))
        epsilon = np.float32(float(rate) * spacing * float(np.float32(0.001)))
        physical = (torch.arange(size, dtype=torch.float64, device=self.device) * spacing + origin).float()
        p = ((physical.double() - origin) * float(rate)).float()
        p = torch.where((p - float(spans)).abs() <= float(epsilon), float(np.float32(spans) - epsilon), p)
        p = torch.where((p < 0.0) & (p.abs() <= float(epsilon)), 0.0, p)
        if bool(((p < 0.0) | (p >= spans)).any()):
            raise ValueError("physical coordinates are outside the ITK parametric domain")
        base = p.to(torch.int64)
        fraction = p - base.float()
        values = torch.stack([_cubic_kernel((fraction - float(corner)) + 1.0) for corner in range(4)])
        return base, values

    def _build_geometry(self) -> None:
        ax = [self._axis(i) for i in range(3)]
        nx, ny, nz = self.field_shape
        cx, cy, cz = self.control_shape
        # 内部所有点按 ITK iterator 的 x-fast 次序排列。
        bx = ax[0][0][None, None, :].expand(nz, ny, nx).reshape(-1)
        by = ax[1][0][None, :, None].expand(nz, ny, nx).reshape(-1)
        bz = ax[2][0][:, None, None].expand(nz, ny, nx).reshape(-1)
        weights, targets = [], []
        square_sum = torch.zeros(nx * ny * nz, dtype=torch.float32, device=self.device)
        # idx[0] 最快；每点乘 x、y、z，再按 64 项顺序加平方。
        for z in range(4):
            for y in range(4):
                for x in range(4):
                    wx = ax[0][1][x][None, None, :]
                    wy = ax[1][1][y][None, :, None]
                    wz = ax[2][1][z][:, None, None]
                    weight = ((wx * wy) * wz).expand(nz, ny, nx).reshape(-1)
                    square_sum.add_(weight * weight)
                    weights.append(weight)
                    targets.append((bx + x) + (by + y) * cx + (bz + z) * (cx * cy))
        weight = torch.stack(weights, dim=1).reshape(-1)
        target = torch.stack(targets, dim=1).reshape(-1)
        denominator = square_sum[:, None].expand(-1, 64).reshape(-1)
        coefficient = ((weight * weight) * weight) / denominator
        # stable 排序保存同一控制点的 x-fast 残差点次序；分段 FP64 归约
        # 和源 FP32 逐点累加仍有不同的舍入语义，报告单独比较。
        order = torch.argsort(target, stable=True)
        self._lengths = torch.bincount(target, minlength=cx * cy * cz)
        self._point = (order // 64).to(torch.int32)
        self._coefficient = coefficient[order]
        omega = torch.segment_reduce((weight[order] * weight[order]).double(), "sum", lengths=self._lengths)
        self._omega = omega.float()

    @torch.no_grad()
    def fit(self, residual: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        """FP32 xyz 残差 → FP32 xyz 控制点；保持输入设备，无文件 I/O。

        输入 shape=field_shape、dtype=float32、device=构造设备，单位是 log
        intensity。默认检查有限值，会产生一次 GPU→CPU 布尔同步；重复热
        调用在外部已经校验输入时可显式 ``validate=False``。非有限输入、
        shape/dtype/device 不匹配抛 ValueError/TypeError，不返回近似替代。
        """
        if residual.dtype != torch.float32:
            raise TypeError("residual must be float32; no implicit half precision")
        if tuple(residual.shape) != self.field_shape or residual.device != self.device:
            raise ValueError("residual shape/device does not match the frozen geometry")
        if validate and not bool(torch.isfinite(residual).all()):
            raise ValueError("residual must contain finite values")
        flat = residual.permute(2, 1, 0).contiguous().reshape(-1)
        contribution = flat[self._point.long()] * self._coefficient
        delta = torch.segment_reduce(contribution.double(), "sum", lengths=self._lengths).float()
        threshold = float(np.float32(0.1 * np.finfo(np.float32).eps))
        phi = torch.where(self._omega.abs() > threshold, delta / self._omega, 0.0)
        phi = torch.where(torch.isfinite(phi), phi, 0.0)
        cx, cy, cz = self.control_shape
        return phi.reshape(cz, cy, cx).permute(2, 1, 0).contiguous()

    @property
    def cache_bytes(self) -> int:
        """实际持久张量缓存字节数；不含构造/调用临时张量或 CUDA context。"""
        return sum(t.numel() * t.element_size() for t in
                   (self._lengths, self._point, self._coefficient, self._omega))
