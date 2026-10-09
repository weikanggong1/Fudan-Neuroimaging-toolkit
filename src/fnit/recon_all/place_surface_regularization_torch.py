# Modified FNIT implementation of placement formulas, source d932c45.
# FreeSurfer Software License: licenses/FreeSurfer.txt.
"""固定网格 placement 正则梯度的 PyTorch 实现；动态碰撞仍由原优化器处理。"""

from __future__ import annotations

import numpy as np
import torch


def _dot3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def _cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), dim=1)


def _normalize(a: torch.Tensor) -> torch.Tensor:
    length = torch.sqrt(_dot3(a, a))
    reciprocal = torch.where(length < 1e-5, torch.ones_like(length), 1.0 / length)
    return a * reciprocal[:, None]


def _basis(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted = normal[:, [1, 2, 0]]
    first = _cross3(normal, shifted)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    first = torch.where((torch.sqrt(_dot3(first, first)) < .001)[:, None],
                        _cross3(normal, alternate), first)
    second = _cross3(normal, first)
    return _normalize(first), _normalize(second)


def _inverse5(gram: torch.Tensor) -> torch.Tensor:
    """批量复用 VNL float32 QR 的逐列运算次序，不调用不同舍入的 linalg.inverse。"""
    work = gram.clone()
    aux = torch.zeros((len(work), 5), dtype=torch.float32, device=work.device)
    for col in range(4):
        scale = torch.zeros(len(work), dtype=torch.float32, device=work.device)
        ssq = torch.ones_like(scale)
        for row in range(col, 5):
            absolute = work[:, row, col].abs()
            nonzero = absolute != 0
            larger = scale < absolute
            # Source SNRM2 also accepts nonzero subnormal scales.  Inactive
            # zero-divisor branches are masked below; do not clamp tiny values.
            ratio_larger = scale / absolute
            ratio_smaller = absolute / scale
            updated = torch.where(larger, ssq * (ratio_larger * ratio_larger) + 1.0,
                                  ssq + ratio_smaller * ratio_smaller)
            ssq = torch.where(nonzero, updated, ssq)
            scale = torch.where(nonzero & larger, absolute, scale)
        norm = (scale.to(torch.float64) * torch.sqrt(ssq.to(torch.float64))).to(torch.float32)
        active = norm != 0
        norm = torch.where(work[:, col, col] < 0, -norm, norm)
        reciprocal = torch.where(active, 1.0 / norm, torch.zeros_like(norm))
        for row in range(col, 5):
            work[:, row, col] = torch.where(active, work[:, row, col] * reciprocal,
                                            work[:, row, col])
        work[:, col, col] = torch.where(active, work[:, col, col] + 1.0,
                                        work[:, col, col])
        for other in range(col + 1, 5):
            dot = torch.zeros_like(norm)
            for row in range(col, 5):
                dot = dot + work[:, row, col] * work[:, row, other]
            factor = -dot / work[:, col, col]
            for row in range(col, 5):
                work[:, row, other] = torch.where(
                    active, work[:, row, other] + factor * work[:, row, col],
                    work[:, row, other])
        aux[:, col] = torch.where(active, work[:, col, col], aux[:, col])
        work[:, col, col] = torch.where(active, -norm, work[:, col, col])
    inverse = torch.zeros_like(work)
    for rhs in range(5):
        result = torch.zeros_like(aux)
        result[:, rhs] = 1.0
        for col in range(4):
            dot = torch.zeros(len(work), dtype=torch.float32, device=work.device)
            for row in range(col, 5):
                value = aux[:, col] if row == col else work[:, row, col]
                dot = dot + value * result[:, row]
            factor = -dot / aux[:, col]
            for row in range(col, 5):
                value = aux[:, col] if row == col else work[:, row, col]
                result[:, row] = result[:, row] + factor * value
        for col in range(4, -1, -1):
            result[:, col] = result[:, col] / work[:, col, col]
            factor = -result[:, col]
            for row in range(col):
                result[:, row] = result[:, row] + factor * work[:, row, col]
        inverse[:, :, rhs] = result
    return inverse


class PlacementRegularizationTorch:
    """缓存固定邻接的 signed averaging、弹簧与二次曲率梯度。

    输入坐标/法线/梯度为同顶点顺序 (N,3) float32，surface RAS/mm。
    一跳 neighbors/valid 和二跳 offsets/candidates 沿用原 CPU 排序；ripped
    为 (N,) bool。device 默认 cuda:0，chunk_size 默认 32768，均可显式指定。
    每次调用上传当前坐标/法线，不缓存失效几何；缓存仅属于当前网格/掩膜。
    输出 (N,3) float32 NumPy 梯度，保留逐邻接加法和 5x5 QR 次序。
    不使用低精度、矩阵乘法或全局精度开关；CUDA 周围其他阶段仍使用 TF32。
    该函数是 mris_place_surface 内部步骤，没有独立官方 CLI。
    """

    def __init__(self, *, neighbors, valid, offsets, candidates, ripped,
                 device: str = "cuda:0", chunk_size: int = 32768):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            raise ValueError("specify an explicit CUDA device, e.g. cuda:0")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        first = np.asarray(neighbors, dtype=np.int64)
        mask = np.asarray(valid, dtype=np.bool_)
        skip = np.asarray(ripped, dtype=np.bool_)
        offset = np.asarray(offsets, dtype=np.int64)
        flat = np.asarray(candidates, dtype=np.int64)
        count = len(skip)
        if skip.ndim != 1 or first.ndim != 2 or first.shape[0] != count or mask.shape != first.shape \
                or offset.shape != (count + 1,) or offset[0] != 0 \
                or offset[-1] != len(flat) or np.any(np.diff(offset) < 0):
            raise ValueError("inconsistent ordered adjacency arrays")
        if np.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("valid neighbors must precede padding, matching native break order")
        if (first[mask].size and (first[mask].min() < 0 or first[mask].max() >= count)) \
                or (flat.size and (flat.min() < 0 or flat.max() >= count)):
            raise ValueError("adjacency index outside vertex range")
        degree = np.diff(offset)
        second = np.zeros((count, int(degree.max(initial=0))), dtype=np.int64)
        second_valid = np.zeros_like(second, dtype=np.bool_)
        for vertex in range(count):
            size = degree[vertex]
            second[vertex, :size] = flat[offset[vertex]:offset[vertex + 1]]
            second_valid[vertex, :size] = True
        self.neighbors = torch.tensor(first, device=self.device)
        self.valid = torch.tensor(mask, device=self.device)
        self.ripped = torch.tensor(skip, device=self.device)
        self.second = torch.tensor(second, device=self.device)
        self.second_valid = torch.tensor(second_valid, device=self.device)
        self.count = count
        self.chunk_size = chunk_size

    def _array(self, value, name: str) -> torch.Tensor:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.count, 3) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite (N,3)")
        return torch.as_tensor(array, device=self.device)

    def _average(self, gradient: torch.Tensor, iterations: int) -> torch.Tensor:
        current = gradient
        for _ in range(iterations):
            total = current.to(torch.float64)
            count = torch.ones(self.count, dtype=torch.int64, device=self.device)
            for rank in range(self.neighbors.shape[1]):
                other = self.neighbors[:, rank]
                nearby = current[other]
                active = self.valid[:, rank] & ~self.ripped[other] & (_dot3(current, nearby) >= 0)
                total = total + torch.where(active[:, None], nearby.to(torch.float64), 0.0)
                count = count + active.to(torch.int64)
            following = (total / count[:, None]).to(torch.float32)
            current = torch.where(self.ripped[:, None], current, following)
        return current

    def _springs(self, xyz: torch.Tensor, normal: torch.Tensor, weight: float) -> tuple:
        total = torch.zeros_like(xyz)
        count = torch.zeros(self.count, dtype=torch.int64, device=self.device)
        for rank in range(self.neighbors.shape[1]):
            other = self.neighbors[:, rank]
            active = self.valid[:, rank] & ~self.ripped[other]
            total = total + torch.where(active[:, None], xyz[other] - xyz, 0.0)
            count = count + active.to(torch.int64)
        total = total / count.clamp_min(1).to(torch.float32)[:, None]
        component = _dot3(total, normal)
        factor = float(np.float32(weight))
        normal_term = (factor * component.to(torch.float64)[:, None] * normal.to(torch.float64)).to(torch.float32)
        tangent_term = (factor * (total - component[:, None] * normal).to(torch.float64)).to(torch.float32)
        return (torch.where(self.ripped[:, None], 0.0, normal_term),
                torch.where(self.ripped[:, None], 0.0, tangent_term))

    def _curvature(self, xyz: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        first, second = _basis(normal)
        result = torch.zeros(self.count, dtype=torch.float32, device=self.device)
        for start in range(0, self.count, self.chunk_size):
            stop = min(start + self.chunk_size, self.count)
            valid = self.second_valid[start:stop]
            delta = xyz[self.second[start:stop]] - xyz[start:stop, None, :]
            def project(direction):
                return ((delta[:, :, 0] * direction[start:stop, None, 0] +
                         delta[:, :, 1] * direction[start:stop, None, 1]) +
                        delta[:, :, 2] * direction[start:stop, None, 2])
            u, v, height = project(first), project(second), project(normal)
            design = torch.stack((u * u, v * v, u, v, torch.ones_like(u)), dim=2)
            design = torch.where(valid[:, :, None], design, 0.0)
            gram = torch.zeros((stop-start, 5, 5), dtype=torch.float32, device=self.device)
            for rank in range(design.shape[1]):
                row = design[:, rank]
                gram = gram + row[:, :, None] * row[:, None, :]
            inverse = _inverse5(gram)
            scalar = torch.zeros(stop-start, dtype=torch.float32, device=self.device)
            for rank in range(design.shape[1]):
                pseudo = torch.zeros_like(scalar)
                for item in range(5):
                    pseudo = pseudo + inverse[:, 4, item] * design[:, rank, item]
                scalar = scalar + torch.where(valid[:, rank], pseudo * height[:, rank], 0.0)
            active = ~self.ripped[start:stop] & (valid.sum(dim=1) >= 5)
            result[start:stop] = torch.where(active, scalar, 0.0)
        return result

    @torch.inference_mode()
    def regularize(self, *, vertices, normals, gradient, iterations: int,
                   spring_weight: float = .3, after_average=None) -> np.ndarray:
        """返回 averaged→可选自斥力→normal spring→curvature→tangent spring。

        after_average 是 white 的 (N,3) 自斥力；pial 已在 gradient 中合并斥力，
        此处传 None。iterations 非负；失败抛 ValueError/PyTorch 异常。
        一次完整梯度回传 CPU，不改变优化器的异步逐顶点接受及清理。
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if not np.isfinite(spring_weight):
            raise ValueError("spring_weight must be finite")
        xyz, normal = self._array(vertices, "vertices"), self._array(normals, "normals")
        averaged = self._average(self._array(gradient, "gradient"), iterations)
        if after_average is not None:
            averaged = averaged + self._array(after_average, "after_average")
        normal_term, tangent_term = self._springs(xyz, normal, spring_weight)
        curvature = self._curvature(xyz, normal)
        result = ((averaged + normal_term) + curvature[:, None] * normal) + tangent_term
        array = result.cpu().numpy()
        if not np.isfinite(array).all():
            raise FloatingPointError("placement QR produced a nonfinite gradient; no silent regularization fallback")
        return array
