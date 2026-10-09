"""三维控制点扩展的局部 GPU 邻域统计，复用原控制点迭代流程。"""
from __future__ import annotations

import numpy as np
import torch


class NeighborSumTorch:
    """缓存一个固定 ROI 的源强度与缓冲，计算原 six/cube 邻域统计。

    image 为有限 float32 的三维 (x,y,z) 强度图；region 为三个步长为1的
    正向 slice；kernel 仅接受原 six 或 3³ 全1模板。device 必须显式 CUDA。
    邻域单位为体素，nearest 边界与原 SciPy 卷积相同，不做 RAS 变换。
    每次调用输入当前 CPU bool 控制图，GPU 双精度累加源 float32 值，
    写回与旧函数相同的 int16 数量、float32 总量 NumPy 视图。视图在下一次
    调用覆写，仅供既有同步迭代立即消费。控制点不由本类修改；不使用半精度。
    无独立 CLI，属于 mri_normalize 的 MRInormFindControlPoints 内部算子。
    非 CUDA、形状/模板/区域不符合约定或非有限源图抛 ValueError。
    """

    def __init__(self, image: np.ndarray, kernel: np.ndarray,
                 region: tuple[slice, slice, slice], *, device: str):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("NeighborSumTorch requires an explicit CUDA device")
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 3 or not np.isfinite(image).all():
            raise ValueError("image must be a finite float32 3D volume")
        six = np.zeros((3, 3, 3), dtype=np.int16)
        six[0, 1, 1] = six[2, 1, 1] = 1
        six[1, 0, 1] = six[1, 2, 1] = 1
        six[1, 1, 0] = six[1, 1, 2] = 1
        self.six = np.array_equal(kernel, six)
        if not self.six and not np.array_equal(kernel, np.ones((3, 3, 3), dtype=np.int16)):
            raise ValueError("kernel must be the source six-neighbor or 3x3x3 all-one stencil")
        if len(region) != 3 or any(not isinstance(part, slice) or part.step not in (None, 1)
                or part.start is None or part.stop is None or not 0 <= part.start < part.stop <= size
                for part, size in zip(region, image.shape)):
            raise ValueError("region must contain three nonempty bounded unit-step slices")
        self.source_shape = image.shape
        self.halo = tuple(slice(max(0, part.start - 1), min(size, part.stop + 1))
                          for part, size in zip(region, image.shape))
        self.start = tuple(part.start - halo.start for part, halo in zip(region, self.halo))
        self.shape = tuple(part.stop - part.start for part in region)
        self.source = torch.as_tensor(np.ascontiguousarray(image[self.halo]), device=self.device)
        self.control = torch.empty(self.source.shape, dtype=torch.uint8, device=self.device)
        self.count = torch.empty(self.shape, dtype=torch.int16, device=self.device)
        self.total = torch.empty(self.shape, dtype=torch.float32, device=self.device)
        self.cpu_count = torch.empty(self.shape, dtype=torch.int16)
        self.cpu_total = torch.empty(self.shape, dtype=torch.float32)

    def __call__(self, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """上传当前控制图，返回可覆写的 count/total；同步下载后原循环继续。"""
        if control.shape != self.source_shape or control.dtype != np.bool_:
            raise ValueError("control must be a same-grid CPU bool array")
        from ._normalization_neighbors_cuda import neighbor_sum
        self.control.copy_(torch.from_numpy(np.ascontiguousarray(control[self.halo], dtype=np.uint8)))
        neighbor_sum(source=self.source, control=self.control, count=self.count,
                     total=self.total, shape=self.shape, start=self.start, six=self.six)
        self.cpu_count.copy_(self.count)
        self.cpu_total.copy_(self.total)
        return self.cpu_count.numpy(), self.cpu_total.numpy()
