"""球面配准的有序 float32 Jacobi 平均内核；只在 CUDA 后端导入。"""

import triton
import triton.language as tl
import torch


@triton.jit
def _average_gradient_kernel(current, following, neighbors, degrees, reciprocals,
                             size: tl.constexpr, width: tl.constexpr,
                             block: tl.constexpr):
    element = tl.program_id(0) * block + tl.arange(0, block)
    vertex, axis = element // 3, element % 3
    valid = vertex < size
    degree = tl.load(degrees + vertex, mask=valid, other=0)
    value = tl.load(current + element, mask=valid, other=0)
    for column in tl.static_range(width):
        active = valid & (column < degree)
        adjacent = tl.load(neighbors + vertex * width + column, mask=active, other=0)
        sample = tl.load(current + adjacent * 3 + axis, mask=active, other=0)
        value = tl.where(active, value + sample, value)
    reciprocal = tl.load(reciprocals + vertex, mask=valid, other=0)
    tl.store(following + element, value * reciprocal, mask=valid)


def average_on_cuda(current, following, neighbors, degrees, reciprocals, iterations):
    """执行全部 iterations 轮，逐轮交换缓冲区；返回最终 CUDA 张量。

    current/following 为 (N,3) 连续 float32 张量；neighbors 为 (N,K)
    int64，degrees 为 (N,) int64，reciprocals 为 CPU 原规则计算后搬入的
    (N,) float32。全部位于同一显式 CUDA 设备。没有原子累加、邻居截断、
    fastmath、FMA 融合、TF32 矩阵运算或半精度。每轮全局依赖由同一
    CUDA stream 上的 kernel 顺序保证，节点内部保留邻接顺序。
    """
    size, width = neighbors.shape
    with torch.cuda.device(current.device):
        for _ in range(iterations):
            _average_gradient_kernel[(triton.cdiv(size * 3, 256),)](
                current, following, neighbors, degrees, reciprocals,
                size=size, width=width, block=256, num_warps=4,
                enable_fp_fusion=False)
            current, following = following, current
    return current
