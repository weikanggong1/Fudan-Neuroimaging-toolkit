"""Exact source-order CPU averaging for spherical registration gradients."""

import numpy as np
import torch
from numba import njit


@njit(cache=True, fastmath=False)
def three_hop_neighbor_total(neighbors, degrees):
    """统计三跳内不重复邻点；标记数组复用，整数 BFS 不改变邻域定义。

    neighbors 为 (N, K) int64 邻接矩阵，degrees 为 (N,) int64 有效列数。
    返回所有顶点三跳可达邻点数之和，不含顶点自身；支持断连图及孤立点。
    邻接顺序保留。每个顶点使用独立的整数戳和 FIFO，避免 Python set，
    两个 O(N) 数组仅分配一次，没有固定大小截断或低精度计算。
    """
    size = len(degrees)
    marks = np.zeros(size, np.int64)
    queue = np.empty(size, np.int64)
    total = 0
    for vertex in range(size):
        stamp = vertex + 1
        marks[vertex] = stamp
        queue[0] = vertex
        begin, end = 0, 1
        for _ in range(3):
            layer_end = end
            for index in range(begin, layer_end):
                current = queue[index]
                for column in range(degrees[current]):
                    adjacent = neighbors[current, column]
                    if marks[adjacent] != stamp:
                        marks[adjacent] = stamp
                        queue[end] = adjacent
                        end += 1
            begin = layer_end
        total += end - 1
    return total


@njit
def _average_numpy(gradient, neighbors, degrees, reciprocals, iterations):
    current = gradient.copy()
    following = np.empty_like(current)
    for _ in range(iterations):
        for vertex in range(len(current)):
            degree = degrees[vertex]
            for axis in range(3):
                value = current[vertex, axis]
                for index in range(degree):
                    value = np.float32(value + current[neighbors[vertex, index], axis])
                following[vertex, axis] = np.float32(value * reciprocals[vertex])
        current, following = following, current
    return current


@torch.no_grad()
def average_gradients_exact_cpu(gradient: torch.Tensor,
                                neighbors: torch.Tensor, degrees: torch.Tensor,
                                iterations: int) -> torch.Tensor:
    """Match MRISaverageGradients float32 arithmetic with an ordered CPU loop."""
    if gradient.device.type != "cpu":
        raise ValueError("source-order averaging requires CPU tensors")
    reciprocals = (1.0 / (degrees + 1).float()).numpy()
    result = _average_numpy(gradient.numpy(), neighbors.numpy(), degrees.numpy(),
                            reciprocals, iterations)
    return torch.from_numpy(result)


class RegistrationGradientAverager:
    """复用同一表面邻接，在 CPU 或显式 CUDA 上执行原规则平均。

    neighbors: (N,K) 有序 int64 邻接；degrees: (N,) int64 有效邻接数；
    device 默认 cpu。CUDA 构造时仅一次搬运邻接和 CPU float32 倒数。
    每次调用输入 (N,3) float32 gradient 与非负 iterations，返回同
    shape/dtype/device 的新张量，不修改输入。CUDA 只读旧缓冲区、写新
    缓冲区，保留每顶点邻接累加顺序，未开启半精度或改变全局 TF32 策略。
    返回 CPU 输入的结果会完成必要 D2H 同步；完整阶段必须计入此传输。
    Triton 已由主页 Conda 声明；CUDA 不可用、参数无效或编译失败时抛错。
    """

    def __init__(self, neighbors: torch.Tensor, degrees: torch.Tensor,
                 device: str = "cpu"):
        if (neighbors.ndim != 2 or degrees.shape != (len(neighbors),)
                or neighbors.dtype != torch.int64 or degrees.dtype != torch.int64):
            raise ValueError("neighbors/degrees must be int64 with shapes (N,K)/(N,)")
        if torch.any(degrees < 0) or torch.any(degrees > neighbors.shape[1]):
            raise ValueError("degrees must be between zero and neighbor width")
        columns = torch.arange(neighbors.shape[1], device=neighbors.device)
        active = columns.unsqueeze(0) < degrees.to(neighbors.device).unsqueeze(1)
        if torch.any(active & ((neighbors < 0) | (neighbors >= len(neighbors)))):
            raise ValueError("active neighbor indices must be within [0,N)")
        self.device = torch.device(device)
        self.neighbors, self.degrees = neighbors, degrees
        self.kernel = None
        if self.device.type == "cuda":
            from .mris_register_average_gpu import average_on_cuda
            self.kernel = average_on_cuda
            self.gpu_neighbors = neighbors.to(device=self.device, dtype=torch.int64).contiguous()
            self.device = self.gpu_neighbors.device
            self.gpu_degrees = degrees.to(device=self.device, dtype=torch.int64).contiguous()
            self.gpu_reciprocals = (1.0 / (degrees.cpu() + 1).float()).to(self.device)
        elif self.device.type != "cpu":
            raise ValueError("averaging device must be cpu or cuda")

    @torch.no_grad()
    def __call__(self, gradient: torch.Tensor, iterations: int) -> torch.Tensor:
        if gradient.dtype != torch.float32 or gradient.shape != (len(self.degrees), 3):
            raise ValueError("gradient must be float32 with shape (N, 3)")
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        if iterations == 0:
            return gradient.clone()
        if self.device.type == "cpu":
            return average_gradients_exact_cpu(gradient, self.neighbors, self.degrees, iterations)
        current = gradient.to(self.device).contiguous()
        if current.data_ptr() == gradient.data_ptr():
            current = current.clone()
        following = torch.empty_like(current)
        result = self.kernel(current, following, self.gpu_neighbors, self.gpu_degrees,
                             self.gpu_reciprocals, iterations)
        return result.to(gradient.device)
