"""固定 mris_inflate 配方的实验 Torch 积分；复用有序法向和平均器。

坐标始终为 float32，标量全局统计和原规则动量的指定项为 float64。
保留完整二环、档次、停止和 sulc 更新；不改变 TF32 或启用半精度。
生产调度尚未接入，必须先完成同输入 surface/sulc 回归。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .inflate_topology import inflate_neighbor_tables
from .mris_register_average_numba import RegistrationGradientAverager
from .place_surface_normals import TorchFaceNormalTopology


class TorchInflationContext:
    """缓存同一有序面整数邻域；每轮坐标、法向和距离重新计算。

    faces 为整数(F,3)，nvertices为N，device默认cuda:0。缓存一/二环
    索引与度数、有效掩膜、有序法向/平均上下文；不跨拓扑复用。无文件
    读写、不保存参考结果；CUDA不可用、非法索引或孤立顶点会抛异常。
    """
    def __init__(self, *, faces: np.ndarray, nvertices: int, device: str = "cuda:0"):
        triangles = np.asarray(faces)
        self.normals = TorchFaceNormalTopology(triangles=triangles, nvertices=nvertices,
                                               device=device)
        self.device = self.normals.device
        one_indices, one_degree, two_indices, two_degree = inflate_neighbor_tables(
            normal_topology=self.normals)
        if np.any(one_degree == 0) or np.any(two_degree == 0):
            raise ValueError("standard inflation requires a nonisolated mesh")
        self.average = RegistrationGradientAverager(
            neighbors=torch.from_numpy(one_indices.astype(np.int64)),
            degrees=torch.from_numpy(one_degree.astype(np.int64)), device=str(self.device))
        self.faces = torch.as_tensor(triangles.astype(np.int64), device=self.device)
        self.one = torch.as_tensor(one_indices.astype(np.int64), device=self.device)
        self.two = torch.as_tensor(two_indices.astype(np.int64), device=self.device)
        self.one_degree = torch.as_tensor(one_degree, device=self.device)
        self.two_degree = torch.as_tensor(two_degree, device=self.device)
        self.one_valid = (torch.arange(self.one.shape[1], device=self.device)[None, :]
                          < self.one_degree[:, None])
        self.two_valid = (torch.arange(self.two.shape[1], device=self.device)[None, :]
                          < self.two_degree[:, None])
        self.inverse_average_degree = torch.tensor(
            np.float32(1 / np.float32(np.sum(two_degree, dtype=np.int64) / nvertices)),
            dtype=torch.float32, device=self.device)
        self.nvertices = nvertices

    @staticmethod
    def _length(vector):
        return torch.sqrt((vector[..., 0] * vector[..., 0]
                           + vector[..., 1] * vector[..., 1])
                          + vector[..., 2] * vector[..., 2])

    def face_area(self, xyz):
        a = xyz[self.faces[:, 1]] - xyz[self.faces[:, 0]]
        b = xyz[self.faces[:, 2]] - xyz[self.faces[:, 0]]
        cross = torch.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                             a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                             a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), dim=1)
        return (self._length(cross) * .5).sum(dtype=torch.float64).to(torch.float32)

    def distances(self, xyz):
        length = self._length(xyz[:, None, :] - xyz[self.two])
        return torch.where(self.two_valid, length, torch.zeros_like(length))

    def distance_gradient(self, xyz, normals, original_dist, current_dist,
                          original_area, current_area, weight):
        gradient = torch.zeros_like(xyz)
        if weight == 0:
            return gradient
        scale = torch.sqrt(original_area / current_area)
        for column in range(self.two.shape[1]):
            direction = xyz[self.two[:, column]] - xyz
            length = self._length(direction)
            reciprocal = torch.ones_like(length) / torch.where(
                length > 0, length, torch.ones_like(length))
            delta = current_dist[:, column] - original_dist[:, column] / scale
            term = (direction * reciprocal[:, None]) * delta[:, None]
            gradient = gradient + torch.where(self.two_valid[:, column, None],
                                               term, torch.zeros_like(term))
        gradient = gradient * self.inverse_average_degree
        normal_component = ((gradient[:, 0] * normals[:, 0]
                             + gradient[:, 1] * normals[:, 1])
                            + gradient[:, 2] * normals[:, 2])
        gradient = gradient - normal_component[:, None] * normals
        return gradient * torch.tensor(weight, dtype=torch.float32, device=self.device)

    def add_spring(self, gradient, xyz, normals, original_area, current_area):
        spring = torch.zeros_like(xyz)
        for column in range(self.one.shape[1]):
            delta = xyz[self.one[:, column]] - xyz
            spring = spring + torch.where(self.one_valid[:, column, None],
                                           delta, torch.zeros_like(delta))
        scale = torch.sqrt(original_area / current_area)
        spring = spring * (scale / self.one_degree.to(torch.float32))[:, None]
        dot = ((spring[:, 0] * normals[:, 0] + spring[:, 1] * normals[:, 1])
               + spring[:, 2] * normals[:, 2])
        dot_average = (dot.sum(dtype=torch.float64) / self.nvertices).to(torch.float32)
        return (gradient + spring) - dot_average * normals

    def rms_height(self, xyz, normals):
        # Source first subtracts FP32 coordinates, then promotes differences
        # to double for length and tangent-plane statistics.
        delta = (xyz[self.two] - xyz[:, None, :]).to(torch.float64)
        squared = (delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1])
        squared = squared + delta[..., 2] * delta[..., 2]
        n = normals.to(torch.float64)[:, None, :]
        dot = ((delta[..., 0] * n[..., 0] + delta[..., 1] * n[..., 1])
               + delta[..., 2] * n[..., 2])
        valid = self.two_valid & (squared > 1e-12)
        term = dot * dot / torch.where(valid, squared, torch.ones_like(squared))
        total = torch.where(valid, term, torch.zeros_like(term)).sum()
        return torch.sqrt(total / valid.sum())

    @torch.no_grad()
    def finalize(self, *, coordinates: torch.Tensor, original_vertices: torch.Tensor,
                 sulc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """复用原规则包围盒居中/面积缩放，并用double均值校正sulc。

        三个输入须是本上下文设备的FP32；坐标(N,3)mm、sulc(N,)mm。
        返回同设备同形状新张量；sulc不改为欧氏径向距离。
        """
        for array in (coordinates, original_vertices):
            if array.shape != (self.nvertices, 3) or array.dtype != torch.float32 or array.device != self.device:
                raise ValueError("final coordinates must match cached device/shape/dtype")
        if sulc.shape != (self.nvertices,) or sulc.dtype != torch.float32 or sulc.device != self.device:
            raise ValueError("sulc must match cached vertex order/device")
        current_area = self.face_area(coordinates)
        original_area = self.face_area(original_vertices)
        center = ((coordinates.amin(dim=0).double() + coordinates.amax(dim=0).double())
                  .float() * .5)
        centered = coordinates - center
        scale = torch.sqrt(original_area / current_area)
        if float(scale) != 1:
            center = ((centered.amin(dim=0).double() + centered.amax(dim=0).double())
                      .float() * .5)
            centered = (centered - center) * scale + center
        centered_sulc = (sulc.double() - sulc.double().sum() / self.nvertices).float()
        return centered, centered_sulc

    @torch.no_grad()
    def integrate(self, *, vertices: torch.Tensor, niterations: int = 10,
                  rms_target: float = .015, profile: bool = False,
                  callback=None) -> dict:
        """FP32(N,3)surface RAS/mm→完整更新/sulc与逐轮状态，不改输入。

        niterations默认每档10、rms_target默认0.015、profile默认False；
        callback默认None，可只读接收(step,coordinates)。返回字典包括
        同设备coordinates/sulc、步数、RMS、停止状态、timings_seconds。
        profile逐段同步显式设备；停止检查本身每步需读取单个标量。
        """
        if (vertices.shape != (self.nvertices, 3) or vertices.dtype != torch.float32
                or vertices.device != self.device):
            raise ValueError("vertices must be float32(N,3) on cached device")
        if not isinstance(niterations, int) or isinstance(niterations, bool) or niterations < 1:
            raise ValueError("niterations must be a positive integer")
        if not np.isfinite(rms_target) or rms_target < 0:
            raise ValueError("rms_target must be finite and nonnegative")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("coordinates must be finite")
        timings = {}
        def synchronize():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        def timed(name, function, *args):
            if not profile:
                return function(*args)
            synchronize(); begin = time.perf_counter()
            result = function(*args)
            synchronize(); timings[name] = timings.get(name, 0.0) + time.perf_counter() - begin
            return result
        synchronize(); begin = time.perf_counter()
        xyz = vertices.clone()
        previous, sulc = torch.zeros_like(xyz), torch.zeros_like(xyz[:, 0])
        area = original_area = timed("face_area", self.face_area, xyz)
        normals = timed("normals", self.normals.evaluate_tensor, xyz)
        current_dist = original_dist = timed("distances", self.distances, xyz)
        momentum = torch.tensor(np.float32(.9), device=self.device, dtype=torch.float32)
        step, reached, rms = 0, False, None
        for averages in (16, 8, 4, 2, 1, 0):
            weight = np.float32(np.float32(.1) * np.sqrt(averages))
            for _ in range(niterations):
                gradient = timed("distance_gradient", self.distance_gradient, xyz, normals,
                                 original_dist, current_dist, original_area, area, weight)
                gradient = timed("ordered_averaging", self.average, gradient, averages)
                gradient = timed("normalized_spring", self.add_spring, gradient, xyz,
                                 normals, original_area, area)
                previous = (gradient.to(torch.float64) * momentum.to(torch.float64)
                            + (momentum * previous).to(torch.float64)).to(torch.float32)
                xyz = xyz + previous
                projection = ((previous[:, 0] * normals[:, 0]
                               + previous[:, 1] * normals[:, 1])
                              + previous[:, 2] * normals[:, 2])
                sulc = sulc + projection
                normals = timed("normals", self.normals.evaluate_tensor, xyz)
                area = timed("face_area", self.face_area, xyz)
                current_dist = timed("distances", self.distances, xyz)
                step += 1
                if callback is not None:
                    callback(step, xyz)
                rms = float(timed("rms_tangent_height", self.rms_height, xyz, normals))
                if not np.isfinite(rms):
                    raise FloatingPointError("inflation RMS became nonfinite")
                if rms < rms_target:
                    reached = True
                    break
            if reached:
                break
        synchronize()
        return {"coordinates": xyz, "sulc": sulc, "steps": step,
                "final_rms": rms, "stopped_by_rms": reached,
                "timings_seconds": timings,
                "integration_wall_seconds": time.perf_counter() - begin}
