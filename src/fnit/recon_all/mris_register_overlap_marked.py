"""完整球面径向翻折清理的显式Torch候选：只计算marked行邻域位移。

保留固定源码负面选择、一步扩张、dt、接受/停止顺序及全体径向投影。
不是几何近似；原dense实现继续承担默认和同输入参考作用。
整数一环复用已验证的inflation CSR/Numba构造，缓存与有序面绑定。
坐标surface RAS/mm；API输入/输出(N,3) FP32，面(F,3)整数。
"""
from __future__ import annotations

import numpy as np
import torch

from .mris_register_kernels import project_sphere


@torch.no_grad()
def negative_sphere_faces(positions: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """返回(F,)bool，严格保留原FP32面积小于0的判定。

输入(N,3)FP32坐标与(F,3)同设备整数面，无读写；非法索引抛异常。
只省去判定不需要的sqrt/单位法向。squared>0保护FP32平方下溢：
原length为0时即使orientation负，signed area也是-0，不能算负面。
NaN、零面积及inf沿用原比较行为，不加epsilon或几何clamp。
此函数属于mris_sphere/mris_register末段内部步骤，无独立官方CLI。
"""
    if positions.dtype != torch.float32:
        raise ValueError("negative_sphere_faces requires source float32 coordinates")
    corners = positions[faces]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    cross = torch.stack((first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1],
                         first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2],
                         first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]), dim=1)
    squared = (cross[:, 0] * cross[:, 0] + cross[:, 1] * cross[:, 1]
               + cross[:, 2] * cross[:, 2])
    centroid = corners.sum(dim=1)
    orientation_negative = (centroid * cross).sum(dim=1) < 0.0
    return orientation_negative & (squared > 0)


class OverlapTopology:
    """缓存同版本有序面的完整一环，法向/坐标/标记不跨迭代缓存。

triangles为(F,3)CPU/CUDA整数张量，nvertices为总顶点数；构建一次
CPU整数CSR/Numba一环，再返回原设备int64邻居(N,K)、度数(N)、
有效掩膜(N,K)。邻居无截断；面顺序、重复和孤立行按原实现保留。
输入非法由FaceNormalTopology或Tensor索引检查抛异常，无文件读写。
"""
    def __init__(self, *, triangles: torch.Tensor, nvertices: int):
        from .inflate_topology import _one_ring
        from .place_surface_normals import FaceNormalTopology
        self.faces = triangles.detach().to(dtype=torch.int64).clone()
        self.nvertices = nvertices
        cpu = FaceNormalTopology(np.asarray(self.faces.cpu().numpy(), np.int32), nvertices)
        one, degree = _one_ring(cpu.faces, cpu.face_ids, cpu.corners, cpu.offsets)
        self.neighbors = torch.from_numpy(one.astype(np.int64)).to(self.faces.device)
        self.degrees = torch.from_numpy(degree.astype(np.int64)).to(self.faces.device)
        self.active = (torch.arange(self.neighbors.shape[1], device=self.faces.device)[None, :]
                       < self.degrees[:, None])


@torch.no_grad()
def remove_overlap_sphere_marked(positions: torch.Tensor, faces: torch.Tensor, *,
                                start_iteration: int = 0, max_iterations: int = 1000,
                                topology: OverlapTopology | None = None
                                ) -> tuple[torch.Tensor, list[int]]:
    """完整清理，返回同设备FP32新坐标及每步更新前负面数。

positions=(N,3)，faces=(F,3)同设备；与原实现一致转换FP32并clone，
不原地改输入。start_iteration默认0，max_iterations默认1000，原
停止比较顺序完整保留，实际步数还受min_iteration规则影响。topology
默认None每次构造；可显式复用同有序面/设备/顶点数的整数缓存，
不满足身份条件直接报错。所有顶点仍执行每轮原径向投影，包括未标
顶点；只省略被torch.where丢弃的未标顶点邻居位移，不改变dt或扩张。
没有独立CLI；对应mris_sphere/mris_register末段内部清理。
    """
    xyz = positions.float().clone()
    if topology is not None and (
            topology.nvertices != len(xyz) or topology.faces.device != xyz.device
            or topology.faces.shape != faces.shape or not torch.equal(topology.faces, faces)):
        raise ValueError("overlap topology belongs to a different ordered surface/device")
    negative = negative_sphere_faces(xyz, faces)
    count = int(negative.sum())
    history: list[int] = []
    if count == 0:
        return xyz, history
    if topology is None:
        topology = OverlapTopology(triangles=faces, nvertices=len(xyz))
    neighbors, degrees, active = topology.neighbors, topology.degrees, topology.active
    min_negative, min_iteration = count, 0
    iteration, last_expand, same, max_neighbors = start_iteration, 0, 0, 0
    dt = 0.99
    while count > 0:
        history.append(count)
        marked = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
        marked[faces[negative].reshape(-1)] = True
        if max_neighbors:
            marked |= (marked[neighbors] & active).any(dim=1)
        marked_indices = torch.nonzero(marked, as_tuple=False).flatten()
        xyz_double = xyz.double()
        centers = xyz_double[marked_indices]
        selected_neighbors = neighbors[marked_indices]
        selected_active = active[marked_indices]
        displacement = torch.zeros_like(centers)
        for index in range(neighbors.shape[1]):
            difference = xyz_double[selected_neighbors[:, index]] - centers
            displacement = torch.where(selected_active[:, index, None],
                                       displacement + difference, displacement)
        displacement = (displacement / degrees[marked_indices, None].double()).float()
        moved = (centers + dt * displacement.double()).float()
        pre_projection = xyz.clone()
        pre_projection[marked_indices] = moved
        xyz = project_sphere(pre_projection)

        old_count = count
        iteration += 1
        negative = negative_sphere_faces(xyz, faces)
        count = int(negative.sum())
        if count < min_negative:
            min_negative, min_iteration = count, iteration
        elif (iteration - min_iteration) % 10 == 0 and iteration > min_iteration:
            if dt > 0.01:
                dt *= 0.95
        elif iteration > min_iteration + 50 and iteration > last_expand + 25:
            if max_neighbors < 1:
                dt, max_neighbors = 0.99, 1
            last_expand, same = iteration, 0
        if iteration > min_iteration + 1000:
            break
        if old_count == count:
            if same > 25 and max_neighbors < 1:
                max_neighbors, dt, last_expand, same = 1, 0.99, iteration, 0
            else:
                same += 1
        else:
            same = 0
        if iteration - start_iteration > max_iterations:
            break
    return xyz, history
