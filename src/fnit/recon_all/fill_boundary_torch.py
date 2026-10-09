"""CC外部fast marching的完整边界初始化；保持原生首次访问次序。"""
from __future__ import annotations

import torch


@torch.no_grad()
def initialize_cc_boundary_torch(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """以整数first-visit键复现z/y/x扫描的边界状态和有序alive列表。

    target为非空3D bool张量，x/y/z局部裁切网格，CPU或明确CUDA。
    返回同shape/device的float32 distance（目标0、边界0.5、其余100）、
    uint8 state（目标3 forbidden、边界2 alive、其余0 far），以及Nx3
    int64 alive局部体素坐标。顺序严格为旧扫描第一次add_alive的次序；
    不使用堆、不更新eikonal、不近似距离、不改变target或精度策略。
    维度/dtype/空图像或整数键范围错误抛ValueError；CUDA失败原样传播。
    """
    if target.ndim != 3 or target.dtype != torch.bool or not target.numel():
        raise ValueError("expected a nonempty 3D bool target")
    if target.numel() > torch.iinfo(torch.int64).max // 4:
        raise ValueError("voxel count exceeds exact first-visit key range")
    sx, sy, sz = target.shape
    # z/y/x scan gives r=(z*sy+y)*sx+x. Slots 0,1,2,3 are +x,+y,+z,self;
    # a key is unique for each possible source call, so ties cannot arise.
    x = torch.arange(sx, dtype=torch.int64, device=target.device)[:, None, None]
    y = torch.arange(sy, dtype=torch.int64, device=target.device)[None, :, None]
    z = torch.arange(sz, dtype=torch.int64, device=target.device)[None, None, :]
    rank = ((z * sy + y) * sx + x) * 4
    absent = torch.iinfo(torch.int64).max
    first = torch.full(target.shape, absent, dtype=torch.int64, device=target.device)
    # An outside point may first be reached as a positive neighbor of a target
    # at the previous x/y/z coordinate. Crop edges have no extra neighbors.
    first[1:] = torch.where(~target[1:] & target[:-1], rank[1:] - 4, first[1:])
    first[:, 1:] = torch.minimum(first[:, 1:], torch.where(
        ~target[:, 1:] & target[:, :-1], rank[:, 1:] - 4 * sx + 1, absent))
    first[:, :, 1:] = torch.minimum(first[:, :, 1:], torch.where(
        ~target[:, :, 1:] & target[:, :, :-1], rank[:, :, 1:] - 4 * sx * sy + 2, absent))
    positive = torch.zeros_like(target)
    positive[:-1] |= target[1:]
    positive[:, :-1] |= target[:, 1:]
    positive[:, :, :-1] |= target[:, :, 1:]
    first = torch.minimum(first, torch.where(~target & positive, rank + 3, absent))
    seed = first != absent
    distance = torch.full(target.shape, 100., dtype=torch.float32, device=target.device)
    distance.masked_fill_(target, 0.).masked_fill_(seed, .5)
    state = torch.zeros_like(target, dtype=torch.uint8)
    state.masked_fill_(target, 3).masked_fill_(seed, 2)
    points = seed.nonzero()
    order = first[points[:, 0], points[:, 1], points[:, 2]].argsort()
    alive = points[order]
    return distance, state, alive
