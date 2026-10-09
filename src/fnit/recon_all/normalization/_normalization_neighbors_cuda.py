"""控制点邻域的融合 Triton 内核；不改变跨迭代控制图反馈。"""
from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _neighbor_sum(source, control, count_out, total_out,
                  HX: tl.constexpr, HY: tl.constexpr, HZ: tl.constexpr,
                  CX: tl.constexpr, CY: tl.constexpr, CZ: tl.constexpr,
                  OX: tl.constexpr, OY: tl.constexpr, OZ: tl.constexpr,
                  SIX: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < CX * CY * CZ
    x = index // (CY * CZ) + OX
    y = (index // CZ) % CY + OY
    z = index % CZ + OZ
    count = tl.full((BLOCK,), 0, tl.int32)
    total = tl.full((BLOCK,), 0., tl.float64)
    for dx in tl.static_range(-1, 2):
        xx = tl.minimum(tl.maximum(x + dx, 0), HX - 1)
        for dy in tl.static_range(-1, 2):
            yy = tl.minimum(tl.maximum(y + dy, 0), HY - 1)
            for dz in tl.static_range(-1, 2):
                if not SIX or dx * dx + dy * dy + dz * dz == 1:
                    zz = tl.minimum(tl.maximum(z + dz, 0), HZ - 1)
                    neighbor = (xx * HY + yy) * HZ + zz
                    selected = tl.load(control + neighbor, valid, other=0) != 0
                    value = tl.load(source + neighbor, valid, other=0).to(tl.float64)
                    count += selected.to(tl.int32)
                    total += tl.where(selected, value, 0.)
    tl.store(count_out + index, count.to(tl.int16), valid)
    tl.store(total_out + index, total.to(tl.float32), valid)


def neighbor_sum(*, source, control, count, total, shape, start, six):
    """写满既有同设备缓冲；float64 累加→float32 与 SciPy 的存储约定一致。"""
    # Triton 使用当前 CUDA stream；设备参数不能只停留在张量分配上。
    # 暂时进入目标设备并恢复调用方当前设备，兼容已初始化的 Python API。
    with torch.cuda.device(source.device):
        _neighbor_sum[(triton.cdiv(shape[0] * shape[1] * shape[2], 256),)](
            source, control, count, total, *source.shape, *shape, *start, six,
            BLOCK=256, num_warps=4, enable_fp_fusion=False)
