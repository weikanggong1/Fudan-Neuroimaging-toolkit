"""纯 PyTorch 的 mri_em_register 候选评估后端。

本模块只实现候选仿射变换的 GPU 采样和批量评分。它不声称复现
FreeSurfer ``mri_em_register`` 的 GCA/EM 参数估计，也不会自动替换现有
Conda C++ 阶段。候选参数、参考图像和掩膜可以全部驻留在同一个 CUDA
设备上，避免逐候选的 CPU/GPU 往返；调用方完成同输入 LTA 和 downstream
``norm`` 回归后，才可以把它接入生产 profile。

坐标约定：``source_to_target`` 是目标体素坐标到源体素坐标的 4x4
矩阵（与 ``grid_sample`` 的采样方向一致），体素索引顺序为 ``x,y,z``，
体积张量顺序为 ``z,y,x``。``source_affine`` 与 ``target_affine`` 将体素
索引映射到同一世界坐标系；不提供时使用单位矩阵。
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TorchAffineScoreConfig:
    """批量采样参数。

    ``candidate_chunk`` 控制峰值显存；``padding_mode`` 必须和参考实现
    的边界语义匹配，默认 ``border``，避免外部候选产生未定义 NaN。
    ``use_tf32`` 只控制 matmul/grid 相关的全局开关，默认保持项目 TF32
    策略；不启用 FP16/BF16。
    """

    candidate_chunk: int = 16
    padding_mode: str = "border"
    align_corners: bool = True
    use_tf32: bool = True

    def __post_init__(self) -> None:
        if self.candidate_chunk < 1:
            raise ValueError("candidate_chunk must be >= 1")
        if self.padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("unsupported grid_sample padding_mode")


@contextmanager
def _tf32_scope(enabled: bool):
    """短暂应用本模块的 TF32 策略并恢复调用方设置。"""

    if not torch.cuda.is_available():
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _as_volume(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 3:
        value = value[None, None]
    elif value.ndim == 4:
        value = value[None]
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [D,H,W], [1,D,H,W], or [N,1,D,H,W]")
    if not value.is_floating_point():
        value = value.float()
    return value.contiguous()


def _as_affine(value: Optional[Tensor], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if value is None:
        return torch.eye(4, device=device, dtype=dtype)
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.shape != (4, 4):
        raise ValueError("voxel/world affine must have shape [4,4]")
    return value


def _target_grid(
    source_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
    source_affine: Tensor,
    target_affine: Tensor,
    source_to_target: Tensor,
    *,
    align_corners: bool,
) -> Tensor:
    """Create a ``grid_sample`` grid for a batch of target-to-source matrices."""

    # ``source_to_target`` is named for compatibility with LTA terminology,
    # but its operational direction is target voxel -> source voxel.
    if source_to_target.ndim == 2:
        source_to_target = source_to_target[None]
    if source_to_target.ndim != 3 or source_to_target.shape[-2:] != (4, 4):
        raise ValueError("source_to_target must have shape [K,4,4]")
    k = source_to_target.shape[0]
    td, th, tw = target_shape
    sd, sh, sw = source_shape
    z, y, x = torch.meshgrid(
        torch.arange(td, device=source_to_target.device, dtype=source_to_target.dtype),
        torch.arange(th, device=source_to_target.device, dtype=source_to_target.dtype),
        torch.arange(tw, device=source_to_target.device, dtype=source_to_target.dtype),
        indexing="ij",
    )
    # Grid coordinates are x,y,z while volume storage is z,y,x.
    target_vox = torch.stack((x, y, z, torch.ones_like(x)), dim=-1).reshape(-1, 4)
    # When affines are provided, map target voxel -> world -> source voxel.
    world_from_target = target_affine
    source_from_world = torch.linalg.inv(source_affine)
    base = target_vox @ world_from_target.T
    target_to_source_world = source_from_world @ base.T
    target_to_source_world = target_to_source_world.T.reshape(-1, 4)
    source_vox = torch.einsum("kij,nj->kni", source_to_target, target_to_source_world)
    source_vox = source_vox[..., :3] / source_vox[..., 3:].clamp_min(torch.finfo(source_vox.dtype).eps)
    sx, sy, sz = source_vox.unbind(-1)
    if align_corners:
        nx = 2 * sx / max(sw - 1, 1) - 1
        ny = 2 * sy / max(sh - 1, 1) - 1
        nz = 2 * sz / max(sd - 1, 1) - 1
    else:
        nx = (2 * sx + 1) / sw - 1
        ny = (2 * sy + 1) / sh - 1
        nz = (2 * sz + 1) / sd - 1
    return torch.stack((nx, ny, nz), dim=-1).reshape(k, td, th, tw, 3)


def sample_candidates(
    source: Tensor,
    source_to_target: Tensor,
    *,
    target_shape: Optional[Tuple[int, int, int]] = None,
    source_affine: Optional[Tensor] = None,
    target_affine: Optional[Tensor] = None,
    config: TorchAffineScoreConfig = TorchAffineScoreConfig(),
) -> Tensor:
    """批量采样候选仿射结果，返回 ``[K,D,H,W]`` float32 张量。

    所有候选会按 ``candidate_chunk`` 分块，峰值显存与候选数线性无关。
    ``source`` 可以是 CPU 或 CUDA 张量，矩阵与图像会统一到 source 设备；
    生产调用应在进入此函数前把数据放到目标 GPU，以免隐式传输。
    """

    source5 = _as_volume(source, name="source")
    device = source5.device
    dtype = torch.float32 if source5.dtype not in (torch.float32, torch.float64) else source5.dtype
    source5 = source5.to(dtype=dtype)
    matrices = torch.as_tensor(source_to_target, device=device, dtype=dtype)
    if matrices.ndim == 2:
        matrices = matrices[None]
    if matrices.ndim != 3 or matrices.shape[-2:] != (4, 4):
        raise ValueError("source_to_target must have shape [K,4,4]")
    if target_shape is None:
        target_shape = tuple(int(x) for x in source5.shape[-3:])
    if len(target_shape) != 3 or min(target_shape) < 1:
        raise ValueError("target_shape must be a positive (D,H,W) tuple")
    src_aff = _as_affine(source_affine, device=device, dtype=dtype)
    tgt_aff = _as_affine(target_affine, device=device, dtype=dtype)
    outputs = []
    with _tf32_scope(config.use_tf32):
        for start in range(0, matrices.shape[0], config.candidate_chunk):
            chunk = matrices[start : start + config.candidate_chunk]
            grid = _target_grid(
                tuple(int(x) for x in source5.shape[-3:]),
                target_shape,
                src_aff,
                tgt_aff,
                chunk,
                align_corners=config.align_corners,
            )
            repeated = source5.expand(chunk.shape[0], -1, -1, -1, -1)
            outputs.append(
                F.grid_sample(
                    repeated,
                    grid,
                    mode="bilinear",
                    padding_mode=config.padding_mode,
                    align_corners=config.align_corners,
                )[:, 0]
            )
    return torch.cat(outputs, dim=0)


def _masked_center(value: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if mask is None:
        mask = torch.ones_like(value, dtype=torch.bool)
    else:
        mask = mask.to(device=value.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask[None]
        if mask.ndim == 5:
            mask = mask[:, 0]
        if mask.ndim != 4:
            raise ValueError("mask must have shape [D,H,W] or [K,D,H,W]")
        if mask.shape[0] == 1 and value.shape[0] != 1:
            mask = mask.expand(value.shape[0], -1, -1, -1)
        if mask.shape != value.shape:
            raise ValueError("mask shape must match sampled candidates")
    count = mask.sum(dim=(-3, -2, -1)).clamp_min(1).to(value.dtype)
    mean = (value * mask).sum(dim=(-3, -2, -1)) / count
    return value - mean[:, None, None, None], mask


def score_candidates(
    sampled: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    metric: str = "ncc",
) -> Tensor:
    """在 GPU 上批量计算候选分数，返回 ``[K]``，值越大越优。

    ``ncc`` 是零均值归一化相关，``ssd`` 返回负均方误差。此函数只负责
    候选排序，不能替代 GCA/EM 的分类器估计。
    """

    if not isinstance(sampled, Tensor) or sampled.ndim != 4:
        raise ValueError("sampled must have shape [K,D,H,W]")
    if not sampled.is_floating_point():
        sampled = sampled.float()
    sampled = sampled.contiguous()
    target5 = _as_volume(target, name="target")
    target3 = target5[0, 0]
    if sampled.shape[-3:] != target3.shape:
        raise ValueError("sampled and target spatial shapes must match")
    centered, mask4 = _masked_center(sampled, mask)
    target4 = target3[None].expand(sampled.shape[0], -1, -1, -1)
    target_centered, _ = _masked_center(target4, mask4)
    count = mask4.sum(dim=(-3, -2, -1)).clamp_min(1).to(sampled.dtype)
    if metric == "ncc":
        numerator = (centered * target_centered * mask4).sum(dim=(-3, -2, -1))
        denom = (
            (centered.square() * mask4).sum(dim=(-3, -2, -1))
            * (target_centered.square() * mask4).sum(dim=(-3, -2, -1))
        ).sqrt().clamp_min(torch.finfo(sampled.dtype).eps)
        return numerator / denom
    if metric == "ssd":
        return -(((sampled - target3[None]) ** 2) * mask4).sum(dim=(-3, -2, -1)) / count
    raise ValueError("metric must be 'ncc' or 'ssd'")


def select_best_candidate(scores: Tensor) -> int:
    """将 GPU 分数安全地转为一个 Python 索引，只在候选搜索末端同步一次。"""

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    return int(torch.argmax(scores).item())


def evaluate_affine_candidates(
    source: Tensor,
    target: Tensor,
    source_to_target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    source_affine: Optional[Tensor] = None,
    target_affine: Optional[Tensor] = None,
    metric: str = "ncc",
    config: TorchAffineScoreConfig = TorchAffineScoreConfig(),
) -> Tuple[Tensor, Tensor, int]:
    """采样、评分并返回 ``(sampled, scores, best_index)``。

    这是 mri_em_register 的可插拔候选搜索内核。它没有隐式 ``.cpu()`` 或
    逐候选 Python 循环；只有 ``best_index`` 的最终选择会产生同步。
    """

    sampled = sample_candidates(
        source,
        source_to_target,
        target_shape=target_shape,
        source_affine=source_affine,
        target_affine=target_affine,
        config=config,
    )
    scores = score_candidates(sampled, target, mask=mask, metric=metric)
    return sampled, scores, select_best_candidate(scores)


def run_torch_candidate(
    source: Tensor,
    target: Tensor,
    source_to_target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    source_affine: Optional[Tensor] = None,
    target_affine: Optional[Tensor] = None,
    metric: str = "ncc",
    config: TorchAffineScoreConfig = TorchAffineScoreConfig(),
    report_path: Optional[str] = None,
) -> dict:
    """运行一次纯 PyTorch 候选评分并写出机器可读报告。

    该入口明确标记为 ``experimental-candidate-only``：它只替换 GCA
    候选采样/排序，未实现完整 GCA/EM、LTA 写出或 ``norm`` 下游阶段，
    因此不会被 recon-all 默认调度调用。报告包含设备、dtype、TF32、
    候选数、分块大小和包含 GPU 同步的评分墙钟时间，便于同输入对照。
    返回字典中保留 ``sampled`` 和 ``scores`` 张量，避免为了写报告把整
    个体数据复制到 CPU；报告中的标量统计才会同步到 Python。
    """

    if not isinstance(source, Tensor) or not isinstance(target, Tensor):
        raise TypeError("source and target must be torch.Tensor")
    device = source.device
    target_device = target.device
    if target_device != device:
        raise ValueError(
            "source and target must be on the same device; move them explicitly "
            "before run_torch_candidate to make data transfer visible"
        )
    matrices = torch.as_tensor(source_to_target, device=device)
    if matrices.ndim == 2:
        candidate_count = 1
    elif matrices.ndim == 3:
        candidate_count = int(matrices.shape[0])
    else:
        candidate_count = 0
    tf32_before = None
    cudnn_tf32_before = None
    if device.type == "cuda":
        tf32_before = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_tf32_before = bool(torch.backends.cudnn.allow_tf32)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    sampled, scores, best_index = evaluate_affine_candidates(
        source,
        target,
        source_to_target,
        mask=mask,
        target_shape=target_shape,
        source_affine=source_affine,
        target_affine=target_affine,
        metric=metric,
        config=config,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = {
        "schema": "fnit.recon_all.mri_em_register_torch.candidate.v1",
        "algorithm": "torch-grid-sample-candidate-score",
        "equivalence_status": "experimental-candidate-only",
        "production_default": False,
        "native_stage_replaced": "GCA candidate sampling and ranking only",
        "complete_em_lta": False,
        "device": str(device),
        "source_dtype": str(source.dtype),
        "target_dtype": str(target.dtype),
        "candidate_dtype": str(matrices.dtype),
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "candidate_shape": list(matrices.shape),
        "candidate_count": candidate_count,
        "candidate_chunk": config.candidate_chunk,
        "metric": metric,
        "padding_mode": config.padding_mode,
        "align_corners": config.align_corners,
        "tf32_requested": config.use_tf32,
        "tf32_matmul_before": tf32_before,
        "tf32_cudnn_before": cudnn_tf32_before,
        "elapsed_seconds_synchronised": elapsed,
        "best_index": best_index,
        "best_score": float(scores[best_index].detach().item()),
        "score_min": float(scores.detach().min().item()),
        "score_max": float(scores.detach().max().item()),
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"sampled": sampled, "scores": scores, "best_index": best_index, "report": report}


__all__ = [
    "TorchAffineScoreConfig",
    "sample_candidates",
    "score_candidates",
    "select_best_candidate",
    "evaluate_affine_candidates",
    "run_torch_candidate",
]
