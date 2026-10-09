"""复用WM分割的局部直方图规则，以分块PyTorch张量执行独立候选。

只迁移MRIhistoSegment内部子阶段。图像只读，每个候选直方图相互独立；
不改变有序strand/diagonal扫描，不替换完整mri_segment默认后端。
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .mri_segment import _histogram_kernel


def _check(image: torch.Tensor, labels: torch.Tensor, window: int, batch_size: int) -> None:
    if (image.ndim != 3 or image.shape != labels.shape or image.device != labels.device
            or image.dtype != torch.uint8 or labels.dtype != torch.uint8
            or not image.numel()):
        raise ValueError("expected nonempty same-device 3D uint8 image and labels")
    if not isinstance(window, int) or window < 1 or not window % 2:
        raise ValueError("window must be a positive odd voxel count")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")


def _batch_histograms(image: torch.Tensor, points: torch.Tensor, offsets: torch.Tensor):
    """分块3D候选生成精确int32的256-bin计数，图像边界裁切而非重复采样。

    image为3D uint8，points为Bx3整数体素，offsets为Px3整数窗口偏移，
    三者同device。返回计数Bx256、每块最小/最大灰度B，不修改输入。
    内部函数由公开入口验证参数；没有空间插值或标签反馈。
    """
    coordinate = [points[:, axis, None] + offsets[None, :, axis] for axis in range(3)]
    valid = torch.ones(coordinate[0].shape, dtype=torch.bool, device=image.device)
    for axis, bound in enumerate(image.shape):
        valid &= (coordinate[axis] >= 0) & (coordinate[axis] < bound)
        coordinate[axis].clamp_(0, bound - 1)
    linear = coordinate[0] * (image.shape[1] * image.shape[2])
    linear.add_(coordinate[1] * image.shape[2]).add_(coordinate[2])
    samples = image.reshape(-1)[linear.long()]
    counts = torch.zeros((len(points), 256), dtype=torch.int32, device=image.device)
    counts.scatter_add_(1, samples.long(), valid.to(torch.int32))
    low = torch.where(valid, samples, 255).amin(dim=1).long()
    high = torch.where(valid, samples, 0).amax(dim=1).long()
    return counts, low, high


def _smooth_source_order(counts, low, high, kernel):
    """按固定源码25个核位置依次float32累加；不使用TF32卷积或半精度。"""
    bins = torch.arange(256, device=counts.device)[None, :]
    in_range = (bins >= low[:, None]) & (bins <= high[:, None])
    source = counts.float()
    padded = F.pad(source, (12, 12))
    active = F.pad(in_range.float(), (12, 12))
    total, norm = torch.zeros_like(source), torch.zeros_like(source)
    # The fixed sigma=3 kernel has 25 entries. Eager mul then add keeps the
    # source float32 accumulation order and avoids a GEMM/conv precision policy.
    for index, weight in enumerate(kernel.unbind()):
        total.add_(padded[:, index:index + 256] * weight)
        norm.add_(active[:, index:index + 256] * weight)
    smooth = torch.where(in_range, total / norm.clamp_min(torch.finfo(torch.float32).tiny), 0)
    return smooth, in_range


def _classify_histograms(counts, low, high, center, kernel, *, wm_low, wm_hi, gray_hi):
    """保留last-peak、bimodal回溯、整数valley及±2强度不决定规则。"""
    smooth, in_range = _smooth_source_order(counts, low, high, kernel)
    bins = torch.arange(256, device=counts.device)[None, :]
    maximum = smooth.amax(dim=1)
    neighbor_max = F.max_pool1d(smooth[:, None], 11, stride=1, padding=5)[:, 0]
    padded = F.pad(smooth, (5, 5))
    total = torch.zeros_like(smooth)
    for index in range(11):
        total.add_(padded[:, index:index + 256])
    mean = total / 11
    peak = in_range & (maximum[:, None] > 0) & (neighbor_max <= smooth)
    # MIN_STD is a C double literal; preserve that comparison after the
    # source float32 subtraction rather than rounding the literal to float32.
    peak &= (mean >= .15 * maximum[:, None]) & ((smooth - mean).double() >= 1.9)
    white = torch.where(peak & (bins >= wm_low - 1) & (bins <= wm_hi - 6),
                        bins, -1).amax(dim=1)
    gray_prefix = torch.cummax(torch.where(peak & (bins >= 71), bins, -1), dim=1).values
    # P(k) is the last eligible gray peak below k-9. Binary lifting follows
    # exactly the monotonically decreasing source loop with five device
    # jumps, without 29 rounds or host synchronization for each candidate.
    parent = gray_prefix.gather(1, (bins - 9).clamp(0, 255).expand_as(gray_prefix))
    parent = torch.where(bins - 9 >= 71, parent, -1)
    jumps = [parent]
    for _ in range(4):
        previous = jumps[-1]
        next_parent = previous.gather(1, previous.clamp(0, 255))
        jumps.append(torch.where(previous >= 0, next_parent, -1))
    for jump in reversed(jumps):
        destination = jump.gather(1, white.clamp(0, 255)[:, None])[:, 0]
        white = torch.where((white >= 0) & (destination > gray_hi), destination, white)
    gray = parent.gather(1, white.clamp(0, 255)[:, None])[:, 0]
    gray = torch.where(white >= 0, gray, -1)
    integer = smooth.to(torch.int32)
    outside = torch.iinfo(torch.int32).max
    interior = torch.where(in_range, integer, outside)
    previous = F.pad(interior[:, :-1], (1, 0), value=outside)
    following = F.pad(interior[:, 1:], (0, 1), value=outside)
    valley_ok = in_range & (integer <= previous) & (integer <= following)
    valley_ok &= (bins >= gray[:, None] + 1) & (bins <= white[:, None] - 1)
    valley = torch.where(valley_ok, bins, 256).amin(dim=1)
    decisive = ((white >= 0) & (gray >= 0) & (integer.amax(dim=1) > 0)
                & (valley < gray_hi) & ((center.long() - valley).abs() > 2))
    return torch.where(decisive, torch.where(center >= valley, 255, 1), 128).to(torch.uint8)


@torch.no_grad()
def histogram_segmentation_torch(image: torch.Tensor, labels: torch.Tensor, *,
                                  wm_low: float, wm_hi: float, gray_hi: float,
                                  window: int = 13, batch_size: int = 2048) -> torch.Tensor:
    """以PyTorch分批解析全部AMBIGUOUS=128体素的局部直方图。

    image与labels须同3D网格/device的uint8张量；坐标为x/y/z体素。
    labels含NOT_WHITE=1、AMBIGUOUS=128、WHITE=255。wm_low/wm_hi/
    gray_hi为灰度阈值，按原生int参数截断；window默认13且为正奇数
    体素数；batch_size默认2048，只控制临时内存，不截断候选。
    输出新uint8标签张量，shape/device不变，不修改输入。不读文件或
    调用原生程序。CPU用于回归诊断；CUDA失败传播，不静默回退。
    阈值须在0..255且wm_low<=wm_hi。输入网格/dtype/window/batch或
    阈值错误抛ValueError。此函数不是完整WM分割。
    """
    _check(image, labels, window, batch_size)
    if (not all(math.isfinite(float(value)) for value in (wm_low, wm_hi, gray_hi))
            or not 0 <= wm_low <= wm_hi <= 255 or not 0 <= gray_hi <= 255):
        raise ValueError("expected finite 0..255 thresholds and wm_low <= wm_hi")
    wm_low, wm_hi, gray_hi = int(wm_low), int(wm_hi), int(gray_hi)
    result = labels.clone()
    points = (labels == 128).nonzero()
    index_dtype = torch.int32 if image.numel() <= torch.iinfo(torch.int32).max else torch.int64
    point_index = points.to(index_dtype)
    half = window // 2
    axis = torch.arange(-half, half + 1, dtype=index_dtype, device=image.device)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    kernel = torch.as_tensor(_histogram_kernel(), device=image.device)
    for start in range(0, len(points), batch_size):
        block = point_index[start:start + batch_size]
        counts, low, high = _batch_histograms(image, block, offsets)
        center = image[block[:, 0].long(), block[:, 1].long(), block[:, 2].long()]
        classified = _classify_histograms(counts, low, high, center, kernel,
                                          wm_low=wm_low, wm_hi=wm_hi, gray_hi=gray_hi)
        result[block[:, 0].long(), block[:, 1].long(), block[:, 2].long()] = classified
    return result
