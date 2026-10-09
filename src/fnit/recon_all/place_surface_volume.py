"""复用固定源码亮区规则，准备white/pial边界搜索的uint8强度图。"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def prepare_placement_volume(
    brain: np.ndarray,
    wm: np.ndarray,
    *,
    surface: str,
    mid_gray: float,
    restore_255: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """返回放置CBV强度图与亮区标签，同shape/dtype uint8三维数组。

    brain和wm是同一conform体素网格的三维数组，调用方负责affine一致；
    此函数转换为uint8，不做原始T1强度缩放或空间变换。surface必填white/
    pial；mid_gray为pial边界标签100的目标强度，按原整数写出规则舍入，
    white分支不使用它。restore_255默认True，仅white恢复原255体素为110。
    labels为0/100/130（非亮区/亮区边界/亮区）；返回体积保留输入网格。
    非法surface或不匹配3D shape抛ValueError。不调用外部程序。

    对应mris_place_surface内部MRIclipBrightWM/MRIfindBrightNonWM。
    源代码邻居阈值实际整数((27-1)/2)=13，不能照其错误注释写14。
    """
    if surface not in ("white", "pial"):
        raise ValueError("surface must be white or pial")
    source = np.asarray(brain, dtype=np.uint8)
    wm_mask = np.asarray(wm, dtype=np.uint8)
    if source.shape != wm_mask.shape or source.ndim != 3:
        raise ValueError("brain and wm must be matching 3D volumes")

    volume = source.copy()
    volume[(wm_mask >= 5) & (volume > 110)] = 110
    neighborhood = np.ones((3, 3, 3), dtype=np.uint8)
    wm_neighbors = ndimage.convolve((wm_mask >= 5).astype(np.uint8), neighborhood, mode="nearest")
    minimum_white_neighbors = (3 * 3 * 3 - 1) // 2
    seed = (wm_mask < 5) & (volume > 125) & (wm_neighbors < minimum_white_neighbors)
    closed = ndimage.minimum_filter(
        ndimage.maximum_filter(seed.astype(np.uint8), size=3, mode="nearest"),
        size=3, mode="nearest",
    ).astype(bool)
    expanded = closed | (ndimage.maximum_filter(closed, size=3, mode="nearest") & (volume >= 100))
    labels = np.zeros(volume.shape, dtype=np.uint8)
    labels[expanded & ~closed] = 100
    labels[closed] = 130
    for _ in range(3):
        labels[(volume == 0) & ndimage.maximum_filter(labels == 130, size=3, mode="nearest")] = 130

    if surface == "white":
        volume[(labels == 100) | (labels == 130)] = 0
        if restore_255:
            volume[source == 255] = 110
        return volume, labels

    volume[labels == 100] = np.uint8(np.floor(mid_gray + 0.5))
    volume[labels == 130] = 255
    return volume, labels
