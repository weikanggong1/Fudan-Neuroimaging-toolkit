"""温和归一化的白质控制点，沿用 MRInormGentlyFindControlPoints 规则。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from .normalize_3d_controls import _remove_outliers_ordered


def gentle_controls(image: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """按整数强度窗口选择白质控制点，并执行有序原地离群清理。

    ``image`` 是三维 (x, y, z) 强度张量，空间和强度单位沿用输入，不做
    RAS 变换。输入在 CPU 转为 float32 后截断为整数强度，以 7/5 体素
    窗口判断均匀区。离群清理复用三维控制点的串行 Numba 内核，保留
    z/y/x 顺序和裁剪边界；不改变 TF32 设置。返回同 shape、同设备的
    uint8 控制图（0/1），以及三个锚点/清理计数构成的 dict。没有可调
    算法参数；非三维输入抛出 ValueError。属于 ``mri_normalize -g 1``
    的内部步骤，没有独立官方 CLI；后续 Voronoi 和偏置场另行计算。
    """
    if image.ndim != 3:
        raise ValueError("expected a 3D volume")
    # Native val0/val are int, so float input intensities truncate on read.
    src = np.trunc(np.asarray(image.detach().cpu(), dtype=np.float32))
    control = np.zeros(src.shape, dtype=bool)
    counts = []
    # Native low_thresh/hi_thresh are BUFTYPE, truncating fractional bounds.
    for width, low, high in ((7, int(110 - 1.5 * 7.5), int(110 + 1.5 * 25)),
                             (5, int(110 - 7.5), int(110 + 25))):
        minimum = ndimage.minimum_filter(src, size=width, mode="nearest")
        maximum = ndimage.maximum_filter(src, size=width, mode="nearest")
        control |= (minimum >= low) & (maximum <= high)
        counts.append(int(control.sum()))
    _remove_outliers_ordered(control)
    return torch.as_tensor(control.astype(np.uint8), device=image.device), {
        "first_7x7x7": counts[0], "after_5x5x5": counts[1],
        "after_outlier_removal": int(control.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = nib.load(str(args.input))
    values = torch.as_tensor(np.asarray(source.dataobj).astype(np.float32, copy=True), device=args.device)
    control, details = gentle_controls(values)
    # The source is a float MGH; controls need the native uint8 MGH type.
    nib.save(nib.MGHImage(control.cpu().numpy(), source.affine), str(args.output))
    print(json.dumps(details))


if __name__ == "__main__":
    main()
