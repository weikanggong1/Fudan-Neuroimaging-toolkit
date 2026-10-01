"""三维白质控制点：复用 MRInormFindControlPoints 的阈值和有序更新规则。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit
from scipy import ndimage

from .normalize_tissue_peaks import tissue_peaks


def _homogeneous(image: np.ndarray, width: int, low: float, high: float) -> np.ndarray:
    return (ndimage.minimum_filter(image, size=width, mode="nearest") >= low) & (
        ndimage.maximum_filter(image, size=width, mode="nearest") <= high)


def _candidate_region(eligible: np.ndarray) -> tuple[slice, slice, slice]:
    bounds = []
    for axis in range(3):
        populated = np.any(eligible, axis=tuple(i for i in range(3) if i != axis))
        locations = np.flatnonzero(populated)
        bounds.append(slice(int(locations[0]), int(locations[-1]) + 1))
    return tuple(bounds)


def _neighbor_sum(image: np.ndarray, control: np.ndarray, kernel: np.ndarray,
                  region: tuple[slice, slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    # Only eligible voxels can be added; one halo voxel preserves their 3³ neighborhood.
    halo = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1))
                 for s, n in zip(region, image.shape))
    core = tuple(slice(s.start - h.start, s.stop - h.start) for s, h in zip(region, halo))
    count = ndimage.convolve(control[halo].astype(np.int16), kernel, mode="nearest")[core]
    total = ndimage.convolve((image[halo] * control[halo]).astype(np.float32),
                             kernel, mode="nearest")[core]
    return count, total


@njit(cache=True)
def _remove_outliers_ordered(control: np.ndarray) -> int:
    """原地删除少于两个邻居的控制点，返回删除数量。

    ``control`` 是 (x, y, z) 体素网格的三维 bool 数组；邻域半径为一个
    体素，图像边界处裁剪，不重复边界值。按 z/y/x 顺序访问并立即删除，
    后面的体素读取更新后的邻域。这是 ``mri_normalize`` 的
    ``mriRemoveOutliers`` 内部步骤，没有独立官方命令。仅使用 CPU 整数
    计数；不并行、不启用 fastmath，不改变图像空间或强度精度。
    """
    size_x, size_y, size_z = control.shape
    removed = 0
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                if not control[x, y, z]:
                    continue
                count = 0
                for neighbor_x in range(max(0, x - 1), min(size_x, x + 2)):
                    for neighbor_y in range(max(0, y - 1), min(size_y, y + 2)):
                        for neighbor_z in range(max(0, z - 1), min(size_z, z + 2)):
                            count += int(control[neighbor_x, neighbor_y, neighbor_z])
                if count - 1 < 2:
                    control[x, y, z] = False
                    removed += 1
    return removed


def controls_3d(source: np.ndarray, wm_peak: float | None = None,
                gm_peak: float | None = None) -> tuple[np.ndarray, dict]:
    """从归一化强度图选择三维白质控制点，不调用外部软件。

    输入 ``source`` 为三维 (x, y, z) 强度数组，使用输入体素网格，无 RAS
    变换；内部强度为 float32，邻域窗口沿用上游整数转换。``wm_peak``、
    ``gm_peak`` 是白质、灰质强度峰（归一化强度单位），默认 None；任一
    未给出时重新估计两者。返回同 shape 的 uint8 控制图（0/1）和 dict，
    其中各键记录锚点、扩展及离群清理数量，自动估计时另含组织峰报告。
    非三维输入抛出 ValueError；没有可选组织区域时保留现有失败行为。
    对应 ``mri_normalize`` 的内部三维控制点阶段，参数不改变输出空间。
    """
    if source.ndim != 3:
        raise ValueError("expected a 3D float image")
    raw = np.asarray(source, dtype=np.float32)
    image = raw.astype(np.int16)  # InWindow assigns MRIgetVoxVal to int.
    control = np.zeros(image.shape, dtype=bool)
    details = {}

    # First anchors estimate the tissue peaks. Reproduction of that estimate
    # is intentionally separate; the source clears these anchors afterwards.
    control |= _homogeneous(image, 7, 87, 147)
    details["initial_7"] = int(control.sum())
    control |= _homogeneous(image, 5, 95, 135)
    details["initial_5"] = int(control.sum())

    if wm_peak is None or gm_peak is None:
        wm_peak, gm_peak, peaks = tissue_peaks(raw, control)
        details["tissue_peaks"] = peaks

    control[:] = False
    control |= _homogeneous(image, 7, 87, 148)
    details["adaptive_7"] = int(control.sum())
    below = np.floor((wm_peak - gm_peak) / 3.0)
    low_5 = max(110 - 1.5 * 15, 110 - below)
    control |= _homogeneous(image, 5, np.floor(110 - np.ceil(110 - low_5)), 135)
    details["adaptive_5"] = int(control.sum())

    adaptive = (wm_peak - gm_peak) / 4.0
    lower = 110 - adaptive
    upper = 135
    six = np.zeros((3, 3, 3), dtype=np.int16)
    six[0, 1, 1] = six[2, 1, 1] = 1
    six[1, 0, 1] = six[1, 2, 1] = 1
    six[1, 1, 0] = six[1, 1, 2] = 1
    neighborhood_ok = (ndimage.minimum_filter(raw, size=3, mode="nearest") > lower) & (
        ndimage.maximum_filter(raw, size=3, mode="nearest") < upper)
    eligible = (raw >= lower) & (raw <= upper) & neighborhood_ok
    region = _candidate_region(eligible)
    candidates, selected, values = eligible[region], control[region], raw[region]
    three_added = 0
    while True:
        count, total = _neighbor_sum(raw, control, six, region)
        mean = np.divide(total, count, out=np.zeros(count.shape, dtype=np.float32), where=count>0)
        additions = candidates & ~selected & (count > 0) & ((values >= 110) | (mean - values < adaptive / 2))
        added = int(additions.sum())
        if added == 0:
            break
        selected |= additions
        three_added += added
    details["three_added"] = three_added

    lower = max(lower, 110 - 15)
    six_ok = (ndimage.minimum_filter(raw, footprint=six.astype(bool), mode="nearest") >= lower) & (
        ndimage.maximum_filter(raw, footprint=six.astype(bool), mode="nearest") <= upper)
    eligible = (raw >= lower) & (raw <= upper) & six_ok
    region = _candidate_region(eligible)
    candidates, selected, values = eligible[region], control[region], raw[region]
    cube = np.ones((3, 3, 3), dtype=np.int16)
    six_added = 0
    while True:
        count, total = _neighbor_sum(raw, control, cube, region)
        mean = np.divide(total, count, out=np.zeros(count.shape, dtype=np.float32), where=count>0)
        additions = candidates & ~selected & (count >= 4) & ((values >= 110) | (mean - values < adaptive / 2))
        added = int(additions.sum())
        if added == 0:
            break
        selected |= additions
        six_added += added
    details["six_added"] = six_added
    details["before_outlier_removal"] = int(control.sum())

    _remove_outliers_ordered(control)
    details["after_outlier_removal"] = int(control.sum())
    return control.astype(np.uint8), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wm-peak", type=float)
    parser.add_argument("--gm-peak", type=float)
    args = parser.parse_args()
    image = nib.load(str(args.input))
    result, details = controls_3d(np.asarray(image.dataobj), args.wm_peak, args.gm_peak)
    nib.save(nib.MGHImage(result, image.affine), str(args.output))
    print(json.dumps(details))


if __name__ == "__main__":
    main()
