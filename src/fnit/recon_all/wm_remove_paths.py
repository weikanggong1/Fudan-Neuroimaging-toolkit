"""按固定源顺序阻断 MTL→皮层路径，保留 MRI 的数值/字节双重访问。

本模块为 FNIT 自有 NumPy/Numba 实现，不加载原版程序或共享库。
临时 ROI、hippo 和 filled 保留 aseg 的原生四字节类型；XYZ 影像
转为内部 ZYX 行存储，使 row[x] 字节与 row[x*4] 数值共享同一缓冲。
"""

from __future__ import annotations

import sys
import time

import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation

from .edit_wm_aseg_core_python import (
    _clamp, _distance_to_label, _distance_to_wm, _superior_edge, _anterior_edge,
)


@njit(cache=True)
def _lateral_edge(seg, label, x, y, z, left):
    """沿源指定左右方向寻找九个连续无该标签的窗口，返回体素距离。"""
    direction = 1 if left else -1
    for distance in range(seg.shape[0]):
        xx = _clamp(x + direction * distance, seg.shape[0])
        if _distance_to_label(seg, label, xx, y, z, direction, 0, 0, 10) >= 10:
            return distance
    return seg.shape[0]


@njit(cache=True)
def _inferior_edge(seg, label, x, y, z):
    """沿 y 增加方向寻找源标签边界；完整扫描未找到时返回 −1。"""
    for distance in range(seg.shape[1]):
        yy = _clamp(y + distance, seg.shape[1])
        if _distance_to_label(seg, label, x, yy, z, 0, 1, 0, 10) >= 10:
            return distance
    return -1


@njit(cache=True)
def _clean_roi_bytes(roi_bytes, aseg, left):
    """顺序执行五项 ROI 清理；写入是 row[x] 字节，不是数值体素。"""
    width, height, depth = aseg.shape
    for x in range(width):
        for y in range(1, height):
            for z in range(depth - 1):
                label = aseg[x, y, z]
                if label != 18 and label != 54:
                    continue
                is_left = label == 18
                if (_anterior_edge(aseg, label, x, y, z) > 3 and
                        _superior_edge(aseg, label, x, y, z) > 2):
                    continue
                if _inferior_edge(aseg, label, x, y, z) <= 3:
                    continue
                if _lateral_edge(aseg, label, x, y, z, is_left) <= 3:
                    continue
                roi_bytes[z, y, x] = 0
                roi_bytes[z + 1, y, x] = 0
                roi_bytes[z, y - 1, x] = 0

    # xl is deliberately reset for each x slice, exactly as in the source.
    for x in range(width):
        xl = 0 if left else width - 1
        for y in range(1, height):
            for z in range(depth - 1):
                label = aseg[x, y, z]
                if left and label == 17 and x > xl:
                    xl = x
                elif not left and label == 53 and x < xl:
                    xl = x
        for y in range(1, height):
            for z in range(depth - 1):
                label = aseg[x, y, z]
                if label != 17 and label != 53:
                    continue
                if (left and x > xl - 5) or (not left and x < xl + 5):
                    continue
                if _anterior_edge(aseg, label, x, y, z) > 3:
                    continue
                roi_bytes[z, y, x] = 0
                roi_bytes[z + 1, y, x] = 0
                roi_bytes[z, y - 1, x] = 0

    for z in range(depth):
        xmin, xmax = width - 1, 0
        for x in range(width):
            for y in range(height):
                label = aseg[x, y, z]
                if ((left and (label == 17 or label == 18)) or
                        (not left and (label == 53 or label == 54))):
                    xmin = min(xmin, x)
                    xmax = max(xmax, x)
        xmid = (2 * xmin + xmax) // 3 if left else (2 * xmax + xmin) // 3
        for x in range(width):
            for y in range(height):
                label = aseg[x, y, z]
                if left:
                    if ((label == 18 or label == 17 or label == 3) and x < xmid):
                        roi_bytes[z, y, x] = 0
                    if label == 0 and x < xmin + 3:
                        roi_bytes[z, y, x] = 0
                else:
                    if ((label == 54 or label == 53 or label == 42) and x > xmid):
                        roi_bytes[z, y, x] = 0
                    if label == 0 and x > xmax - 3:
                        roi_bytes[z, y, x] = 0

    for z in range(depth):
        xmin, xmax = width - 1, 0
        for x in range(width):
            for y in range(height):
                label = aseg[x, y, z]
                if ((left and (label == 17 or label == 18)) or
                        (not left and (label == 53 or label == 54))):
                    xmin = min(xmin, x)
                    xmax = max(xmax, x)
        xmid = (xmin + xmax) // 2
        if left:
            xmax = xmid
        else:
            xmin = xmid
        for x in range(xmin, xmax + 1):
            for y in range(height):
                if roi_bytes[z, y, x] == 0 or aseg[x, y, z] != 0:
                    continue
                if _distance_to_wm(aseg, x, y, z, 0, 1, 0, 6, True) >= 6:
                    roi_bytes[z, y, x] = 0

    for z in range(depth):
        for x in range(width):
            for y in range(height):
                if roi_bytes[z, y, x] == 0:
                    continue
                label = aseg[x, y, z]
                if label != 3 and label != 42:
                    continue
                amygdala = 18 if label == 3 else 54
                if (_distance_to_label(aseg, amygdala, x, y, z, 0, 0, -1, 10) <= 2 and
                        _distance_to_label(aseg, amygdala, x, y, z, 0, 0, 1, 10) >= 10):
                    roi_bytes[z, y, x] = 0


@njit(cache=True)
def _six_on(wm, x, y, z):
    """MRIneighborsOn 的六邻域 >=5，重复的边界索引也逐项计数。"""
    return (int(wm[_clamp(x - 1, wm.shape[0]), y, z] >= 5) +
            int(wm[_clamp(x + 1, wm.shape[0]), y, z] >= 5) +
            int(wm[x, _clamp(y + 1, wm.shape[1]), z] >= 5) +
            int(wm[x, _clamp(y - 1, wm.shape[1]), z] >= 5) +
            int(wm[x, y, _clamp(z + 1, wm.shape[2])] >= 5) +
            int(wm[x, y, _clamp(z - 1, wm.shape[2])] >= 5))


@njit(cache=True)
def _search_hemisphere(wm, brain, aseg, roi_bytes, roi_values,
                       hippo_bytes, filled_bytes, filled_values, bounds, left):
    """固定 x/y/z 扫描即时传播；返回填充次数、外层轮次和停止原因。"""
    width, height, depth = wm.shape
    x1, x2, y1, y2, z1, z2 = bounds
    total = 0
    niter = 0
    while True:
        hippo_bytes[:] = 0
        xmedial = width if left else 0
        zanterior = 0
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    if wm[x, y, z] > 5:
                        continue
                    label = aseg[x, y, z]
                    seed = ((left and (label == 17 or label == 18 or label == 5)) or
                            (not left and (label == 53 or label == 54 or label == 44)))
                    if seed and wm[x, y + 1, z] > 5:
                        hippo_bytes[z, y, x] = 128
                        if left:
                            xmedial = min(xmedial, x)
                        else:
                            xmedial = max(xmedial, x)
                        zanterior = max(zanterior, z)
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                for z in range(z1, z2 + 1):
                    if hippo_bytes[z, y, x] and (
                            (left and x < xmedial + 10) or
                            (not left and x > xmedial - 10) or z > zanterior - 15):
                        hippo_bytes[z, y, x] = 0
        filled_bytes[:] = hippo_bytes
        path = False
        xgm = ygm = zgm = -1
        while True:
            nfilled = 0
            nmutated = 0
            lo = xmedial if left else x1
            hi = x2 if left else xmedial
            for x in range(lo, hi + 1):
                if path:
                    break
                for y in range(y1, y2 + 1):
                    if path:
                        break
                    for z in range(z2 + 1):
                        if path:
                            break
                        # The source LH uses typed reads; RH uses MRIvox here.
                        if left:
                            if filled_values[z, y, x] == 0 or roi_values[z, y, x] == 0:
                                continue
                        elif filled_bytes[z, y, x] == 0 or roi_bytes[z, y, x] == 0:
                            continue
                        for xk in range(-1, 2):
                            if path:
                                break
                            xi = _clamp(x + xk, width)
                            for yk in range(2):
                                if path:
                                    break
                                yi = _clamp(y + yk, height)
                                for zk in range(-1, 2):
                                    if path:
                                        break
                                    zi = _clamp(z + zk, depth)
                                    if left:
                                        if roi_values[zi, yi, xi] == 0 or filled_values[zi, yi, xi] != 0:
                                            continue
                                    elif filled_bytes[zi, yi, xi] != 0 or roi_bytes[zi, yi, xi] == 0:
                                        continue
                                    if wm[xi, yi, zi] > 5:
                                        continue
                                    label = aseg[xi, yi, zi]
                                    if label == 3 or label == 42:
                                        path = True
                                        xgm, ygm, zgm = xi, yi, zi
                                    elif left:
                                        if (xi > xmedial + 10 and
                                                (label == 0 or label == 3 or label == 17 or
                                                 label == 18 or label == 5 or label == 2) and
                                                roi_values[zi, yi, xi] > 0):
                                            nfilled += 1
                                            if filled_bytes[zi, yi, xi] != 128:
                                                nmutated += 1
                                            filled_bytes[zi, yi, xi] = 128
                                    elif xi < xmedial - 10 and roi_values[zi, yi, xi] > 0:
                                        nfilled += 1
                                        if filled_bytes[zi, yi, xi] != 128:
                                            nmutated += 1
                                        filled_bytes[zi, yi, xi] = 128
            if nfilled == 0 or path:
                break
            if nmutated == 0:
                raise RuntimeError("source path growth repeats an unchanged byte buffer without reaching cortex")
        if path:
            brightest = int(brain[xgm, ygm, zgm])
            xb, yb, zb = xgm, ygm, zgm
            for xk in range(-3, 4):
                xi = _clamp(xgm + xk, width)
                for yk in range(-3, 1):
                    yi = _clamp(ygm + yk, height)
                    for zk in range(-3, 4):
                        zi = _clamp(zgm + zk, depth)
                        if filled_values[zi, yi, xi] == 0 or wm[xi, yi, zi] > 5:
                            continue
                        if _six_on(wm, xi, yi, zi) == 0:
                            continue
                        intensity = int(brain[xi, yi, zi])
                        if intensity > brightest:
                            brightest = intensity
                            xb, yb, zb = xi, yi, zi
            wm[xb, yb, zb] = 250
            hippo_bytes[zb, yb, xb] = 0
            total += 1
        # Source: path && niter++ < 100000 (the terminating fill is retained).
        if not path:
            return total, niter + 1, False
        previous = niter
        niter += 1
        if previous >= 100000:
            return total, niter, True


def _row_views(shape: tuple[int, int, int], dtype: np.dtype,
               values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """分配原生 ZYX 连续数值行及别名字节视图；XYZ 值按原位置复制。"""
    width, height, depth = shape
    typed = np.zeros((depth, height, width), dtype=dtype)
    if values is not None:
        typed[:] = values.transpose(2, 1, 0)
    raw = typed.view(np.uint8).reshape(depth, height, width * dtype.itemsize)
    return raw, typed


def _bounds(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """按 typed >1 计算源 boundingBox，空 ROI 返回等价的空扫描范围。"""
    indices = np.argwhere(values > 1)
    if not len(indices):
        return np.array([max(shape[0] - 1, 1), 0,
                         max(shape[1] - 1, 1), 0,
                         max(shape[2] - 1, 1), 0], dtype=np.int64)
    lower = indices.min(axis=0)
    upper = indices.max(axis=0)
    return np.array([max(int(lower[2]), 1), min(int(upper[2]), shape[0] - 2),
                     max(int(lower[1]), 1), min(int(upper[1]), shape[1] - 2),
                     max(int(lower[0]), 1), min(int(upper[0]), shape[2] - 2)], dtype=np.int64)


def remove_mtl_paths(*, wm: np.ndarray, brain: np.ndarray, aseg: np.ndarray,
                     aseg_storage_dtype: str | np.dtype = "int32",
                     use_unreachable_proof: bool = True) -> tuple[np.ndarray, dict]:
    """完整有序路径阻断，返回新 uint8 XYZ WM 和阶段诊断。

    wm/brain: 同形 3D uint8 WM/脑强度；aseg: 同形 int32 标签；各轴≥3。
    aseg_storage_dtype: 原 aseg 的 int32 或 float32 四字节存储，默认int32；
    float32 路径要求每个标签能精确表示。只支持 little-endian 运行主机。
    use_unreachable_proof 默认True：五次3³膨胀 ROI 在 row[x] 可及数值
    范围之外时跳过确证无作用的扫描；False完整执行，用于回归该证明。
    输出保持原 XYZ 网格/单位体素，无重采样、输入修改或原程序调用。
    metadata 含实际 wall 秒数、类型、每侧填充/轮次、source cap状态；
    cap保留100000条件，触发时返回原语义结果并显式记录，不额外近似。
    非法 shape/dtype/标签/参数抛ValueError。原源内层在 byte 写入不改变
    缓冲却仍计数时会永久重复；本实现证明该固定状态后抛RuntimeError，
    不返回近似结果。Numba/内存异常传播。
    """
    started = time.perf_counter()
    dtype = np.dtype(aseg_storage_dtype).newbyteorder("=")
    if (not isinstance(use_unreachable_proof, bool) or sys.byteorder != "little" or
            dtype not in (np.dtype(np.int32), np.dtype(np.float32)) or
            not isinstance(wm, np.ndarray) or not isinstance(brain, np.ndarray) or
            not isinstance(aseg, np.ndarray) or wm.ndim != 3 or
            wm.shape != brain.shape or wm.shape != aseg.shape or min(wm.shape) < 3 or
            wm.dtype != np.uint8 or brain.dtype != np.uint8 or
            aseg.dtype.newbyteorder("=") != np.dtype(np.int32) or np.any(aseg < 0)):
        raise ValueError("expected matching 3D uint8 WM/brain, nonnegative int32 labels and int32/float32 storage")
    if dtype == np.dtype(np.float32) and not np.array_equal(aseg.astype(np.float32), aseg):
        raise ValueError("float32 aseg storage must represent labels exactly")
    output = wm.copy()
    sides = []
    byte_max_x = (wm.shape[0] - 1) // 4
    for left, labels in ((True, (17, 18, 5)), (False, (53, 54, 44))):
        seed = np.isin(aseg, labels)
        dilated = binary_dilation(seed, structure=np.ones((3, 3, 3), dtype=bool), iterations=5)
        if use_unreachable_proof and not np.any(dilated[:byte_max_x + 1]):
            sides.append({"hemisphere": "lh" if left else "rh", "proof_unreachable": True,
                          "filled": 0, "outer_iterations": 0, "source_iteration_cap_reached": False})
            continue
        roi_bytes, roi_values = _row_views(wm.shape, dtype, dilated.astype(dtype) * 128)
        _clean_roi_bytes(roi_bytes, aseg, left)
        hippo_bytes, _ = _row_views(wm.shape, dtype)
        filled_bytes, filled_values = _row_views(wm.shape, dtype)
        bounds = _bounds(roi_values, wm.shape)
        count, iterations, capped = _search_hemisphere(output, brain, aseg, roi_bytes,
            roi_values, hippo_bytes, filled_bytes, filled_values, bounds, left)
        sides.append({"hemisphere": "lh" if left else "rh", "proof_unreachable": False,
                      "filled": int(count), "outer_iterations": int(iterations),
                      "source_iteration_cap_reached": bool(capped), "bounding_box_scan": bounds.tolist()})
    return output, {"implementation": "FNIT source-order NumPy/Numba MTL path search",
        "used_native_programs": False, "aseg_storage_dtype": dtype.name,
        "host_byteorder": sys.byteorder, "byte_access_max_numerical_x": byte_max_x,
        "filled": sum(side["filled"] for side in sides), "hemispheres": sides,
        "lh_reads": "typed ROI/filled", "rh_reads": "byte ROI/filled",
        "iteration_cap_condition": "path && niter++ < 100000",
        "total_seconds": time.perf_counter() - started}
