"""原 MRI 与完整候选边表生成局部缺陷强度及校准白灰概率场。

独立 NumPy/Numba 实现固定拓扑源码的 mriDefectVolume、
computeDefectStatistics 强度统计和 type=0 defectVolumeLikelihood。
复用已验证的拓扑 RAS/MRI 采样器，不读取参考局部体积或原程序。
"""
from __future__ import annotations
import numpy as np
from numba import njit
from .topology_edge_mri_score import _sample


@njit(cache=True, fastmath=False)
def _round_away(value):
    return int(np.float64(value) - .5) if value < 0 else int(np.float64(value) + .5)


@njit(cache=True, fastmath=False)
def _statistics(bins, counts, bin_size):
    peak = np.float32(0)
    for value in counts:
        if value > peak:
            peak = value
    total = mean = variance = np.float32(0)
    for index in range(len(counts)):
        value = np.float32(np.float64(bins[index]) - np.float64(bin_size) / 2.0)
        if np.float64(counts[index]) < np.float64(peak) / 100.0:
            continue
        mean = np.float32(mean + np.float32(value * counts[index]))
        variance = np.float32(variance + np.float32(np.float32(value * value) * counts[index]))
        total = np.float32(total + counts[index])
    mean = np.float32(mean / total)
    variance = np.float32(variance / total)
    variance = np.float32(variance - np.float32(mean * mean))
    if variance < 0:
        variance = np.float32(0)
    index = min(len(counts) - 1, max(0, _round_away(mean)))
    mean_log = np.float32(np.log(counts[index]))
    return mean, variance, mean_log


@njit(cache=True, fastmath=False)
def _index(value, origin, scale):
    return int(np.float64(np.float32(scale * np.float32(value - origin))) + .5)


@njit(cache=True, fastmath=False)
def _geometry(xyz, edges, scale):
    lower = np.full(3, np.float32(1000), np.float32)
    upper = np.full(3, np.float32(-1000), np.float32)
    for edge in edges:
        for vertex in edge:
            for axis in range(3):
                value = xyz[vertex, axis]
                if value > upper[axis]:
                    upper[axis] = value
                if value < lower[axis]:
                    lower[axis] = value
    shape = np.empty(3, np.int64)
    for axis in range(3):
        lower[axis] = np.float32(lower[axis] - np.float32(1))
        upper[axis] = np.float32(upper[axis] + np.float32(1))
        shape[axis] = int(np.ceil(np.float32(scale * np.float32(upper[axis] - lower[axis]))))
    return lower, upper, shape


@njit(cache=True, fastmath=False)
def _edge_mask(xyz, edges, origin, scale, shape):
    mask = np.zeros((shape[0], shape[1], shape[2]), np.uint8)
    for first, second in edges:
        a, b = xyz[first].copy(), xyz[second].copy()
        if b[0] < a[0]:
            a, b = b, a
        difference = np.empty(3, np.float32)
        for axis in range(3):
            difference[axis] = np.float32(b[axis] - a[axis])
        square = np.float32(np.float32(difference[0] * difference[0] + difference[1] * difference[1]) + difference[2] * difference[2])
        length = np.float32(scale * np.sqrt(square))
        point = a.copy()
        if abs(length) >= np.float32(np.finfo(np.float32).eps):
            step = np.empty(3, np.float32)
            for axis in range(3):
                step[axis] = np.float32(difference[axis] / np.float32(np.float32(2) * length))
            maximum = int(np.ceil(np.float32(np.float32(2) * length + np.float32(5))))
            index = 0
            # Preserve the original x-only condition: vertical edges do not
            # gain a replacement sampling rule, but their last point remains.
            while point[0] < b[0] and index < maximum:
                i, j, k = _index(point[0], origin[0], scale), _index(point[1], origin[1], scale), _index(point[2], origin[2], scale)
                if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                    mask[i, j, k] = 1
                for axis in range(3):
                    point[axis] = np.float32(point[axis] + step[axis])
                index += 1
        else:
            i, j, k = _index(a[0], origin[0], scale), _index(a[1], origin[1], scale), _index(a[2], origin[2], scale)
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                mask[i, j, k] = 1
        i, j, k = _index(b[0], origin[0], scale), _index(b[1], origin[1], scale), _index(b[2], origin[2], scale)
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
            mask[i, j, k] = 1
    return mask


@njit(cache=True, fastmath=False)
def _prior_fields(volume, matrix, shape, origin, scale, white_counts, gray_counts, white_mean, gray_mean):
    intensity = np.empty((shape[0], shape[1], shape[2]), np.uint8)
    white_log = np.empty(intensity.shape, np.float32)
    gray_log = np.empty(intensity.shape, np.float32)
    # Source writes every local voxel, including zero entries in the edge mask.
    for k in range(shape[2]):
        z = np.float32(origin[2] + np.float32(np.float32(k) / scale))
        for j in range(shape[1]):
            y = np.float32(origin[1] + np.float32(np.float32(j) / scale))
            for i in range(shape[0]):
                x = np.float32(origin[0] + np.float32(np.float32(i) / scale))
                value = _sample(volume, matrix, x, y, z)
                intensity[i, j, k] = np.uint8(value)
                w = _round_away(min(value, np.float64(white_mean)))
                g = _round_away(max(value, np.float64(gray_mean)))
                # Original type=0 lookup has no bounds clamp. Public validation
                # below ensures all possible uint8 MRI samples are valid bins.
                white_log[i, j, k] = np.float32(np.log(white_counts[w]))
                gray_log[i, j, k] = np.float32(np.log(gray_counts[g]))
    return intensity, white_log, gray_log


def _histogram(bins, counts, bin_size, name):
    bins, counts = np.asarray(bins), np.asarray(counts)
    size = np.asarray(bin_size)
    if (bins.dtype != np.float32 or counts.dtype != np.float32 or bins.ndim != 1
            or counts.shape != bins.shape or len(bins) < 256 or not np.isfinite(bins).all()
            or not np.isfinite(counts).all() or np.any(counts <= 0)
            or size.ndim != 0 or not np.isfinite(size) or size < 0):
        raise ValueError(name + ' finite float32 bins/positive counts and nonnegative bin_size required')
    return np.ascontiguousarray(bins), np.ascontiguousarray(counts), np.float32(size)


def defect_volume_priors(*, original_vertices, candidate_edges, volume, ras_to_voxel,
                         white_histogram_bins, white_histogram_counts, white_histogram_bin_size,
                         gray_histogram_bins, gray_histogram_counts, gray_histogram_bin_size,
                         volume_resolution=-1) -> dict:
    """完整原 MRI/候选边表及校准直方图→自产局部强度、白灰 logP 和均值。

    original_vertices 是 float32(N,3) orig surface RAS/mm；candidate_edges
    是完整、有序整数(E,2)边表，不是候选已接受面的边。volume 是 MRI
    网格(X,Y,Z)uint8，ras_to_voxel为实际拓扑surface RAS→MRI连续体素
    float32(4,4)，用 source_topology_ras_to_voxel 从该MRI头生成。
    两组织 histogram_bins/counts 是已校准 float32(B,)（B至少256），
    bin_size为非负scalar；源校准直方图可能保留0，仍按原bins统计，
    不重建bin间距。它们是阶段输入，不能以参考局部logP代替。
    volume_resolution 默认-1对应源scale=2；显式正数是局部坐标到索引
    乘数mm^-1。按全部边两端bbox±1mm、ceil及原边采样生成mask，然后
    完整采样每个体素；uint8写出向零截断，logP基于舍入前double强度。
    返回局部XYZ uint8 intensity/initial_edge_mask、float32白灰logP、
    原始algorithm几何、白灰mean/variance/mean_log。不修改输入/线程，
    不写文件、不调用原程序；不是生成原始校准直方图或完整GA。
    无效输入或越界mean在执行前ValueError；非有限产物FloatingPointError。
    """
    xyz, edges, image, matrix = map(np.asarray, (original_vertices, candidate_edges, volume, ras_to_voxel))
    if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or len(edges) < 1 or edges.min() < 0 or edges.max() >= len(xyz)
            or image.dtype != np.uint8 or image.ndim != 3 or min(image.shape) < 1
            or matrix.dtype != np.float32 or matrix.shape != (4, 4) or not np.isfinite(matrix).all()):
        raise ValueError('finite float32 orig mesh/matrix, complete integer edges and uint8 MRI required')
    resolution = np.asarray(volume_resolution)
    if resolution.ndim != 0 or not np.isfinite(resolution) or not (resolution == -1 or resolution > 0):
        raise ValueError('volume_resolution=-1 or a positive scale required')
    scale = np.float32(2 if resolution == -1 else resolution)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite float32 volume scale required')
    wb, wc, ws = _histogram(white_histogram_bins, white_histogram_counts, white_histogram_bin_size, 'white')
    gb, gc, gs = _histogram(gray_histogram_bins, gray_histogram_counts, gray_histogram_bin_size, 'gray')
    white_mean, white_variance, white_mean_log = _statistics(wb, wc, ws)
    gray_mean, gray_variance, gray_mean_log = _statistics(gb, gc, gs)
    if not (0 <= white_mean <= 255 and 0 <= gray_mean <= 255):
        raise ValueError('calibrated means within uint8 MRI range required for source unclamped lookup')
    xyz, edges = np.ascontiguousarray(xyz), np.ascontiguousarray(edges, np.int64)
    origin, bounds_end, shape = _geometry(xyz, edges, scale)
    if np.any(shape <= 0):
        raise ValueError('source bbox must allocate positive XYZ dimensions')
    mask = _edge_mask(xyz, edges, origin, scale, shape)
    intensity, white_log, gray_log = _prior_fields(np.ascontiguousarray(image), np.ascontiguousarray(matrix), shape, origin, scale, wc, gc, white_mean, gray_mean)
    if not all(np.isfinite(x).all() for x in (white_log, gray_log)):
        raise FloatingPointError('nonfinite calibrated source log probabilities')
    return dict(defect_intensity_volume=intensity, initial_edge_mask=mask,
                white_log_probability=white_log, gray_log_probability=gray_log,
                volume_origin_mm=origin, volume_index_scale_per_mm=np.full(3, scale, np.float32),
                volume_shape=tuple(int(x) for x in shape), bounding_end_mm=bounds_end,
                algorithm_volume_geometry=np.concatenate((origin, np.full(3, scale, np.float32), shape.astype(np.float32) / np.float32(2))),
                white_mean=float(white_mean), gray_mean=float(gray_mean),
                white_variance=float(white_variance), gray_variance=float(gray_variance),
                initial_white_mean_log_probability=float(white_mean_log), initial_gray_mean_log_probability=float(gray_mean_log),
                scope='complete GA type=0 local MRI/logP preparation from calibrated histogram inputs; not histogram generation or full GA')
