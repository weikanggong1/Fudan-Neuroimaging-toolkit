"""拓扑 unMRI 的完整体积聚合和 curvbak 反馈，距离场由调用者提供。

固定 d932 源算法的 k→j→i 有序 float32 累加与原状态反馈；独立
Python/Numba 实现，不运行原软件，也不把距离场/完整 GA 声称为已完成。
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _aggregate(intensity, distance, white, gray, origin, index_scale, sampling_xyz,
               initial_curvbak, vertices, discarded, unset, white_ll, gray_ll, int_w, int_g):
    nwhite = ngray = 1
    max_distance = np.float32(0)
    for k in range(3, distance.shape[2] - 3):
        for j in range(3, distance.shape[1] - 3):
            for i in range(3, distance.shape[0] - 3):
                if intensity[i, j, k] == 0:
                    continue
                value = distance[i, j, k]
                if value == unset:
                    continue
                magnitude = np.float32(abs(value))
                if magnitude > max_distance:
                    max_distance = magnitude
                if value > .5:
                    gray_ll = np.float32(gray_ll + gray[i, j, k])
                    int_g = np.float32(int_g + np.float32(intensity[i, j, k]))
                    ngray += 1
                if value < -.5:
                    white_ll = np.float32(white_ll + white[i, j, k])
                    int_w = np.float32(int_w + np.float32(intensity[i, j, k]))
                    nwhite += 1
    white_ll = np.float32(white_ll / np.float32(nwhite))
    gray_ll = np.float32(gray_ll / np.float32(ngray))
    int_w = np.float32(int_w / np.float32(nwhite))
    int_g = np.float32(int_g / np.float32(ngray))
    raw_max = max_distance
    max_distance = min(np.float32(2), max(np.float32(1), max_distance))
    curvbak = initial_curvbak.copy()
    for slot in range(len(vertices)):
        if discarded[slot]:
            continue
        vertex = vertices[slot]
        # iVOL/jVOL/kVOL use a FP32 difference times FP32 x/y/zsize,
        # followed by double .5, then int
        # truncation; unlike nint, negative half values are not symmetric.
        i = int(np.float64(np.float32(np.float32(sampling_xyz[vertex, 0]-origin[0])*index_scale[0])) + .5)
        j = int(np.float64(np.float32(np.float32(sampling_xyz[vertex, 1]-origin[1])*index_scale[1])) + .5)
        k = int(np.float64(np.float32(np.float32(sampling_xyz[vertex, 2]-origin[2])*index_scale[2])) + .5)
        if i < 0 or i >= intensity.shape[0] or j < 0 or j >= intensity.shape[1] or k < 0 or k >= intensity.shape[2]:
            continue
        value = distance[i, j, k]
        if value == unset:
            value = np.float32(1)
        else:
            value = np.float32(np.float64(abs(value)) / np.float64(max_distance))
        value = min(np.float32(1), value)
        curvbak[vertex] = np.float32(curvbak[vertex] * value)
    unmri_ll = np.float32(white_ll + gray_ll)
    return white_ll, gray_ll, int_w, int_g, max_distance, raw_max, unmri_ll, nwhite, ngray, curvbak


def defect_volume_likelihood(*, defect_intensity_volume, signed_distance_mm,
                              white_log_probability, gray_log_probability,
                              initial_white_mean_log_probability, initial_gray_mean_log_probability,
                              initial_white_mean_intensity, initial_gray_mean_intensity,
                              volume_origin_mm, volume_index_scale_per_mm, vertex_sampling_positions_mm,
                              initial_vertex_curvbak, translated_defect_vertices,
                              discarded_defect_vertices, distance_unset_sentinel_mm) -> dict:
    """完整冻结缺陷体积与有序状态→源 float32 unMRI、均值和 curvbak 副本。

    defect_intensity_volume 是 (X,Y,Z) uint8，0为无体素；signed_distance_mm
    与 white_log_probability/gray_log_probability 是同网格 float32 三维
    数组，分别为原有符号距离/mm和已经求 log 的白/灰组织概率。
    四个 initial_* 标量转换 float32，直接提供原 defect 的初始均值；
    白/灰计数各从1开始，不从观测均值反推。volume_origin_mm 和
    volume_index_scale_per_mm 为有限 (3,) float32 缺陷体积 x/y/z 起点/mm及
    正索引比例/mm^-1，即原 MRI 字段 xsize/ysize/zsize；原 iVOL 使用
    (fx-xstart)*xsize，不把该比例改成间距除法，也不使用主MRI affine。
    vertex_sampling_positions_mm 为 (N,3) float32，即原 fx/fy/fz；
    initial_vertex_curvbak 为同序 (N,) float32，返回其更新副本。
    translated_defect_vertices 为有序 (K,) 整数 vertex_trans 后索引；
    允许重复，重复项按原顺序逐次相乘。discarded_defect_vertices 为
    bool(K,)，只对应原 status==DISCARD_VERTEX，其余状态都保留。
    distance_unset_sentinel_mm 为固定源 NPY 的有限 float32 值；不能
    用新的阈值替换精确相等测试。三体素边缘、±0.5mm分类及1…2mm
    反馈归一化固定，不提供算法开关；k→j→i有序 float32 累加。
    返回 white_ll/gray_ll/white_mean_intensity/gray_mean_intensity、
    max_distance_mm（裁剪后）、raw_max_distance_mm、unmri_ll（源TP
    float32转Python float）、nwhite/ngray整数、vertex_curvbak float32
    (N,)和scope。不修改输入、几何或距离场；调用者显式写回curvbak。
    非法类型、形状、索引、标记或非有限字段抛ValueError；输出
    非有限时抛FloatingPointError。尚不包含距离场生成、候选优化、
    全fitness/GA或网格验收；没有原版回退、文件读写或随机性。
    """
    intensity = np.asarray(defect_intensity_volume)
    distance = np.asarray(signed_distance_mm)
    white, gray = np.asarray(white_log_probability), np.asarray(gray_log_probability)
    if (intensity.ndim != 3 or not all(intensity.shape) or intensity.dtype != np.uint8
            or any(value.shape != intensity.shape or value.dtype != np.float32
                   for value in (distance, white, gray))):
        raise ValueError("uint8 defect volume and matching float32 distance/log-probability volumes required")
    origin = np.asarray(volume_origin_mm, np.float32)
    index_scale = np.asarray(volume_index_scale_per_mm, np.float32)
    xyz = np.asarray(vertex_sampling_positions_mm, np.float32)
    curvbak = np.asarray(initial_vertex_curvbak, np.float32)
    if (origin.shape != (3,) or index_scale.shape != (3,) or np.any(index_scale <= 0)
            or xyz.ndim != 2 or xyz.shape[1] != 3 or curvbak.shape != (len(xyz),)):
        raise ValueError("matching sampling positions/curvbak and three-axis defect-volume geometry required")
    initial = np.asarray([initial_white_mean_log_probability, initial_gray_mean_log_probability,
                          initial_white_mean_intensity, initial_gray_mean_intensity], np.float32)
    unset = np.asarray(distance_unset_sentinel_mm, np.float32)
    if initial.shape != (4,) or unset.ndim != 0 or any(not np.isfinite(value).all()
            for value in (distance, white, gray, origin, index_scale, xyz, curvbak, initial, unset)):
        raise ValueError("finite source fields and scalar initial means/NPY sentinel required")
    vertices, discarded = np.asarray(translated_defect_vertices), np.asarray(discarded_defect_vertices)
    if (vertices.ndim != 1 or not np.issubdtype(vertices.dtype, np.integer)
            or discarded.dtype != bool or discarded.shape != vertices.shape):
        raise ValueError("ordered integer translated vertices and matching bool DISCARD mask required")
    active = vertices[~discarded]
    if active.size and (active.min() < 0 or active.max() >= len(xyz)):
        raise ValueError("active translated defect vertex outside sampling array")
    values = _aggregate(np.ascontiguousarray(intensity), np.ascontiguousarray(distance),
        np.ascontiguousarray(white), np.ascontiguousarray(gray), origin, index_scale,
        np.ascontiguousarray(xyz), np.ascontiguousarray(curvbak), np.ascontiguousarray(vertices, np.int64),
        np.ascontiguousarray(discarded), np.float32(unset), *initial)
    white_ll, gray_ll, int_w, int_g, maximum, raw_max, unmri_ll, nwhite, ngray, feedback = values
    if not np.isfinite((white_ll, gray_ll, int_w, int_g, maximum, raw_max, unmri_ll)).all() or not np.isfinite(feedback).all():
        raise FloatingPointError("nonfinite ordered unMRI aggregate or curvbak feedback")
    return {"white_ll": float(white_ll), "gray_ll": float(gray_ll),
        "white_mean_intensity": float(int_w), "gray_mean_intensity": float(int_g),
        "max_distance_mm": float(maximum), "raw_max_distance_mm": float(raw_max),
        "unmri_ll": float(unmri_ll), "nwhite": int(nwhite), "ngray": int(ngray),
        "vertex_curvbak": feedback,
        "scope": "complete unMRI volume aggregation/feedback only; signed distance and full GA not implemented here"}
