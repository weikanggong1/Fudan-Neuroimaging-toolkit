"""拓扑缺陷的法向点积和主曲率似然；不进行几何更新或 GA 搜索。

按固定 FreeSurfer d932 源算法独立实现，许可见 licenses/FreeSurfer.txt。
主曲率由已有 topology_principal_curvature 或候选几何调用者计算；本模块
只消费同一几何版本的字段，不缓存、重新估计或替换主曲率。
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numba import njit


def _patch_indices(values, count: int) -> np.ndarray:
    vertices = np.asarray(values)
    if (vertices.ndim != 1 or not np.issubdtype(vertices.dtype, np.integer)
            or len(np.unique(vertices)) != len(vertices)
            or vertices.size and (vertices.min() < 0 or vertices.max() >= count)):
        raise ValueError("patch_vertices must contain unique ordered vertex indices")
    return np.ascontiguousarray(vertices, np.int64)


def _areas(values, count: int, vertices: np.ndarray) -> np.ndarray:
    areas = np.asarray(values, np.float32)
    if areas.shape != (count,) or not np.isfinite(areas).all():
        raise ValueError("finite signed vertex areas required")
    # The original plane metric can produce signed vertex areas. Preserve
    # them; the kernels check their own existing ordered double denominator.
    return np.ascontiguousarray(areas)


def _histogram(value, name: str):
    if not isinstance(value, Mapping) or any(key not in value for key in ("bins", "counts", "bin_size")):
        raise ValueError(f"{name} requires bins, counts and bin_size")
    bins = np.asarray(value["bins"], np.float32)
    counts = np.asarray(value["counts"], np.float32)
    spacing = np.asarray(value["bin_size"], np.float32)
    if (bins.ndim != 1 or not bins.size or counts.shape != bins.shape
            or spacing.ndim != 0 or not np.isfinite(spacing) or spacing <= 0
            or not np.isfinite(bins).all() or not np.isfinite(counts).all()
            or np.any(counts <= 0)):
        raise ValueError(f"{name} needs finite float32 bins, positive PDF counts and scalar bin_size")
    return bins[0], np.ascontiguousarray(counts), np.float32(spacing)


@njit(cache=True, fastmath=False)
def _nint(value):
    # Pinned nint(double): half away from zero, then C-style truncation.
    return int(value - .5) if value < 0 else int(value + .5)


@njit(cache=True, fastmath=False, error_model="numpy")
def _normal_dot(xyz, normal, areas, vertices, offsets, neighbors, first, spacing, counts):
    total = weighted = total_area = np.float64(0)
    per_vertex = np.empty(len(vertices), np.float64)
    for index in range(len(vertices)):
        vertex = vertices[index]
        x, y, z = np.float64(xyz[vertex, 0]), np.float64(xyz[vertex, 1]), np.float64(xyz[vertex, 2])
        nx, ny, nz = np.float64(normal[vertex, 0]), np.float64(normal[vertex, 1]), np.float64(normal[vertex, 2])
        likelihood = np.float64(0)
        degree = offsets[vertex + 1] - offsets[vertex]
        for slot in range(offsets[vertex], offsets[vertex + 1]):
            neighbor = neighbors[slot]
            dx = np.float64(xyz[neighbor, 0]) - x
            dy = np.float64(xyz[neighbor, 1]) - y
            dz = np.float64(xyz[neighbor, 2]) - z
            dot = dx * nx + dy * ny + dz * nz
            # The index is clamped after nearest-integer rounding, not floor.
            ratio = (dot - np.float64(first)) / np.float64(spacing)
            bin_index = _nint(ratio)
            bin_index = min(len(counts) - 1, max(0, bin_index))
            likelihood += np.log(np.float64(counts[bin_index]))
        mean = likelihood / degree
        per_vertex[index] = mean
        total += mean
        area = np.float64(areas[vertex])
        total_area += area
        # Preserve source (area * sum) / degree order.
        weighted += area * likelihood / degree
    if len(vertices):
        return np.float32(total / len(vertices)), np.float32(weighted / total_area), per_vertex
    return np.float32(0), np.float32(0), per_vertex


def patch_normal_dot_likelihood(*, original_vertices, vertex_normals, vertex_areas_mm2,
                                patch_vertices, neighbor_offsets, ordered_neighbors,
                                normal_dot_histogram) -> dict:
    """完整有序 patch→源 float32 curv_ll/cll 及逐顶点均值，输入不变。

    original_vertices/vertex_normals 为同序 (N,3) 字段，转换 float32；
    前者是本候选 orig surface RAS/mm，后者是其原流程当前法向、无单位。
    vertex_areas_mm2 为 (N,) float32 mm²；patch_vertices 为唯一有序
    (K,) 整数索引。neighbor_offsets 是 (N+1,) CSR 偏移，ordered_neighbors
    是 (E,) 整数一环邻接，逐顶点与原 vt.v[0:vt.vnum] 顺序相同。
    normal_dot_histogram 为 bins/counts/bin_size 字典：前两者 (B,)
    float32，bin_size 正 float32 mm，counts 是正概率，不重新归一化。
    返回 curv_ll（顶点均值的均值）、cll（源面积加权值），两者经过
    原 TP 字段 float32 写回再转换为 Python float；另返回 (K,) float64
    per_vertex_log_likelihood、vertex_count 及 scope。零 patch 两标量为0。
    本函数不隐式跳过 marked/rip；调用者提供原 tp.vertices。无缓存，
    几何/邻接改变后必须重新提供全部匹配字段。非法/非有限字段、空
    选中邻接或非正PDF 抛 ValueError；负面积和负总面积保留原权重。
    零分母的 cll 诊断保持原 IEEE NaN/Inf，cll_defined 单列有限性，
    cll_float32_bits 保留原位模式；不将诊断写成0。curv_ll 和逐顶点
    logP 非有限时抛 FloatingPointError；此处不替代最终网格质量检查。
    没有设备/精度/线程参数，无文件写出、原版程序或几何更新。
    """
    xyz = np.asarray(original_vertices, np.float32)
    normal = np.asarray(vertex_normals, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normal.shape != xyz.shape:
        raise ValueError("matching (N,3) coordinates and normals required")
    if not np.isfinite(xyz).all() or not np.isfinite(normal).all():
        raise ValueError("finite coordinates and normals required")
    vertices = _patch_indices(patch_vertices, len(xyz))
    areas = _areas(vertex_areas_mm2, len(xyz), vertices)
    offsets, neighbors = np.asarray(neighbor_offsets), np.asarray(ordered_neighbors)
    if (offsets.shape != (len(xyz) + 1,) or neighbors.ndim != 1
            or not np.issubdtype(offsets.dtype, np.integer)
            or not np.issubdtype(neighbors.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(neighbors)
            or np.any(offsets[1:] < offsets[:-1])
            or neighbors.size and (neighbors.min() < 0 or neighbors.max() >= len(xyz))
            or np.any(offsets[vertices + 1] == offsets[vertices])):
        raise ValueError("source-order CSR and nonempty selected one-ring neighbors required")
    first, counts, spacing = _histogram(normal_dot_histogram, "normal_dot_histogram")
    curv_ll, cll, per_vertex = _normal_dot(np.ascontiguousarray(xyz), np.ascontiguousarray(normal),
        areas, vertices, np.ascontiguousarray(offsets, np.int64), np.ascontiguousarray(neighbors, np.int64),
        first, spacing, counts)
    if not np.isfinite(curv_ll) or not np.isfinite(per_vertex).all():
        raise FloatingPointError("nonfinite normal-dot likelihood")
    return {"curv_ll": float(curv_ll), "cll": float(cll),
            "cll_defined": bool(np.isfinite(cll)),
            "cll_float32_bits": int(np.asarray(cll, np.float32).view(np.uint32)),
            "per_vertex_log_likelihood": per_vertex, "vertex_count": len(vertices),
            "scope": "normal-dot component only; not complete GA"}


@njit(cache=True, fastmath=False, error_model="numpy")
def _curvature(k1, k2, areas, vertices, first1, spacing1, counts1, first2, spacing2,
               counts2, pdf, origin, spacing):
    marginal_total = total = weighted = total_area = np.float64(0)
    per_vertex = np.empty(len(vertices), np.float64)
    for index in range(len(vertices)):
        vertex = vertices[index]
        ratio1 = np.float32(np.float32(k1[vertex] - first1) / spacing1)
        bin1 = min(len(counts1)-1, max(0, _nint(np.float64(ratio1))))
        likelihood = np.log(np.float64(counts1[bin1]))
        marginal_total += likelihood
        ratio2 = np.float32(np.float32(k2[vertex] - first2) / spacing2)
        bin2 = min(len(counts2)-1, max(0, _nint(np.float64(ratio2))))
        likelihood += np.log(np.float64(counts2[bin2]))
        marginal_total += likelihood
        # MRI indices use truncated FP32 quotient, unlike histogram rounding.
        ix = min(pdf.shape[0]-1, max(0, int(np.float32(np.float32(k1[vertex]-origin[0])/spacing[0]))))
        iy = min(pdf.shape[1]-1, max(0, int(np.float32(np.float32(k2[vertex]-origin[1])/spacing[1]))))
        likelihood = np.log(np.float64(pdf[ix, iy]))
        per_vertex[index] = likelihood
        total += likelihood
        area = np.float64(areas[vertex])
        total_area += area
        weighted += area * likelihood
    if len(vertices):
        return (np.float32(total / len(vertices)), np.float32(weighted / total_area),
                np.float32(marginal_total / len(vertices)), per_vertex)
    return np.float32(0), np.float32(0), np.float32(0), per_vertex


def patch_curvature_likelihood(*, principal_curvature_k1_mm_inv, principal_curvature_k2_mm_inv,
                               vertex_areas_mm2, patch_vertices, k1_histogram, k2_histogram,
                               joint_curvature_pdf, joint_origin_mm_inv,
                               joint_spacing_mm_inv) -> dict:
    """同几何主曲率/PDF→源 float32 qcurv_ll/qcll 及逐顶点联合 logP。

    principal_curvature_k1_mm_inv/k2 为同序 (N,) float32 mm⁻¹；复用已有
    topology_principal_curvature 的本轮结果，禁止跨坐标版本缓存。
    vertex_areas_mm2 (N,) 和 patch_vertices (K,) 同法向接口。
    k1_histogram/k2_histogram 是 bins/counts/bin_size 字典，bin_size
    单位 mm⁻¹，遵循最近整数索引。joint_curvature_pdf 是 (B1,B2) 或
    (B1,B2,1) 正 float32 联合 PDF，第一轴 k1、第二轴 k2；
    joint_origin_mm_inv 和 joint_spacing_mm_inv 是 (2,) float32起点及
    正间距，等于原 MRI xstart/ystart、xsize/ysize。联合图使用商截断
    后边界裁剪，不插值。返回 qcurv_ll（最终联合 logP 的均值）、qcll
    （联合 logP 的面积加权值）、marginal_qcurv_ll（源被覆盖中间值，
    含两次 k1 项，仅诊断）、(K,) float64 per_vertex_log_likelihood、
    vertex_count/scope。标量按原 TP float32 字段写回后转 Python float；
    空 patch 标量为0。非法/非有限输入或非正PDF/间距抛 ValueError；
    负顶点面积和负总面积按原权重保留。零分母 qcll 诊断保持 IEEE
    NaN/Inf，qcll_defined 与 qcll_float32_bits 单列；qcurv_ll、中间
    均值或逐顶点logP 非有限抛 FloatingPointError。候选评分不替代QC。
    没有可调算法阈值，不修改输入、不估计几何/主曲率或执行 GA。
    """
    k1 = np.asarray(principal_curvature_k1_mm_inv, np.float32)
    k2 = np.asarray(principal_curvature_k2_mm_inv, np.float32)
    if k1.ndim != 1 or k2.shape != k1.shape or not np.isfinite(k1).all() or not np.isfinite(k2).all():
        raise ValueError("matching finite principal-curvature vectors required")
    vertices = _patch_indices(patch_vertices, len(k1))
    areas = _areas(vertex_areas_mm2, len(k1), vertices)
    first1, counts1, spacing1 = _histogram(k1_histogram, "k1_histogram")
    first2, counts2, spacing2 = _histogram(k2_histogram, "k2_histogram")
    pdf = np.asarray(joint_curvature_pdf)
    if pdf.ndim == 3 and pdf.shape[2] == 1:
        pdf = pdf[:, :, 0]
    origin, spacing = np.asarray(joint_origin_mm_inv, np.float32), np.asarray(joint_spacing_mm_inv, np.float32)
    if (pdf.ndim != 2 or pdf.dtype != np.float32 or not all(pdf.shape)
            or not np.isfinite(pdf).all() or np.any(pdf <= 0)
            or origin.shape != (2,) or spacing.shape != (2,)
            or not np.isfinite(origin).all() or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise ValueError("positive float32 joint PDF, finite origin and positive spacing required")
    qcurv_ll, qcll, marginal, per_vertex = _curvature(np.ascontiguousarray(k1), np.ascontiguousarray(k2),
        areas, vertices, first1, spacing1, counts1, first2, spacing2, counts2,
        np.ascontiguousarray(pdf), origin, spacing)
    if not np.isfinite((qcurv_ll, marginal)).all() or not np.isfinite(per_vertex).all():
        raise FloatingPointError("nonfinite curvature likelihood")
    return {"qcurv_ll": float(qcurv_ll), "qcll": float(qcll),
            "qcll_defined": bool(np.isfinite(qcll)),
            "qcll_float32_bits": int(np.asarray(qcll, np.float32).view(np.uint32)),
            "marginal_qcurv_ll": float(marginal), "per_vertex_log_likelihood": per_vertex,
            "vertex_count": len(vertices), "scope": "curvature component only; not complete GA"}
