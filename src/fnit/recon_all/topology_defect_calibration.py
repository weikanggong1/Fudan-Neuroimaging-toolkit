"""复用完整拓扑强度准备和校准直方图，生成自有局部 MRI/logP。

网格识别、候选边生成、重连、配准和 GA 控制不在本模块内。两个网格
索引显式区分：原 nofix 顶点用于采样/校准，corrected 顶点用于局部边表。
"""
from __future__ import annotations

import time
import numpy as np

from .topology_edge_mri_score import prepare_topology_edge_inputs
from .topology_intensity_histogram import first_defect_intensity_histograms
from .topology_defect_volume_priors import defect_volume_priors


def calibrated_defect_volume_priors(*, vertices, faces, defect_vertices,
                                   defect_hull, candidate_edges,
                                   vertex_translation, volume, ras_to_voxel,
                                   volume_resolution=-1) -> dict:
    """原 nofix/MRI/缺陷状态→自产白灰校准及完整局部体积，不读参考 PDF。

    vertices 为有限 float32(N,3) surface RAS/mm，faces 为原有序整数
    (F,3)。defect_vertices 是 bool(N,) 全部缺陷标记，defect_hull 是
    当前缺陷凸壳的有序原顶点整数(H,)，不能用整个半球代替。候选完整
    candidate_edges(E,2) 采用 corrected 索引；vertex_translation(N,)
    是原顶点→corrected 的整数全排列，来自上游自有 base translation。
    volume 是归一化 MRI uint8(X,Y,Z)，ras_to_voxel 是该 MRI 的
    float32(4,4) surface RAS→连续体素变换。volume_resolution 默认-1
    对应局部索引密度2 mm^-1；也接受明确正数，沿用 priors 的原规则。

    复用原法向、±0.5mm采样、排除缺陷的两轮中值、两轮原坐标平均；
    白灰直方图采用原校准区域及 sigma=2，bins 为0..255、bin_size=0。
    不输入局部参考卷或校准 PDF，不重算/改变边表与顶点排列。

    返回 {preparation, calibration, priors, stage_seconds, scope}：
    preparation 的 original_vertices/original_normals 为原 nofix 顺序(N,3)，
    white_values/gray_values 为原顺序 float32(N,)；calibration 包含
    selected、白灰 raw/smooth float32(256,)、bins/bin_size、均值；
    priors 按 corrected 边表生成 XYZ uint8局部卷、float32白灰 logP、
    几何及统计，结构与 defect_volume_priors 完全一致。stage_seconds
    含各函数校验/计算，不含导入和调用者读写。非法排列、索引、空校准
    区域或非有限产物明确失败；不修改输入、线程、精度或设备状态。
    这不是完整 GA，也不宣称最终网格质量或整例指标等效。
    """
    started = time.perf_counter()
    xyz = np.asarray(vertices)
    triangles = np.asarray(faces)
    marked = np.asarray(defect_vertices)
    hull = np.asarray(defect_hull)
    translation = np.asarray(vertex_translation)
    edges = np.asarray(candidate_edges)
    if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3
            or not np.isfinite(xyz).all() or len(xyz) == 0
            or marked.shape != (len(xyz),) or marked.dtype != np.bool_
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.size == 0 or triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError('finite float32 nofix mesh and bool all-defect mask required')
    if (hull.ndim != 1 or not np.issubdtype(hull.dtype, np.integer) or len(hull) == 0
            or hull.min() < 0 or hull.max() >= len(xyz)):
        raise ValueError('nonempty ordered original-index defect_hull required')
    if (translation.shape != (len(xyz),)
            or not np.issubdtype(translation.dtype, np.integer)
            or not np.array_equal(np.sort(translation), np.arange(len(xyz)))):
        raise ValueError('vertex_translation must map original vertices to a complete corrected permutation')
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or len(edges) == 0 or edges.min() < 0 or edges.max() >= len(xyz)):
        raise ValueError('complete corrected-index candidate_edges required')
    prepared = prepare_topology_edge_inputs(vertices=xyz, faces=triangles,
        defect_vertices=marked, volume=volume, ras_to_voxel=ras_to_voxel)
    preparation_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    # The mature selector uses the source's nested two two-ring union; empty
    # calibration is invalid before division. Source counters are integer-exact.
    from .topology_intensity_histogram import histogram_vertices
    selected = histogram_vertices(hull.tolist(), prepared['neighbors'], marked)
    if len(selected) == 0:
        raise ValueError('defect calibration contains no unmarked vertices')
    calibration = first_defect_intensity_histograms(
        white=prepared['white_values'], gray=prepared['gray_values'],
        hull=hull.tolist(), neighbors=prepared['neighbors'], marked=marked)
    bins = np.arange(256, dtype=np.float32)
    calibration.update(bins=bins, bin_size=np.float32(0))
    if not all(np.isfinite(calibration[name]).all() for name in (
            'white_raw', 'gray_raw', 'white_smooth', 'gray_smooth')):
        raise FloatingPointError('nonfinite independently calibrated histograms')
    calibration_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    # The creation volume reads corrected origXYZ; the same permutation must
    # be used as the complete candidate edges, never inferred by geometry.
    reverse = np.argsort(translation)
    priors = defect_volume_priors(
        original_vertices=np.ascontiguousarray(prepared['original_vertices'][reverse]),
        candidate_edges=edges, volume=volume, ras_to_voxel=ras_to_voxel,
        white_histogram_bins=bins, white_histogram_counts=calibration['white_smooth'],
        white_histogram_bin_size=calibration['bin_size'],
        gray_histogram_bins=bins, gray_histogram_counts=calibration['gray_smooth'],
        gray_histogram_bin_size=calibration['bin_size'], volume_resolution=volume_resolution)
    prior_seconds = time.perf_counter() - tick
    return dict(preparation=prepared, calibration=calibration, priors=priors,
        stage_seconds=dict(preparation=preparation_seconds, calibration=calibration_seconds,
                           priors=prior_seconds, complete_api=time.perf_counter() - started),
        scope='complete nofix/MRI gray-white calibration and local priors; not topology search or full GA')
