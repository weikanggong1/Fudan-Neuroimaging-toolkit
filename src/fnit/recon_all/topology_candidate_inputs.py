"""拓扑候选的每表面准备与每缺陷局部输入；复用成熟自产原语。"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import time

import numpy as np

from .topology_edge_mri_score import prepare_topology_edge_inputs, _validated_volume
from .topology_intensity_histogram import first_defect_intensity_histograms, histogram_vertices
from .topology_curvature_histogram import _source_mean
from .topology_mri_likelihood import joint_gray_white_pdf
from .topology_defect_volume_priors import defect_volume_priors


def _owned(array):
    value = np.array(array, copy=True, order="C")
    value.flags.writeable = False
    return value


def _histogram(bins, counts, bin_size, name):
    x, p = np.asarray(bins), np.asarray(counts)
    width = np.asarray(bin_size)
    if (x.dtype != np.float32 or x.shape != (100,) or p.dtype != np.float32
            or p.shape != (100,) or not np.isfinite(x).all()
            or not np.isfinite(p).all() or np.any(p <= 0) or width.ndim
            or width.dtype.kind not in "fiu"
            or not np.isfinite(width) or width <= 0):
        raise ValueError("complete finite float32 100-bin " + name + " PDF required")
    return MappingProxyType(dict(bins=_owned(x), counts=_owned(p),
                                 bin_size=np.float32(width)))


@dataclass(frozen=True)
class CandidateInputContext:
    """每个原始表面一次的只读自产准备；corrected 当前坐标不缓存于此。"""
    preparation: Mapping
    defect_vertices: np.ndarray
    vertex_translation: np.ndarray
    volume: np.ndarray
    ras_to_voxel: np.ndarray
    gray_white_pdf: np.ndarray
    normal_dot_histogram: Mapping
    k1_histogram: Mapping
    k2_histogram: Mapping
    joint_curvature_pdf: np.ndarray
    joint_origin_mm_inv: np.ndarray
    joint_spacing_mm_inv: np.ndarray
    k1_mean: np.float32
    k2_mean: np.float32
    seconds: float


def prepare_candidate_input_context(*, vertices, faces, defect_vertices,
                                    vertex_translation, volume, ras_to_voxel,
                                    global_priors) -> CandidateInputContext:
    """原 nofix/MRI及自产全局先验→每表面一次的采样/中值/joint-PDF准备。

    vertices为finite float32(N,3) surface RAS/mm raw nofix，faces为有序
    整数(F,3)，defect_vertices为bool(N,)全部缺陷mask，vertex_translation
    为原编号→corrected完整整数(N,)排列。volume为uint8(X,Y,Z)，
    ras_to_voxel为(4,4)拓扑MRI-only tkregister RAS→连续体素变换。
    global_priors是global_topology_priors自产结果，至少包含curvature和
    normal_dot的bins/counts/spacing与联合PDF；不接受参考字段替代。
    七参必填，无线程/设备/精度开关，复用原采样、两次median、ORIGINAL
    平均与joint_gray_white_pdf；不重复拟合SFF或生成全局先验。

    返回只读CandidateInputContext。preparation为原域(N,3)ORIGINAL/
    normal、float32(N,)white/gray、ripped及有序邻域；gray_white_pdf为
    float32(256,256,1)。三个100-bin PDF与联合float32(100,100,1)及
    origin/spacing(mm^-1)复制为只读。k1_mean/k2_mean由首次全局K先验
    的源1%-peak F32 mean计算，非从SFF2顶点数组求平均。seconds包含
    校验/准备/JIT/copy，不含导入、外部文件读写。

    shape/dtype/全排列/非有限/空消费/非正PDF与零曲率mean报ValueError。
    不改变调用方数组、线程、CUDA/TF32；没有原程序、参考读取或完整GA。
    对应mrisComputeJointGrayWhiteBorderDistributions及内部统计准备，
    无独立官方CLI。每缺陷需另传当前corrected坐标，不能用本缓存重置。
    """
    started = time.perf_counter()
    xyz, triangles = np.asarray(vertices), np.asarray(faces)
    marked, translation = np.asarray(defect_vertices), np.asarray(vertex_translation)
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or xyz.dtype != np.float32
            or len(xyz) == 0 or not np.isfinite(xyz).all()
            or marked.shape != (len(xyz),) or marked.dtype != bool
            or triangles.ndim != 2 or triangles.shape[1:] != (3,)
            or triangles.dtype.kind not in "iu" or not len(triangles)
            or triangles.min() < 0 or triangles.max() >= len(xyz)
            or translation.shape != (len(xyz),) or translation.dtype.kind not in "iu"
            or not np.array_equal(np.sort(translation), np.arange(len(xyz)))):
        raise ValueError("same original-domain nofix, all-defect mask and complete mapping required")
    if not isinstance(global_priors, Mapping):
        raise ValueError("explicit self-generated global priors required")
    curvature, dot = global_priors.get("curvature"), global_priors.get("normal_dot")
    if not isinstance(curvature, Mapping) or not isinstance(dot, Mapping):
        raise ValueError("complete curvature/normal_dot mappings required")
    first = _histogram(curvature.get("k1_bins"), curvature.get("k1_counts"),
                       curvature.get("k1_width"), "K1")
    second = _histogram(curvature.get("k2_bins"), curvature.get("k2_counts"),
                        curvature.get("k2_width"), "K2")
    normal = _histogram(dot.get("bins"), dot.get("counts"), dot.get("bin_size"), "dot")
    joint = np.asarray(curvature.get("joint"))
    if (joint.dtype != np.float32 or joint.shape != (100, 100)
            or not np.isfinite(joint).all() or np.any(joint <= 0)):
        raise ValueError("complete positive float32 joint curvature PDF required")
    origin = np.asarray([curvature.get("k1_start"), curvature.get("k2_start")], np.float32)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("finite explicit joint PDF origin required")
    means = [_source_mean(h["bins"], h["counts"], h["bin_size"]) for h in (first, second)]
    if not np.isfinite(means).all() or np.any(np.asarray(means) == 0):
        raise ValueError("finite nonzero source K means required for matching radii")
    image, matrix = _validated_volume(volume, ras_to_voxel)
    prepared = prepare_topology_edge_inputs(vertices=xyz, faces=triangles,
        defect_vertices=marked, volume=image, ras_to_voxel=matrix)
    pdf = joint_gray_white_pdf(white_values=prepared["white_values"],
        gray_values=prepared["gray_values"], excluded_vertices=marked)
    frozen = {name: _owned(value) for name, value in prepared.items() if name != "neighbors"}
    frozen["neighbors"] = tuple(tuple(int(v) for v in row) for row in prepared["neighbors"])
    return CandidateInputContext(preparation=MappingProxyType(frozen),
        defect_vertices=_owned(marked), vertex_translation=_owned(translation),
        volume=_owned(image), ras_to_voxel=_owned(matrix), gray_white_pdf=_owned(pdf),
        normal_dot_histogram=normal, k1_histogram=first, k2_histogram=second,
        joint_curvature_pdf=_owned(joint[:, :, None]), joint_origin_mm_inv=_owned(origin),
        joint_spacing_mm_inv=_owned(np.asarray([first["bin_size"], second["bin_size"]], np.float32)),
        k1_mean=means[0], k2_mean=means[1], seconds=time.perf_counter()-started)


def prepare_defect_candidate_inputs(*, context, original_defect_hull,
                                   candidate_edges, current_corrected_original_vertices,
                                   volume_resolution=-1, mri_weight=1.,
                                   volume_unmri_weight=10., normal_dot_weight=1.,
                                   quadratic_curvature_weight=1.) -> dict:
    """一个defect的当前corrected状态→校准/local-MRI与候选评分公共参数。

    context为自产prepare_candidate_input_context返回值。original_defect_hull
    为有序原编号整数(H,)，candidate_edges是创建时完整corrected编号
    整数(E,2)ETable，不是已接受面边。current_corrected_original_vertices
    为float32(N,3)当前corrected orig surface RAS/mm；必填，保留前面
    缺陷最终几何更新，不从原域prepared按translation重置。volume_resolution
    默认-1（2mm^-1）；其他值按成熟defect_volume_priors接口验证。四个
    原double权重默认MRI1/unMRI10/normal1/quadratic1，允许有限负数。

    原域校准复用cached white/gray与source hull邻域，局部uint8强度卷/
    float32 logP/mean由当前corrected坐标和ETable自产，一defect计算一次。
    返回candidate_kwargs可展开给evaluate_candidate，或作为
    evaluate_candidate_cycle的candidate_inputs字典传入，含
    原volume/ras_to_voxel、local_priors、全局joint gray-white、三个hist、联合curvaturePDF/
    origin/spacing、Kmeans及weights；calibration另含selected与256bin
    raw/smooth/bins/bin_size，stage_seconds含校验/JIT/校准/局部写数组。
    每个candidate复用这些只读全局字段及本defect局部字段，不重做准备。

    非context/原hull越界/空校准/非法edges/current坐标/非有限weights报
    ValueError；子函数原异常保留。输入不变，无隐藏磁盘缓存或参考读取，
    不改线程/TF32、无原FS运行；完整GA/最终QC由调用方处理。对应内部
    mrisComputeGrayWhiteBorderDistributions、computeDefectStatistics及
    mriDefectVolume/defectVolumeLikelihood，无独立原软件CLI。
    """
    started = time.perf_counter()
    if not isinstance(context, CandidateInputContext):
        raise ValueError("self-generated CandidateInputContext required")
    n = len(context.vertex_translation)
    hull, edges, current = (np.asarray(original_defect_hull), np.asarray(candidate_edges),
                            np.asarray(current_corrected_original_vertices))
    weights = np.asarray([mri_weight, volume_unmri_weight, normal_dot_weight,
                          quadratic_curvature_weight], np.float64)
    if (hull.ndim != 1 or hull.dtype.kind not in "iu" or len(hull) == 0
            or hull.min() < 0 or hull.max() >= n or edges.ndim != 2
            or edges.shape[1:] != (2,) or edges.dtype.kind not in "iu"
            or len(edges) == 0 or edges.min() < 0 or edges.max() >= n
            or current.shape != (n, 3) or current.dtype != np.float32
            or not np.isfinite(current).all() or not np.isfinite(weights).all()):
        raise ValueError("ordered original hull, complete corrected edges/current state and finite weights required")
    tick = time.perf_counter()
    prepared = context.preparation
    selected = histogram_vertices(hull.tolist(), prepared["neighbors"], context.defect_vertices)
    if not len(selected):
        raise ValueError("defect calibration has no unmarked original vertices")
    calibrated = first_defect_intensity_histograms(white=prepared["white_values"],
        gray=prepared["gray_values"], hull=hull.tolist(), neighbors=prepared["neighbors"],
        marked=context.defect_vertices)
    bins = np.arange(256, dtype=np.float32)
    calibrated.update(bins=bins, bin_size=np.float32(0))
    calibration_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    local = defect_volume_priors(original_vertices=current, candidate_edges=edges,
        volume=context.volume, ras_to_voxel=context.ras_to_voxel,
        white_histogram_bins=bins, white_histogram_counts=calibrated["white_smooth"],
        white_histogram_bin_size=calibrated["bin_size"], gray_histogram_bins=bins,
        gray_histogram_counts=calibrated["gray_smooth"],
        gray_histogram_bin_size=calibrated["bin_size"], volume_resolution=volume_resolution)
    prior_seconds = time.perf_counter()-tick
    kwargs = dict(volume=context.volume, ras_to_voxel=context.ras_to_voxel,
        local_priors=local, gray_white_pdf=context.gray_white_pdf,
        k1_mean=context.k1_mean, k2_mean=context.k2_mean,
        normal_dot_histogram=context.normal_dot_histogram,
        k1_histogram=context.k1_histogram, k2_histogram=context.k2_histogram,
        joint_curvature_pdf=context.joint_curvature_pdf,
        joint_origin_mm_inv=context.joint_origin_mm_inv,
        joint_spacing_mm_inv=context.joint_spacing_mm_inv,
        mri_weight=float(weights[0]), volume_unmri_weight=float(weights[1]),
        normal_dot_weight=float(weights[2]), quadratic_curvature_weight=float(weights[3]))
    return dict(candidate_kwargs=kwargs, calibration=calibrated,
        stage_seconds=dict(calibration=calibration_seconds, local_priors=prior_seconds,
                           complete_api=time.perf_counter()-started),
        source_phase="immutable original calibration; current corrected per-defect MRI",
        complete_ga=False, production_default_changed=False)
