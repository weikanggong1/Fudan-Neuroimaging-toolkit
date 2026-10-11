"""首次自产 SFF 与平滑 ORIGINAL 的全局拓扑先验组合，不重做拟合。"""
from __future__ import annotations

import time

import numpy as np

from .normal_pending_python import repair_vertex_normals
from .topology_curvature_histogram import curvature_histograms
from .topology_global_dot_prior import global_normal_dot_prior


def global_topology_priors(*, principal_curvature_fields,
                          smoothed_original_vertices, faces,
                          neighbor_offsets, neighbors, vertex_rip,
                          surface_status="sphere") -> dict:
    """首次 SFF1 自产 K 与两轮平滑 ORIGINAL→完整 K/dot 先验。

    principal_curvature_fields 是上游 global SFF1 返回的字典，必含
    float32(N,) k1/k2（mm^-1）；不接受后续 SFF2 或参考输出代替首次拟合。
    smoothed_original_vertices 为有限 float32(N,3) surface RAS/mm，
    faces 为原有序整数(F,3)；二者必须仍在同一原始域，不能用 corrected
    base 逆推。neighbor_offsets/ neighbors 为原有序 vnum 一环 CSR，
    整数(N+1)/整数(E)；不将 SFF 的 vtotal 二环当作一环。vertex_rip
    为 bool/整数(N)，只跳过直方图中心，源 dot 仍消费中心的 ripped 邻点。
    六个数据参数必填。surface_status 默认 sphere，传给成熟 normal
    pending 核；支持该核声明的非平面/平面状态，不改线程或精度。

    返回 curvature（两个100-bin PDF、100×100联合PDF及原元数据）、
    normal_dot（100-bin平滑PDF及每有向边诊断）、smoothed_metric
    （完整 NormalRepairResult，坐标mm、面积mm²、法向无量纲）和 seconds。
    seconds 含校验、metric/JIT和两项先验；step_seconds 分列各子项。
    正常输入不原地修改。源五轮退化修复的实际坐标变化、未解决顶点及
    各轮 pending 通过 smoothed_metric 返回；dot 使用这个实际状态，
    不静默丢掉位移或额外重算法向。调用方的 raw/TMP/ORIGINAL 槽保持不变，
    本函数不代替随后 raw metric 或原 mris 的 SFF2，也不向 clone 回写 SFF2。

    格式/索引/空消费/非有限消费K或零宽范围抛 ValueError；normal/dot
    的异常沿用原接口，源100-bin越界明确失败。rip/无邻点未更新K可在
    被跳过中心保留非有限初值。该API只组合已自产的阶段输入，不生成
    MRI、raw SFF或GA；对应两个内部distribution函数，无独立官方CLI。
    """
    started = time.perf_counter()
    xyz = np.asarray(smoothed_original_vertices)
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or xyz.dtype != np.float32
            or len(xyz) == 0 or not np.isfinite(xyz).all()):
        raise ValueError("finite float32 smoothed ORIGINAL (N,3) required")
    n = len(xyz)
    rip = np.asarray(vertex_rip)
    if rip.shape != (n,) or rip.dtype.kind not in "biu":
        raise ValueError("same-order bool/integer vertex_rip (N,) required")
    if not isinstance(principal_curvature_fields, dict):
        raise ValueError("self-generated first-SFF k1/k2 fields required")
    values = []
    for name in ("k1", "k2"):
        raw = np.asarray(principal_curvature_fields.get(name))
        if raw.shape != (n,) or raw.dtype != np.float32:
            raise ValueError("same-order float32 first-SFF k1/k2 (N,) required")
        values.append(raw[rip == 0])
    if not len(values[0]) or any(not np.isfinite(value).all() for value in values):
        raise ValueError("at least one finite non-ripped curvature vertex required")
    for value in values:
        low = np.maximum(np.float32(-3), np.min(value))
        high = np.minimum(np.float32(3), np.max(value))
        if np.float32(high-low) == 0:
            raise ValueError("source curvature histogram range has zero width")
    curvature_start = time.perf_counter()
    curvature = curvature_histograms(*values)
    curvature_seconds = time.perf_counter()-curvature_start
    metric_start = time.perf_counter()
    metric = repair_vertex_normals(positions=xyz, faces=faces,
        status=surface_status, unitize=True, fix_vertex_area=True,
        ripped=np.ascontiguousarray(rip != 0))
    metric_seconds = time.perf_counter()-metric_start
    dot = global_normal_dot_prior(smoothed_original_vertices=metric.positions,
        metric_vertex_normals=metric.normals, neighbor_offsets=neighbor_offsets,
        neighbors=neighbors, vertex_rip=rip)
    return dict(curvature=curvature, normal_dot=dot, smoothed_metric=metric,
        seconds=time.perf_counter()-started,
        step_seconds=dict(curvature_pdf=curvature_seconds,
            smoothed_original_metric=metric_seconds, normal_dot_pdf=dot["seconds"]),
        source_phase="raw mixed-normal SFF1 -> smoothed ORIGINAL metric/vnum priors",
        restored_raw_metric_included=False, original_sff2_included=False,
        complete_ga=False, production_default_changed=False)
