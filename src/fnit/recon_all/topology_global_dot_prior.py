"""全局 normal-dot 曲率先验；消费平滑 orig 和该阶段 metric 后法向。"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

from .topology_vnl_svd import add, div, f, mul, sub
from .topology_intensity_histogram import _smooth_histogram


@njit(cache=True, fastmath=False)
def _projected_dots(xyz, normals, offsets, neighbors, ripped):
    values=np.full(len(neighbors),np.nan,np.float32)
    consumed=np.zeros(len(neighbors),np.bool_)
    for vertex in range(len(xyz)):
        if ripped[vertex]:continue
        for slot in range(offsets[vertex],offsets[vertex+1]):
            other=neighbors[slot]
            dx=sub(xyz[other,0],xyz[vertex,0])
            dy=sub(xyz[other,1],xyz[vertex,1])
            dz=sub(xyz[other,2],xyz[vertex,2])
            # This prior's pinned expression differs from the three-term
            # candidate log-likelihood projection. Keep the original order.
            yz=mul(mul(mul(dy,normals[vertex,1]),dz),normals[vertex,2])
            values[slot]=add(mul(dx,normals[vertex,0]),yz)
            consumed[slot]=True
    return values,consumed


@njit(cache=True, fastmath=False, error_model="numpy")
def _bins_and_counts(values,consumed):
    low,high=f(100000),f(-100000)
    num=0
    for slot in range(len(values)):
        if not consumed[slot]:continue
        low=min(low,values[slot]);high=max(high,values[slot]);num+=1
    spacing=div(sub(high,low),98)
    if not np.isfinite(spacing) or spacing<=0:
        raise ValueError("source dot-prior range must have positive finite width")
    bins=np.empty(100,np.float32)
    value=sub(low,spacing)
    for i in range(100):
        bins[i]=value;value=add(value,spacing)
    counts=np.zeros(100,np.float32)
    indices=np.full(len(values),-1,np.int32)
    for slot in range(len(values)):
        if not consumed[slot]:continue
        index=int(div(sub(values[slot],bins[0]),spacing))
        index=min(100,max(0,index))
        # The source allows index==nbins although storage is nbins long.
        # Explicitly fail rather than copying undefined memory writes.
        if index>=100:raise ValueError("source dot-prior index exceeds allocated 100 bins")
        indices[slot]=index;counts[index]=add(counts[index],1)
    for i in range(100):
        if counts[i]==0:counts[i]=f(.01)
        counts[i]=div(counts[i],num)
    return bins,counts,spacing,indices,low,high,num


def global_normal_dot_prior(*, smoothed_original_vertices, metric_vertex_normals,
                            neighbor_offsets, neighbors, vertex_rip) -> dict:
    """平滑ORIGINAL及metric后法向→完整100bin原normal-dot先验，输入不修改。

    smoothed_original_vertices 转float32(N,3)，surface RAS/mm；不是raw
    current或canonical。metric_vertex_normals转float32(N,3)，无量纲，必须
    对应该平滑网格执行实际metric后得到。neighbor_offsets整数(N+1)、
    neighbors整数(E)是原有序vt.v[0:vnum]一环，不是SFF的vtotal。vertex_rip
    bool/整数(N)，仅跳过中心rip；原邻点rip仍被消费。所有参数均必填。

    返回bins/counts/raw_counts float32(100)、bin_size/min/max float32，
    sampled_edges整数、projected_values float32(E)、consumed bool(E)、
    bin_indices int32(E)；未消费边值NaN、bin_index=-1。counts复用成熟
    sigma=2/HISTOsmooth，raw_counts是加0.01地板并按边数归一化的原频率。
    原F32表达式dx*nx + dy*ny*dz*nz及截断分bin被保留，不改候选likelihood；
    此legacy表达式混合mm与mm²项，不将其数值冒称三维投影距离(mm)。
    seconds含输入校验和首次JIT，不含外部读写。无上游metric/球面/MRI计算，
    不调用原版程序，不改变线程/TF32/CUDA。格式/索引、空消费、非有限产物、
    零宽或源index==100越界报ValueError/FloatingPointError，不输出占位PDF。
    对应mrisComputeNormalDotDistribution内部步骤，无独立原软件CLI。
    本接口尚待真实阶段捕获，不改变旧默认或宣称完整GA。
    """
    started=time.perf_counter()
    xyz=np.ascontiguousarray(smoothed_original_vertices,dtype=np.float32)
    normals=np.ascontiguousarray(metric_vertex_normals,dtype=np.float32)
    if xyz.ndim!=2 or xyz.shape[1:]!=(3,):raise ValueError("smoothed_original_vertices (N,3) required")
    n=len(xyz);rip=np.asarray(vertex_rip)
    if (normals.shape!=xyz.shape or not np.isfinite(xyz).all() or not np.isfinite(normals).all()
            or rip.shape!=(n,) or rip.dtype.kind not in "biu"):
        raise ValueError("same-order finite smoothed ORIGINAL/metric normal and rip fields required")
    converted=[]
    for value in (neighbor_offsets,neighbors):
        raw=np.asarray(value)
        if (raw.ndim!=1 or raw.dtype.kind not in "iu" or np.any(raw<0)
                or np.any(raw>np.iinfo(np.int32).max)):
            raise ValueError("original ordered integer vnum CSR required")
        converted.append(np.ascontiguousarray(raw,dtype=np.int32))
    offsets,adjacent=converted
    if (offsets.shape!=(n+1,) or offsets[0]!=0 or offsets[-1]!=len(adjacent)
            or np.any(np.diff(offsets.astype(np.int64))<0)
            or (adjacent.size and adjacent.max()>=n)):
        raise ValueError("valid full ordered vnum CSR required")
    values,consumed=_projected_dots(xyz,normals,offsets,adjacent,np.ascontiguousarray(rip!=0))
    if not consumed.any():raise ValueError("at least one non-ripped center edge required")
    if not np.isfinite(values[consumed]).all():raise FloatingPointError("nonfinite source dot-prior projection")
    bins,raw,spacing,indices,low,high,num=_bins_and_counts(values,consumed)
    counts=_smooth_histogram(raw)
    return dict(bins=bins,counts=counts,raw_counts=raw,bin_size=np.float32(spacing),
        minimum_projection=np.float32(low),maximum_projection=np.float32(high),sampled_edges=int(num),
        projected_values=values,consumed=consumed,bin_indices=indices,
        projection_units="source legacy numeric expression combines mm and mm^2 terms",
        seconds=time.perf_counter()-started,
        scope="prepared smoothed-ORIGINAL/metric/vnum prior; not raw-SFF K prior or GA",
        real_stage_validation="pending",production_default_changed=False)
