"""全局二阶曲面拟合的显式阶段接口；复用已验证的 VNL/SVD/EISPACK 内核。

调用方提供当前坐标、法向、rip 和原有序 vtotal 邻域。本模块不重建上游
几何，不读取参考曲率，不调用原版程序；不改变旧拓扑接口或生产默认。
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

from .topology_candidate_sff import _cross3, _dot3, _eigen2
from .topology_vnl_svd import add, div, f, mul, sub, svdc_3, svd_inverse_3

_EPS = np.float32(np.finfo(np.float32).eps)


@njit(cache=True, fastmath=False)
def _global_tangent(normal):
    """全局 source tangent：normal×[ny,nz,nx]，先叉乘再倒数乘法归一化。"""
    other = np.array([normal[1], normal[2], normal[0]], np.float32)
    first = _cross3(normal, other)
    # VectorLen adds its squares to a float32 zero in index order.
    squared = f(0)
    for j in range(3):
        squared = add(squared, mul(first[j], first[j]))
    if np.sqrt(squared) < .001:
        other[1] = f(-normal[2])
        first = _cross3(normal, other)
    second = _cross3(normal, first)
    for vector in (first, second):
        length = np.sqrt(_dot3(vector, vector))
        reciprocal = f(1) if abs(length) < _EPS else div(1, length)
        for j in range(3):
            vector[j] = mul(vector[j], reciprocal)
    return first, second


@njit(cache=True, fastmath=False, error_model="numpy")
def _global_fit(xyz, normals, offsets, neighbors, ripped, initial_k1, initial_k2):
    count = len(xyz)
    k1, k2 = initial_k1.copy(), initial_k2.copy()
    e1, e2 = np.empty_like(xyz), np.empty_like(xyz)
    # Source computes tangent planes for ripped/no-neighbor vertices too.
    for vertex in range(count):
        e1[vertex], e2[vertex] = _global_tangent(normals[vertex])
    conditions = np.full(count, np.nan, np.float32)
    kinds = np.zeros(count, np.int8)
    valid_counts = np.zeros(count, np.int32)
    retry_counts = np.zeros(count, np.int16)
    grams = np.zeros((count, 3, 3), np.float32)
    rhs_all = np.zeros((count, 3), np.float32)
    hessians = np.zeros((count, 2, 2), np.float32)
    for vertex in range(count):
        if ripped[vertex]:
            kinds[vertex], valid_counts[vertex] = -1, -1
            continue
        degree = offsets[vertex + 1] - offsets[vertex]
        if degree <= 0:
            continue
        first, second = e1[vertex].copy(), e2[vertex].copy()
        design = np.zeros((degree, 3), np.float32)
        heights = np.zeros(degree, np.float32)
        rounds = 0
        while True:
            minimum, maximum = f(10000), f(-10000)
            valid = 0
            for i in range(degree):
                neighbor = neighbors[offsets[vertex] + i]
                if ripped[neighbor]:
                    continue
                delta = np.empty(3, np.float32)
                for j in range(3):
                    delta[j] = sub(xyz[neighbor, j], xyz[vertex, j])
                u = np.float64(_dot3(delta, first))
                v = np.float64(_dot3(delta, second))
                # Source writes a row before validating rsq; invalid rows can
                # be overwritten by the next neighbor at the same row index.
                design[valid, 0] = f(u * u)
                design[valid, 1] = f(2. * u * v)
                design[valid, 2] = f(v * v)
                heights[valid] = _dot3(normals[vertex], delta)
                radius2 = f(u * u + v * v)
                if abs(radius2) >= _EPS and radius2 > 0:
                    local = div(heights[valid], radius2)
                    maximum = max(maximum, local)
                    minimum = min(minimum, local)
                    valid += 1
            rounds += 1
            # pct_thresh=-1 gives a fixed zero threshold. The source's
            # postincrement `niter++ > 100` stops after 102 attempts.
            if rounds >= 102 or valid >= 4:
                break
        valid_counts[vertex], retry_counts[vertex] = valid, rounds
        gram = np.zeros((3, 3), np.float32)
        for i in range(3):
            for j in range(3):
                for row in range(degree):
                    gram[i, j] = add(gram[i, j], mul(design[row, i], design[row, j]))
        grams[vertex] = gram
        zero = True
        for i in range(3):
            for j in range(3):
                if np.float64(abs(gram[i, j])) > 1.e-11:
                    zero = False
        if zero:
            # Global SFF explicitly resets m_eigen to identity here. The
            # candidate-SFF routine has different state semantics.
            conditions[vertex] = f(1.e10)
            k1[vertex], k2[vertex], kinds[vertex] = f(0), f(0), 3
            eigenvectors = np.eye(2, dtype=np.float32)
        else:
            _, singular, _ = svdc_3(gram)
            low, high = singular[0], singular[0]
            for j in range(1, 3):
                value = abs(singular[j])
                low, high = min(low, value), max(high, value)
            conditions[vertex] = f(1.e8) if abs(low) < _EPS else div(high, low)
            inverse = svd_inverse_3(gram)
            rhs = np.zeros(3, np.float32)
            coefficients = np.zeros(3, np.float32)
            for i in range(3):
                for row in range(degree):
                    rhs[i] = add(rhs[i], mul(design[row, i], heights[row]))
            rhs_all[vertex] = rhs
            for i in range(3):
                for j in range(3):
                    coefficients[i] = add(coefficients[i], mul(inverse[i, j], rhs[j]))
            q = np.empty((2, 2), np.float32)
            q[0, 0], q[0, 1] = mul(2, coefficients[0]), mul(2, coefficients[1])
            q[1, 0], q[1, 1] = q[0, 1], mul(2, coefficients[2])
            hessians[vertex] = q
            if conditions[vertex] >= 500000.:
                k1[vertex], k2[vertex], kinds[vertex] = maximum, minimum, 2
                continue
            values, eigenvectors = _eigen2(q)
            k1[vertex], k2[vertex], kinds[vertex] = values[0], values[1], 1
        for j in range(3):
            e1[vertex, j] = add(mul(first[j], eigenvectors[0, 0]),
                                 mul(second[j], eigenvectors[1, 0]))
            e2[vertex, j] = add(mul(first[j], eigenvectors[0, 1]),
                                 mul(second[j], eigenvectors[1, 1]))
    return k1, k2, e1, e2, conditions, kinds, valid_counts, retry_counts, grams, rhs_all, hessians


def global_second_fundamental_form(*, current_vertices, vertex_normals,
                                   neighbor_offsets, neighbors, vertex_rip,
                                   initial_k1, initial_k2) -> dict:
    """当前完整网格的 SFF 阶段，所有参数必填，输入不修改。

    current_vertices 转 float32(N,3)，当前 surface RAS/mm，非 orig 备份；
    vertex_normals 转 float32(N,3)，无量纲、由调用方实际前序提供。CSR 为
    原有序 vt.v[0:vtotal]：neighbor_offsets 整数(N+1)、neighbors 整数(E)，
    不排序、不由 faces 重建。vertex_rip 为bool/整数(N)，非零不拟合；
    initial_k1/k2 转 float32(N)，单位mm⁻¹，ripped或无邻点的初始值保留。
    输出完整k1/k2、tangent_e1/e2、condition_number、fit_kind(-1rip/0无邻点/
    1拟合/2病态回退/3零矩阵)、valid_neighbors、search_attempts，以及gram
    (N,3,3)mm⁴、rhs(N,3)mm³、hessian(N,2,2)mm⁻¹和含JIT/校验的seconds。
    上游投影/法向/拓扑不在本接口重算；只支持原pct_thresh=-1非平面流程。
    非法shape/CSR/索引/坐标或法向非有限报ValueError；SVD不收敛报RuntimeError。
    初始K可在未更新顶点非有限并原样保留，消费直方图时另行检查有限性。
    本阶段不改变线程/精度/CUDA，复用原有VNL/EISPACK，不依赖原FS二进制。
    对应MRIScomputeSecondFundamentalForm内部步骤，没有独立原软件CLI。
    本接口尚待完整真实before/after阶段验收，不作为旧默认的替代。
    """
    started = time.perf_counter()
    xyz, normal, k1, k2 = [np.ascontiguousarray(value, dtype=np.float32) for value in
                           (current_vertices, vertex_normals, initial_k1, initial_k2)]
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError("current_vertices must have shape (N,3)")
    n = len(xyz)
    rip = np.asarray(vertex_rip)
    if (normal.shape != xyz.shape or k1.shape != (n,) or k2.shape != (n,)
            or not np.isfinite(xyz).all() or not np.isfinite(normal).all()
            or rip.shape != (n,) or rip.dtype.kind not in "biu"):
        raise ValueError("same-order finite current geometry/normals and rip/K fields required")
    converted = []
    for value in (neighbor_offsets, neighbors):
        raw = np.asarray(value)
        if (raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < 0)
                or np.any(raw > np.iinfo(np.int32).max)):
            raise ValueError("original integer vtotal CSR required")
        converted.append(np.ascontiguousarray(raw, dtype=np.int32))
    offsets, adjacent = converted
    if (offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(adjacent)
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or (adjacent.size and adjacent.max() >= n)):
        raise ValueError("valid full same-order vtotal CSR required")
    values = _global_fit(xyz, normal, offsets, adjacent, np.ascontiguousarray(rip != 0), k1, k2)
    names = ("k1", "k2", "tangent_e1", "tangent_e2", "condition_number", "fit_kind",
             "valid_neighbors", "search_attempts", "gram", "rhs", "hessian")
    output = dict(zip(names, values))
    output.update(seconds=time.perf_counter() - started,
                  scope="prepared global SFF; actual pre-stage geometry/normal/vtotal required",
                  real_stage_validation="pending", production_default_changed=False)
    return output


def global_curvature_priors(*, current_vertices, vertex_normals, neighbor_offsets,
                            neighbors, vertex_rip, initial_k1, initial_k2) -> dict:
    """自产全局SFF→现有100×100曲率PDF；输入、空间与SFF接口相同。

    返回sff完整字段、histogram原接口字段及seconds。PDF只消费非rip的自产
    K1/K2，不能传入参考输出K代替计算。无可消费顶点、非有限消费K或源夹取
    范围零宽时抛ValueError，不用均匀PDF或删掉异常顶点。只生成曲率先验，
    不生成normal-dot/强度先验、MRI、候选或完整GA；无独立原软件CLI。
    """
    from .topology_curvature_histogram import curvature_histograms
    started = time.perf_counter()
    sff = global_second_fundamental_form(current_vertices=current_vertices,
        vertex_normals=vertex_normals, neighbor_offsets=neighbor_offsets, neighbors=neighbors,
        vertex_rip=vertex_rip, initial_k1=initial_k1, initial_k2=initial_k2)
    selected = np.asarray(vertex_rip) == 0
    first, second = sff["k1"][selected], sff["k2"][selected]
    if not len(first) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("at least one finite non-ripped curvature vertex required")
    for values in (first, second):
        start = np.maximum(np.float32(-3), np.min(values))
        end = np.minimum(np.float32(3), np.max(values))
        if np.float32(end - start) == 0:
            raise ValueError("source curvature histogram range has zero width")
    histogram = curvature_histograms(first, second)
    return dict(sff=sff, histogram=histogram, seconds=time.perf_counter() - started,
                scope="prepared global curvature prior only; not complete topology/GA")


@njit(cache=True, fastmath=False, error_model="numpy")
def _global_state_fields(k1, k2, kinds, initial_h, initial_gaussian, area):
    h, gaussian = initial_h.copy(), initial_gaussian.copy()
    kmin = hmin = f(10000)
    kmax = hmax = f(-10000)
    ktotal = f(0)
    total_area = np.float64(0)
    for vertex in range(len(k1)):
        kind = kinds[vertex]
        if kind <= 0:
            continue
        gaussian[vertex] = mul(k1[vertex], k2[vertex])
        h[vertex] = div(add(k1[vertex], k2[vertex]), 2)
        # The ill-conditioned branch updates vertex H/K then continues;
        # it does not enter MRIS summary accumulation or principal rotation.
        if kind == 2:
            continue
        if gaussian[vertex] < kmin:
            kmin = gaussian[vertex]
        if gaussian[vertex] > kmax:
            kmax = gaussian[vertex]
        if h[vertex] < hmin:
            hmin = h[vertex]
        if h[vertex] > hmax:
            hmax = h[vertex]
        contribution = np.float64(k1[vertex]) * np.float64(k2[vertex]) * np.float64(area[vertex])
        ktotal = f(np.float64(ktotal) + contribution)
        total_area += np.float64(area[vertex])
    return h, gaussian, kmin, kmax, hmin, hmax, ktotal, total_area


def global_second_fundamental_form_state(*, current_vertices, vertex_normals,
        neighbor_offsets, neighbors, vertex_rip, initial_k1, initial_k2,
        initial_h, initial_gaussian_curvature, vertex_area) -> dict:
    """全局SFF及源实际H/K与MRIS汇总写回；输入均必填，不修改调用方状态。

    前七个参数与global_second_fundamental_form相同。initial_h与
    initial_gaussian_curvature转float32(N)，分别为未更新顶点H/mm⁻¹和
    K/mm⁻²；vertex_area转float32(N)，是实际前序的有限面积/mm²，可为0。
    返回原SFF字段，另加h、gaussian_curvature float32(N)和surface_statistics：
    Kmin/Kmax/Hmin/Hmax/Ktotal均float32，total_area_mm2为float64诊断。
    rip/无邻域保留初始H/K；病态回退更新H/K但不计入surface汇总；其余按
    顶点序累加double(k1)*double(k2)*double(area)后逐次写回float32 Ktotal。
    初始H/K允许非有限，以保留未消费状态。curv/curvbak/valbak/法向/面积
    从未写回，调用方应保留这些原字段；不得用h或K自动覆盖它们。
    非法shape或非有限面积报ValueError，其余失败同SFF；seconds含校验、
    JIT和全函数，不含外部IO。对应MRIScomputeSecondFundamentalForm内部
    步骤，无独立CLI，真实完整阶段尚待before/after捕获，不切生产默认。
    """
    started = time.perf_counter()
    xyz = np.asarray(current_vertices)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError("current_vertices must have shape (N,3)")
    count = len(xyz)
    h, gaussian, area = [np.ascontiguousarray(value, dtype=np.float32) for value in
        (initial_h, initial_gaussian_curvature, vertex_area)]
    if (h.shape != (count,) or gaussian.shape != (count,) or area.shape != (count,)
            or not np.isfinite(area).all()):
        raise ValueError("same-order initial H/K and finite vertex_area (N) required")
    output = global_second_fundamental_form(current_vertices=current_vertices,
        vertex_normals=vertex_normals, neighbor_offsets=neighbor_offsets, neighbors=neighbors,
        vertex_rip=vertex_rip, initial_k1=initial_k1, initial_k2=initial_k2)
    fields = _global_state_fields(output["k1"], output["k2"], output["fit_kind"], h, gaussian, area)
    output["h"], output["gaussian_curvature"] = fields[:2]
    output["surface_statistics"] = dict(zip(
        ("Kmin", "Kmax", "Hmin", "Hmax", "Ktotal", "total_area_mm2"),
        [np.float32(value) for value in fields[2:7]] + [np.float64(fields[7])]))
    output["unmodified_state_fields"] = ("curv", "curvbak", "valbak", "vertex_normals", "vertex_area")
    output["seconds"] = time.perf_counter() - started
    return output
