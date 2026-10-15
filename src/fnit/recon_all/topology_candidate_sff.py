"""候选 patch 的完整二阶曲面拟合；不运行原版程序，不生成上游法向。

复用现有 source-order VNL float32 3×3 SVD。2×2 对称 Hessian 按
VNL 的 double EISPACK 运算边界求解，再写回 float32 主曲率和切向量。
这是拓扑候选的内部阶段，不能直接用初始 qsphere 主曲率接口替代。
"""
from __future__ import annotations

import time
import numpy as np
from numba import njit

from .topology_vnl_svd import add, div, f, mul, sub, svdc_3, svd_inverse_3

_EPS = np.float32(np.finfo(np.float32).eps)


@njit(cache=True, fastmath=False)
def _dot3(a, b):
    return add(add(mul(a[0], b[0]), mul(a[1], b[1])), mul(a[2], b[2]))


@njit(cache=True, fastmath=False)
def _cross3(a, b):
    c = np.empty(3, np.float32)
    c[0] = sub(mul(a[1], b[2]), mul(a[2], b[1]))
    c[1] = sub(mul(a[2], b[0]), mul(a[0], b[2]))
    c[2] = sub(mul(a[0], b[1]), mul(a[1], b[0]))
    return c


@njit(cache=True, fastmath=False)
def _tangent(normal):
    axis = np.array([1., 0., 0.], np.float32)
    c = _cross3(normal, axis)
    squared = _dot3(c, c)
    if abs(squared) < _EPS:
        axis[0], axis[1] = f(0), f(1)
        c = _cross3(normal, axis)
        squared = _dot3(c, c)
    length = np.sqrt(squared)
    if abs(length) < _EPS:
        length = f(1)
    first = np.empty(3, np.float32)
    for j in range(3):
        first[j] = div(c[j], length)
    # The second cross product consumes c before normalization.
    second = _cross3(normal, c)
    length = np.sqrt(_dot3(second, second))
    if abs(length) < _EPS:
        length = f(1)
    for j in range(3):
        second[j] = div(second[j], length)
    return first, second


@njit(cache=True, fastmath=False)
def _stable_norm2(a, b):
    """Double EISPACK stable norm; preserve its ordered refinement."""
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("nonfinite candidate SFF eigen norm input")
    high = max(abs(a), abs(b))
    if high == 0.:
        return high
    ratio = min(abs(a), abs(b)) / high
    square = ratio * ratio
    while True:
        total = square + 4.
        if total == 4.:
            return high
        fraction = square / total
        scale = fraction * 2. + 1.
        high = scale * high
        ratio = fraction / scale
        square = ratio * ratio * square


@njit(cache=True, fastmath=False)
def _eigen2(q):
    """Specialized symmetric 2×2 EISPACK QL, then source abs ordering."""
    if not np.isfinite(q).all():
        raise ValueError("nonfinite candidate SFF Hessian")
    d = np.array([np.float64(q[0, 0]), np.float64(q[1, 1])])
    e = np.array([np.float64(q[1, 0]), 0.])
    vectors = np.eye(2, dtype=np.float64)
    offset = 0.
    test = 0.
    for left in range(2):
        size = abs(d[left]) + abs(e[left])
        if test < size:
            test = size
        right = left
        while True:
            # The two-element EISPACK sentinel must remain finite. Guard the
            # index before reading it, including after a failed rotation.
            if right >= len(e):
                raise ValueError("candidate SFF eigen scan exceeds two-element domain")
            if not np.isfinite(test) or not np.isfinite(e[right]):
                raise ValueError("nonfinite candidate SFF eigen scan")
            if test + abs(e[right]) == test:
                break
            right += 1
        if right != left:
            old = d[left]
            p = (d[left + 1] - old) / (e[left] * 2.)
            norm = _stable_norm2(p, 1.)
            signed = norm if p >= 0. else -norm
            d[left] = e[left] / (p + signed)
            d[left + 1] = e[left] * (p + signed)
            next_diag = d[left + 1]
            offset += old - d[left]
            p = d[right]
            c = 1.
            s = 0.
            next_e = e[left + 1]
            for i in range(right - 1, left - 1, -1):
                previous_c = c
                previous_s = s
                g = c * e[i]
                h = c * p
                norm = _stable_norm2(p, e[i])
                e[i + 1] = s * norm
                s = e[i] / norm
                c = p / norm
                p = c * d[i] - s * g
                d[i + 1] = h + s * (c * g + s * d[i])
                for row in range(2):
                    h = vectors[row, i + 1]
                    vectors[row, i + 1] = s * vectors[row, i] + c * h
                    vectors[row, i] = c * vectors[row, i] - s * h
            # For n=2 the loop has one rotation and next_e==0.
            p = -s * previous_s * previous_c * next_e * e[left] / next_diag
            e[left] = s * p
            d[left] = c * p
        d[left] += offset
    if d[1] < d[0]:
        d[0], d[1] = d[1], d[0]
        for row in range(2):
            vectors[row, 0], vectors[row, 1] = vectors[row, 1], vectors[row, 0]
    values = d.astype(np.float32)
    vectors32 = vectors.astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(vectors32).all():
        raise ValueError("nonfinite candidate SFF eigen result")
    # The source qsort retains the first item on an equal-absolute pair.
    if abs(values[0]) < abs(values[1]):
        values[0], values[1] = values[1], values[0]
        for row in range(2):
            vectors32[row, 0], vectors32[row, 1] = vectors32[row, 1], vectors32[row, 0]
    return values, vectors32


@njit(cache=True, fastmath=False, error_model="numpy")
def _fit(xyz, normals, offsets, neighbors, selected, initial_k1, initial_k2, initial_e1, initial_e2):
    k1, k2 = initial_k1.copy(), initial_k2.copy()
    e1, e2 = initial_e1.copy(), initial_e2.copy()
    conditions = np.full(len(selected), np.nan, np.float32)
    kinds = np.zeros(len(selected), np.int8)  # 0 no-neighbor, 1 fitted, 2 ill, 3 zero matrix
    previous_eigenvectors = np.zeros((2, 2), np.float32)
    grams = np.zeros((len(selected), 3, 3), np.float32)
    rhs_all = np.zeros((len(selected), 3), np.float32)
    hessian_all = np.zeros((len(selected), 2, 2), np.float32)
    for p in range(len(selected)):
        vertex = selected[p]
        first, second = _tangent(normals[vertex])
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("nonfinite candidate SFF tangent")
        e1[vertex], e2[vertex] = first, second
        count = offsets[vertex + 1] - offsets[vertex]
        if count == 0:
            continue
        design = np.empty((count, 3), np.float32)
        heights = np.empty(count, np.float32)
        minimum, maximum = f(10000), f(-10000)
        for row in range(count):
            neighbor = neighbors[offsets[vertex] + row]
            delta = np.empty(3, np.float32)
            for j in range(3):
                delta[j] = sub(xyz[neighbor, j], xyz[vertex, j])
            u = np.float64(_dot3(delta, first))
            v = np.float64(_dot3(delta, second))
            design[row, 0] = f(u * u)
            design[row, 1] = f(2. * u * v)
            design[row, 2] = f(v * v)
            heights[row] = _dot3(normals[vertex], delta)
            radius2 = f(u * u + v * v)
            if (not np.isfinite(design[row]).all() or not np.isfinite(heights[row])
                    or not np.isfinite(radius2)):
                raise ValueError("nonfinite candidate SFF design or radius")
            if abs(radius2) >= _EPS:
                local = div(heights[row], radius2)
                if not np.isfinite(local):
                    raise ValueError("nonfinite candidate SFF local curvature")
                if local > maximum:
                    maximum = local
                if local < minimum:
                    minimum = local
        gram = np.zeros((3, 3), np.float32)
        for i in range(3):
            for j in range(3):
                for row in range(count):
                    gram[i, j] = add(gram[i, j], mul(design[row, i], design[row, j]))
        if not np.isfinite(gram).all():
            raise ValueError("nonfinite candidate SFF Gram matrix")
        grams[p] = gram
        zero = True
        for i in range(3):
            for j in range(3):
                if np.float64(abs(gram[i, j])) > 1.e-11:
                    zero = False
        values = np.zeros(2, np.float32)
        if zero:
            conditions[p] = f(1.e10)
            kinds[p] = 3
        else:
            _, singular, _ = svdc_3(gram)
            if not np.isfinite(singular).all():
                raise ValueError("nonfinite candidate SFF singular values")
            low, high = singular[0], singular[0]
            for j in range(1, 3):
                value = abs(singular[j])
                if value > high:
                    high = value
                if value < low:
                    low = value
            conditions[p] = f(1.e8) if abs(low) < _EPS else div(high, low)
            if not np.isfinite(conditions[p]):
                raise ValueError("nonfinite candidate SFF condition number")
            inverse = svd_inverse_3(gram)
            if not np.isfinite(inverse).all():
                raise ValueError("nonfinite candidate SFF inverse")
            rhs = np.zeros(3, np.float32)
            coefficients = np.zeros(3, np.float32)
            for i in range(3):
                for row in range(count):
                    rhs[i] = add(rhs[i], mul(design[row, i], heights[row]))
            if not np.isfinite(rhs).all():
                raise ValueError("nonfinite candidate SFF right hand side")
            rhs_all[p] = rhs
            for i in range(3):
                for j in range(3):
                    coefficients[i] = add(coefficients[i], mul(inverse[i, j], rhs[j]))
            q = np.empty((2, 2), np.float32)
            q[0, 0], q[0, 1] = mul(2, coefficients[0]), mul(2, coefficients[1])
            q[1, 0], q[1, 1] = q[0, 1], mul(2, coefficients[2])
            if not np.isfinite(q).all():
                raise ValueError("nonfinite candidate SFF Hessian")
            hessian_all[p] = q
            if conditions[p] >= 500000.:
                k1[vertex], k2[vertex] = maximum, minimum
                kinds[p] = 2
                continue
            values, previous_eigenvectors = _eigen2(q)
            kinds[p] = 1
        k1[vertex], k2[vertex] = values[0], values[1]
        for j in range(3):
            e1[vertex, j] = add(mul(first[j], previous_eigenvectors[0, 0]),
                                 mul(second[j], previous_eigenvectors[1, 0]))
            e2[vertex, j] = add(mul(first[j], previous_eigenvectors[0, 1]),
                                 mul(second[j], previous_eigenvectors[1, 1]))
    return k1, k2, e1, e2, conditions, kinds, grams, rhs_all, hessian_all


def candidate_second_fundamental_form(*, original_vertices, vertex_normals,
                                     neighbor_offsets, neighbors, patch_vertices,
                                     initial_k1, initial_k2, initial_tangent_e1,
                                     initial_tangent_e2) -> dict:
    """原有序候选字段→自产完整 patch 二阶曲面拟合，输入不被改写。

    original_vertices=float32(N,3)，surface RAS/mm；vertex_normals 和两个
    initial_tangent_* 为 float32(N,3) 无量纲当前状态。initial_k1/k2 是
    float32(N) mm⁻¹，供 patch 外或无邻点顶点保持原状态。neighbor_offsets
    int32(N+1)、neighbors int32(E) 必须是原 vt.v[0:vtotal] 有序CSR，
    不可用排序邻域或仅 vnum 替代。patch_vertices int32(P) 依原 TP 次序。
    所有参数均必填；不重算法向、不读 MRI/参考字段，不执行 GA。
    返回完整 k1/k2/tangent_e1/e2 副本、patch 有序 condition_number/
    fit_kind(0无邻点、1拟合、2病态回退、3零矩阵)、gram/rhs/hessian
    诊断、API seconds；gram 单位mm⁴、rhs mm³、Hessian mm⁻¹。
    malformed shape/CSR/越界/重复patch/非有限输入或拟合中间量抛 ValueError；VNL
    不收敛抛 RuntimeError。source zero-normal 与零矩阵按原状态分支
    保留，不填参考或占位输出。不会改变全局线程、TF32或CUDA设置。
    对应固定 mris_fix_topology 内部 computeDefectSecondFundamentalForm，
    没有独立原软件命令。真实同输入结果见专属说明页。
    """
    started = time.perf_counter()
    arrays = [np.ascontiguousarray(value, dtype=np.float32) for value in
              (original_vertices, vertex_normals, initial_k1, initial_k2,
               initial_tangent_e1, initial_tangent_e2)]
    xyz, normal, k1, k2, e1, e2 = arrays
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError("original_vertices must have shape (N,3)")
    n = len(xyz)
    if (xyz.shape != (n, 3) or normal.shape != (n, 3) or e1.shape != (n, 3)
            or e2.shape != (n, 3) or k1.shape != (n,) or k2.shape != (n,)
            or any(not np.isfinite(value).all() for value in arrays)):
        raise ValueError("finite full same-order candidate geometry and state required")
    converted = []
    for value in (neighbor_offsets, neighbors, patch_vertices):
        raw = np.asarray(value)
        if (raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < 0)
                or np.any(raw > np.iinfo(np.int32).max)):
            raise ValueError("one-dimensional original int32 candidate CSR/patch required")
        converted.append(np.ascontiguousarray(raw, dtype=np.int32))
    offsets, adjacent, selected = converted
    if (offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(adjacent)
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or (adjacent.size and (adjacent.min() < 0 or adjacent.max() >= n))
            or (selected.size and (selected.min() < 0 or selected.max() >= n))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("valid same-order vtotal CSR and unique patch indices required")
    result = _fit(xyz, normal, offsets, adjacent, selected, k1, k2, e1, e2)
    names = ("k1", "k2", "tangent_e1", "tangent_e2", "condition_number", "fit_kind", "gram", "rhs", "hessian")
    output = dict(zip(names, result))
    output["seconds"] = time.perf_counter() - started
    output["scope"] = "candidate SFF only; caller supplies actual normals and ordered vtotal"
    return output
