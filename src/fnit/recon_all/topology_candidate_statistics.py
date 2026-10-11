"""拓扑候选的有序顶点反馈统计；独立 NumPy，不运行原版程序。"""
from __future__ import annotations

import numpy as np

from .topology_angle_pseudonormals import _indices


_FINAL_VERTEX = 2  # 固定 mrisurf_defect.h 的统计标记，非 retess 的 USED_VERTEX。


def update_candidate_statistics(*, vertex_marks, vertex_curvbak,
        candidate_edges, edge_used, translated_defect_vertices,
        discarded_defect_vertices, translated_border_vertices,
        use_counts, vertex_fitness, candidate_fitness) -> dict:
    """一次候选完整状态 → 新 marks、RP 使用次数和 F32 有序运行均值。

    vertex_marks 为 int32(N,) 原标记，vertex_curvbak 为有限float32(N,)
    unMRI之后的完整反馈；后者为原统计反馈量，不是当前曲率或距离图。
    candidate_edges 为整数(E,2)当前有序边，edge_used 为整数(E,)源0…6
    状态，仅3/4标记端点为FINAL_VERTEX。translated_defect_vertices 为
    整数(K,)原缺陷到当前网格映射；discarded_defect_vertices 为bool(K,)
    源status==DISCARD，丢弃条目不会查询映射。translated_border_vertices
    为整数(B,)当前border映射。use_counts 是非负int32(K,) RP.nused，
    vertex_fitness 是有限float32(K,) RP.vertex_fitness，均按原缺陷顺序。
    candidate_fitness 为有限实数候选分数；固定原函数将其权重和归一化
    都覆盖为1，因此它不参与运行均值，不能用分数重新加权。

    返回新字典：vertex_marks int32(N,)、use_counts int32(K,)、
    vertex_fitness float32(K,)、updated_defect_entries int64(A,)原缺陷序号。
    先标记3/4边端点，再清border；逐原缺陷更新并立即清mark，保留重复
    映射的源顺序。DISCARD和域外marks不强制清零。输入不修改，不创建
    隐含的全零基底，不恢复DVS，不删顶点，也不复位边状态。
    全参数必填，无设备/线程/阈值默认；RP反馈单位沿用输入。非法类型、
    形状、映射/非有限输入抛ValueError，计数或F32均值溢出抛
    FloatingPointError；失败不改输入。不改GPU、TF32、线程或autocast。
    对应mris_fix_topology内部updateVertexStatistics，无独立官方CLI。
    """
    marks, feedback = np.asarray(vertex_marks), np.asarray(vertex_curvbak)
    if (marks.dtype != np.int32 or marks.ndim != 1
            or feedback.dtype != np.float32 or feedback.shape != marks.shape
            or not np.isfinite(feedback).all()):
        raise ValueError('int32 marks and finite float32 curvbak (N,) required')
    n = len(marks)
    edges = np.asarray(candidate_edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('candidate_edges integer (E,2) required')
    # _indices validates values without silently truncating float endpoints.
    edges = _indices(edges.reshape(-1), n, 'candidate_edges').reshape(-1, 2)
    used = np.asarray(edge_used)
    if (used.shape != (len(edges),) or not np.issubdtype(used.dtype, np.integer)
            or np.any(used < 0) or np.any(used > 6)):
        raise ValueError('source edge_used integer (E,) states 0...6 required')
    mapped, discarded = np.asarray(translated_defect_vertices), np.asarray(discarded_defect_vertices)
    if (mapped.ndim != 1 or not np.issubdtype(mapped.dtype, np.integer)
            or discarded.dtype != np.bool_ or discarded.shape != mapped.shape):
        raise ValueError('integer defect mapping and bool discarded (K,) required')
    # Source skips DISCARD before dereferencing the mapping (which may be -1).
    _indices(mapped[~discarded], n, 'active translated_defect_vertices')
    border = _indices(translated_border_vertices, n, 'translated_border_vertices')
    counts, fitness = np.asarray(use_counts), np.asarray(vertex_fitness)
    if (counts.dtype != np.int32 or counts.shape != mapped.shape or np.any(counts < 0)
            or fitness.dtype != np.float32 or fitness.shape != mapped.shape
            or not np.isfinite(fitness).all()):
        raise ValueError('nonnegative int32 use_counts and finite float32 fitness (K,) required')
    score = np.asarray(candidate_fitness)
    if (score.ndim or score.dtype.kind not in 'iuf' or not np.isfinite(score)):
        raise ValueError('finite real scalar candidate_fitness required')

    new_marks, new_counts, new_fitness = marks.copy(), counts.copy(), fitness.copy()
    new_marks[edges[(used == 3) | (used == 4)].reshape(-1)] = _FINAL_VERTEX
    new_marks[border] = 0
    updated = []
    with np.errstate(over='ignore', invalid='ignore'):
        for entry in range(len(mapped)):
            if discarded[entry]:
                continue
            vertex = int(mapped[entry])
            if new_marks[vertex] == _FINAL_VERTEX:
                count = int(new_counts[entry])
                if count == np.iinfo(np.int32).max:
                    raise FloatingPointError('candidate use count exceeds source int32')
                count_f32 = np.float32(count)
                # Fixed source overwrites weight and total_vertex_fitness to 1.
                # Keep the source F32 multiply/add/divide order for RP means.
                weighted_previous = np.float32(count_f32 * new_fitness[entry])
                numerator = np.float32(feedback[vertex] + weighted_previous)
                denominator = np.float32(count_f32 + np.float32(1))
                mean = np.float32(numerator / denominator)
                if not np.isfinite(mean):
                    raise FloatingPointError('nonfinite candidate running mean')
                new_fitness[entry] = mean
                new_counts[entry] = count + 1
                updated.append(entry)
            # This reset occurs inside the ordered loop, including aliases.
            new_marks[vertex] = 0
    return dict(vertex_marks=new_marks, use_counts=new_counts, vertex_fitness=new_fitness,
                updated_defect_entries=np.asarray(updated, np.int64))
