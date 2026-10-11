"""拓扑 GA 的最差顶点剔除；独立 NumPy，静态阈值改为显式状态。"""
from __future__ import annotations

import numpy as np


def eliminate_worst_vertices(*, discarded_defect_vertices,
        translated_defect_vertices, vertex_rip, use_counts, vertex_fitness,
        fraction, deletion_count, threshold):
    """完整 RP/缺陷状态 → 新 DISCARD/rip 状态和下一次 F32 threshold。

    discarded_defect_vertices 是 bool(K,) 原 status==DISCARD；
    translated_defect_vertices 为整数(K,) 同序到当前网格映射，DISCARD
    条目的负映射不查询。vertex_rip 为 int32(N,) 的完整 signed-char
    解码值[-128,127]；use_counts 是非负int32(K,)，vertex_fitness 是
    有限float32(K,)原RP反馈，不是GA候选fitness。fraction 为[0,1]
    有限实数，进入源float字段；deletion_count为int32范围整数，<=0
    按源改1，赋rip时按已核固定构建的signed8bit char存储。threshold
    为必填非负float32标量；控制器初值4并跨缺陷传递，不隐式全局缓存。

    先剔除nused==0的非DISCARD条目，再按原有效条目数计算最差反馈的
    删除次数。有效条目不足10时返回计数0，但前面的剔除仍保留。
    原停止条件比较max_i索引与threshold，不比较反馈max；严格>选择
    使同分时较早条目优先。源float fraction与double0.1比较，因此
    float32(0.1)也触发threshold/2；保持源定义，不事后修成另一个算法。

    返回dict：discarded_defect_vertices bool(K,)、vertex_rip int32(N,)、
    threshold float32标量、source_reported_deleted int、
    unused_deleted_entries/fitness_deleted_entries int64(U,)/(W,)、
    effective_deletion_count int及stored_rip_value int。
    输入不改；RP均值/计数不改；失败不更新外部阈值。无坐标/设备/精度
    默认，不执行原版、不修改CUDA/TF32/线程，无原软件独立CLI，属于
    mris_fix_topology的deleteWorstVertices。非法输入抛ValueError。
    本函数不是完整GA、DVS恢复、最终网格删除或网格质量验收。
    """
    discarded = np.asarray(discarded_defect_vertices)
    mapped, rip = np.asarray(translated_defect_vertices), np.asarray(vertex_rip)
    counts, feedback = np.asarray(use_counts), np.asarray(vertex_fitness)
    if (discarded.dtype != np.bool_ or discarded.ndim != 1
            or mapped.shape != discarded.shape or mapped.dtype.kind not in 'iu'
            or rip.ndim != 1 or rip.dtype != np.int32
            or np.any(rip < -128) or np.any(rip > 127)):
        raise ValueError('bool discarded, integer mapping and int32 signed-char rip required')
    if np.any(mapped[~discarded] < 0) or np.any(mapped[~discarded] >= len(rip)):
        raise ValueError('active defect mapping outside complete vertex domain')
    if (counts.dtype != np.int32 or counts.shape != discarded.shape or np.any(counts < 0)
            or feedback.dtype != np.float32 or feedback.shape != discarded.shape
            or not np.isfinite(feedback).all()):
        raise ValueError('nonnegative int32 counts and finite float32 RP fitness required')
    for value, name in [(fraction, 'fraction'), (threshold, 'threshold')]:
        scalar = np.asarray(value)
        if scalar.ndim or scalar.dtype.kind not in 'iuf' or not np.isfinite(scalar):
            raise ValueError(name+' requires a finite real scalar')
    with np.errstate(over='ignore', invalid='ignore'):
        fraction_f32, threshold_f32 = np.float32(fraction), np.float32(threshold)
    if (not 0 <= float(fraction) <= 1 or not np.isfinite(fraction_f32)
            or not np.isfinite(threshold_f32) or threshold_f32 < 0):
        raise ValueError('fraction in [0,1] and nonnegative finite F32 threshold required')
    count_array = np.asarray(deletion_count)
    if (count_array.ndim or count_array.dtype.kind not in 'iu'
            or int(count_array) < np.iinfo(np.int32).min
            or int(count_array) > np.iinfo(np.int32).max):
        raise ValueError('deletion_count requires a source int32 integer')
    count = max(1, int(count_array))
    # 固定参考是signed plain-char/8bit；避免NumPy越界转换依赖或告警。
    rip_value = ((count + 128) % 256) - 128
    new_discarded, new_rip = discarded.copy(), rip.copy()
    active_count = 0
    unused, worst = [], []
    for entry in range(len(mapped)):
        if new_discarded[entry]:
            continue
        active_count += 1
        if counts[entry] == 0:
            new_discarded[entry] = True
            new_rip[int(mapped[entry])] = rip_value
            unused.append(entry)
    reported = len(unused)
    if active_count < 10:
        reported = 0
    else:
        if float(fraction_f32) > 0.1:
            threshold_f32 = np.float32(threshold_f32 / np.float32(2))
        iterations = max(1, int(np.float32(fraction_f32 * np.float32(active_count))))
        initial_iterations, changed = iterations, True
        while iterations and changed:
            changed = False
            maximum, maximum_entry = np.float32(0), -1
            for entry in range(len(mapped)):
                if not new_discarded[entry] and feedback[entry] > maximum:
                    maximum, maximum_entry = feedback[entry], entry
            if maximum_entry < float(threshold_f32) and 2 * iterations < initial_iterations:
                break
            if maximum_entry >= 0:
                new_discarded[maximum_entry] = True
                new_rip[int(mapped[maximum_entry])] = rip_value
                reported += 1
                worst.append(maximum_entry)
                changed = True
            iterations -= 1
    return dict(discarded_defect_vertices=new_discarded, vertex_rip=new_rip,
                threshold=threshold_f32, source_reported_deleted=reported,
                unused_deleted_entries=np.asarray(unused, np.int64),
                fitness_deleted_entries=np.asarray(worst, np.int64),
                effective_deletion_count=count, stored_rip_value=rip_value)
