"""独立 NumPy 拓扑 GA 父代选择；复用源顺序排名，不执行原版程序。"""
from __future__ import annotations

import math
import numpy as np

from .topology_fitness_search import rank_patch_fitness


def _draw(callback, upper):
    if not callable(callback):
        raise ValueError('explicit draw_random_number(low, high) callback required')
    value = np.asarray(callback(0.0, float(upper)))
    if (value.ndim or value.dtype.kind not in 'iuf'
            or not np.isfinite(value) or not 0 <= float(value) <= upper):
        raise ValueError('randomNumber result outside requested finite real interval')
    return float(value)


def plan_generation_selection(*, fitness, draw_random_number):
    """完整旧种群 F64 fitness → 精英、变异副本与交叉父代表。

    fitness 为有限实数(P,)，P为2..1000，按当前种群顺序转float64；
    log-P加权评分无单位。draw_random_number 为必填callable(low,high)，
    返回闭区间内有限double，控制器负责种子、版本和随机流。仅父代表
    未填满时每个空位调用一次(0,1)，其余不抽样；不隐式创建NumPy RNG。

    固定配方为10%精英、50%可交叉候选、10%变异副本，正数nint采用
    floor(x+0.5)。逐对排名复用rank_patch_fitness的FLT_EPSILON同分规则；
    若该规则产生重叠排名，原ranks表会有未定义槽，此接口明确失败。
    不能用稳定排序悄悄替换此情况。均值/方差按原循环double顺序计算。

    返回dict：rank_by_patch及patch_by_rank为int32(P,)；elite_indices
    int32(E,)和replacement_parent_indices int32(R,)；selected_parents
    int32(C,)可重复；eligible_parent_indices int32(S,)；mean/sigma、
    two_sigma_sq/fitness_norm及eligible_probabilities float64(S,)；
    random_draws float64(D,3)每行[low,high,value]，distinct_parent_count
    与counts字典。C=P-E-R。源尾部补选使用exp(fitness/scale)，与初始
    配额使用exp((fitness-best)/scale)不同，保留这条规则及下溢后选末项。

    输入不改；无文件、坐标、设备或精度设置。类型/形状/范围错误及
    不唯一排名抛ValueError，累计统计溢出抛FloatingPointError；外部
    callback异常传播且随机流不回滚。不是完整GA或网格质量验收；
    对应mris_fix_topology -ga内部父代选择，无独立原CLI。
    """
    raw = np.asarray(fitness)
    if raw.dtype.kind not in 'iuf' or raw.ndim != 1 or not 2 <= len(raw) <= 1000:
        raise ValueError('finite real fitness vector with 2..1000 patches required')
    values = raw.astype(np.float64, copy=True)
    if not np.isfinite(values).all() or not callable(draw_random_number):
        raise ValueError('finite fitness and explicit random callback required')
    with np.errstate(over='ignore'):
        ranks = rank_patch_fitness(values)
    if not np.array_equal(np.sort(ranks), np.arange(len(values))):
        raise ValueError('source epsilon ranks collide; original inverse table is undefined')
    inverse = np.empty(len(values), np.int32)
    inverse[ranks] = np.arange(len(values), dtype=np.int32)
    count = len(values)
    elite = max(1, math.floor(.10 * count + .5))
    eligible = max(2, math.floor(.50 * count + .5))
    replacements = min(math.floor(.10 * count + .5), count - elite)
    crossovers = count - elite - replacements
    total = squared = 0.0
    for value in values:
        total += float(value)
        squared += float(value) * float(value)
    mean = total / float(np.float32(count))
    variance = squared / count - mean * mean
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise FloatingPointError('source population statistics overflow')
    sigma = 0.0 if variance < 0 else math.sqrt(variance)
    candidates = inverse[:eligible]
    best = float(values[inverse[0]])
    scale = best - float(values[candidates[-1]])
    if abs(scale) < float(np.finfo(np.float32).eps):
        scale = 1.0
    weights = [math.exp((float(values[index]) - best) / scale) for index in candidates]
    normalization = 0.0
    for weight in weights:
        normalization += weight
    probabilities = np.asarray([weight / normalization for weight in weights], np.float64)
    selected, draws = [], []
    for index, probability in zip(candidates, probabilities):
        copies = math.floor(float(probability) * crossovers + .5)
        if copies >= crossovers:
            copies = crossovers - 1
        elif copies == 0:
            copies = 1
        for _ in range(max(0, min(copies, crossovers - len(selected)))):
            selected.append(int(index))
    while len(selected) < crossovers:
        value = _draw(draw_random_number, 1.0)
        draws.append((0.0, 1.0, value))
        cumulative, index = 0.0, int(candidates[-1])
        for candidate in candidates:
            index = int(candidate)
            # 故意保留源补选的绝对fitness指数；不是前面的平移指数。
            with np.errstate(over='ignore', under='ignore'):
                weight = float(np.exp(np.float64(values[index] / scale))) / normalization
            cumulative += weight
            if cumulative > value:
                break
        selected.append(index)
    selected = np.asarray(selected, np.int32)
    return dict(rank_by_patch=ranks, patch_by_rank=inverse,
        elite_indices=inverse[:elite].copy(),
        replacement_parent_indices=inverse[:replacements].copy(),
        eligible_parent_indices=candidates.copy(), selected_parents=selected,
        eligible_probabilities=probabilities, mean=mean, sigma=sigma,
        two_sigma_sq=scale, fitness_norm=normalization,
        random_draws=np.asarray(draws, np.float64).reshape(-1, 3),
        distinct_parent_count=len(np.unique(selected)),
        counts=dict(population=count, elite=elite, eligible=eligible,
                    replacements=replacements, crossovers=crossovers))


def choose_distinct_parent(*, selected_parents, first_parent, draw_random_number):
    """有序父代表 → 一个不同于first_parent的索引和所有重试draw。

    selected_parents为非负整数(C,)，保持原重复项；first_parent为其中
    一个非负整数标量。draw_random_number为必填闭区间double回调。
    逐次调用(0,C-0.001)，截断为槽，命中相同父代时继续抽样，不去重、
    不额外调用、不擅自截断重试。回调应能最终产生不同父代；若表中
    不存在第二种父代则在抽样前抛ValueError，避免原代码无限循环。

    返回dict：parent_index为Python int；random_draws为float64(D,3)
    [low,high,value]。输入不改，异常不回滚外部随机流，无默认RNG、
    文件、坐标、GPU或精度设置。输入/非法draw抛ValueError。它只选
    第二父代，交叉本体复用topology_ga_ordering.crossover_ordering；
    属于mris_fix_topology -ga内部步骤，没有独立原CLI。
    """
    selected, first = np.asarray(selected_parents), np.asarray(first_parent)
    if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or np.any(selected < 0)
            or first.ndim or first.dtype.kind not in 'iu' or int(first) < 0
            or not np.any(selected == first) or not np.any(selected != first)):
        raise ValueError('integer parent pool must contain first and a distinct second parent')
    first = int(first)
    upper = len(selected) - .001
    draws = []
    while True:
        value = _draw(draw_random_number, upper)
        draws.append((0.0, upper, value))
        parent = int(selected[int(value)])
        if parent != first:
            return dict(parent_index=parent, random_draws=np.asarray(draws, np.float64))
