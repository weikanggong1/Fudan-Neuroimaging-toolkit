"""GCSA 稀疏概率表与有序Gibbs单轮内核，复用原顶点排列和反馈规则。

属于mris_ca_label内部步骤，无独立原软件CLI。仅显式numba后端调用；
没有并行/Jacobi、fastmath、近似邻域或低精度概率。几何保持surface RAS。
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


def pack_model(model, atlas) -> dict:
    """把既有GibbsModel只读字典编成CSR，返回数值数组字典。

    输入为原模型和对应InitialAtlas，labels/feature不复制以保留逐顶点反馈。概率和
    高斯参数float64，特征float32，索引int64/标签int32；边顺序保持原
    neighbors/edge_slots。完整读取/构造属于阶段墙钟，不隐藏准备成本。
    缺少概率表、非法边槽或非正方差/先验抛ValueError；该候选要求atlas
    的这些参数有效，零邻居概率仍保留原-10000000惩罚。
    """
    classifier_offsets, classifier_labels, means, variances = [0], [], [], []
    for node in model.classifier:
        for label, (mean, variance) in node.items():
            if not variance > 0:
                raise ValueError("Numba Gibbs requires positive classifier variance")
            classifier_labels.append(label); means.append(mean); variances.append(variance)
        classifier_offsets.append(len(classifier_labels))
    prior_offsets, prior_labels, probabilities = [0], [], []
    direction_offsets, chance_labels, chances = [0], [], []
    for node in model.prior:
        for label, (probability, directions) in node.items():
            if not probability > 0 or len(directions) != 2:
                raise ValueError("Numba Gibbs requires positive prior and two directions")
            prior_labels.append(label); probabilities.append(probability)
            for direction in directions:
                for neighbor_label, chance in direction.items():
                    if chance < 0:
                        raise ValueError("Numba Gibbs requires nonnegative neighbor chance")
                    chance_labels.append(neighbor_label); chances.append(chance)
                direction_offsets.append(len(chance_labels))
        prior_offsets.append(len(prior_labels))
    choice_offsets, choice_labels = [0], []
    for node in atlas.prior_nodes:
        choice_labels.extend(label for label, _ in node)
        choice_offsets.append(len(choice_labels))
    neighbor_offsets, neighbor_indices, edge_slots = [0], [], []
    for neighbors, slots in zip(model.neighbors, model.edge_slots):
        if len(neighbors) != len(slots) or any(int(slot) not in (0, 1) for slot in slots):
            raise ValueError("invalid Gibbs edge slots")
        neighbor_indices.extend(neighbors); edge_slots.extend(slots)
        neighbor_offsets.append(len(neighbor_indices))
    integer = dict(classifier_offsets=classifier_offsets, classifier_labels=classifier_labels,
        prior_offsets=prior_offsets, prior_labels=prior_labels, direction_offsets=direction_offsets,
        chance_labels=chance_labels, neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices, edge_slots=edge_slots,
        choice_offsets=choice_offsets, choice_labels=choice_labels)
    result = {key: np.asarray(value, dtype=np.int64) for key, value in integer.items()}
    result.update(means=np.asarray(means, np.float64), variances=np.asarray(variances, np.float64),
        probabilities=np.asarray(probabilities, np.float64), chances=np.asarray(chances, np.float64),
        classifier_indices=np.asarray(model.classifier_indices, np.int64),
        prior_indices=np.asarray(model.prior_indices, np.int64))
    return result


@njit(cache=True, fastmath=False)
def _find_label(offsets, labels, node, label):
    for index in range(offsets[node], offsets[node + 1]):
        if labels[index] == label:
            return index
    return -1


@njit(cache=True, fastmath=False)
def _vertex_likelihood(vertex, input_value, labels, classifier_indices, prior_indices,
                       classifier_offsets, classifier_labels, means, variances,
                       prior_offsets, prior_labels, probabilities, direction_offsets,
                       chance_labels, chances, neighbor_offsets, neighbor_indices, edge_slots):
    label = labels[vertex]
    prior = _find_label(prior_offsets, prior_labels, prior_indices[vertex], label)
    classifier = _find_label(classifier_offsets, classifier_labels, classifier_indices[vertex], label)
    if prior < 0 or classifier < 0:
        return -10000000.0
    # 原函数先float32量化mean-input，再用Python float64计算整个likelihood。
    diff = np.float64(np.float32(means[classifier] - input_value))
    variance = variances[classifier]
    value = -0.5 * diff * diff / variance - 0.5 * math.log(variance)
    for edge in range(neighbor_offsets[vertex], neighbor_offsets[vertex + 1]):
        direction = prior * 2 + edge_slots[edge]
        chance = _find_label(direction_offsets, chance_labels, direction, labels[neighbor_indices[edge]])
        if chance >= 0 and chances[chance] != 0:
            value += math.log(chances[chance])
        else:
            value += -10000000.0
    return value + math.log(probabilities[prior])


@njit(cache=True, fastmath=False)
def _neighborhood_likelihood(vertex, candidate, input_value, labels, classifier_indices,
                            prior_indices, classifier_offsets, classifier_labels, means,
                            variances, prior_offsets, prior_labels, probabilities,
                            direction_offsets, chance_labels, chances, neighbor_offsets,
                            neighbor_indices, edge_slots):
    old = labels[vertex]
    labels[vertex] = candidate
    value = _vertex_likelihood(vertex, input_value, labels, classifier_indices, prior_indices,
        classifier_offsets, classifier_labels, means, variances, prior_offsets, prior_labels,
        probabilities, direction_offsets, chance_labels, chances, neighbor_offsets,
        neighbor_indices, edge_slots)
    for edge in range(neighbor_offsets[vertex], neighbor_offsets[vertex + 1]):
        # 原算法中心input_value也用于邻居，不能换成feature[neighbor]。
        value += _vertex_likelihood(neighbor_indices[edge], input_value, labels,
            classifier_indices, prior_indices, classifier_offsets, classifier_labels, means,
            variances, prior_offsets, prior_labels, probabilities, direction_offsets,
            chance_labels, chances, neighbor_offsets, neighbor_indices, edge_slots)
    labels[vertex] = old
    return value


@njit(cache=True, fastmath=False)
def _ordered_sweep(permutation, mark, feature, labels, classifier_indices, prior_indices,
                   classifier_offsets, classifier_labels, means, variances, prior_offsets,
                   prior_labels, probabilities, direction_offsets, chance_labels, chances,
                   neighbor_offsets, neighbor_indices, edge_slots, choice_offsets, choice_labels):
    changed, examined = 0, 0
    for vertex in permutation:
        if mark[vertex] == 0:
            continue
        mark[vertex] = 0
        examined += 1
        node = prior_indices[vertex]
        if choice_offsets[node + 1] - choice_offsets[node] <= 1:
            continue
        old = labels[vertex]
        best = old
        input_value = float(feature[vertex])
        maximum = _neighborhood_likelihood(vertex, old, input_value, labels,
            classifier_indices, prior_indices, classifier_offsets, classifier_labels, means,
            variances, prior_offsets, prior_labels, probabilities, direction_offsets,
            chance_labels, chances, neighbor_offsets, neighbor_indices, edge_slots)
        for choice in range(choice_offsets[node], choice_offsets[node + 1]):
            candidate = choice_labels[choice]
            likelihood = _neighborhood_likelihood(vertex, candidate, input_value, labels,
                classifier_indices, prior_indices, classifier_offsets, classifier_labels, means,
                variances, prior_offsets, prior_labels, probabilities, direction_offsets,
                chance_labels, chances, neighbor_offsets, neighbor_indices, edge_slots)
            if likelihood > maximum:
                maximum = likelihood
                best = candidate
        if best != old:
            labels[vertex] = best
            mark[vertex] = 1
            changed += 1
    return changed, examined


def ordered_sweep(*, packed: dict, permutation: np.ndarray, mark: np.ndarray,
                  feature: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """按既有permutation执行一轮原位反馈，返回changed/examined整数。

    packed来自pack_model；其余数组来自原reclassify，不能跨模型标签版本
    复用。对当前标签立即更新，尚未访问顶点会读取最新标签；没有GPU/线程
    并行或另建种子。Numba首次编译计入调用墙钟，缓存仅包含编译程序。
    """
    return _ordered_sweep(permutation, mark, feature, labels, **packed)


def neighborhood_likelihood(*, packed: dict, vertex: int, candidate: int,
                            input_value: float, labels: np.ndarray) -> float:
    """同一原模型单次诊断评分，临时设置候选后恢复labels，不改几何。"""
    arrays = {key: value for key, value in packed.items()
              if key not in ("choice_offsets", "choice_labels")}
    return _neighborhood_likelihood(vertex, candidate, input_value, labels, **arrays)
