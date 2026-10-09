"""复用有序face CSR生成inflation的一/完整二环整数缓存。

只优化整数遍历，不改变邻域、面/角点顺序、同分选择或浮点算法。
对应mrisCompleteTopology_old和mris扩展二环内部步骤，无独立CLI。
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _one_ring(faces, face_ids, corners, offsets):
    count = len(offsets) - 1
    width = 0
    for vertex in range(count):
        width = max(width, 2 * (offsets[vertex + 1] - offsets[vertex]))
    table = np.zeros((count, width), np.int32)
    degree = np.zeros(count, np.int32)
    for vertex in range(count):
        for entry in range(offsets[vertex], offsets[vertex + 1]):
            face = face_ids[entry]
            corner = corners[entry]
            for position in (2, 1):
                neighbor = faces[face, (corner + position) % 3]
                found = False
                for column in range(degree[vertex]):
                    if table[vertex, column] == neighbor:
                        found = True
                        break
                if not found:
                    table[vertex, degree[vertex]] = neighbor
                    degree[vertex] += 1
    maximum = 0
    for vertex in range(count):
        maximum = max(maximum, degree[vertex])
    return table[:, :maximum].copy(), degree


@njit(cache=True)
def _two_ring(one, degree):
    count = len(degree)
    marks = np.zeros(count, np.int64)
    sizes = degree.copy()
    for vertex in range(count):
        stamp = vertex + 1
        marks[vertex] = stamp
        for column in range(degree[vertex]):
            marks[one[vertex, column]] = stamp
        for column in range(degree[vertex]):
            adjacent = one[vertex, column]
            for position in range(degree[adjacent]):
                candidate = one[adjacent, position]
                if marks[candidate] != stamp:
                    marks[candidate] = stamp
                    sizes[vertex] += 1
    maximum = 0
    for vertex in range(count):
        maximum = max(maximum, sizes[vertex])
    table = np.zeros((count, maximum), np.int32)
    marks[:] = 0
    for vertex in range(count):
        stamp = vertex + 1
        marks[vertex] = stamp
        end = degree[vertex]
        for column in range(degree[vertex]):
            table[vertex, column] = one[vertex, column]
            marks[one[vertex, column]] = stamp
        for column in range(degree[vertex]):
            adjacent = one[vertex, column]
            for position in range(degree[adjacent]):
                candidate = one[adjacent, position]
                if marks[candidate] != stamp:
                    marks[candidate] = stamp
                    table[vertex, end] = candidate
                    end += 1
    return table, sizes


def inflate_neighbor_tables(*, normal_topology) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """从FaceNormalTopology冻结CSR返回int32一环/二环矩阵和度数。

    返回(one_indices(N,K1),one_degree(N),two_indices(N,K2),two_degree(N))。
    有效列严格保留旧ordered_neighbors与two_ring_neighbors的顺序，未用
    固定邻点截断；无坐标计算/文件读写。缓存必须与原有序面版本绑定。
    缺少所需CSR字段抛属性异常；拓扑合法性由成熟FaceNormalTopology验证。
    """
    one, one_degree = _one_ring(normal_topology.faces, normal_topology.face_ids,
                                normal_topology.corners, normal_topology.offsets)
    two, two_degree = _two_ring(one, one_degree)
    return one, one_degree, two, two_degree
