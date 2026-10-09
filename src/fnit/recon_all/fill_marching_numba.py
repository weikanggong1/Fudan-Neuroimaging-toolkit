"""复用现有有序fast marching循环的Numba编译；不采用并行Jacobi。"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def march_cc_distance_numba(distance: np.ndarray, state: np.ndarray,
                             alive: np.ndarray, query: np.ndarray) -> int:
    """修改同局部XYZ网格的float32 distance/u8 state并返回未确定CC数。

    alive为原首次访问顺序Nx3 int64坐标；query为同shape bool查询。
    按既有堆同分次序、相邻更新顺序、float32逐步舍入传播，全部查询
    alive后停止；CPU单线程，不改变输入alive/query，不调用原生程序。
    fastmath关闭；首次JIT独立计时。内部函数，暂无独立CLI。
    """
    sx, sy, sz = distance.shape
    heap = [(0, 0, 0)]
    heap.pop()
    def priority(point: tuple[int, int, int]) -> float:
        return float(distance[point])

    def push(point: tuple[int, int, int]) -> None:
        heap.append(point)
        index = len(heap) - 1
        while index:
            parent = (index - 1) // 2
            if priority(heap[parent]) <= priority(point):
                break
            heap[index] = heap[parent]
            index = parent
        heap[index] = point

    def pop() -> tuple[int, int, int]:
        first = heap[0]
        last = heap.pop()
        if heap:
            index = 0
            length = len(heap)
            while 2 * index + 1 < length:
                child = 2 * index + 1
                if child + 1 < length and priority(heap[child + 1]) <= priority(heap[child]):
                    child += 1
                heap[index] = heap[child]
                index = child
            while index:
                parent = (index - 1) // 2
                if priority(heap[parent]) <= priority(last):
                    break
                heap[index] = heap[parent]
                index = parent
            heap[index] = last
        return first

    def update(x: int, y: int, z: int) -> None:
        if state[x, y, z] >= 2:
            return
        raw_a = min(distance[x - 1, y, z] if x else 100,
                distance[x + 1, y, z] if x + 1 < sx else 100)
        raw_b = min(distance[x, y - 1, z] if y else 100,
                distance[x, y + 1, z] if y + 1 < sy else 100)
        raw_c = min(distance[x, y, z - 1] if z else 100,
                distance[x, y, z + 1] if z + 1 < sz else 100)
        # 边界100整数字面量使Numba的raw_*提升为float64。保留独立
        # 变量确保sorted后的a/b/c真为float32，不与raw变量类型合并。
        a, b, c = sorted((np.float32(raw_a), np.float32(raw_b), np.float32(raw_c)))
        one = np.float32(1)
        value = np.float32(a + one)
        sum3 = np.float32(np.float32(a + b) + c)
        squares3 = np.float32(np.float32(a * a + b * b) + c * c)
        delta = np.float32(sum3 * sum3 - np.float32(3) * np.float32(squares3 - one))
        solved = False
        if delta >= 0:
            # Python float(np.float32)为float64；Numba的float()仍可能
            # 保持float32，故明确提升sqrt及和，之后按原步骤舍回float32。
            solution = np.float32((np.float64(sum3) + np.sqrt(np.float64(delta))) / 3.0)
            if np.float32(solution + np.float32(1e-6)) >= c:
                value = solution
                solved = True
        if not solved:
            sum2 = np.float32(a + b)
            squares2 = np.float32(a * a + b * b)
            delta = np.float32(sum2 * sum2 - np.float32(2) * np.float32(squares2 - one))
            if delta >= 0:
                solution = np.float32((np.float64(sum2) + np.sqrt(np.float64(delta))) / 2.0)
                if np.float32(solution + np.float32(1e-6)) >= b:
                    value = solution
        distance[x, y, z] = np.float32(value)
        if state[x, y, z] == 0:
            state[x, y, z] = 1
            push((x, y, z))

    def update_neighbors(x: int, y: int, z: int) -> None:
        for xx, yy, zz in ((x - 1, y, z), (x + 1, y, z),
                           (x, y - 1, z), (x, y + 1, z),
                           (x, y, z - 1), (x, y, z + 1)):
            if 0 <= xx < sx and 0 <= yy < sy and 0 <= zz < sz:
                update(xx, yy, zz)

    for x, y, z in alive:
        # 初始trial只更新far；其后的传播还会更新trial。明确分离，
        # 避免Numba嵌套函数默认参数内联合并两种调用的布尔条件。
        for xx, yy, zz in ((x - 1, y, z), (x + 1, y, z),
                           (x, y - 1, z), (x, y + 1, z),
                           (x, y, z - 1), (x, y, z + 1)):
            if 0 <= xx < sx and 0 <= yy < sy and 0 <= zz < sz and state[xx, yy, zz] == 0:
                update(xx, yy, zz)
    remaining = int(np.count_nonzero(query & (state != 2)))
    while remaining and heap:
        x, y, z = pop()
        state[x, y, z] = 2
        if query[x, y, z]:
            remaining -= 1
        update_neighbors(x, y, z)
    return remaining
