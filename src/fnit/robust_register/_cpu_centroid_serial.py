"""验证候选：按固定原实现的顺序累加 CPU Double 质心。

values 是 Float32、X 最快的只读一维数组；三个尺寸按 XYZ 给出，outside
是 Double。返回一基体素坐标中的三个 Double 值。此模块位于 validation，
未接入 FNIT 默认后端；调用者应先证明输入有限、总质量为正及数组长度正确。
实际通过范围及冷编译开销见同目录 README.md。
"""
import math
import numpy as np
from numba import njit


@njit(fastmath=False, parallel=False, cache=False, error_model="numpy")
def centroid_serial(values, width, height, depth, outside):
    total = np.float64(0.0)
    xsum = np.float64(0.0)
    ysum = np.float64(0.0)
    zsum = np.float64(0.0)
    epsilon = outside / np.float64(255.0)
    index = 0
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                value = np.float64(values[index])
                if math.fabs(value - outside) < epsilon:
                    value = np.float64(0.0)
                total += value
                xsum += np.float64(w + 1) * value
                ysum += np.float64(h + 1) * value
                zsum += np.float64(d + 1) * value
                index += 1
    xsum /= total
    ysum /= total
    zsum /= total
    if math.isnan(xsum + ysum + zsum):
        raise ValueError("source-defined centroid is NaN")
    return xsum, ysum, zsum
