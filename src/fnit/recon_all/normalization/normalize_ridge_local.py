"""仅计算 medial ridge 实际读取的背景距离，复用完整有序Numba传播。"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from .normalize_aseg_ridge import _march_pass, _nonmax


def ridge_required_distance(aseg: np.ndarray) -> tuple[np.ndarray, dict]:
    """返回ridge所需距离图和传播计数；不是完整signed_distance的替代。

    aseg为三维XYZ数值标签（含MGH float32保存的整数标签），2/41表示WM；
    各维至少2。输入网格由调用者
    保证1mm，各距离单位体素。所有WM内部、以及WM外侧26邻域距离完整；
    远处背景为负limit，不允许当成通用有符号距离场使用。
    原堆排序、动态试探值更新和float32算术不变，所有背景查询settled后
    才结束正向pass；负向WM pass完整执行。返回CPU float32同shape数组
    和dict：marching两pass(alive,processed)、outside_query_voxels。
    不修改输入，不调用原生程序；无效维度/dtype或查询未确定抛异常。
    属于mri_normalize -aseg内部步骤，没有独立原软件CLI。
    """
    aseg = np.asarray(aseg)
    if aseg.ndim != 3 or any(size < 2 for size in aseg.shape):
        raise ValueError("ridge requires a 3D grid with each extent at least 2")
    if not np.issubdtype(aseg.dtype, np.number) or np.issubdtype(aseg.dtype, np.complexfloating):
        raise ValueError("aseg must contain real numeric labels")
    mask = np.ascontiguousarray((aseg == 2) | (aseg == 41), dtype=np.uint8)
    # _nonmax的归一化梯度每分量绝对值<=1，三线性读取至多是26邻域。
    # 任何非boundary的WM试探点六邻居都在WM内；完整内侧传播不读远背景。
    query = np.ascontiguousarray((maximum_filter(mask, size=3, mode="nearest") != 0) & (mask == 0), dtype=np.uint8)
    distance = np.zeros(mask.size, np.float32)
    first = _march_pass(mask.ravel(), distance, 1, *mask.shape, query.ravel())
    second = _march_pass(mask.ravel(), distance, -1, *mask.shape)
    return -distance.reshape(mask.shape), {
        "marching": (first, second), "outside_query_voxels": int(np.count_nonzero(query)),
        "distance_scope": "all WM and its external 26-neighbor band; distant background sentinel"}


def medial_ridge_local(aseg: np.ndarray) -> tuple[np.ndarray, dict]:
    """标签2/41→与完整传播相同的uint8 ridge及限域计数。

    输入、失败和空间条件见ridge_required_distance。输出同网格0/1控制候选；
    复用原非极大值选择，无近似、无动态并行更新，无额外依赖。
    """
    distance, details = ridge_required_distance(aseg=aseg)
    ridge = _nonmax(np.ascontiguousarray(distance))
    details["ridge_voxels"] = int(np.count_nonzero(ridge))
    details["marching_backend"] = "local-consumer"
    return ridge, details
