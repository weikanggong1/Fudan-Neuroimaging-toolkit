"""WM planar holes：Torch缓存静态平面索引，Numba保留逐点动态反馈。"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.ndimage import maximum_filter
import torch


def plane_indices_torch(points: np.ndarray, offsets: np.ndarray, shape: tuple[int, int, int], *,
                        device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """预计算全部候选的22×25平面线性索引，XYZ网格、不采样动态标签。

    points为N×3 int64体素坐标，offsets为22×5×5×3 float32原配方偏移。
    shape为XYZ网格；device默认cpu，可显式cuda；batch_size默认256，
    正整数且不截断候选。返回N×22×25 int32 C-order CPU索引，GPU
    路径含上传/下载；float32相加后float64向最近体素舍入，边界裁切。
    不修改输入、不改变TF32、不使用半精度；参数错误抛ValueError。
    """
    if (points.ndim != 2 or points.shape[1:] != (3,) or points.dtype != np.int64
            or offsets.shape != (22, 5, 5, 3) or offsets.dtype != np.float32
            or len(shape) != 3 or not all(isinstance(axis, (int, np.integer)) for axis in shape)
            or min(shape) < 1 or np.prod(shape, dtype=np.int64) > np.iinfo(np.int32).max
            or not np.isfinite(offsets).all() or np.any(points < 0) or np.any(points >= np.asarray(shape))
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("invalid point/offset/grid/batch geometry")
    indices = np.empty((len(points), 22, 25), np.int32)
    offsets_tensor = torch.from_numpy(offsets).to(device)
    limits = torch.tensor(shape, dtype=torch.int64, device=device) - 1
    for start in range(0, len(points), batch_size):
        point_tensor = torch.from_numpy(points[start:start + batch_size].astype(np.float32)).to(device)
        positions = point_tensor[:, None, None, None, :] + offsets_tensor[None]
        nearest = torch.floor(positions.to(torch.float64) + .5).to(torch.int64)
        nearest.clamp_(min=0)
        nearest = torch.minimum(nearest, limits)
        linear = (nearest[..., 0] * shape[1] + nearest[..., 1]) * shape[2] + nearest[..., 2]
        indices[start:start + len(point_tensor)] = linear.reshape(-1, 22, 25).to(torch.int32).cpu().numpy()
    return indices


@njit(cache=True, fastmath=False)
def _sample_ray(binary, px, py, pz):
    x = min(max(np.float64(px), 0.), np.float64(binary.shape[0] - 1))
    y = min(max(np.float64(py), 0.), np.float64(binary.shape[1] - 1))
    z = min(max(np.float64(pz), 0.), np.float64(binary.shape[2] - 1))
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = min(x0 + 1, binary.shape[0] - 1), min(y0 + 1, binary.shape[1] - 1), min(z0 + 1, binary.shape[2] - 1)
    dx1, dy1, dz1 = x - x0, y - y0, z - z0
    dx0, dy0, dz0 = 1. - dx1, 1. - dy1, 1. - dz1
    value = (dx0*dy0*dz0*np.float64(binary[x0,y0,z0])
             + dx0*dy0*dz1*binary[x0,y0,z1]
             + dx0*dy1*dz0*binary[x0,y1,z0]
             + dx0*dy1*dz1*binary[x0,y1,z1]
             + dx1*dy0*dz0*binary[x1,y0,z0]
             + dx1*dy0*dz1*binary[x1,y0,z1]
             + dx1*dy1*dz0*binary[x1,y1,z0]
             + dx1*dy1*dz1*binary[x1,y1,z1])
    return np.float32(value)


@njit(cache=True, fastmath=False)
def _ordered_planar_fill(result, binary, border, points, plane_indices, directions):
    flat = binary.reshape(-1)
    total_added = 0
    passes = 0
    while True:
        added = 0
        for point_index in range(len(points)):
            x, y, z = points[point_index]
            if result[x, y, z] > 5 or not border[x, y, z]:
                continue
            binary_count = 0
            result_count = 0
            for xx, yy, zz in ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z),
                               (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)):
                binary_count += int(binary[xx, yy, zz] >= 1)
                result_count += int(result[xx, yy, zz] >= 5)
            if binary_count != result_count:
                continue
            best_count = -1
            vertex = -1
            for candidate in range(22):
                count = 0
                for offset in range(25):
                    count += int(flat[plane_indices[point_index, candidate, offset]])
                # 原配方平分时使用最后一个平面。
                if count >= best_count:
                    best_count, vertex = count, candidate
            accepted = True
            for ray in range(4):
                found = False
                for distance in (np.float32(.75), np.float32(1.5)):
                    px = np.float32(np.float32(x) + np.float32(directions[vertex, ray, 0] * distance))
                    py = np.float32(np.float32(y) + np.float32(directions[vertex, ray, 1] * distance))
                    pz = np.float32(np.float32(z) + np.float32(directions[vertex, ray, 2] * distance))
                    if _sample_ray(binary, px, py, pz) > .5:
                        found = True
                if not found:
                    accepted = False
                    break
            if not accepted:
                continue
            result[x, y, z] = 200
            binary[x, y, z] = 1
            border[x, y, z] = False
            added += 1
        total_added += added
        passes += 1
        if added == 0:
            break
    return passes, total_added


def fill_planar_holes_cached(result: np.ndarray, strand: np.ndarray, *,
                             device: str = "cpu", batch_size: int = 256) -> dict:
    """复用已有22平面配方，有序原地更新result，strand不修改。

    result/strand为同三维XYZ uint8个体体素网格，原规则label 200。
    device仅选择静态索引生成设备；Numba动态更新为单线程CPU，默认
    cpu。batch_size默认256，仅控制几何临时内存。返回候选数/轮数/
    添加体素数/device；geometry_candidates另记初始result<=5、需要
    静态几何的数量。两轮边界膨胀限于strand包围盒外扩2；初始result>5
    在全部轮次都必然跳过，只省其几何，保留其余候选顺序和动态回读。
    首次JIT单列。几何仅在本次strand版本内缓存，不跨组件复用标签状态。
    网格/dtype错误抛ValueError，原配方正边缘越界显式抛IndexError，
    CUDA错误传播。内部子阶段，没有独立原软件CLI。
    """
    if (result.ndim != 3 or not result.size or result.dtype != np.uint8
            or strand.shape != result.shape or strand.dtype != np.uint8
            or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("require matching 3D uint8 result and strand")
    from .mri_segment import _plane_bases, _plane_positions
    binary = np.ascontiguousarray((strand >= 5).astype(np.uint8))
    occupied = np.argwhere(binary)
    if not len(occupied):
        return {"candidates": 0, "geometry_candidates": 0, "passes": 1, "added_voxels": 0,
                "geometry_device": str(device), "dynamic_update_device": "cpu", "batch_size": batch_size}
    # 两轮3邻域的影响半径为2。框外初始值均为0；框触及原图边缘时
    # 保持nearest边界，因此裁切膨胀与整幅膨胀相同。
    lower = np.maximum(occupied.min(axis=0) - 2, 0)
    upper = np.minimum(occupied.max(axis=0) + 3, result.shape)
    roi = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    local_binary = binary[roi]
    local_border = maximum_filter(maximum_filter(local_binary, size=3, mode="nearest"),
                                  size=3, mode="nearest") > 0
    local_border &= local_binary == 0
    border = np.zeros(result.shape, dtype=np.bool_)
    border[roi] = local_border
    points = np.argwhere(local_border) + lower
    points = points[np.lexsort((points[:, 0], points[:, 1], points[:, 2]))]
    border_candidates = len(points)
    # 本函数只把result升至200，从不降低标签。初始>5点在每轮均
    # 被原规则跳过，可省去其静态几何而不改变后续候选或相对顺序。
    points = points[result[tuple(points.T)] <= 5]
    if np.any(points == np.asarray(result.shape) - 1):
        raise IndexError("active planar candidate outside original positive neighbor bounds")
    _, first, second = _plane_bases()
    offsets = _plane_positions(np.zeros((22, 3), np.float32), first, second, 5)
    directions = np.stack((first, -first, second, -second), axis=1)
    indices = plane_indices_torch(points, offsets, result.shape, device=device, batch_size=batch_size)
    passes, added = _ordered_planar_fill(result, binary, border, points, indices, directions)
    return {"candidates": border_candidates, "geometry_candidates": len(points),
            "passes": int(passes), "added_voxels": int(added),
            "geometry_device": str(device), "dynamic_update_device": "cpu", "batch_size": batch_size}
