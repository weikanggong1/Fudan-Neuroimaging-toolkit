"""固定表面状态的白质/软脑膜目标强度搜索，自有 Num ba 实现。

输入为已预处理 MRI、surface RAS 坐标/法向和 rip 标记；此处不执行
MRI 预处理、rip 或表面优化。复用固定源码的候选扩展及峰选择规则。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from types import FunctionType


_LH_LABELS = np.array([2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31], dtype=np.int32)
_RH_LABELS = np.array([41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63], dtype=np.int32)


@njit(cache=True)
def _sample(volume: np.ndarray, x: float, y: float, z: float) -> float:
    width, height, depth = volume.shape
    if x < -0.5 or y < -0.5 or z < -0.5 or x >= width - 0.5 or y >= height - 0.5 or z >= depth - 0.5:
        return 0.0
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    z = min(max(z, 0.0), depth - 1.0)
    xm, ym, zm = int(x), int(y), int(z)
    xp, yp, zp = min(width - 1, xm + 1), min(height - 1, ym + 1), min(depth - 1, zm + 1)
    xmd, ymd, zmd = x - xm, y - ym, z - zm
    xpd, ypd, zpd = 1.0 - xmd, 1.0 - ymd, 1.0 - zmd
    return (
        xpd * ypd * zpd * float(volume[xm, ym, zm])
        + xpd * ypd * zmd * float(volume[xm, ym, zp])
        + xpd * ymd * zpd * float(volume[xm, yp, zm])
        + xpd * ymd * zmd * float(volume[xm, yp, zp])
        + xmd * ypd * zpd * float(volume[xp, ym, zm])
        + xmd * ypd * zmd * float(volume[xp, ym, zp])
        + xmd * ymd * zpd * float(volume[xp, yp, zm])
        + xmd * ymd * zmd * float(volume[xp, yp, zp])
    )


@njit(cache=True)
def _derivative(volume: np.ndarray, x: float, y: float, z: float, dx: float, dy: float, dz: float, sigma: float) -> float:
    width, height, depth = volume.shape
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    z = min(max(z, 0.0), depth - 1.0)
    step = max(0.25, sigma / 5.0)
    vp = vm = total_weight = total_distance = 0.0
    distance = step
    while distance <= max(2.0 * sigma, step):
        weight = np.exp(-distance * distance / (2.0 * sigma * sigma)) if sigma else 1.0
        total_weight += weight
        total_distance += distance
        vp += weight * _sample(volume, x + distance * dx, y + distance * dy, z + distance * dz)
        vm += weight * _sample(volume, x - distance * dx, y - distance * dy, z - distance * dz)
        distance += step
    return (vp / total_weight - vm / total_weight) / (2.0 * total_distance / total_weight)


@njit(cache=True)
def _voxel(affine: np.ndarray, x: float, y: float, z: float) -> tuple[float, float, float]:
    # MatrixMultiply receives a float32 vector and accumulates in float32.
    point = np.empty(4, dtype=np.float32)
    point[0], point[1], point[2], point[3] = np.float32(x), np.float32(y), np.float32(z), np.float32(1.0)
    result = np.empty(3, dtype=np.float32)
    for row in range(3):
        value = np.float32(0.0)
        for col in range(4):
            value = np.float32(value + np.float32(affine[row, col] * point[col]))
        result[row] = value
    return float(result[0]), float(result[1]), float(result[2])


@njit(cache=True)
def _opposite(label: int, left: bool) -> bool:
    labels = _RH_LABELS if left else _LH_LABELS
    for value in labels:
        if label == value:
            return True
    return False


@njit(cache=True)
def _search(
    volume: np.ndarray,
    tmp: np.ndarray,
    seg: np.ndarray,
    affine: np.ndarray,
    xyz: np.ndarray,
    normals: np.ndarray,
    original: np.ndarray,
    ripped: np.ndarray,
    previous_values: np.ndarray,
    left: bool,
    pial: bool,
    thresholds: np.ndarray,
    sigma: float,
    max_thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nvertices = len(xyz)
    values = previous_values.copy()
    distances = np.zeros(nvertices, dtype=np.float32)
    means = np.zeros(nvertices, dtype=np.float32)
    target = xyz.copy()
    marked = np.zeros(nvertices, dtype=np.int32)
    used_sigma = np.zeros(nvertices, dtype=np.float32)
    inside_hi, border_hi, border_low, outside_low, outside_hi = thresholds
    # 未启用parallel的dispatcher将prange按原串行range执行。
    for v in prange(nvertices):
        if ripped[v]:
            continue
        px, py, pz = float(xyz[v, 0]), float(xyz[v, 1]), float(xyz[v, 2])
        nx, ny, nz = float(normals[v, 0]), float(normals[v, 1]), float(normals[v, 2])
        xw, yw, zw = _voxel(affine, px, py, pz)
        xw1, yw1, zw1 = _voxel(affine, px + nx, py + ny, pz + nz)
        vx, vy, vz = np.float32(xw1 - xw), np.float32(yw1 - yw), np.float32(zw1 - zw)
        vnorm = np.float32(np.sqrt(np.float32(vx * vx + vy * vy + vz * vz)))
        if vnorm == 0:
            continue
        vx, vy, vz = np.float32(vx / vnorm), np.float32(vy / vnorm), np.float32(vz / vnorm)
        orig_dist = abs((px - float(original[v, 0])) * nx + (py - float(original[v, 1])) * ny + (pz - float(original[v, 2])) * nz)
        current_sigma = sigma
        inward = 0.25
        outward = -0.25
        while current_sigma <= 10.0 * sigma:
            dist = np.float32(0.0)
            while dist > -max_thickness:
                if abs(float(dist)) + orig_dist > max_thickness:
                    break
                x, y, z = _voxel(affine, px + nx * float(dist), py + ny * float(dist), pz + nz * float(dist))
                mag = _derivative(tmp, x, y, z, vx, vy, vz, current_sigma)
                if mag >= 0.0 or _sample(volume, x, y, z) > border_hi:
                    break
                dist = np.float32(dist - 0.5)
            inward = float(dist) + 0.25
            dist = np.float32(0.0)
            while dist < max_thickness:
                if abs(float(dist)) + orig_dist > max_thickness:
                    break
                x, y, z = _voxel(affine, px + nx * float(dist), py + ny * float(dist), pz + nz * float(dist))
                mag = _derivative(tmp, x, y, z, vx, vy, vz, current_sigma)
                if mag >= 0.0 or _sample(volume, x, y, z) < border_low:
                    break
                dist = np.float32(dist + 0.5)
            outward = float(dist) - 0.25
            if inward <= 0 or outward >= 0:
                break
            current_sigma *= 2.0
        if inward > 0 and outward < 0:
            current_sigma = sigma
        used_sigma[v] = current_sigma

        max_mag_val, max_mag, min_val, min_dist = -10.0, 0.0, 10000.0, 0.0
        local_max, best_dist = False, 0.0
        dist = np.float32(inward)
        while float(dist) <= outward:
            d = float(dist)
            x, y, z = _voxel(affine, px + nx * d, py + ny * d, pz + nz * d)
            prev_x, prev_y, prev_z = _voxel(affine, px + nx * (d - 0.1), py + ny * (d - 0.1), pz + nz * (d - 0.1))
            previous = _sample(volume, prev_x, prev_y, prev_z)
            if previous < inside_hi and previous >= border_low:
                value = _sample(volume, x, y, z)
                if value < min_val:
                    min_val, min_dist = value, d
                next_x, next_y, next_z = _voxel(affine, px + nx * (d + 0.1), py + ny * (d + 0.1), pz + nz * (d + 0.1))
                next_mag = _derivative(tmp, next_x, next_y, next_z, vx, vy, vz, sigma)
                prev_mag = _derivative(tmp, prev_x, prev_y, prev_z, vx, vy, vz, sigma)
                mag = _derivative(tmp, x, y, z, vx, vy, vz, sigma)
                if 0 <= x < seg.shape[0] and 0 <= y < seg.shape[1] and 0 <= z < seg.shape[2]:
                    label = int(seg[int(np.floor(x + 0.5)), int(np.floor(y + 0.5)), int(np.floor(z + 0.5))])
                    if _opposite(label, left):
                        break
                if pial and _sample(volume, next_x, next_y, next_z) < border_low:
                    next_mag = 0.0
                if abs(mag) > abs(prev_mag) and abs(mag) > abs(next_mag) and border_low <= value <= border_hi:
                    # 固定源码 dist+1 的1是整数，先以float计算距离、乘积和
                    # 坐标；d+1.0在Numba提升double，会改变外侧阈值筛选。
                    ox, oy, oz = _voxel(affine, np.float32(px + np.float32(nx * np.float32(d + 1.0))), np.float32(py + np.float32(ny * np.float32(d + 1.0))), np.float32(pz + np.float32(nz * np.float32(d + 1.0))))
                    next_val = _sample(volume, ox, oy, oz)
                    if outside_low <= next_val <= border_hi and next_val <= outside_hi and (not local_max or max_mag < abs(mag)):
                        local_max = True
                        best_dist, max_mag, max_mag_val = d, abs(mag), value
                elif not local_max and abs(mag) > max_mag and border_low <= value <= border_hi:
                    ox, oy, oz = _voxel(affine, np.float32(px + np.float32(nx * np.float32(d + 1.0))), np.float32(py + np.float32(ny * np.float32(d + 1.0))), np.float32(pz + np.float32(nz * np.float32(d + 1.0))))
                    next_val = _sample(volume, ox, oy, oz)
                    if outside_low <= next_val <= border_hi and next_val < outside_hi:
                        best_dist, max_mag, max_mag_val = d, abs(mag), value
            dist = np.float32(dist + 0.1)

        if pial and not local_max and best_dist > 0:
            allgray = True
            outlen = np.float32(best_dist)
            while float(outlen) < best_dist + 2.0:
                ox, oy, oz = _voxel(affine, px + nx * float(outlen), py + ny * float(outlen), pz + nz * float(outlen))
                outval = _sample(volume, ox, oy, oz)
                if outval < outside_hi or outval > border_hi:
                    allgray = False
                    break
                outlen = np.float32(outlen + 0.1)
            if allgray:
                max_mag_val, best_dist = -10.0, 0.0

        if max_mag_val > 0:
            values[v] = np.float32(max(max_mag_val, border_low))
            distances[v] = np.float32(best_dist)
            means[v] = np.float32(max_mag)
            marked[v] = 1
        elif min_val < 1000:
            values[v] = np.float32(max(min_val, border_low))
            distances[v] = np.float32(min_dist)
            marked[v] = 1
        elif values[v] >= 0:
            marked[v] = 1
        target[v, 0] = np.float32(px + nx * float(distances[v]))
        target[v, 1] = np.float32(py + ny * float(distances[v]))
        target[v, 2] = np.float32(pz + nz * float(distances[v]))
    return values, distances, means, target, marked, used_sigma


def _parallel_search_dispatcher():
    """从同一函数体创建并行dispatcher，使用独立缓存名，不复制搜索算法。

    只改函数身份和编译选项；每顶点内部运算及float32舍入保留。
    线程数由调用方既定Numba预算控制，此处不修改全局线程或精度。
    """
    source = _search.py_func
    function = FunctionType(source.__code__.replace(co_name="_search_parallel"),
                            source.__globals__, "_search_parallel",
                            source.__defaults__, source.__closure__)
    function.__qualname__ = "_search_parallel"
    function.__module__ = __name__
    return njit(cache=True, parallel=True, fastmath=False)(function)


_search_parallel = _parallel_search_dispatcher()


def compute_border_values_first_pass(
    volume: np.ndarray,
    segmentation: np.ndarray,
    surface_xyz: np.ndarray,
    surface_normals: np.ndarray,
    original_xyz: np.ndarray,
    ripped: np.ndarray,
    previous_values: np.ndarray,
    sras2vox: np.ndarray,
    thresholds: np.ndarray,
    *,
    hemisphere: str,
    surface: str = "white",
    sigma: float = 2.0,
    max_thickness: float = 10.0,
    target_search_backend: str = "serial",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算一次固定状态下的目标值，供完整四轮放置流程复用。

    输入：volume为已预处理(W,H,D)强度图，转换为uint8；segmentation为
    同网格标签图，转换为int32；surface_xyz、surface_normals、original_xyz
    分别为(N,3)的surface RAS毫米坐标、单位法向及初始坐标，转换为float32；
    ripped与previous_values为(N,)的排除标记和上轮目标值；sras2vox为(4,4)
    surface RAS毫米到体素索引仿射；thresholds为五项强度阈值，顺序为
    inside_hi/border_hi/border_low/outside_low/outside_hi，计算使用float64。
    hemisphere仅lh/rh；surface默认white，也支持pial；sigma默认2.0，单位
    为体素；max_thickness默认10.0mm，限定法向搜索总距离（包含初始位移）。
    target_search_backend默认serial；parallel只并行独立顶点搜索，使用同一
    函数体和已有Numba线程预算，fastmath=False。有序目标平均与顶点接受
    不在此函数内，不因本选项改变。首次JIT属于调用墙钟，无静默回退。

    输出依次为(N,)float32目标强度、法向距离mm、峰导数、(N,3)float32
    目标surface RAS毫米坐标、(N,)int32有效标记、(N,)float32实际sigma。
    峰导数仅本次找到峰时更新；本接口没有上轮mean输入，因此回退时返回0，
    完整放置不使用该诊断量。没有找到新目标时保留previous_values。

    hemisphere/surface/target_search_backend非法抛ValueError；不可转换输入抛NumPy转换异常。
    其余输入须满足上述shape、同网格及有限值契约，Numba路径不对所有
    越界或非有限输入提供保护。对应mris_place_surface内部
    MRIScomputeBorderValues_new，无独立官方CLI；真实逐轮证据见
    docs/recon_all/WHITE_BORDER_TARGET_PRECISION_20261010.md。
    """
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    if surface not in ("white", "pial"):
        raise ValueError("surface must be white or pial")
    if target_search_backend not in ("serial", "parallel"):
        raise ValueError("target_search_backend must be serial or parallel")
    source_volume = np.asarray(volume, dtype=np.uint8)
    search = _search if target_search_backend == "serial" else _search_parallel
    return search(
        source_volume,
        np.where(source_volume == 255, 0, source_volume).astype(np.uint8),
        np.asarray(segmentation, dtype=np.int32),
        np.asarray(sras2vox, dtype=np.float32),
        np.asarray(surface_xyz, dtype=np.float32),
        np.asarray(surface_normals, dtype=np.float32),
        np.asarray(original_xyz, dtype=np.float32),
        np.asarray(ripped, dtype=np.bool_),
        np.asarray(previous_values, dtype=np.float32),
        hemisphere == "lh",
        surface == "pial",
        np.asarray(thresholds, dtype=np.float64),
        float(sigma),
        float(max_thickness),
    )
