"""Source-order signed fast marching and medial WM ridge for mri_normalize."""

from __future__ import annotations

import math

from numba import njit
import numpy as np


@njit(cache=True)
def _heap_push(heap, count, item, distance, sign):
    hole = count
    count += 1
    while hole:
        parent = (hole - 1) // 2
        if sign * distance[heap[parent]] <= sign * distance[item]:
            break
        heap[hole] = heap[parent]
        hole = parent
    heap[hole] = item
    return count


@njit(cache=True)
def _heap_pop(heap, count, distance, sign):
    top = heap[0]
    count -= 1
    if count == 0:
        return top, count
    item = heap[count]
    hole = 0
    child = 2
    while child < count:
        if sign * distance[heap[child]] > sign * distance[heap[child - 1]]:
            child -= 1
        heap[hole] = heap[child]
        hole = child
        child = 2 * (hole + 1)
    if child == count:
        heap[hole] = heap[child - 1]
        hole = child - 1
    while hole:
        parent = (hole - 1) // 2
        if sign * distance[heap[parent]] <= sign * distance[item]:
            break
        heap[hole] = heap[parent]
        hole = parent
    heap[hole] = item
    return top, count


@njit(cache=True)
def _neighbor_value(index, distance, status, limit):
    if status[index] == 2:
        return limit
    return distance[index]


@njit(cache=True)
def _update_value(index, distance, status, sign, limit, sx, sy, sz):
    stride = sy * sz
    x = index // stride
    y = (index // sz) % sy
    z = index % sz
    if x == 0:
        a = np.float32(sign * _neighbor_value(index + stride, distance, status, limit))
    elif x == sx - 1:
        a = np.float32(sign * _neighbor_value(index - stride, distance, status, limit))
    else:
        a = min(np.float32(sign * _neighbor_value(index - stride, distance, status, limit)),
                np.float32(sign * _neighbor_value(index + stride, distance, status, limit)))
    if y == 0:
        b = np.float32(sign * _neighbor_value(index + sz, distance, status, limit))
    elif y == sy - 1:
        b = np.float32(sign * _neighbor_value(index - sz, distance, status, limit))
    else:
        b = min(np.float32(sign * _neighbor_value(index - sz, distance, status, limit)),
                np.float32(sign * _neighbor_value(index + sz, distance, status, limit)))
    if z == 0:
        c = np.float32(sign * _neighbor_value(index + 1, distance, status, limit))
    elif z == sz - 1:
        c = np.float32(sign * _neighbor_value(index - 1, distance, status, limit))
    else:
        c = min(np.float32(sign * _neighbor_value(index - 1, distance, status, limit)),
                np.float32(sign * _neighbor_value(index + 1, distance, status, limit)))
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    result = np.float32(sign * np.float32(a + np.float32(1)))
    bb = np.float32(-(a + b + c))
    cc = np.float32(a * a + b * b + c * c - np.float32(1))
    delta = np.float32(bb * bb - np.float32(3) * cc)
    if delta >= 0:
        solution = np.float32((-np.float64(bb) + math.sqrt(np.float64(delta))) / 3.0)
        if solution + np.float32(1e-6) >= c:
            return np.float32(sign * solution)
    bb = np.float32(-(a + b))
    cc = np.float32(a * a + b * b - np.float32(1))
    delta = np.float32(bb * bb - np.float32(2) * cc)
    if delta >= 0:
        solution = np.float32((-np.float64(bb) + math.sqrt(np.float64(delta))) / 2.0)
        if solution + np.float32(1e-6) >= b:
            result = np.float32(sign * solution)
    return result


@njit(cache=True)
def _add_alive(index, status, distance, alive, count, sign):
    if status[index] == 2:
        status[index] = 0
        distance[index] = np.float32(sign * 0.5)
        alive[count] = index
        count += 1
    return count


@njit(cache=True)
def _march_pass(mask, distance, sign, sx, sy, sz, stop_query=None):
    nvox = sx * sy * sz
    stride = sy * sz
    limit = np.float32(sign * 2 * max(sx, sy, sz))
    status = np.empty(nvox, np.uint8)
    for index in range(nvox):
        if (mask[index] != 0) == (sign < 0):
            status[index] = 2
            distance[index] = limit
        else:
            status[index] = 3
    alive = np.empty(nvox, np.int32)
    alive_count = 0
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                index = x * stride + y * sz + z
                added = False
                if x < sx - 1 and mask[index] != mask[index + stride]:
                    alive_count = _add_alive(index + stride, status, distance, alive, alive_count, sign)
                    added = True
                if y < sy - 1 and mask[index] != mask[index + sz]:
                    alive_count = _add_alive(index + sz, status, distance, alive, alive_count, sign)
                    added = True
                if z < sz - 1 and mask[index] != mask[index + 1]:
                    alive_count = _add_alive(index + 1, status, distance, alive, alive_count, sign)
                    added = True
                if added:
                    alive_count = _add_alive(index, status, distance, alive, alive_count, sign)
    heap = np.empty(nvox, np.int32)
    heap_count = 0
    for ai in range(alive_count):
        index = alive[ai]
        x = index // stride
        y = (index // sz) % sy
        z = index % sz
        for neighbor, valid in ((index - stride, x > 0), (index + stride, x < sx - 1),
                                (index - sz, y > 0), (index + sz, y < sy - 1),
                                (index - 1, z > 0), (index + 1, z < sz - 1)):
            if valid and status[neighbor] == 2:
                status[neighbor] = 1
                distance[neighbor] = _update_value(neighbor, distance, status, sign,
                                                   limit, sx, sy, sz)
                heap_count = _heap_push(heap, heap_count, neighbor, distance, sign)
    processed = 0
    # 默认完整距离场不变。限域调用只在所有下游查询点已确定时结束；
    # 已pop的状态0永不重新更新，所以前缀逐步与完整堆完全相同。
    remaining_query = -1
    if stop_query is not None:
        remaining_query = 0
        for index in range(nvox):
            if stop_query[index] and status[index] != 0:
                remaining_query += 1
    while heap_count:
        if remaining_query == 0:
            break
        index = heap[0]
        if np.float32(sign * distance[index]) >= np.float32(sign * limit):
            break
        index, heap_count = _heap_pop(heap, heap_count, distance, sign)
        status[index] = 0
        processed += 1
        if stop_query is not None and stop_query[index]:
            remaining_query -= 1
        x = index // stride
        y = (index // sz) % sy
        z = index % sz
        for neighbor, valid in ((index - stride, x > 0), (index + stride, x < sx - 1),
                                (index - sz, y > 0), (index + sz, y < sy - 1),
                                (index - 1, z > 0), (index + 1, z < sz - 1)):
            if not valid:
                continue
            if status[neighbor] == 2:
                status[neighbor] = 1
                distance[neighbor] = _update_value(neighbor, distance, status, sign,
                                                   limit, sx, sy, sz)
                heap_count = _heap_push(heap, heap_count, neighbor, distance, sign)
            elif status[neighbor] == 1:
                distance[neighbor] = _update_value(neighbor, distance, status, sign,
                                                   limit, sx, sy, sz)
    if stop_query is not None and remaining_query > 0:
        raise RuntimeError("marching terminated before every requested point settled")
    while heap_count:
        index, heap_count = _heap_pop(heap, heap_count, distance, sign)
        status[index] = 2
        distance[index] = limit
    return alive_count, processed


def signed_distance(mask: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Replay ``MRIdistanceTransform(..., DTRANS_MODE_SIGNED)`` at 1 mm."""
    mask = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    distance = np.zeros(mask.size, np.float32)
    flat = mask.ravel()
    first = _march_pass(flat, distance, 1, *mask.shape)
    second = _march_pass(flat, distance, -1, *mask.shape)
    return -distance.reshape(mask.shape), (first, second)


@njit(cache=True)
def _sample(source, x, y, z):
    sx, sy, sz = source.shape
    # MRIdistanceTransform keeps outside_val=-1 after the in-place sign flip.
    if (x < 0 or x > sx - 1 or y < 0 or y > sy - 1 or z < 0 or z > sz - 1):
        if (np.rint(x) < 0 or np.rint(x) >= sx or np.rint(y) < 0 or
                np.rint(y) >= sy or np.rint(z) < 0 or np.rint(z) >= sz):
            return -1.0
    # MRIsampleVolume takes its nearest-neighbor path at integer coordinates.
    if (abs(x - int(x)) < np.finfo(np.float32).eps and
            abs(y - int(y)) < np.finfo(np.float32).eps and
            abs(z - int(z)) < np.finfo(np.float32).eps):
        return source[int(x), int(y), int(z)]
    x = min(max(x, 0.0), sx - 1.0)
    y = min(max(y, 0.0), sy - 1.0)
    z = min(max(z, 0.0), sz - 1.0)
    xm, ym, zm = int(x), int(y), int(z)
    xp, yp, zp = min(xm + 1, sx - 1), min(ym + 1, sy - 1), min(zm + 1, sz - 1)
    xd, yd, zd = x - xm, y - ym, z - zm
    ix, iy, iz = 1.0 - xd, 1.0 - yd, 1.0 - zd
    return (ix * iy * iz * source[xm, ym, zm] + ix * iy * zd * source[xm, ym, zp]
            + ix * yd * iz * source[xm, yp, zm] + ix * yd * zd * source[xm, yp, zp]
            + xd * iy * iz * source[xp, ym, zm] + xd * iy * zd * source[xp, ym, zp]
            + xd * yd * iz * source[xp, yp, zm] + xd * yd * zd * source[xp, yp, zp])


@njit(cache=True)
def _nonmax(distance):
    sx, sy, sz = distance.shape
    ridge = np.zeros(distance.shape, np.uint8)
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                value = distance[x, y, z]
                if value < 1:
                    continue
                gx = (distance[min(x + 1, sx - 1), y, z]
                      - distance[max(x - 1, 0), y, z]) * 0.5
                gy = (distance[x, min(y + 1, sy - 1), z]
                      - distance[x, max(y - 1, 0), z]) * 0.5
                gz = (distance[x, y, min(z + 1, sz - 1)]
                      - distance[x, y, max(z - 1, 0)]) * 0.5
                if abs(gz) >= abs(gy) and abs(gz) >= abs(gx):
                    denominator = gz
                elif abs(gy) >= abs(gx):
                    denominator = gy
                else:
                    denominator = gx
                if abs(denominator) < 1e-15:
                    ridge[x, y, z] = 1
                    continue
                dx, dy, dz = gx / denominator, gy / denominator, gz / denominator
                if _sample(distance, x + dx, y + dy, z + dz) > value:
                    continue
                if _sample(distance, x - dx, y - dy, z - dz) > value:
                    continue
                ridge[x, y, z] = 1
    return ridge


def medial_ridge(aseg: np.ndarray) -> tuple[np.ndarray, dict]:
    """Generate the pre-outlier WM controls independently from aseg labels 2/41."""
    distance, passes = signed_distance((aseg == 2) | (aseg == 41))
    ridge = _nonmax(np.ascontiguousarray(distance))
    return ridge, {"marching": passes, "ridge_voxels": int(np.count_nonzero(ridge))}
