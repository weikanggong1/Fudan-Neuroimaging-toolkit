"""Aseg-guided hemisphere fill from FreeSurfer's mri_fill (8.2.0)."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage as ndi
import time


_LEFT = (2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 19, 20, 25, 30, 26, 28, 31, 27,
         32, 33, 34, 35, 36, 37, 38, 39)
_RIGHT = (41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55, 56, 57, 62, 58, 60, 63,
          59, 64, 65, 66, 67, 68, 69, 70, 71)
_ERASE = (7, 8, 16, 46, 47)
_WMSA = (77, 78, 79, 87, 88)


def _cc_outside_distance(seg: np.ndarray, cc: np.ndarray, label: int, *,
                         boundary_backend: str = "python", device: str = "cpu",
                         marching_backend: str = "python",
                         profile: dict | None = None) -> np.ndarray:
    """固定标签的外部fast marching距离，所有CC查询决定后停止。

    seg/cc为同三维x/y/z网格整数分割和bool查询，label为2或41。
    boundary_backend默认python保留原扫描，torch仅替换完整边界初始化；
    device默认cpu、torch后端可明确cuda。返回同网格float32体素距离；
    marching_backend默认python，numba编译同有序heap/eikonal，关闭
    fastmath并明确double sqrt提升。profile默认None；指定dict写分段秒，
    不改变计算。无目标标签仍保留原空数组失败，非法后端抛ValueError。
    """
    if boundary_backend not in {"python", "torch"}:
        raise ValueError("boundary_backend must be python or torch")
    if marching_backend not in {"python", "numba"}:
        raise ValueError("marching_backend must be python or numba")
    started = time.perf_counter() if profile is not None else 0.
    points = np.argwhere(cc)
    label_points = np.argwhere(seg == label)
    lo = np.maximum(np.minimum(points.min(axis=0), label_points.min(axis=0)) - 12, 0)
    hi = np.minimum(np.maximum(points.max(axis=0), label_points.max(axis=0)) + 13, seg.shape)
    area = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    target = seg[area] == label
    query = cc[area]
    prepared = time.perf_counter() if profile is not None else 0.
    # FreeSurfer initializes outside boundary voxels at +0.5 voxel.
    distance = np.full(target.shape, np.float32(100.0), dtype=np.float32)
    state = np.zeros(target.shape, dtype=np.uint8)  # 0 far, 1 trial, 2 alive
    state[target] = 3  # forbidden
    distance[target] = 0.0
    heap: list[tuple[int, int, int]] = []
    sx, sy, sz = target.shape

    alive: list[tuple[int, int, int]] = []

    def add_alive(x: int, y: int, z: int) -> None:
        if state[x, y, z] == 3:
            return
        distance[x, y, z] = np.float32(0.5)
        if state[x, y, z] == 0:
            state[x, y, z] = 2
            alive.append((x, y, z))

    if boundary_backend == "python":
        for z in range(sz):
            for y in range(sy):
                for x in range(sx):
                    val = target[x, y, z]
                    changed = False
                    if x + 1 < sx and val != target[x + 1, y, z]:
                        changed = True
                        add_alive(x + 1, y, z)
                    if y + 1 < sy and val != target[x, y + 1, z]:
                        changed = True
                        add_alive(x, y + 1, z)
                    if z + 1 < sz and val != target[x, y, z + 1]:
                        changed = True
                        add_alive(x, y, z + 1)
                    if changed:
                        add_alive(x, y, z)
    else:
        import torch
        from .fill_boundary_torch import initialize_cc_boundary_torch
        boundary = initialize_cc_boundary_torch(torch.from_numpy(target).to(device))
        distance, state = (tensor.cpu().numpy() for tensor in boundary[:2])
        alive = [tuple(point) for point in boundary[2].cpu().tolist()]
    surface = state == 2
    initialized = time.perf_counter() if profile is not None else 0.

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
        a = min(distance[x - 1, y, z] if x else 100,
                distance[x + 1, y, z] if x + 1 < sx else 100)
        b = min(distance[x, y - 1, z] if y else 100,
                distance[x, y + 1, z] if y + 1 < sy else 100)
        c = min(distance[x, y, z - 1] if z else 100,
                distance[x, y, z + 1] if z + 1 < sz else 100)
        a, b, c = sorted((np.float32(a), np.float32(b), np.float32(c)))
        one = np.float32(1)
        value = np.float32(a + one)
        sum3 = np.float32(np.float32(a + b) + c)
        squares3 = np.float32(np.float32(a * a + b * b) + c * c)
        delta = np.float32(sum3 * sum3 - np.float32(3) * np.float32(squares3 - one))
        solved = False
        if delta >= 0:
            solution = np.float32((float(sum3) + float(np.sqrt(float(delta)))) / 3.0)
            if np.float32(solution + np.float32(1e-6)) >= c:
                value = solution
                solved = True
        if not solved:
            sum2 = np.float32(a + b)
            squares2 = np.float32(a * a + b * b)
            delta = np.float32(sum2 * sum2 - np.float32(2) * np.float32(squares2 - one))
            if delta >= 0:
                solution = np.float32((float(sum2) + float(np.sqrt(float(delta)))) / 2.0)
                if np.float32(solution + np.float32(1e-6)) >= b:
                    value = solution
        distance[x, y, z] = np.float32(value)
        if state[x, y, z] == 0:
            state[x, y, z] = 1
            push((x, y, z))

    def update_neighbors(x: int, y: int, z: int, initial: bool = False) -> None:
        for xx, yy, zz in ((x - 1, y, z), (x + 1, y, z),
                           (x, y - 1, z), (x, y + 1, z),
                           (x, y, z - 1), (x, y, z + 1)):
            if 0 <= xx < sx and 0 <= yy < sy and 0 <= zz < sz:
                if not initial or state[xx, yy, zz] == 0:
                    update(xx, yy, zz)

    if marching_backend == "numba":
        from .fill_marching_numba import march_cc_distance_numba
        trial_initialized = initialized
        remaining = march_cc_distance_numba(distance, state,
            np.asarray(alive, dtype=np.int64).reshape(-1, 3), query)
    else:
        for x, y, z in alive:
            update_neighbors(x, y, z, initial=True)
        remaining = int(np.count_nonzero(query & ~surface))
        trial_initialized = time.perf_counter() if profile is not None else 0.
        while remaining and heap:
            x, y, z = pop()
            state[x, y, z] = 2
            if query[x, y, z]:
                remaining -= 1
            update_neighbors(x, y, z)
    iterated = time.perf_counter() if profile is not None else 0.
    result = np.full(seg.shape, np.float32(100.0), dtype=np.float32)
    result[area] = distance
    if profile is not None:
        finished = time.perf_counter()
        profile.update({"label": int(label), "boundary_backend": boundary_backend,
            "marching_backend": marching_backend,
            "numba_heap_timing_includes_trial_initialization": marching_backend == "numba",
            "boundary_device": str(device) if boundary_backend == "torch" else "cpu",
            "crop_shape": [int(axis) for axis in target.shape], "alive_count": len(alive),
            "preparation_seconds": prepared - started,
            "boundary_initialization_seconds": initialized - prepared,
            "trial_initialization_seconds": trial_initialized - initialized,
            "heap_eikonal_seconds": iterated - trial_initialized,
            "output_assembly_seconds": finished - iterated,
            "total_seconds": finished - started, "unsettled_cc_queries": remaining})
    return result


@njit(cache=True)
def _voronoi_round(values: np.ndarray, distances: np.ndarray,
                   coordinates: np.ndarray, radius: int) -> None:
    sx, sy, sz = values.shape
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        total = 0
        count = 0
        for dx in (-1, 0, 1):
            xi = min(max(x + dx, 0), sx - 1)
            for dy in (-1, 0, 1):
                yi = min(max(y + dy, 0), sy - 1)
                for dz in (-1, 0, 1):
                    zi = min(max(z + dz, 0), sz - 1)
                    if distances[xi, yi, zi] < radius:
                        total += int(values[xi, yi, zi])
                        count += 1
        if count:
            values[x, y, z] = total // count


@njit(cache=True)
def _edited_on_votes(fill: np.ndarray, wm: np.ndarray, aseg: np.ndarray,
                     voxel_xsize: float) -> None:
    sx, sy, sz = wm.shape
    coords = np.argwhere(wm == 255)
    for i in range(len(coords)):
        x, y, z = coords[i]
        radius = int(np.ceil(5.0 / voxel_xsize))
        while True:
            left = 0
            right = 0
            for dx in range(-radius, radius + 1):
                xi = min(max(x + dx, 0), sx - 1)
                for dy in range(-radius, radius + 1):
                    yi = min(max(y + dy, 0), sy - 1)
                    for dz in range(-radius, radius + 1):
                        zi = min(max(z + dz, 0), sz - 1)
                        label = aseg[xi, yi, zi]
                        if label == 2 or label == 3:
                            left += 1
                        elif label == 41 or label == 42:
                            right += 1
            if left or right or radius > 20:
                break
            radius += 1
        fill[x, y, z] = 255 if left > right else 127


def _largest_then_fill_holes(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, 2))
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == np.argmax(sizes)
    return ndi.binary_fill_holes(mask, structure=ndi.generate_binary_structure(3, 1))


def _replace_cc_with_wm(aseg: np.ndarray, *, boundary_backend: str = "python",
                         device: str = "cpu", marching_backend: str = "python",
                         profiles: list[dict] | None = None) -> np.ndarray:
    """将251..255 CC按有序外部距离分成2/41；返回新int32分割。

    aseg为三维整数语义x/y/z网格。boundary_backend默认python、torch只
    改边界初始化；device默认cpu；profiles默认None，可追加两侧分段秒。
    marching_backend默认python，可指定numba编译同一有序传播。
    不改变输入、标签同距选右规则或heap；无CC时直接返回副本。
    """
    if boundary_backend not in {"python", "torch"}:
        raise ValueError("boundary_backend must be python or torch")
    if marching_backend not in {"python", "numba"}:
        raise ValueError("marching_backend must be python or numba")
    seg = np.asarray(aseg, dtype=np.int32).copy()
    if np.any((seg >= 251) & (seg <= 255)):
        cc = (seg >= 251) & (seg <= 255)
        for label in (2, 41):
            row = {} if profiles is not None else None
            distance = _cc_outside_distance(seg, cc, label,
                boundary_backend=boundary_backend, device=device,
                marching_backend=marching_backend, profile=row)
            if profiles is not None:
                profiles.append(row)
            if label == 2:
                left_distance = distance
            else:
                right_distance = distance
        seg[cc] = np.where(left_distance[cc] < right_distance[cc], 2, 41)
    return seg


def _fill_preclassified(wm: np.ndarray, seg: np.ndarray, voxel_xsize: float,
                        cc_cut_mask: np.ndarray | None) -> np.ndarray:

    image = wm.copy()
    if cc_cut_mask is not None:
        if cc_cut_mask.shape != wm.shape:
            raise ValueError("cc_cut_mask shape must match wm")
        image[np.asarray(cc_cut_mask, dtype=bool) & np.isin(image, (200, 210, 220, 230, 240))] = 0
    image[np.isin(seg, _ERASE)] = 0
    left = np.isin(seg, _LEFT)
    right = np.isin(seg, _RIGHT)
    control = left | right
    fill = np.zeros(wm.shape, dtype=np.uint8)
    fill[left] = 255
    fill[right] = 127

    active = (image >= 5) | np.isin(seg, (25, 57, *_WMSA))
    if not np.any(control) or not np.any(active):
        return np.zeros(wm.shape, dtype=np.uint8)
    distances = ndi.distance_transform_cdt(~control, metric="chessboard")
    for radius in range(1, int(distances[active].max()) + 1):
        coordinates = np.argwhere(distances == radius)
        _voronoi_round(fill, distances, coordinates, radius)

    _edited_on_votes(fill, image, seg, voxel_xsize)
    active &= image != 1
    left_mask = active & (fill != 127)
    right_mask = active & (fill == 127)
    left_mask = _largest_then_fill_holes(left_mask)
    right_mask = _largest_then_fill_holes(right_mask)
    result = np.zeros(wm.shape, dtype=np.uint8)
    result[right_mask] = 127
    result[left_mask] = 255
    return result


def fill_with_aseg(wm: np.ndarray, aseg: np.ndarray, voxel_xsize: float = 1.0,
                  cc_cut_mask: np.ndarray | None = None, *,
                   cc_boundary_backend: str = "python", device: str = "cpu",
                   cc_marching_backend: str = "python") -> np.ndarray:
    """既有完整aseg引导填充，返回新三维uint8的0/127/255标签。

    wm/aseg为同x/y/z体素网格，wm为uint8；voxel_xsize默认1.0mm；
    cc_cut_mask默认None、存在时为同网格切割bool；cc_boundary_backend
    默认python，torch仅替换CC距离边界；device默认cpu。
    cc_marching_backend默认python，numba复用同序传播，首次JIT单列。
    输入不改、
    有序Voronoi/heap/孔填充不变；非法输入抛ValueError，CUDA失败传播。
    """
    if wm.shape != aseg.shape or wm.ndim != 3 or wm.dtype != np.uint8:
        raise ValueError("wm and aseg must be equal-shape 3D volumes, wm uint8")
    seg = _replace_cc_with_wm(aseg, boundary_backend=cc_boundary_backend, device=device,
                            marching_backend=cc_marching_backend)
    return _fill_preclassified(wm, seg, voxel_xsize, cc_cut_mask)
