"""Source-order CPU port of the fixed ``mri_edit_wm_with_aseg`` call."""

import argparse
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from numba import njit


@njit(cache=True)
def _clamp(value: int, upper: int) -> int:
    return min(max(value, 0), upper - 1)


@njit(cache=True)
def _distance_to_label(seg: np.ndarray, label: int, x: int, y: int, z: int,
                       dx: int, dy: int, dz: int, maximum: int) -> int:
    for distance in range(1, maximum + 1):
        xx = _clamp(x + distance * dx, seg.shape[0])
        yy = _clamp(y + distance * dy, seg.shape[1])
        zz = _clamp(z + distance * dz, seg.shape[2])
        if seg[xx, yy, zz] == label:
            return distance
    return maximum + 1


@njit(cache=True)
def _superior_edge(seg: np.ndarray, label: int, x: int, y: int, z: int) -> int:
    for distance in range(seg.shape[1]):
        yy = _clamp(y - distance, seg.shape[1])
        if _distance_to_label(seg, label, x, yy, z, 0, -1, 0, 10) >= 10:
            return distance
    return -1


@njit(cache=True)
def _anterior_edge(seg: np.ndarray, label: int, x: int, y: int, z: int) -> int:
    for distance in range(seg.shape[1]):
        zz = _clamp(z + distance, seg.shape[2])
        if _distance_to_label(seg, label, x, y, zz, 0, 0, 1, 10) >= 10:
            return distance
    return -1


@njit(cache=True)
def _spackle(wm: np.ndarray, aseg: np.ndarray) -> int:
    changed = 0
    for x in range(wm.shape[0]):
        for y in range(1, wm.shape[1]):
            for z in range(wm.shape[2]):
                label = int(aseg[x, y, z])
                if label not in (17, 53, 18, 54):
                    continue
                left = label == 17 or label == 18
                if _superior_edge(aseg, label, x, y, z) > 1:
                    continue
                if (_anterior_edge(aseg, label, x, y, z) < 2 and
                        _anterior_edge(aseg, label, x, y + 1, z) < 2):
                    continue
                yy = y - 1
                above = int(aseg[x, yy, z])
                if above in (3, 42) and wm[x, yy, z] < 5:
                    wm[x, yy, z] = 250
                    changed += 1
                xx = x + 1 if left else x - 1
                if label in (18, 54):
                    lateral = int(aseg[xx, y, z])
                    if lateral in (3, 42) and wm[xx, y, z] < 5:
                        wm[xx, y, z] = 250
                        changed += 1
                    diagonal = int(aseg[xx, yy, z])
                    if diagonal in (3, 42) and wm[xx, yy, z] < 5:
                        wm[xx, yy, z] = 250
                        changed += 1
    return changed


def spackle_wm_superior_to_mtl(wm: np.ndarray, aseg: np.ndarray) -> np.ndarray:
    """Replay the final unconditional core function on a same-grid WM array."""
    if wm.shape != aseg.shape or wm.ndim != 3:
        raise ValueError("Expected matching 3D WM and aseg volumes")
    output = wm.copy()
    _spackle(output, aseg.astype(np.int32, copy=False))
    return output


@njit(cache=True)
def _neighbor(seg: np.ndarray, x: int, y: int, z: int, radius: int, label: int) -> bool:
    for zz in range(max(0, z - radius), min(seg.shape[2], z + radius + 1)):
        for yy in range(max(0, y - radius), min(seg.shape[1], y + radius + 1)):
            for xx in range(max(0, x - radius), min(seg.shape[0], x + radius + 1)):
                if seg[xx, yy, zz] == label:
                    return True
    return False


@njit(cache=True)
def _unknown_nbhd(seg: np.ndarray, x: int, y: int, z: int) -> bool:
    count = 0
    for dz in range(-2, 3):
        zz = _clamp(z + dz, seg.shape[2])
        for dy in range(-2, 3):
            yy = _clamp(y + dy, seg.shape[1])
            for dx in range(-2, 3):
                xx = _clamp(x + dx, seg.shape[0])
                if seg[xx, yy, zz] == 0:
                    count += 1
    return count >= 124


@njit(cache=True)
def _distance_to_wm(wm: np.ndarray, x: int, y: int, z: int,
                    dx: int, dy: int, dz: int, maximum: int, nonzero: bool) -> int:
    for distance in range(1, maximum + 1):
        xx = _clamp(x + distance * dx, wm.shape[0])
        yy = _clamp(y + distance * dy, wm.shape[1])
        zz = _clamp(z + distance * dz, wm.shape[2])
        if (wm[xx, yy, zz] >= 5) == nonzero:
            return distance
    return maximum + 1


@njit(cache=True)
def _medial_hippo(seg: np.ndarray, x: int, y: int, z: int, dx: int) -> bool:
    vdc = 28 if dx < 0 else 60
    if (_distance_to_label(seg, 0, x, y, z, dx, 0, 0, 10) < 10 or
            _distance_to_label(seg, 16, x, y, z, dx, 0, 0, 10) < 10 or
            _distance_to_label(seg, vdc, x, y, z, dx, 0, 0, 10) < 10):
        for step in range(5, 11):
            label = seg[_clamp(x + step * dx, seg.shape[0]), y, z]
            if label in (17, 53, 18, 54):
                return False
        return True
    return False


@njit(cache=True)
def _set_filled(wm: np.ndarray, filled: np.ndarray, x: int, y: int, z: int) -> None:
    wm[x, y, z] = 250
    filled[x, y, z] = 250


@njit(cache=True)
def _edit_until_propagation(wm: np.ndarray, seg: np.ndarray,
                             fill_seg_wm: bool = False,
                             fill_seed=None, propagate_wm: bool = True) -> np.ndarray:
    """原有序首两遍编辑及可选WM传播，原位修改wm并返回uint8 filled。

    wm为3D uint8，seg为同网格整数标签；fill_seg_wm默认False。
    fill_seed默认None，可给同网格bool静态候选，仅在首遍WM分支消费。
    propagate_wm默认True；False留下固定filled供独立GPU传播。
    此私有Numba内核不做参数验证，调用者须先检查网格与dtype。
    """
    width, height, depth = wm.shape
    filled = np.zeros(wm.shape, dtype=np.uint8)
    for z in range(depth):
        for y in range(height - 2, 0, -1):
            for x in range(1, width - 1):
                label = int(seg[x, y, z])
                if label in (0, 6, 7, 8, 45, 46, 47, 85):
                    if wm[x, y, z] < 5:
                        continue
                    if label == 0 and not _unknown_nbhd(seg, x, y, z):
                        continue
                    if not _neighbor(seg, x, y, z, 1, 3) and not _neighbor(seg, x, y, z, 1, 42):
                        wm[x, y, z] = 0
                elif label in (4, 43, 5, 44):
                    if label in (4, 43):
                        if (_neighbor(seg, x, y, z, 1, 2) or _neighbor(seg, x, y, z, 1, 41)) and wm[x, y, z] < 5:
                            _set_filled(wm, filled, x, y, z)
                            continue
                        if wm[x, y, z] < 5:
                            _set_filled(wm, filled, x, y, z)
                    xi = _clamp(x + (1 if label == 5 else -1), width)
                    if seg[xi, y, z] in (3, 42) and wm[xi, y, z] < 5:
                        _set_filled(wm, filled, xi, y, z)
                    yi = _clamp(y + 1, height)
                    if seg[xi, yi, z] in (3, 42) and wm[xi, yi, z] < 5:
                        _set_filled(wm, filled, xi, yi, z)
                    if _distance_to_wm(wm, x, y, z, 0, -1, 0, 5, True) <= 3:
                        continue
                    zi = _clamp(z - 1, depth)
                    if seg[x, yi, zi] in (3, 42) and wm[x, yi, zi] < 5:
                        _set_filled(wm, filled, x, yi, zi)
                    zi = _clamp(z + 1, depth)
                    if seg[x, yi, zi] in (3, 42) and wm[x, yi, zi] < 5:
                        _set_filled(wm, filled, x, yi, zi)
                    hlabel = 17 if label in (4, 5) else 53
                    if _distance_to_label(seg, hlabel, x, y, z, 0, 1, 0, 10) < 10:
                        continue
                    if (_neighbor(seg, x, y, z, 1, 3) and _neighbor(seg, x, y, z, 1, 42)
                            and wm[x, y, z] < 5):
                        _set_filled(wm, filled, x, y, z)
                    inferior = int(seg[x, yi, z])
                    if inferior in (0, 2, 3, 41, 42) and wm[x, yi, z] < 5:
                        _set_filled(wm, filled, x, yi, z)
                    if inferior in (5, 44):
                        yi = _clamp(y + 1, height)
                        inferior = int(seg[x, yi, z])
                        if inferior in (2, 3, 41, 42) and wm[x, yi, z] < 5:
                            _set_filled(wm, filled, x, yi, z)
                            yi = _clamp(y + 2, height)
                            if wm[x, yi, z] < 5:
                                _set_filled(wm, filled, x, yi, z)
                elif label in (17, 53):
                    left = label == 17
                    if _medial_hippo(seg, x, y, z, -1 if left else 1):
                        continue
                    if _distance_to_wm(wm, x, y, z, 0, -1, 0, 5, True) <= 3:
                        continue
                    if (_distance_to_label(seg, 2 if left else 41, x, y, z, 0, -1, 0, 5) < 3 or
                            _distance_to_label(seg, 10 if left else 49, x, y, z, 0, -1, 0, 5) < 3 or
                            _distance_to_label(seg, 28 if left else 60, x, y, z, 0, -1, 0, 5) < 3):
                        continue
                    xi = _clamp(x + (1 if left else -1), width)
                    yi = _clamp(y + 1, height)
                    if seg[xi, yi, z] in (3, 42) and wm[xi, yi, z] < 5:
                        _set_filled(wm, filled, xi, yi, z)
                    inferior = int(seg[x, yi, z])
                    if inferior in (2, 3, 41, 42) and wm[x, yi, z] < 5:
                        _set_filled(wm, filled, x, yi, z)
                        yi = _clamp(y + 2, height)
                        if wm[x, yi, z] < 5:
                            _set_filled(wm, filled, x, yi, z)
                elif label in (26, 58, 11, 50, 30, 62, 12, 51, 13, 52, 10, 49, 28, 60):
                    if wm[x, y, z] < 5:
                        _set_filled(wm, filled, x, y, z)
                elif label in (2, 41):
                    if seg[x, y - 1, z] in (5, 44) and wm[x, y, z] < 5:
                        _set_filled(wm, filled, x, y, z)
                    if fill_seg_wm:
                        # Native FillSegWM is inside this ordered first scan,
                        # after the inferior-lateral-ventricle condition.
                        if fill_seed is None:
                            selected = (not _neighbor(seg, x, y, z, 1, 3)
                                        and not _neighbor(seg, x, y, z, 1, 42))
                        else:
                            selected = fill_seed[x, y, z]
                        if selected:
                            _set_filled(wm, filled, x, y, z)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                label = int(seg[x, y, z])
                if label == 0:
                    if ((_neighbor(seg, x, y, z, 1, 4) and _neighbor(seg, x, y, z, 1, 2)) or
                            (_neighbor(seg, x, y, z, 1, 43) and _neighbor(seg, x, y, z, 1, 41))):
                        _set_filled(wm, filled, x, y, z)
                elif label in (4, 43):
                    if _neighbor(seg, x, y, z, 2, 2 if label == 4 else 41):
                        _set_filled(wm, filled, x, y, z)

    if propagate_wm:
        _propagate_from_filled(wm, seg, filled)
    return filled


@njit(cache=True)
def _propagate_from_filled(wm: np.ndarray, seg: np.ndarray,
                            filled: np.ndarray) -> None:
    """固定filled的26邻域WM传播，原位修改uint8 wm，不扩展filled。

    seg为同网格整数标签，filled为uint8非零标志。坐标为x/y/z体素，
    邻域按原生边缘夹取；私有内核不验证输入。包含IS_WM的186/187。
    """
    width, height, depth = wm.shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if filled[x, y, z] == 0:
                    continue
                for dx in range(-1, 2):
                    xx = _clamp(x + dx, width)
                    for dy in range(-1, 2):
                        yy = _clamp(y + dy, height)
                        for dz in range(-1, 2):
                            zz = _clamp(z + dz, depth)
                            label = int(seg[xx, yy, zz])
                            if label in (2, 41, 186, 187, 28, 60, 7, 46,
                                         251, 252, 253, 254, 255) and wm[xx, yy, zz] < 5:
                                wm[xx, yy, zz] = 250


def edit_until_propagation(wm: np.ndarray, aseg: np.ndarray) -> np.ndarray:
    """Replay the first three native `edit_segmentation` loops, before MTL spackling."""
    if wm.shape != aseg.shape or wm.ndim != 3 or wm.dtype != np.uint8:
        raise ValueError("Expected matching 3D aseg and uint8 WM volumes")
    output = wm.copy()
    _edit_until_propagation(output, aseg.astype(np.int32, copy=False))
    return output


@njit(cache=True)
def _add_aseg_wm_below_hippocampus(wm: np.ndarray, seg: np.ndarray) -> None:
    """Source lines 1956-2011; `IS_WHITE_CLASS` is labels 2 and 41."""
    for z in range(1, wm.shape[2]):
        for y in range(1, wm.shape[1] - 1):
            for x in range(2, wm.shape[0] - 2):
                if wm[x, y, z] >= 5:
                    continue
                label = int(seg[x, y, z])
                if label not in (2, 41):
                    continue
                hlabel = 17 if label == 2 else 53
                found = False
                for xx in range(x - 1, x + 2):
                    for zz in range(z - 1, z + 2):
                        if seg[xx, y - 1, zz] == hlabel:
                            found = True
                            break
                    if found:
                        break
                if found:
                    wm[x, y, z] = 250


@njit(cache=True)
def _nonfilled_neighbors(wm: np.ndarray, x: int, y: int, z: int) -> int:
    count = 0
    for zz in range(z - 1, z + 2):
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                value = wm[xx, yy, zz]
                if value >= 5 and value != 250:
                    count += 1
    return count


@njit(cache=True)
def _anterior_edge_amygdala(seg: np.ndarray, x: int, y: int, z: int) -> bool:
    label = int(seg[x, y, z])
    return label in (18, 54) and _distance_to_label(seg, label, x, y, z, 0, 0, 1, 10) >= 10


@njit(cache=True)
def _post_spackle_early(wm: np.ndarray, brain: np.ndarray, seg: np.ndarray) -> None:
    """Active source rules 1008-1441, preserving scan order and T1 comparisons."""
    width, height, depth = wm.shape
    for z in range(1, depth - 1):
        for y in range(height - 2, 0, -1):
            for x in range(2, width - 2):
                if wm[x, y, z] >= 5:
                    continue
                label = int(seg[x, y, z])
                if label not in (0, 3, 42):
                    continue
                left = label == 3
                if wm[x, y - 1, z] < 5:
                    continue
                xi = x - 1 if left else x + 1
                alabel = 18 if left else 54
                if seg[xi, y, z] != alabel:
                    continue
                if _distance_to_label(seg, alabel, x, y, z, 0, 1, 0, 8) < 8:
                    continue
                wm[x, y, z] = 250

    for z in range(depth):
        for y in range(height - 2, 0, -1):
            for x in range(2, width - 2):
                if wm[x, y, z] >= 5:
                    continue
                label = int(seg[x, y, z])
                if label not in (5, 17, 18, 44, 53, 54):
                    continue
                left = label in (5, 17, 18)
                if wm[x, y + 1, z] < 5:
                    continue
                if _distance_to_wm(wm, x, y, z, 0, 1, 0, 4, False) > 2:
                    continue
                cortex = 3 if left else 42
                if (_distance_to_label(seg, 0, x, y, z, 0, 1, 0, 10) > 5 and
                        _distance_to_label(seg, cortex, x, y, z, 0, 1, 0, 10) > 5):
                    continue
                if _distance_to_wm(wm, x, y, z, 0, -1, 0, 5, True) <= 3:
                    continue
                if (_distance_to_label(seg, 2 if left else 41, x, y, z, 0, -1, 0, 5) < 3 or
                        _distance_to_label(seg, 10 if left else 49, x, y, z, 0, -1, 0, 5) < 3 or
                        _distance_to_label(seg, 28 if left else 60, x, y, z, 0, -1, 0, 5) < 3):
                    continue
                if _medial_hippo(seg, x, y, z, -1 if left else 1):
                    continue
                if wm[x, y + 1, z - 1] < 5:
                    if brain[x, y, z] > brain[x, y + 1, z - 1]:
                        wm[x, y, z] = 250
                    else:
                        wm[x, y + 1, z - 1] = 250
                if wm[x, y + 1, z + 1] < 5:
                    if brain[x, y, z] > brain[x, y + 1, z + 1]:
                        wm[x, y, z] = 250
                    else:
                        wm[x, y + 1, z + 1] = 250
                if wm[x - 1, y + 1, z] < 5:
                    if brain[x, y, z] > brain[x - 1, y + 1, z]:
                        wm[x, y, z] = 250
                    else:
                        wm[x - 1, y + 1, z] = 250
                if wm[x + 1, y + 1, z] < 5:
                    if brain[x, y, z] > brain[x + 1, y + 1, z]:
                        wm[x, y, z] = 250
                    else:
                        # The pinned C++ source writes x-1 here, despite testing x+1.
                        wm[x - 1, y + 1, z] = 250

    for z in range(depth):
        for y in range(1, height - 1):
            for x in range(2, width - 2):
                label = int(seg[x, y, z])
                if label not in (3, 42) or wm[x, y, z] >= 5:
                    continue
                left = label == 3
                if _medial_hippo(seg, x, y, z, -1 if left else 1):
                    continue
                if wm[x, y - 1, z] < 5:
                    continue
                xi = x - 1 if left else x + 1
                alabel = 18 if left else 54
                if seg[xi, y, z] != alabel:
                    continue
                if _distance_to_label(seg, alabel, x, y, z, 0, 1, 0, 8) < 8:
                    continue
                wm[x, y, z] = 250


@njit(cache=True)
def _post_spackle_late(wm: np.ndarray, seg: np.ndarray) -> None:
    """Active source rules 1592-1888, in three separate ordered scans."""
    width, height, depth = wm.shape
    for direction in (1, -1):
        for z in range(1, depth - 1):
            for y in range(1, height - 1):
                for x in range(2, width - 2):
                    label = int(seg[x, y, z])
                    if label not in (3, 42):
                        continue
                    left = label == 3
                    if _medial_hippo(seg, x, y, z, -1 if left else 1):
                        continue
                    other = int(seg[x, y - 1, z + direction])
                    if wm[x, y - 1, z + direction] >= 5:
                        continue
                    alabel = 18 if left else 54
                    hlabel = 17 if left else 53
                    if other not in (alabel, hlabel):
                        continue
                    if direction == -1 and other == alabel and _anterior_edge_amygdala(seg, x, y - 1, z - 1):
                        continue
                    if (_distance_to_label(seg, alabel, x, y, z, 0, 1, 0, 8) < 8 or
                            _distance_to_label(seg, hlabel, x, y, z, 0, 1, 0, 8) < 8):
                        continue
                    if _nonfilled_neighbors(wm, x, y, z) < 2:
                        continue
                    wm[x, y, z] = 250


    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(2, width - 2):
                label = int(seg[x, y, z])
                if label == 0:
                    left = seg[x - 1, y - 1, z] == 17
                    if left:
                        xi, hlabel = x - 1, 17
                    elif seg[x + 1, y - 1, z] == 53:
                        xi, hlabel = x + 1, 53
                    else:
                        continue
                    dx = -1 if left else 1
                    distance = _distance_to_label(seg, 0, x, y, z, dx, 0, 0, 10)
                    if distance < 10 and wm[x + dx * distance, y, z] < 5:
                        continue
                    if _distance_to_label(seg, hlabel, x, y, z, 0, 1, 0, 8) < 8:
                        continue
                    if _distance_to_wm(wm, x, y, z, 0, 1, 0, 8, True) < 7:
                        continue
                    wm[xi, y - 1, z] = 250
                elif label == 42:
                    if _medial_hippo(seg, x, y, z, 1):
                        continue
                    other = int(seg[x, y - 1, z + 1])
                    if wm[x, y - 1, z + 1] >= 5 or other not in (54, 53):
                        continue
                    if other == 54 and _anterior_edge_amygdala(seg, x, y - 1, z + 1):
                        continue
                    if (_distance_to_label(seg, 54, x, y, z, 0, 1, 0, 8) < 8 or
                            _distance_to_label(seg, 53, x, y, z, 0, 1, 0, 8) < 8):
                        continue
                    if _nonfilled_neighbors(wm, x, y, z) < 2:
                        continue
                    wm[x, y, z] = 250


def edit_segmentation_no_fill(wm: np.ndarray, brain: np.ndarray,
                              aseg: np.ndarray) -> np.ndarray:
    """Replay the native core segmentation after ``remove_paths_to_cortex``.

    This does not include the optional ``-fill-seg-wm`` branch or the later
    ``spackle_wm_superior_to_mtl`` call.
    """
    return edit_segmentation_profile(wm, brain, aseg, fill_seg_wm=False)


def edit_segmentation_profile(wm: np.ndarray, brain: np.ndarray,
                               aseg: np.ndarray, *,
                               fill_seg_wm: bool = False) -> np.ndarray:
    """运行有序edit_segmentation，fill在原生第一遍WM分支处执行。

    输入同网格3D uint8 WM、强度brain、整数aseg；fill_seg_wm默认False。
    返回新uint8 WM，不修改输入；包含WM传播及海马下方规则，不包含
    remove_paths或末尾spackle。网格/dtype错误抛ValueError。
    """
    if not (wm.ndim == brain.ndim == aseg.ndim == 3 and
            wm.shape == brain.shape == aseg.shape and wm.dtype == np.uint8):
        raise ValueError("Expected matching 3D WM, brain, aseg volumes and uint8 WM")
    result = wm.copy()
    labels = aseg.astype(np.int32, copy=False)
    _edit_until_propagation(result, labels, fill_seg_wm=fill_seg_wm)
    _post_spackle_early(result, brain, labels)
    _post_spackle_late(result, labels)
    _add_aseg_wm_below_hippocampus(result, labels)
    return result



def remove_paths_to_cortex(wm: np.ndarray, brain: np.ndarray,
                           aseg: np.ndarray, *,
                           aseg_storage_dtype: str | np.dtype = "int32") -> tuple[np.ndarray, int]:
    """完整有序 MTL 路径阻断，复用 FNIT NumBa 双视图实现。

    输入同3D XYZ网格：wm/brain uint8、aseg int32语义标签，各轴至少3。
    aseg_storage_dtype默认int32，可显式float32表示原四字节MGH存储；
    不能先改原存储类型再模拟MRIvox。返回新WM和实际填充次数，
    无重采样或输入修改，坐标及半径单位为体素。保留通用不可达证明
    快速路径，其他输入执行源顺序搜索，不按被试或哈希选择算法。
    非法输入抛ValueError，确定不终止的byte固定状态抛RuntimeError；
    其他计算异常传播。对应固定mri_edit_wm_with_aseg的首次子步骤。
    """
    from .wm_remove_paths import remove_mtl_paths
    output, report = remove_mtl_paths(wm=wm, brain=brain, aseg=aseg,
                                     aseg_storage_dtype=aseg_storage_dtype)
    return output, int(report["filled"])


def edit_wm_aseg_profile(wm: np.ndarray, brain: np.ndarray,
                         aseg: np.ndarray, entowm: np.ndarray, *,
                         fill_seg_wm: bool = True,
                         aseg_storage_dtype: str | np.dtype = "int32") -> tuple[np.ndarray, dict]:
    """完整标准WM编辑：已有有序Numba核心与Python后编辑，无原生程序。

    输入同3D XYZ网格：WM/brain为uint8强度，aseg/entowm为int32标签。
    fill_seg_wm默认True，在首次WM分支内消费，不能在核心完成后补填。
    返回新uint8 WM与实际子段秒数、路径证明及参数字典，输入不修改。
    包含SCM-HA半径1、keep-in、EntoWM level3和ACJ；无其他命令选项。
    aseg_storage_dtype默认int32，file API按原aseg存储传入float32或int32。
    remove_paths使用完整双视图有序搜索与通用不可达证明；非法输入
    抛ValueError，源搜索确定不终止时抛RuntimeError，不返回近似WM。
    对应mri_edit_wm_with_aseg标准单T1 recipe，空间单位为体素。
    """
    from .edit_wm_aseg_late_python import apply_late_wm_edits
    if (not isinstance(fill_seg_wm, bool) or wm.ndim != 3
            or wm.shape != brain.shape or wm.shape != aseg.shape
            or wm.shape != entowm.shape or wm.dtype != np.uint8
            or brain.dtype != np.uint8 or aseg.dtype.newbyteorder("=") != np.dtype(np.int32)
            or entowm.dtype.newbyteorder("=") != np.dtype(np.int32)):
        raise ValueError("expected matching 3D uint8 WM/brain, int32 labels and boolean fill_seg_wm")
    started = time.perf_counter()
    after_paths, changed = remove_paths_to_cortex(wm, brain, aseg,
                                                  aseg_storage_dtype=aseg_storage_dtype)
    after_paths_time = time.perf_counter()
    core = edit_segmentation_profile(after_paths, brain, aseg, fill_seg_wm=fill_seg_wm)
    after_core_time = time.perf_counter()
    core = spackle_wm_superior_to_mtl(core, aseg)
    after_mtl_time = time.perf_counter()
    output = apply_late_wm_edits(core, aseg, entowm, wm, fill_seg_wm=False)
    finished = time.perf_counter()
    return output, {"implementation": "FNIT ordered Numba and Python standard WM profile",
        "device": "cpu", "used_native_programs": False, "fill_seg_wm": fill_seg_wm,
        "fill_insertion": "first ordered WM-case scan; late fill disabled",
        "path_proof_changed": int(changed), "paths_filled": int(changed),
        "aseg_storage_dtype": np.dtype(aseg_storage_dtype).newbyteorder("=").name,
        "total_seconds": finished-started,
        "timing_parts_seconds": {"remove_paths_proof": after_paths_time-started,
            "ordered_core": after_core_time-after_paths_time,
            "superior_mtl": after_mtl_time-after_core_time,
            "late_edits": finished-after_mtl_time}}


def write_wm_asegedit_profile(wm_file: str | Path, brain_file: str | Path,
                              aseg_file: str | Path, entowm_file: str | Path,
                              output_file: str | Path, *,
                              fill_seg_wm: bool = True) -> dict:
    """完整CPU标准WM文件API，保留原MGH头/尾及毫米affine，不调用FS。

    前四输入为同网格3D MGH/MGZ；WM/brain原存储uint8，aseg原存储
    int32或float32精确整数，EntoWM为精确整数。输出为新uint8 MGZ；
    fill_seg_wm默认True，规则/限制同edit_wm_aseg_profile。返回实际
    分步、数组耗时及含校验、加载、写出的total_seconds和output路径。
    保留aseg原存储类型用于临时字节访问；输入读写失败传播，非法
    几何/标签抛ValueError，源搜索固定不终止抛RuntimeError；不读参考。
    """
    from .mgh_compat import save_same_dtype_mgh
    from .wm_aseg_io import integer_label_array, validate_aseg_storage
    started = time.perf_counter()
    images = [nib.load(str(path)) for path in (wm_file, brain_file, aseg_file, entowm_file)]
    first = images[0]
    if any(not isinstance(image, nib.MGHImage) or len(image.shape) != 3
           or image.shape != first.shape
           or not np.allclose(image.affine, first.affine, rtol=0, atol=1e-4)
           for image in images):
        raise ValueError("all inputs must share a 3D MGH grid and affine")
    if any(image.get_data_dtype() != np.dtype(np.uint8) for image in images[:2]):
        raise ValueError("WM and brain input storage must be uint8")
    validate_aseg_storage(images[2])
    wm, brain = [np.array(image.dataobj, dtype=np.uint8, order="C", copy=True) for image in images[:2]]
    aseg, entowm = [integer_label_array(image) for image in images[2:]]
    output, report = edit_wm_aseg_profile(wm, brain, aseg, entowm,
        fill_seg_wm=fill_seg_wm, aseg_storage_dtype=images[2].get_data_dtype())
    save_same_dtype_mgh(wm_file, output_file, output)
    report.update(array_stage_seconds=report["total_seconds"],
                  total_seconds=time.perf_counter()-started, output=str(output_file), file_io_included=True)
    return report


def edit_wm_aseg_core_no_fill(wm: np.ndarray, brain: np.ndarray,
                              aseg: np.ndarray) -> np.ndarray:
    """无fill的兼容核心API，复用一般输入几何证明和原有序三函数。

    wm/brain/aseg为同3D网格数组，返回新uint8 WM；不含标准后编辑。
    完整byte-access路径按int32存储执行，不按被试或输入哈希选算法。
    """
    after_paths, _ = remove_paths_to_cortex(wm, brain, aseg)
    return spackle_wm_superior_to_mtl(
        edit_segmentation_no_fill(after_paths, brain, aseg), aseg)


def apply_wm_asegedit_fixed(wm: np.ndarray, brain: np.ndarray,
                             aseg: np.ndarray, entowm: np.ndarray) -> np.ndarray:
    """标准固定recipe兼容名称，复用一般输入完整API，不再晚置fill。

    输入/失败条件同edit_wm_aseg_profile；返回新uint8数组，内部fill
    开启。旧冻结被试哈希与晚置fill实现已被新完整标准函数替代。
    """
    return edit_wm_aseg_profile(wm, brain, aseg, entowm, fill_seg_wm=True)[0]


def write_wm_asegedit_fixed(wm_file: str | Path, brain_file: str | Path,
                             aseg_file: str | Path, entowm_file: str | Path,
                             output_file: str | Path) -> Path:
    """Write the fixed call without a FreeSurfer runtime, preserving MGH tags."""
    write_wm_asegedit_profile(wm_file=wm_file, brain_file=brain_file,
        aseg_file=aseg_file, entowm_file=entowm_file, output_file=output_file, fill_seg_wm=True)
    return Path(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wm_file", "brain_file", "aseg_file", "entowm_file", "output_file"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_wm_asegedit_fixed(args.wm_file, args.brain_file, args.aseg_file,
                             args.entowm_file, args.output_file)


if __name__ == "__main__":
    main()
