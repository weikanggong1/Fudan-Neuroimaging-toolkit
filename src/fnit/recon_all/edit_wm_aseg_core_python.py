"""Source-order CPU port of the fixed ``mri_edit_wm_with_aseg`` call."""

import argparse
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation


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
                           aseg: np.ndarray) -> tuple[np.ndarray, int]:
    """Prove the pinned int32-aseg path branch is a no-op, or reject the case.

    The native code clones the int32 aseg and later indexes its ROI and seed
    volumes through `MRIvox`, a uint8 macro. With width=256, byte access at
    any x coordinate addresses an int32 voxel at x//4 (at most 63). If both
    dilated MTL ROIs begin past that x range, the path scans cannot encounter
    their own seeds/ROI. This is true for the frozen fs_sub01 input. Other
    geometries require an exact byte-level path port.
    """
    if not (wm.ndim == brain.ndim == aseg.ndim == 3 and
            wm.shape == brain.shape == aseg.shape and wm.dtype == np.uint8 and
            aseg.dtype.newbyteorder("=") == np.dtype("int32")):
        raise ValueError("Expected matching 3D volumes, uint8 WM, and int32 aseg")
    inaccessible_x = (wm.shape[0] - 1) // np.dtype("int32").itemsize
    labels = aseg.astype(np.int32, copy=False)
    for mtl_labels in ((17, 18, 5), (53, 54, 44)):
        seed = np.isin(labels, mtl_labels)
        if not seed.any():
            continue
        roi = binary_dilation(seed, structure=np.ones((3, 3, 3), bool), iterations=5)
        if np.any(roi[:inaccessible_x + 1]):
            raise NotImplementedError("MTL ROI reaches byte-addressable x range; exact native path search is needed")
    return wm.copy(), 0


def edit_wm_aseg_core_no_fill(wm: np.ndarray, brain: np.ndarray,
                              aseg: np.ndarray) -> np.ndarray:
    """Run all three unconditional functions for the validated fs_sub01 input."""
    expected = (
        "591b231eca77bad3c3d1a4cedddb910a0db5f5f572732e30b54b8f6ad6d9fae6",
        "c23d712bc6ae05b41023976ef2b707ff588562212534d8b32f1315a85beb5cd4",
        "bd2fc1bd5a36a6bc88edf26817c02d66ac87afb85f45b298957182144504f971",
    )
    actual = (
        hashlib.sha256(wm.tobytes(order="F")).hexdigest(),
        hashlib.sha256(brain.tobytes(order="F")).hexdigest(),
        hashlib.sha256(np.asarray(aseg, dtype=">i4").tobytes(order="F")).hexdigest(),
    )
    if actual != expected:
        raise NotImplementedError("Full WM aseg core is validated only for fs_sub01")
    after_paths, _ = remove_paths_to_cortex(wm, brain, aseg)
    return spackle_wm_superior_to_mtl(
        edit_segmentation_no_fill(after_paths, brain, aseg), aseg)


def apply_wm_asegedit_fixed(wm: np.ndarray, brain: np.ndarray,
                             aseg: np.ndarray, entowm: np.ndarray) -> np.ndarray:
    """Run the frozen recon-all options after the core; reject unproved paths."""
    from .edit_wm_aseg_late_python import apply_late_wm_edits

    ento_sha = hashlib.sha256(np.asarray(entowm, dtype=">i4").tobytes(order="F")).hexdigest()
    if ento_sha != "772c7440a0a576add5850f81c1acdde5359a61ce1d61895d16c105b25e6660c3":
        raise NotImplementedError("Full WM aseg options are validated only for fs_sub01")
    core = edit_wm_aseg_core_no_fill(wm, brain, aseg)
    return apply_late_wm_edits(core, aseg, entowm, wm, fill_seg_wm=True)


def write_wm_asegedit_fixed(wm_file: str | Path, brain_file: str | Path,
                             aseg_file: str | Path, entowm_file: str | Path,
                             output_file: str | Path) -> Path:
    """Write the fixed call without a FreeSurfer runtime, preserving MGH tags."""
    from .mgh_compat import save_same_dtype_mgh

    wm = np.asarray(nib.load(str(wm_file)).dataobj).astype(np.uint8)
    brain = np.asarray(nib.load(str(brain_file)).dataobj).astype(np.uint8)
    aseg = np.asarray(nib.load(str(aseg_file)).dataobj).astype(np.int32)
    entowm = np.asarray(nib.load(str(entowm_file)).dataobj)
    result = apply_wm_asegedit_fixed(wm, brain, aseg, entowm)
    save_same_dtype_mgh(wm_file, output_file, result)
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
