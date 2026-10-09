"""Isolated Python port of the fixed FreeSurfer 8.2 ``mri_segment`` call.

The full ``wm.seg.mgz`` voxels match the frozen 1 mm MP-RAGE subject; the
stage remains outside the native-free recon-all entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

import torch
import numpy as np
from numba import njit
from scipy.ndimage import convolve, maximum_filter, minimum_filter


@dataclass(frozen=True)
class IntensityStats:
    white_mean: float
    white_sigma: float
    gray_mean: float
    gray_sigma: float
    wm_low: float
    gray_hi: float


def intensity_segmentation(
    image: torch.Tensor, *, wm_low: float, wm_hi: float, gray_hi: float
) -> torch.Tensor:
    """Classify a UCHAR T1 as NOT_WHITE=1, AMBIGUOUS=128, or WHITE=255."""
    if image.ndim != 3 or image.dtype != torch.uint8:
        raise ValueError("expected a 3D uint8 FreeSurfer intensity volume")
    value = image.to(torch.float32)
    in_wm_range = (value >= wm_low) & (value <= wm_hi)
    result = torch.full_like(image, 1)
    result[in_wm_range & (value <= gray_hi)] = 128
    result[in_wm_range & (value > gray_hi)] = 255
    return result


def _histogram_kernel(sigma: float = 3.0) -> np.ndarray:
    half = round(8 * sigma) // 2
    weights = []
    for x in range(-half, half + 1):
        distance = abs(x)
        if distance <= 2 * sigma:
            weight = np.float32(np.exp(-x * x / (2 * sigma * sigma)))
        elif distance <= 4 * sigma:
            weight = np.float32((4 - distance / sigma) ** 4 / (16 * np.e**2))
        else:
            weight = np.float32(0)
        weights.append(weight)
    total = np.float32(0)
    for weight in weights:
        total = np.float32(total + weight)
    return np.asarray([np.float32(weight / total) for weight in weights])


def _smooth_histogram(counts: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half = len(kernel) // 2
    total = np.convolve(counts, kernel, "full")[half:half + len(counts)]
    norm = np.convolve(np.ones_like(counts), kernel, "full")[half:half + len(counts)]
    return total / norm


def _last_peak(counts: np.ndarray, first_bin: int, low: int, high: int) -> int:
    maximum = counts.max()
    if maximum == 0:
        return -1
    start = max(0, low - first_bin - 1)
    stop = min(len(counts) - 1, high - first_bin + 1)
    for index in range(stop, start - 1, -1):
        center = counts[index]
        neighbors = counts[max(0, index - 5):min(len(counts), index + 6)]
        if np.any(neighbors > center):
            continue
        mean = np.float32(neighbors.sum(dtype=np.float32) / 11)
        if mean >= np.float32(0.15 * maximum) and center - mean >= 1.9:
            return first_bin + index
    return -1


def _first_valley(counts: np.ndarray, first_bin: int, low: int, high: int) -> int:
    # HISTOfindValley stores each smoothed count in an int before comparing.
    integer_counts = counts.astype(np.int32)
    if integer_counts.max() == 0:
        return -1
    start = max(0, low - first_bin - 1)
    stop = min(len(counts) - 1, high - first_bin + 1)
    for index in range(start, stop + 1):
        neighbors = integer_counts[max(0, index - 1):min(len(counts), index + 2)]
        if not np.any(neighbors < integer_counts[index]):
            return first_bin + index
    return -1


def histogram_segmentation(
    image: torch.Tensor, labels: torch.Tensor, *, wm_low: float, wm_hi: float,
    gray_hi: float, window: int = 13, limit: int | None = None
) -> torch.Tensor:
    """Resolve trinary labels with FreeSurfer's local histogram peaks.

    ``limit`` is only for comparison of a leading subset during development.
    The fixed production call uses all ambiguous voxels.
    """
    if image.shape != labels.shape or image.dtype != torch.uint8 or labels.dtype != torch.uint8:
        raise ValueError("expected matching uint8 intensity and label volumes")
    if window % 2 != 1:
        raise ValueError("window must be odd")
    source = image.cpu().numpy()
    result = labels.cpu().numpy().copy()
    half = window // 2
    kernel = _histogram_kernel()
    points = np.argwhere(result == 128)
    if limit is not None:
        points = points[:limit]
    wm_low, wm_hi, gray_hi = int(wm_low), int(wm_hi), int(gray_hi)
    for x, y, z in points:
        patch = source[max(0, x - half):min(source.shape[0], x + half + 1),
                       max(0, y - half):min(source.shape[1], y + half + 1),
                       max(0, z - half):min(source.shape[2], z + half + 1)]
        low, high = int(patch.min()), int(patch.max())
        counts = np.bincount(patch.ravel(), minlength=high + 1)[low:high + 1].astype(np.float32)
        smooth = _smooth_histogram(counts, kernel)
        white_peak = _last_peak(smooth, low, wm_low, wm_hi - 7)
        gray_peak = _last_peak(smooth, low, 72, white_peak - 10)
        while gray_peak > gray_hi and white_peak >= 0:
            white_peak = gray_peak
            gray_peak = _last_peak(smooth, low, 72, white_peak - 10)
        if white_peak < 0 or gray_peak < 0:
            continue
        valley = _first_valley(smooth, low, gray_peak + 2, white_peak - 2)
        if valley < 0 or valley >= gray_hi or abs(int(source[x, y, z]) - valley) <= 2:
            continue
        result[x, y, z] = 255 if source[x, y, z] >= valley else 1
    return torch.from_numpy(result).to(image.device)


def detect_intensity_thresholds(image: torch.Tensor, labels: torch.Tensor) -> IntensityStats:
    """Compute FreeSurfer's border WM/GM statistics after histogram pass one."""
    source = image.cpu().numpy()
    classified = labels.cpu().numpy()
    white = classified == 255
    gray = classified == 1
    white_border = white & maximum_filter(gray, size=3, mode="nearest")
    gray_border = gray & maximum_filter(white, size=3, mode="nearest")
    white_values = source[white_border & (source >= 70)].astype(np.float64)
    gray_values = source[gray_border & (source >= 30) & (source <= 110)].astype(np.float64)
    if len(white_values) == 0 or len(gray_values) == 0:
        raise ValueError("empty border class in intensity statistics")
    wm_mean64 = white_values.mean()
    gm_mean64 = gray_values.mean()
    wm = np.float32(wm_mean64)
    gm = np.float32(gm_mean64)
    wm_sigma = np.float32(np.sqrt(np.mean(white_values * white_values) - wm_mean64 ** 2))
    gm_sigma = np.float32(np.sqrt(np.mean(gray_values * gray_values) - gm_mean64 ** 2))
    wm_low = np.float32(gm + gm_sigma)
    gray_hi = np.float32(min(np.float32(gm + np.float32(2) * gm_sigma), wm - 1))
    return IntensityStats(float(wm), float(wm_sigma), float(gm), float(gm_sigma),
                          float(wm_low), float(gray_hi))


_ICO_VERTICES = np.asarray([
    (0, 0, 1), (.8944, 0, .4472), (.2764, .8507, .4472),
    (-.7236, .5257, .4472), (-.7236, -.5257, .4472),
    (.2764, -.8507, .4472), (-.4253, -.309, .8507),
    (-.8507, 0, .5257), (-.4253, .309, .8507),
    (.1625, -.5, .8507), (-.2629, -.809, .5257),
    (.5257, 0, .8507), (.6882, -.5, .5257),
    (.1625, .5, .8507), (.6882, .5, .5257),
    (-.2629, .809, .5257), (-.5878, .809, 0),
    (0, 1, 0), (-.9511, .309, 0), (-.9511, -.309, 0),
    (-.5878, -.809, 0), (1, 0, 0),
], dtype=np.float32)


def _scale_to_voxel(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.abs(vector)
    largest = x if x > y and x > z else y if y > z else z
    return np.float32(vector * np.float32(1 / largest))


def _plane_bases() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals, first, second = [], [], []
    for x, y, z in _ICO_VERTICES:
        normal = np.asarray((x, y, z), dtype=np.float32)
        e1 = np.asarray((np.float32(z*z - x*y), np.float32(x*x - y*z),
                         np.float32(y*y - z*x)), dtype=np.float32)
        e1 = _scale_to_voxel(e1)
        e2 = np.asarray((np.float32(e1[1]*z - e1[2]*y),
                         np.float32(e1[0]*z - e1[2]*x),
                         np.float32(e1[1]*x - e1[0]*y)), dtype=np.float32)
        normals.append(_scale_to_voxel(normal))
        first.append(e1)
        second.append(_scale_to_voxel(e2))
    return np.asarray(normals), np.asarray(first), np.asarray(second)


def _plane_positions(points: np.ndarray, first: np.ndarray, second: np.ndarray,
                     window: int) -> np.ndarray:
    offsets = np.arange(-(window // 2), window // 2 + 1, dtype=np.float32)
    base = points[:, None, :] + offsets[None, :, None] * second[:, None, :]
    return base[:, :, None, :] + offsets[None, None, :, None] * first[:, None, None, :]


def _select_normals(image: np.ndarray, points: np.ndarray,
                    first: np.ndarray, second: np.ndarray, window: int) -> np.ndarray:
    point_float = points.astype(np.float32)
    variances = np.empty((len(points), len(first)), dtype=np.float32)
    for vertex in range(len(first)):
        position = _plane_positions(point_float,
                                    np.broadcast_to(first[vertex], point_float.shape),
                                    np.broadcast_to(second[vertex], point_float.shape), window)
        index = np.floor(position.astype(np.float64) + .5).astype(np.int32)
        for axis, bound in enumerate(image.shape):
            index[..., axis].clip(0, bound - 1, out=index[..., axis])
        values = image[index[..., 0], index[..., 1], index[..., 2]].astype(np.int32)
        total = values.sum(axis=(1, 2)).astype(np.float32)
        total_sq = (values * values).sum(axis=(1, 2)).astype(np.float32)
        mean = np.float32(total / np.float32(window * window))
        variances[:, vertex] = np.float32(total_sq / np.float32(window * window)
                                           - np.float32(mean * mean))
    return np.argmin(variances, axis=1)


def _sample_trilinear(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    position = positions.reshape(-1, 3).astype(np.float64)
    position = np.clip(position, 0, np.asarray(image.shape, dtype=np.float64) - 1)
    lower = position.astype(np.int32)
    upper = np.minimum(lower + 1, np.asarray(image.shape) - 1)
    delta = position - lower
    left = 1 - delta
    x0, y0, z0 = lower.T
    x1, y1, z1 = upper.T
    dx0, dy0, dz0 = left.T
    dx1, dy1, dz1 = delta.T
    result = (dx0*dy0*dz0*image[x0, y0, z0].astype(np.float64)
              + dx0*dy0*dz1*image[x0, y0, z1]
              + dx0*dy1*dz0*image[x0, y1, z0]
              + dx0*dy1*dz1*image[x0, y1, z1]
              + dx1*dy0*dz0*image[x1, y0, z0]
              + dx1*dy0*dz1*image[x1, y0, z1]
              + dx1*dy1*dz0*image[x1, y1, z0]
              + dx1*dy1*dz1*image[x1, y1, z1])
    return result.astype(np.float32).reshape(positions.shape[:-1])


def median_curve_center(
    image: torch.Tensor, labels: torch.Tensor, *, gray_hi: float, wm_low: float,
    window: int = 5, batch_size: int = 512, limit: int | None = None
) -> torch.Tensor:
    """Resolve only voxels whose least-variance central-plane median is decisive."""
    source = image.cpu().numpy()
    result = labels.cpu().numpy().copy()
    points = np.argwhere(result == 128)
    if limit is not None:
        points = points[:limit]
    _, first, second = _plane_bases()
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        vertex = _select_normals(source, batch, first, second, window)
        plane = _plane_positions(batch.astype(np.float32), first[vertex],
                                 second[vertex], window)
        sampled = _sample_trilinear(source, plane)
        median = np.partition(sampled.reshape(len(batch), -1), window*window//2,
                              axis=1)[:, window*window//2]
        white = median >= gray_hi
        gray = median <= wm_low
        selected = tuple(batch.T)
        current = result[selected]
        current[white] = 255
        current[~white & gray] = 1
        result[selected] = current
    return torch.from_numpy(result).to(image.device)


def median_curve_segmentation(
    image: torch.Tensor, labels: torch.Tensor, *, gray_hi: float, wm_low: float,
    window: int = 5, length: float = 3, batch_size: int = 128,
    limit: int | None = None
) -> torch.Tensor:
    """Resolve ambiguous voxels using the 22 least-variance plane directions."""
    source = image.cpu().numpy()
    result = labels.cpu().numpy().copy()
    points = np.argwhere(result == 128)
    if limit is not None:
        points = points[:limit]
    normals, first, second = _plane_bases()
    distances = np.arange(-length, length + .125, .25, dtype=np.float32)
    offsets = np.arange(-(window // 2), window // 2 + 1, dtype=np.float32)
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        vertex = _select_normals(source, batch, first, second, window)
        normal = normals[vertex]
        magnitude = np.float32(np.sqrt(np.float32(
            np.float32(normal[:, 0] * normal[:, 0] + normal[:, 1] * normal[:, 1])
            + normal[:, 2] * normal[:, 2])))
        unit = np.float32(normal / magnitude[:, None])
        centers = np.float32(batch[:, None, :] + unit[:, None, :] * distances[None, :, None])
        base = np.float32(centers[:, :, None, :]
                          + offsets[None, None, :, None] * second[vertex, None, None, :])
        positions = np.float32(base[:, :, :, None, :]
                               + offsets[None, None, None, :, None]
                               * first[vertex, None, None, None, :])
        samples = _sample_trilinear(source, positions).reshape(len(batch),
                                                                len(distances), window * window)
        medians = np.partition(samples, window * window // 2, axis=2)[:, :, window * window // 2]
        center = medians[:, len(distances) // 2]
        classification = np.full(len(batch), 128, dtype=np.uint8)
        classification[center >= gray_hi] = 255
        classification[center <= wm_low] = 1
        undecided = np.flatnonzero(classification == 128)
        if len(undecided):
            values = medians[undecided]
            center_values = center[undecided]
            absolute = np.abs(distances)
            white_mask = values >= gray_hi
            gray_mask = values <= wm_low
            white_index = np.argmin(np.where(white_mask, absolute, 1000), axis=1)
            gray_index = np.argmin(np.where(gray_mask, absolute, 1000), axis=1)
            white_dist = np.where(white_mask.any(axis=1), absolute[white_index], 1000)
            gray_dist = np.where(gray_mask.any(axis=1), absolute[gray_index], 1000)
            white_val = np.where(white_mask.any(axis=1),
                                 values[np.arange(len(values)), white_index], -1)
            gray_val = np.where(gray_mask.any(axis=1),
                                values[np.arange(len(values)), gray_index], -1)
            missing_white = white_dist > length
            peak_index = np.argmax(np.where(values > wm_low, values, -1), axis=1)
            has_peak = values[np.arange(len(values)), peak_index] > wm_low
            white_dist = np.where(missing_white & has_peak, absolute[peak_index], white_dist)
            white_val = np.where(missing_white & has_peak,
                                 values[np.arange(len(values)), peak_index], white_val)
            missing_gray = gray_dist > length
            trough_index = np.argmin(np.where(values < gray_hi, values, 10000), axis=1)
            has_trough = values[np.arange(len(values)), trough_index] < gray_hi
            gray_dist = np.where(missing_gray & has_trough, absolute[trough_index], gray_dist)
            gray_val = np.where(missing_gray & has_trough,
                                values[np.arange(len(values)), trough_index], gray_val)
            choice = np.where(white_dist < gray_dist - .75, 255,
                              np.where(gray_dist < white_dist - .75, 1,
                                       np.where(white_dist > 2, 1,
                                                np.where(np.abs(white_val - center_values)
                                                         > np.abs(gray_val - center_values), 1, 255))))
            classification[undecided] = choice
        result[tuple(batch.T)] = classification
    return torch.from_numpy(result).to(image.device)


def _border_labels(labels: np.ndarray) -> np.ndarray:
    white = labels == 255
    gray = labels == 1
    border = np.full(labels.shape, 128, dtype=np.uint8)
    border[white & maximum_filter(gray, size=3, mode="nearest")] = 255
    border[gray & maximum_filter(white, size=3, mode="nearest")] = 1
    return border


def _summed_volume(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values.astype(np.int64), ((1, 0),) * 3)
    return padded.cumsum(0).cumsum(1).cumsum(2)


def _box_sum(volume: np.ndarray, points: np.ndarray, half: int) -> np.ndarray:
    low = points - half
    high = points + half + 1
    x0, y0, z0 = low.T
    x1, y1, z1 = high.T
    return (volume[x1, y1, z1] - volume[x0, y1, z1]
            - volume[x1, y0, z1] - volume[x1, y1, z0]
            + volume[x0, y0, z1] + volume[x0, y1, z0]
            + volume[x1, y0, z0] - volume[x0, y0, z0])


def _border_probabilities(image: np.ndarray, border: np.ndarray,
                          volumes: tuple[np.ndarray, ...], points: np.ndarray,
                          window: int) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points.T
    intensity = image[x, y, z].astype(np.float32)
    white_center = border[x, y, z] == 255
    gray_center = border[x, y, z] == 1
    wc, ws, wq, gc, gs, gq = (_box_sum(volume, points, window // 2)
                               for volume in volumes)
    wc -= white_center
    ws -= np.where(white_center, intensity, 0).astype(np.int64)
    wq -= np.where(white_center, intensity * intensity, 0).astype(np.int64)
    gc -= gray_center
    gs -= np.where(gray_center, intensity, 0).astype(np.int64)
    gq -= np.where(gray_center, intensity * intensity, 0).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        wmean = np.where(wc > 0, np.float32(ws / wc), np.float32(1))
        wvar = np.where(wc > 0, np.float32(wq / wc) - wmean * wmean, np.float32(1))
        gmean = np.where(gc > 0, np.float32(gs / gc), np.float32(1))
        white_dist = np.float32(intensity - wmean)
        gray_dist = np.float32(intensity - gmean)
        pw = np.exp(np.float32(-white_dist * white_dist / (2 * wvar)).astype(np.float64)).astype(np.float32)
        pg = np.exp(np.float32(-gray_dist * gray_dist / (2 * wvar)).astype(np.float64)).astype(np.float32)
        total = np.float32(pw + pg)
        pg = np.where(np.abs(total) >= np.finfo(np.float32).eps, np.float32(pg / total), pg)
        pw = np.float32(pw / total)
    return pw, pg


def reclassify_border(
    image: torch.Tensor, labels: torch.Tensor, *, wm_low: float, gray_hi: float,
    window: int = 13, limit: int | None = None
) -> torch.Tensor:
    """Reclassify intensities near the WM/GM threshold using border statistics."""
    source = image.cpu().numpy()
    result = labels.cpu().numpy().copy()
    points = np.argwhere((source > wm_low) & (source < gray_hi))
    if limit is not None:
        points = points[:limit]
    border = _border_labels(result)
    white = border == 255
    gray = border == 1
    squared = source.astype(np.int32) ** 2
    volumes = tuple(_summed_volume(values) for values in (
        white, np.where(white, source, 0), np.where(white, squared, 0),
        gray, np.where(gray, source, 0), np.where(gray, squared, 0)))
    pw, pg = _border_probabilities(source, border, volumes, points, window)
    classification = np.where(pg > pw, 1, 255)
    confidence = np.where(pw >= pg, pw, pg)
    multi = np.flatnonzero(confidence < .7)
    if len(multi):
        selected = points[multi]
        white_total = np.zeros(len(multi), dtype=np.float32)
        gray_total = np.zeros(len(multi), dtype=np.float32)
        for scale in (5, 9, 13, 17):
            wprob, gprob = _border_probabilities(source, border, volumes,
                                                 selected, scale)
            white_total = np.float32(white_total + wprob)
            gray_total = np.float32(gray_total + gprob)
        classification[multi] = np.where(gray_total > white_total, 1, 255)
    result[tuple(points.T)] = classification
    return torch.from_numpy(result).to(image.device)


def mask_white_labels(image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Keep T1 intensity at WHITE voxels and zero out NOT_WHITE voxels."""
    result = torch.zeros_like(image)
    result[labels == 255] = image[labels == 255]
    result[labels == 128] = 128
    return result


def recover_bright_white(
    image: torch.Tensor, masked: torch.Tensor, *, wm_low: float, wm_hi: float,
    white_sigma: float
) -> torch.Tensor:
    """Restore upper-tail white voxels surrounded by at least nine WM intensities."""
    source = image.cpu().numpy()
    result = masked.cpu().numpy().copy()
    candidate = ((source > wm_hi) & (source <= wm_hi + white_sigma)
                 & (result < 5))
    neighbor = (source >= wm_low) & (source <= wm_hi)
    counts = convolve(neighbor.astype(np.uint8),
                      np.ones((3, 3, 3), dtype=np.uint8), mode="nearest")
    result[candidate & (counts >= 9)] = source[candidate & (counts >= 9)]
    return torch.from_numpy(result).to(image.device)


def remove_wrong_direction(masked: torch.Tensor, *, low: float,
                           high: float) -> torch.Tensor:
    """Remove WM voxels with positive 3x3x3 local gradient direction."""
    source = masked.cpu().numpy()
    result = source.copy()
    points = np.argwhere((source >= low) & (source <= high))
    # MRIgaussian1d(0.25, 100) has the exact kernel [0, 1, 0].
    padded = np.pad(source.astype(np.int16), 1, mode="edge")
    gx = np.float32((2 * padded[2:, 1:-1, 1:-1]
                     + padded[2:, :-2, 1:-1] + padded[2:, 2:, 1:-1]
                     - 2 * padded[:-2, 1:-1, 1:-1]
                     - padded[:-2, :-2, 1:-1] - padded[:-2, 2:, 1:-1]) / 8)
    gy = np.float32((2 * padded[:-2, 2:, 1:-1]
                     + padded[1:-1, 2:, 1:-1] + padded[2:, 2:, 1:-1]
                     - 2 * padded[:-2, :-2, 1:-1]
                     - padded[1:-1, :-2, 1:-1] - padded[2:, :-2, 1:-1]) / 8)
    gz = np.float32((2 * padded[:-2, 1:-1, 2:]
                     + padded[1:-1, 1:-1, 2:] + padded[2:, 1:-1, 2:]
                     - 2 * padded[:-2, 1:-1, :-2]
                     - padded[1:-1, 1:-1, :-2] - padded[2:, 1:-1, :-2]) / 8)
    x, y, z = points.T
    center_x, center_y, center_z = gx[x, y, z], gy[x, y, z], gz[x, y, z]
    direction = np.zeros(len(points), dtype=np.float32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                first = np.float32(dx * center_x + dy * center_y + dz * center_z)
                second = np.float32(gx[x + dx, y + dy, z + dz] * center_x
                                    + gy[x + dx, y + dy, z + dz] * center_y
                                    + gz[x + dx, y + dy, z + dz] * center_z)
                direction = np.float32(direction + np.float32(first * second))
    result[x[direction > 0], y[direction > 0], z[direction > 0]] = 0
    return torch.from_numpy(result).to(masked.device)


def remove_1d_structures(masked: torch.Tensor) -> torch.Tensor:
    """Prune WM voxels with fewer than two face neighbors in native scan order."""
    result = masked.cpu().numpy().copy()
    active = result >= 5
    kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1
    counts = convolve(active.astype(np.uint8), kernel, mode="constant")
    width, height, depth = result.shape
    def priority(x: int, y: int, z: int) -> int:
        return (z * height + y) * width + x

    pending = [(priority(int(x), int(y), int(z)), int(x), int(y), int(z))
               for x, y, z in np.argwhere(active & (counts < 2))]
    heapq.heapify(pending)
    while pending:
        next_pass = []
        while pending:
            current, x, y, z = heapq.heappop(pending)
            if not active[x, y, z]:
                continue
            active[x, y, z] = False
            result[x, y, z] = 0
            for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                               (0, 1, 0), (0, 0, -1), (0, 0, 1)):
                xi, yi, zi = x + dx, y + dy, z + dz
                if not (0 <= xi < width and 0 <= yi < height and 0 <= zi < depth):
                    continue
                previous = counts[xi, yi, zi]
                counts[xi, yi, zi] = previous - 1
                if active[xi, yi, zi] and previous == 2:
                    item = (priority(xi, yi, zi), xi, yi, zi)
                    heapq.heappush(pending if item[0] > current else next_pass, item)
        pending = next_pass
    return torch.from_numpy(result).to(masked.device)


def remove_bright_nonwhite(image: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Apply the fixed ``MRIfindBrightNonWM`` mask to the thickened WM volume."""
    source = image.cpu().numpy()
    result = masked.cpu().numpy().copy()
    white = result >= 5
    white_neighbors = convolve(white.astype(np.uint8),
                               np.ones((3, 3, 3), dtype=np.uint8), mode="nearest")
    bright = (~white) & (source > 125) & (white_neighbors < 14)
    bright = minimum_filter(maximum_filter(bright, size=3, mode="nearest"),
                            size=3, mode="nearest")
    expanded = bright | (maximum_filter(bright, size=3, mode="nearest")
                         & (source >= 100))
    labels = np.where(bright, 130, np.where(expanded, 100, 0)).astype(np.uint8)
    for _ in range(3):
        grown = maximum_filter(labels == 130, size=3, mode="nearest") & (source == 0)
        labels[grown] = 130
    result[labels >= 5] = 0
    return torch.from_numpy(result).to(masked.device)


@njit(cache=True)
def _diagonal_present(volume: np.ndarray, x: int, y: int, z: int,
                      offsets: np.ndarray) -> bool:
    width, height, depth = volume.shape
    for offset in offsets:
        dx, dy, dz = offset
        xi = min(max(x + dx, 0), width - 1)
        yi = min(max(y + dy, 0), height - 1)
        zi = min(max(z + dz, 0), depth - 1)
        if volume[xi, yi, zi] == 0:
            continue
        if dx != 0 and volume[xi, y, z] >= 5:
            continue
        if dy != 0 and volume[x, yi, z] >= 5:
            continue
        if dz != 0 and volume[x, y, zi] >= 5:
            continue
        return True
    return False


@njit(cache=True)
def _filter_diagonal_scan(volume: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    result = volume.copy()
    width, height, depth = result.shape
    while True:
        added = 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if result[x, y, z] < 5:
                        continue
                    for offset in offsets:
                        dx, dy, dz = offset
                        xi = min(max(x + dx, 0), width - 1)
                        yi = min(max(y + dy, 0), height - 1)
                        zi = min(max(z + dz, 0), depth - 1)
                        if result[xi, yi, zi] < 5:
                            continue
                        if dx != 0 and result[xi, y, z] >= 5:
                            continue
                        if dy != 0 and result[x, yi, z] >= 5:
                            continue
                        if dz != 0 and result[x, y, zi] >= 5:
                            continue
                        for corner in range(3):
                            if corner == 0:
                                fx, fy, fz = xi, y, z
                            elif corner == 1:
                                fx, fy, fz = x, yi, z
                            else:
                                fx, fy, fz = x, y, zi
                            if fx == x and fy == y and fz == z:
                                continue
                            result[fx, fy, fz] = 230
                            if _diagonal_present(result, fx, fy, fz, offsets):
                                result[fx, fy, fz] = 0
                            else:
                                added += 1
        if added == 0:
            return result


def filter_diagonal_morphology(masked: torch.Tensor) -> torch.Tensor:
    """Fill diagonal WM gaps in source scan order with compiled CPU loops."""
    offsets = np.asarray([(dx, dy, dz) for dz in (-1, 0, 1)
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                          if abs(dx) + abs(dy) + abs(dz) > 1], dtype=np.int8)
    result = _filter_diagonal_scan(masked.cpu().numpy(), offsets)
    return torch.from_numpy(result).to(masked.device)


def _closed_strand_volume(source: np.ndarray) -> np.ndarray:
    on = source > 0
    neighbors = convolve(on.astype(np.uint8),
                         np.ones((3, 3, 3), dtype=np.uint8), mode="nearest")
    isolated = np.where(on & (neighbors <= 3), 0,
                        np.where(~on & (neighbors >= 24), 200, source)).astype(np.uint8)
    return minimum_filter(maximum_filter(isolated, size=3, mode="nearest"),
                          size=3, mode="nearest")


def thin_strand_candidates(masked: torch.Tensor, *, thickness: int = 4) -> torch.Tensor:
    """Find the candidate strands before FreeSurfer's 20-component thickening."""
    source = masked.cpu().numpy()
    closed = _closed_strand_volume(source) >= 5
    up = np.zeros(source.shape, dtype=np.uint8)
    down = np.zeros(source.shape, dtype=np.uint8)
    for distance in range(1, thickness + 1):
        up_slice = np.zeros_like(closed)
        down_slice = np.zeros_like(closed)
        up_slice[:, :-distance, :] = ~closed[:, distance:, :]
        down_slice[:, distance:, :] = ~closed[:, :-distance, :]
        up[(up == 0) & up_slice] = distance
        down[(down == 0) & down_slice] = distance
    # The candidate center is tested in the original WM volume. The searches
    # on either side use the closed volume; the upstream source comment swaps them.
    thin = (source > 0) & (up > 0) & (down > 0) & (up + down <= thickness + 1)
    return torch.from_numpy(thin.astype(np.uint8)).to(masked.device)


def _strand_segments(thin: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Build 6-connected segments with FreeSurfer's scan and merge order."""
    labels = np.full(thin.shape, -1, dtype=np.int32)
    segments: list[list[tuple[int, int, int]]] = []
    active_count = 0
    points = np.argwhere(thin)
    points = points[np.lexsort((points[:, 0], points[:, 1], points[:, 2]))]
    for x, y, z in points:
        x, y, z = int(x), int(y), int(z)
        prior = []
        for xi, yi, zi in ((x, y, z - 1), (x, y - 1, z), (x - 1, y, z)):
            if min(xi, yi, zi) >= 0:
                found = int(labels[xi, yi, zi])
                if found >= 0 and found not in prior:
                    prior.append(found)
        if not prior:
            active_count += 1
            label = next((i for i in range(active_count)
                          if i >= len(segments) or not segments[i]), active_count - 1)
            if label == len(segments):
                segments.append([])
        else:
            label = prior[0]
            for merged in prior[1:]:
                for voxel in segments[merged]:
                    labels[voxel] = label
                segments[label].extend(segments[merged])
                segments[merged] = []
                active_count -= 1
        segments[label].append((x, y, z))
        labels[x, y, z] = label
    return [segment for segment in segments if segment]


def _dilate_strand_segments(
    segments: list[list[tuple[int, int, int]]], source: np.ndarray, passes: int = 5
) -> None:
    occupied = np.zeros(source.shape, dtype=bool)
    for segment in segments:
        for point in segment:
            occupied[point] = source[point] != 0
    width, height, depth = source.shape
    for _ in range(passes):
        for segment in segments:
            prior_length = len(segment)
            for x, y, z in segment[:prior_length]:
                for dx, dy, dz in ((-1, 0, 0), (0, -1, 0), (0, 0, -1),
                                   (0, 0, 1), (0, 1, 0), (1, 0, 0)):
                    xi, yi, zi = x + dx, y + dy, z + dz
                    if (0 <= xi < width and 0 <= yi < height and 0 <= zi < depth
                            and source[xi, yi, zi] and not occupied[xi, yi, zi]):
                        occupied[xi, yi, zi] = True
                        segment.append((xi, yi, zi))


def _thicken_strands_core(
    image: np.ndarray, source: np.ndarray, closed: np.ndarray,
    segments: list[list[tuple[int, int, int]]], *, count: int = 20,
    wm_hi: int = 125, planar_holes: bool = False,
    planar_backend: str = "python", planar_device: str = "cpu",
    planar_batch_size: int = 256,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """按原顺序沿MRI y轴加厚最多count个最大组件，并可作平面补洞。

    image/source/closed为同XYZ三维uint8强度、当前WM、闭运算WM；
    segments为有序组件列表，每项为XYZ体素三元组。count默认20，
    同大小组件按原索引排列；wm_hi默认125灰度。planar_holes默认False，
    True才执行22平面规则。planar_backend默认python，cached显式复用
    Torch静态几何与Numba有序反馈；planar_device默认cpu，仅控制静态
    几何；planar_batch_size默认256正整数。返回新uint8 WM和按选择顺序
    的uint8组件图列表。输入不修改，不重采样，更新仍为CPU顺序。
    无效后端/批量抛ValueError，底层几何或CUDA错误传播；无隐式回退。
    属于mri_segment内部步骤，没有独立原软件CLI。
    """
    if planar_backend not in {"python", "cached"}:
        raise ValueError("planar_backend must be python or cached")
    if not isinstance(planar_batch_size, int) or planar_batch_size < 1:
        raise ValueError("planar_batch_size must be positive")
    planar_fill = _fill_planar_holes
    if planar_backend == "cached":
        from .mri_segment_planar_torch import fill_planar_holes_cached

        def planar_fill(result, strand):
            return fill_planar_holes_cached(result=result, strand=strand,
                device=planar_device, batch_size=planar_batch_size)
    result = source.copy()
    selected = sorted(range(len(segments)), key=lambda i: (-len(segments[i]), i))[:count]
    segment_images = []
    height = source.shape[1]
    for index in selected:
        strand = np.zeros(source.shape, dtype=np.uint8)
        for point in segments[index]:
            strand[point] = closed[point]
        for x, y, z in segments[index]:
            if source[x, y, z] < 5:
                continue
            down = up = 3.0
            for distance in range(4):
                if result[x, max(0, y - distance), z] < 5:
                    down = distance - .5
                    break
            for distance in range(4):
                if result[x, min(height - 1, y + distance), z] < 5:
                    up = distance - .5
                    break
            while up + down <= 2:
                up_added = down_added = False
                if up <= down:
                    outer = y + int(up + 1.5)
                    if outer < height and result[x, outer, z] == 0:
                        up_added = True
                        inner = y + int(up + .5)
                        if image[x, inner, z] < wm_hi:
                            result[x, inner, z] = 200
                            strand[x, inner, z] = 200
                if up >= down:
                    outer = y - int(down + 1.5)
                    if outer >= 0 and result[x, outer, z] == 0:
                        down_added = True
                        inner = y - int(down + .5)
                        if image[x, inner, z] < wm_hi:
                            result[x, inner, z] = 200
                            strand[x, inner, z] = 200
                if up_added:
                    up += 1
                if down_added:
                    down += 1
                if not up_added and not down_added:
                    break
        _fill_strand_neighborhood(image, result, strand, wm_hi=wm_hi)
        if planar_holes:
            planar_fill(result, strand)
        segment_images.append(strand)
    return result, segment_images


def _fill_strand_neighborhood(
    image: np.ndarray, result: np.ndarray, strand: np.ndarray, *, wm_hi: int
) -> None:
    """Fill within-component single-voxel gaps in native scan order."""
    x0, y0, z0 = np.argwhere(strand > 0).min(axis=0)
    x1, y1, z1 = np.argwhere(strand > 0).max(axis=0) + 1
    x0, y0, z0 = max(1, x0 - 1), max(1, y0 - 1), max(1, z0 - 1)
    x1, y1, z1 = min(result.shape[0] - 1, x1 + 1), min(result.shape[1] - 1, y1 + 1), min(result.shape[2] - 1, z1 + 1)
    local = strand[x0:x1, y0:y1, z0:z1] > 0
    pair = np.zeros(local.shape, dtype=bool)
    pair[1:-1] |= local[2:] & local[:-2]
    pair[:, 1:-1] |= local[:, 2:] & local[:, :-2]
    pair[:, :, 1:-1] |= local[:, :, 2:] & local[:, :, :-2]
    candidate = pair & (result[x0:x1, y0:y1, z0:z1] == 0) & (image[x0:x1, y0:y1, z0:z1] < wm_hi)
    width, height, _ = result.shape

    def rank(x: int, y: int, z: int) -> int:
        return (z * height + y) * width + x

    pending = [rank(int(x + x0), int(y + y0), int(z + z0))
               for x, y, z in np.argwhere(candidate)]
    seen = set(pending)
    heapq.heapify(pending)
    while pending:
        next_pass = []
        while pending:
            current = heapq.heappop(pending)
            x = current % width
            y = current // width % height
            z = current // (width * height)
            if result[x, y, z] or image[x, y, z] >= wm_hi:
                continue
            if not ((strand[x - 1, y, z] and strand[x + 1, y, z])
                    or (strand[x, y - 1, z] and strand[x, y + 1, z])
                    or (strand[x, y, z - 1] and strand[x, y, z + 1])):
                continue
            if np.any((result[x-1:x+2, y-1:y+2, z-1:z+2] != 0)
                      & (strand[x-1:x+2, y-1:y+2, z-1:z+2] == 0)):
                continue
            result[x, y, z] = 210
            strand[x, y, z] = 210
            for xi, yi, zi in ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z),
                               (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)):
                if not (1 <= xi < width - 1 and 1 <= yi < height - 1
                        and 1 <= zi < result.shape[2] - 1):
                    continue
                item = rank(xi, yi, zi)
                if item in seen or result[xi, yi, zi] or image[xi, yi, zi] >= wm_hi:
                    continue
                if ((strand[xi - 1, yi, zi] and strand[xi + 1, yi, zi])
                        or (strand[xi, yi - 1, zi] and strand[xi, yi + 1, zi])
                        or (strand[xi, yi, zi - 1] and strand[xi, yi, zi + 1])):
                    seen.add(item)
                    heapq.heappush(pending if item > current else next_pass, item)
        pending = next_pass


def _fill_planar_holes(result: np.ndarray, strand: np.ndarray) -> None:
    """Fill strand-plane holes with the native 22-plane and four-ray rules."""
    binary = (strand >= 5).astype(np.uint8)
    border = maximum_filter(maximum_filter(binary, size=3, mode="nearest"),
                            size=3, mode="nearest") > 0
    border &= binary == 0
    points = np.argwhere(border)
    points = points[np.lexsort((points[:, 0], points[:, 1], points[:, 2]))]
    _, first, second = _plane_bases()
    plane_offsets = _plane_positions(np.zeros((len(first), 3), dtype=np.float32),
                                     first, second, 5)
    directions = np.stack((first, -first, second, -second), axis=1)
    distances = np.asarray((.75, 1.5), dtype=np.float32)
    shape = np.asarray(result.shape)
    while True:
        added = 0
        for x, y, z in points:
            if result[x, y, z] > 5 or not border[x, y, z]:
                continue
            faces = ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z),
                     (x, y + 1, z), (x, y, z - 1), (x, y, z + 1))
            if sum(binary[point] >= 1 for point in faces) != sum(result[point] >= 5 for point in faces):
                continue
            point = np.asarray((x, y, z), dtype=np.float32)
            positions = point + plane_offsets
            index = np.floor(positions.astype(np.float64) + .5).astype(np.int32)
            for axis in range(3):
                index[..., axis].clip(0, shape[axis] - 1, out=index[..., axis])
            counts = binary[index[..., 0], index[..., 1], index[..., 2]].sum(axis=(1, 2))
            vertex = np.flatnonzero(counts == counts.max())[-1]
            ray_positions = point + directions[vertex, :, None, :] * distances[None, :, None]
            if not np.all(np.any(_sample_trilinear(binary, ray_positions) > .5, axis=1)):
                continue
            result[x, y, z] = 200
            binary[x, y, z] = 1
            border[x, y, z] = False
            added += 1
        if added == 0:
            break


def segment_white_matter(image: torch.Tensor, *, device: str | torch.device | None = None,
                         histogram_backend: str = "cpu",
                         histogram_batch_size: int = 2048,
                         planar_backend: str = "python",
                         planar_batch_size: int = 256) -> torch.Tensor:
    """执行固定单T1的 mri_segment -wsizemm 13 -mprage 分割。

    image为非空三维uint8、x/y/z体素网格。device默认None保留输入设备；
    histogram_backend默认cpu使用原NumPy参考，torch复用分块PyTorch，
    histogram_batch_size默认2048正整数，只控制torch临时内存。
    planar_backend默认python；cached复用静态平面索引缓存与Numba顺序
    扫描，几何位于目标device；planar_batch_size默认256正整数。
    返回新uint8 WM，shape不变，device为明确目标。不改输入、不重采样。
    strand及有序填充保持CPU更新顺序；不是完整GPU转写。输入/backend/
    batch错误抛ValueError；CUDA错误传播，不自动回退。TF32策略由调用者
    管理，本函数不全局修改精度，不使用半精度。
    """
    if image.ndim != 3 or image.dtype != torch.uint8 or not image.numel():
        raise ValueError("expected a 3D uint8 FreeSurfer intensity volume")
    if histogram_backend not in {"cpu", "torch"}:
        raise ValueError("histogram_backend must be cpu or torch")
    if not isinstance(histogram_batch_size, int) or histogram_batch_size < 1:
        raise ValueError("histogram_batch_size must be positive")
    if planar_backend not in {"python", "cached"}:
        raise ValueError("planar_backend must be python or cached")
    if not isinstance(planar_batch_size, int) or planar_batch_size < 1:
        raise ValueError("planar_batch_size must be positive")
    histogram = histogram_segmentation
    histogram_options = {}
    if histogram_backend == "torch":
        from .mri_segment_histogram_torch import histogram_segmentation_torch
        histogram = histogram_segmentation_torch
        histogram_options = {"batch_size": histogram_batch_size}
    target = image if device is None else image.to(device)
    first = intensity_segmentation(target, wm_low=79, wm_hi=125, gray_hi=99)
    first = histogram(target, first, wm_low=79, wm_hi=125,
                      gray_hi=99, **histogram_options)
    thresholds = detect_intensity_thresholds(target, first)
    second = intensity_segmentation(target, wm_low=thresholds.wm_low,
                                    wm_hi=125, gray_hi=thresholds.gray_hi)
    second = histogram(target, second, wm_low=thresholds.wm_low,
                       wm_hi=125, gray_hi=thresholds.gray_hi, **histogram_options)
    labels = median_curve_segmentation(target, second, gray_hi=thresholds.gray_hi,
                                       wm_low=thresholds.wm_low)
    labels = reclassify_border(target, labels, wm_low=thresholds.wm_low - 5,
                               gray_hi=thresholds.gray_hi)
    masked = mask_white_labels(target, labels)
    masked = recover_bright_white(target, masked, wm_low=thresholds.wm_low,
                                  wm_hi=125, white_sigma=thresholds.white_sigma)
    masked = remove_wrong_direction(masked, low=thresholds.wm_low - 5,
                                    high=thresholds.gray_hi)
    masked = remove_1d_structures(masked)
    source = masked.cpu().numpy()
    closed = _closed_strand_volume(source)
    thin = thin_strand_candidates(masked).cpu().numpy()
    segments = _strand_segments(thin)
    _dilate_strand_segments(segments, source)
    thickened, _ = _thicken_strands_core(target.cpu().numpy(), source, closed,
        segments, planar_holes=True, planar_backend=planar_backend,
        planar_device=str(target.device), planar_batch_size=planar_batch_size)
    masked = remove_bright_nonwhite(target, torch.from_numpy(thickened).to(target.device))
    return filter_diagonal_morphology(masked).to(target.device)


def segment_white_matter_mgz(source_path: str | Path, output_path: str | Path, *,
                             device: str | torch.device = "cpu",
                             histogram_backend: str = "cpu",
                             histogram_batch_size: int = 2048,
                             planar_backend: str = "python",
                             planar_batch_size: int = 256) -> dict:
    """nibabel读取3D uint8 MGZ，执行固定WM配方，保留MGH头与affine。

    source_path为强度文件；output_path为新WM文件，uint8、原网格和mm
    affine，不作scanner/surface RAS变换。device默认cpu；histogram_backend
    默认cpu、torch为显式分块候选；histogram_batch_size默认2048。
    planar_backend默认python，cached显式复用静态几何缓存与有序Numba；
    planar_batch_size默认256，仅控制缓存几何批量，目标为相同device。
    返回implementation/device/backend/batch/ordered_cpu_rules/output字典。
    标签语义及有序CPU扫描不变；读取/写出/CUDA异常传播，错误dtype或网格
    抛ValueError。没有官方文件读取或原生程序调用；无生产额外同步。
    """
    import nibabel as nib

    source_image = nib.load(str(source_path))
    source = np.asarray(source_image.dataobj)
    if source.ndim != 3 or source.dtype != np.uint8:
        raise ValueError("expected a 3D uint8 MGH input")
    segmented = segment_white_matter(
        torch.from_numpy(source.copy()), device=device,
        histogram_backend=histogram_backend,
        histogram_batch_size=histogram_batch_size,
        planar_backend=planar_backend, planar_batch_size=planar_batch_size).cpu().numpy()
    nib.save(nib.MGHImage(segmented, source_image.affine,
                          header=source_image.header.copy()), str(output_path))
    return {"implementation": "FNIT PyTorch hybrid", "device": str(device),
            "histogram_backend": histogram_backend,
            "histogram_batch_size": histogram_batch_size,
            "planar_backend": planar_backend, "planar_batch_size": planar_batch_size,
            "ordered_cpu_rules": True, "output": str(output_path)}
