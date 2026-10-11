"""First-defect white/gray histograms for pinned FreeSurfer topology repair.

The caller supplies original-surface white/gray vertex intensities after the
source's two median passes, the first defect's convex hull, and one-ring
neighbors. This does not perform topology search.
"""

from __future__ import annotations

from math import e, exp

import numpy as np


def histogram_vertices(hull: list[int], neighbors: list[list[int]],
                       marked: np.ndarray) -> np.ndarray:
    """Collect two nested two-ring ``vtotal`` neighborhoods, without defects.

    The optimal-retessellation path sets neighborhood size 2 before this
    call. Histogram count increments are integer-exact, so the order in which
    each reached vertex is first visited cannot alter any bin.
    """
    reached = np.zeros(len(marked), np.bool_)
    frontier = list(dict.fromkeys(hull))
    reached[frontier] = True
    for _ in range(4):
        next_frontier = []
        for vertex in frontier:
            for neighbor in neighbors[vertex]:
                if not reached[neighbor]:
                    reached[neighbor] = True
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return np.flatnonzero(reached & ~np.asarray(marked, np.bool_)).astype(np.int32)


def _raw_histogram(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    counts = np.zeros(256, np.float32)
    for vertex in selected:
        value = float(np.float32(values[vertex]))
        bin_index = min(255, max(0, int(np.floor(value + 0.5))))
        counts[bin_index] = np.float32(counts[bin_index] + np.float32(1))
    counts[counts == 0] = np.float32(0.1)
    return counts / np.float32(len(selected))


def _smooth_histogram(raw: np.ndarray) -> np.ndarray:
    """Source sigma2 有序平滑，支持输入实际bin数；256-bin原调用逐值兼容。"""
    kernel = np.empty(17, np.float32)
    norm = np.float32(0)
    for index in range(17):
        offset = float(index - 8)
        if abs(offset) <= 4:
            value = np.float32(exp(-offset * offset / 8.0))
        else:
            value = np.float32(np.float32(1.0 / (16.0 * e * e))
                               * np.float32((4.0 - abs(offset) / 2.0) ** 4))
        kernel[index] = value
        norm = np.float32(norm + value)
    kernel = kernel / norm
    result = np.empty(len(raw), np.float32)
    for bin_index in range(len(raw)):
        total = norm = np.float32(0)
        for index, weight in enumerate(kernel):
            source = bin_index + index - 8
            if 0 <= source < len(raw):
                norm = np.float32(norm + weight)
                total = np.float32(total + np.float32(weight * raw[source]))
        result[bin_index] = np.float32(total / norm)
    return result


def histogram_mean(counts: np.ndarray) -> np.float32:
    """Source computeDefectStatistics 1%-of-peak truncated float32 mean."""
    peak = np.max(counts)
    total = weighted = np.float32(0)
    for bin_index, count in enumerate(counts):
        if count < float(peak) / 100.0:
            continue
        weighted = np.float32(weighted + np.float32(np.float32(bin_index) * count))
        total = np.float32(total + count)
    return np.float32(weighted / total)


def first_defect_intensity_histograms(white: np.ndarray, gray: np.ndarray,
                                      hull: list[int], neighbors: list[list[int]],
                                      marked: np.ndarray) -> dict[str, object]:
    """Build raw/smoothed white/gray bins and means without native plots."""
    selected = histogram_vertices(hull, neighbors, marked)
    white_raw = _raw_histogram(white, selected)
    gray_raw = _raw_histogram(gray, selected)
    white_smooth = _smooth_histogram(white_raw)
    gray_smooth = _smooth_histogram(gray_raw)
    return {"selected": selected, "white_raw": white_raw,
            "gray_raw": gray_raw, "white_smooth": white_smooth,
            "gray_smooth": gray_smooth,
            "white_mean": histogram_mean(white_smooth),
            "gray_mean": histogram_mean(gray_smooth)}
