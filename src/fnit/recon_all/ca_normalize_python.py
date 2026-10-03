"""Native-free fixed single-T1 GCA normalization from FreeSurfer 8.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
from numba import njit, prange
from scipy.ndimage import maximum_filter, minimum_filter

from .mri_em_register import (GCA, atlas_label_peak, estimate_image_white_matter_peak,
                              read_gca, _vnl_affine_inverse)


NORMALIZATION_LABELS = (2, 41, 7, 46, 16)
MIN_CANDIDATE_INTENSITY = .75 * .8 * 110
MAX_CANDIDATE_INTENSITY = 1.25 * 1.2 * 110


@dataclass(frozen=True)
class AtlasSamples:
    prior_coordinates: np.ndarray
    source_coordinates: np.ndarray
    labels: np.ndarray
    priors: np.ndarray


def read_voxel_lta(path: str | Path) -> np.ndarray:
    """Read the single VOX_TO_VOX transform used by the fixed recon-all stage."""

    lines = Path(path).read_text().splitlines()
    if not any(line.startswith("type") and line.split("=")[1].split("#")[0].strip() == "0"
               for line in lines):
        raise ValueError("A VOX_TO_VOX LTA is required")
    marker = lines.index("1 4 4")
    return np.asarray([[float(value) for value in lines[marker + i].split()]
                       for i in range(1, 5)], np.float64)


def masked_input(nu_path: str | Path, mask_path: str | Path) -> np.ndarray:
    """Reproduce the close-and-mask operation before GCA histogram scaling."""

    nu = nib.load(str(nu_path))
    mask = nib.load(str(mask_path))
    if nu.shape != mask.shape or not np.allclose(nu.affine, mask.affine, atol=1e-4, rtol=0):
        raise ValueError("Input and mask geometry differ")
    voxels = np.asarray(nu.dataobj)
    if voxels.dtype != np.uint8:
        raise ValueError("Fixed T1 input must be uint8")
    mask_values = np.asarray(mask.dataobj)
    binary = (mask_values != 0) & (mask_values != 1)
    # The native program applies 3x3x3 grayscale closing after changing 1 to 0.
    closed = minimum_filter(maximum_filter(binary, size=3, mode="nearest"),
                            size=3, mode="nearest")
    return np.where(closed, voxels, 0).astype(np.uint8)


def atlas_samples(gca: GCA, voxel_lta: np.ndarray) -> AtlasSamples:
    """Reproduce GCAfindAllSamples labels and atlas-to-source integer coordinates."""

    if gca.prior_shape != (128, 128, 128) or gca.volume_shape != (256, 256, 256):
        raise ValueError("Expected fixed 2020 GCA geometry")
    cells = np.repeat(np.arange(len(gca.prior_training), dtype=np.int32),
                      np.diff(gca.prior_offsets))
    max_prior = np.full(len(gca.prior_training), -1, np.float32)
    np.maximum.at(max_prior, cells, gca.prior_values)
    best = np.full(len(gca.prior_training), -1, np.int32)
    tied = gca.prior_values == max_prior[cells]
    np.maximum.at(best, cells[tied], np.flatnonzero(tied))
    labels = np.full(len(gca.prior_training), -1, np.int32)
    occupied = best >= 0
    labels[occupied] = gca.prior_labels[best[occupied]]
    nonunknown_nearby = maximum_filter((labels.reshape(gca.prior_shape) > 0).astype(np.uint8),
                                       size=3, mode="constant").ravel() != 0
    selected = occupied & ((labels != 0) | nonunknown_nearby)
    chosen = np.flatnonzero(selected)
    prior = np.column_stack(np.unravel_index(chosen, gca.prior_shape)).astype(np.int32)
    source = prior_to_source_coordinates(prior, voxel_lta, gca.prior_spacing,
                                         source_shape=gca.volume_shape)
    return AtlasSamples(prior, source, labels[chosen], gca.prior_values[best[chosen]])


def prior_to_source_coordinates(prior: np.ndarray, voxel_lta: np.ndarray,
                                prior_spacing: float, *,
                                source_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    """固定 GCA 的 prior voxel→source voxel；保持原生 FP32 和 nint 契约。

    prior 为 N×3 整数坐标；voxel_lta 为 source→atlas 4×4 affine；
    prior_spacing 为 prior 到 atlas voxel 的固定缩放（当前图谱为 2）。
    输出 N×3 int32 源体素索引，不做 world/RAS 变换。source_shape 为
    可选源网格；给定时按原生浮点边界 [0, size-1] 判断，越界保留
    GCAfindAllSamples 的初始 atlas 体素坐标，不截到边缘。
    复用 VNL affine inverse，每项乘积和累计都舍入 FP32；nint 前
    将已经得到的 FP32 坐标提升为 double，避免 +0.5 再被 FP32 舍入。
    本函数不修改 TF32、设备或调用方数组；非法 affine/坐标明确拒绝。
    """
    coordinates = np.asarray(prior)
    matrix = np.asarray(voxel_lta, dtype=np.float32)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3
            or not np.issubdtype(coordinates.dtype, np.integer)
            or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], np.array([0, 0, 0, 1], np.float32))
            or not np.isfinite(prior_spacing) or prior_spacing <= 0):
        raise ValueError("Expected integer prior coordinates and finite voxel affine")
    if np.linalg.det(matrix[:3, :3].astype(np.float64)) == 0:
        raise ValueError("Singular source-to-atlas affine")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        mapping = _vnl_affine_inverse(matrix)
        mapping[:, :3] = np.float32(mapping[:, :3] * np.float32(prior_spacing))
        points = coordinates.astype(np.float32)
        source_float = np.zeros((len(points), 3), dtype=np.float32)
        for axis in range(3):
            for component in range(3):
                source_float[:, axis] = np.float32(
                    source_float[:, axis]
                    + np.float32(mapping[axis, component] * points[:, component]))
            source_float[:, axis] = np.float32(source_float[:, axis] + mapping[axis, 3])
    # Native MatrixMultiply outputs float, promoted to double before nint.
    values = source_float.astype(np.float64)
    rounded = np.where(values >= 0, np.floor(values + .5), np.ceil(values - .5))
    bounds = np.iinfo(np.int32)
    if not np.isfinite(rounded).all() or np.any((rounded < bounds.min) | (rounded > bounds.max)):
        raise ValueError("Source coordinates exceed int32 index range")
    result = rounded.astype(np.int32)
    if source_shape is not None:
        shape = np.asarray(source_shape)
        if shape.shape != (3,) or not np.issubdtype(shape.dtype, np.integer) or np.any(shape <= 0):
            raise ValueError("Expected positive integer source shape")
        inside = np.all((values >= 0) & (values <= shape - 1), axis=1)
        # GCAcomputeSampleCoords only replaces the initial atlas coordinate
        # when GCApriorToSourceVoxel succeeds. The fixed GCA atlas voxels are 1 mm.
        initial = np.float32(points * np.float32(prior_spacing))
        result[~inside] = initial[~inside].astype(np.int32)
    return result


def scale_masked_input(voxels: np.ndarray, atlas_peak: int, image_peak: int) -> np.ndarray:
    """FreeSurfer's uint8 ``MRIscalarMulFrame`` operation."""

    factor = np.float32(atlas_peak / image_peak)
    return np.clip(voxels.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _atlas_label_fields(gca: GCA, label: int) -> tuple[np.ndarray, np.ndarray]:
    cells = np.repeat(np.arange(len(gca.prior_training), dtype=np.int32),
                      np.diff(gca.prior_offsets))
    prior = np.zeros(len(gca.prior_training), np.float32)
    matching = gca.prior_labels == label
    prior[cells[matching]] = gca.prior_values[matching]
    nodes = np.repeat(np.arange(len(gca.node_training), dtype=np.int32),
                      np.diff(gca.node_offsets))
    variance = np.full(len(gca.node_training), np.nan, np.float32)
    matching = gca.node_labels == label
    variance[nodes[matching]] = gca.node_variances[matching]
    return prior.reshape(gca.prior_shape), variance.reshape(gca.node_shape)


def _node_label_means(gca: GCA, label: int) -> np.ndarray:
    nodes = np.repeat(np.arange(len(gca.node_training), dtype=np.int32),
                      np.diff(gca.node_offsets))
    means = np.zeros(len(gca.node_training), np.float32)
    matching = gca.node_labels == label
    means[nodes[matching]] = gca.node_means[matching]
    return means.reshape(gca.node_shape)


def _source_node_variance(coordinates: np.ndarray, voxel_lta: np.ndarray,
                          variance: np.ndarray) -> np.ndarray:
    source = np.column_stack((coordinates.astype(np.float64),
                              np.ones(len(coordinates), np.float64)))
    atlas = np.float32(source @ voxel_lta.T)[:, :3]
    prior = np.floor(atlas / np.float32(2) + np.float32(.5)).astype(np.int32)
    node = np.clip(prior // 2, 0, np.asarray(variance.shape) - 1)
    return variance[node[:, 0], node[:, 1], node[:, 2]]


def _prefilter_controls(gca: GCA, samples: AtlasSamples, scaled: np.ndarray,
                        label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior, variance = _atlas_label_fields(gca, label)
    local_min = minimum_filter(prior, size=3, mode="constant")
    candidates = (samples.labels == label) & (samples.priors >= .6)
    p = samples.prior_coordinates[candidates]
    s = samples.source_coordinates[candidates].copy()
    stable = local_min[p[:, 0], p[:, 1], p[:, 2]] >= .5
    p, s = p[stable], s[stable]
    intensity = scaled[s[:, 0], s[:, 1], s[:, 2]]
    valid = (intensity >= MIN_CANDIDATE_INTENSITY) & (intensity <= MAX_CANDIDATE_INTENSITY)
    p, s = p[valid], s[valid]
    return p, s, variance


def _uniform_controls(s: np.ndarray, voxel_lta: np.ndarray,
                      scaled: np.ndarray, variance: np.ndarray,
                      minimum: np.ndarray, maximum: np.ndarray,
                      nsigma: float) -> tuple[np.ndarray, np.ndarray]:
    picked = np.zeros(len(s), bool)
    result = s.copy()
    offsets = [(0, 0, 0)] + [(dx, dy, dz) for dx in range(-1, 2)
                            for dy in range(-1, 2) for dz in range(-1, 2)]
    for dx, dy, dz in offsets:
        remaining = np.flatnonzero(~picked)
        if len(remaining) == 0:
            break
        trial = s[remaining] + np.asarray((dx, dy, dz), np.int32)
        trial = np.clip(trial, 0, np.asarray(scaled.shape) - 1)
        value = scaled[trial[:, 0], trial[:, 1], trial[:, 2]].astype(np.float64)
        spread = np.sqrt(_source_node_variance(trial, voxel_lta, variance).astype(np.float64))
        spread = np.maximum(spread, .05 * value)
        spread = np.minimum(spread, .10 * value)
        difference = (maximum[trial[:, 0], trial[:, 1], trial[:, 2]] -
                      minimum[trial[:, 0], trial[:, 1], trial[:, 2]])
        uniform = np.isfinite(spread) & (difference <= np.float32(nsigma) * spread)
        found = remaining[uniform]
        result[found] = s[found] + np.asarray((dx, dy, dy), np.int32)
        picked[found] = True
    return picked, result


@njit
def _propagate_short_bias(bias: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """The 26-neighbor layered expansion in ``mriBuildVoronoiDiagramShort``."""

    sx, sy, sz = bias.shape
    marked = controls.copy().ravel()
    seen = marked.copy()
    values = bias.copy().ravel()
    frontier = np.empty(values.size, np.int32)
    next_frontier = np.empty(values.size, np.int32)
    count = 0
    for i in range(values.size):
        if marked[i]:
            frontier[count] = i
            count += 1
        else:
            values[i] = 0
    while count:
        next_count = 0
        for k in range(count):
            current = frontier[k]
            x = current // (sy * sz)
            y = (current // sz) % sy
            z = current % sz
            for dx in range(-1, 2):
                xx = min(sx - 1, max(0, x + dx))
                for dy in range(-1, 2):
                    yy = min(sy - 1, max(0, y + dy))
                    for dz in range(-1, 2):
                        zz = min(sz - 1, max(0, z + dz))
                        neighbor = (xx * sy + yy) * sz + zz
                        if not seen[neighbor]:
                            seen[neighbor] = True
                            next_frontier[next_count] = neighbor
                            next_count += 1
        for k in range(next_count):
            current = next_frontier[k]
            x = current // (sy * sz)
            y = (current // sz) % sy
            z = current % sz
            total = 0
            n = 0
            for dz in range(-1, 2):
                zz = min(sz - 1, max(0, z + dz))
                for dy in range(-1, 2):
                    yy = min(sy - 1, max(0, y + dy))
                    for dx in range(-1, 2):
                        xx = min(sx - 1, max(0, x + dx))
                        neighbor = (xx * sy + yy) * sz + zz
                        if marked[neighbor]:
                            total += values[neighbor]
                            n += 1
            if n:
                values[current] = np.int16(np.floor(np.float32(total) / np.float32(n) + .5))
        for k in range(next_count):
            marked[next_frontier[k]] = True
        frontier, next_frontier = next_frontier, frontier
        count = next_count
    return values.reshape(bias.shape)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    length = int(np.floor(8 * sigma + .5)) + 1
    length += length % 2 == 0
    length = min(length, 100)
    centered = np.arange(length, dtype=np.float32) - length // 2
    weights = np.empty(length, np.float32)
    for i, coordinate in enumerate(centered):
        absolute = abs(float(coordinate))
        if absolute <= 2 * sigma:
            weights[i] = np.float32(np.exp(-absolute * absolute / (2 * sigma * sigma)))
        elif absolute <= 4 * sigma:
            weights[i] = np.float32((4 - absolute / sigma) ** 4 / (16 * np.e * np.e))
        else:
            weights[i] = 0
    norm = np.float32(0)
    for weight in weights:
        norm = np.float32(norm + weight)
    for i in range(length):
        weights[i] = np.float32(weights[i] / norm)
    return weights


@njit(parallel=True)
def _convolve_short(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    sx, sy, sz = values.shape
    result = np.empty_like(values)
    half = len(kernel) // 2
    for x in prange(sx):
        for y in range(sy):
            for z in range(sz):
                total = np.float32(0)
                for i in range(len(kernel)):
                    offset = i - half
                    xx = x + offset if axis == 0 else x
                    yy = y + offset if axis == 1 else y
                    zz = z + offset if axis == 2 else z
                    if xx < 0:
                        xx = 0
                    elif xx >= sx:
                        xx = sx - 1
                    if yy < 0:
                        yy = 0
                    elif yy >= sy:
                        yy = sy - 1
                    if zz < 0:
                        zz = 0
                    elif zz >= sz:
                        zz = sz - 1
                    total = np.float32(total + np.float32(kernel[i] * np.float32(values[int(xx), int(yy), int(zz)])))
                result[x, y, z] = np.int16(np.floor(np.float64(total) + .5))
    return result


def _smooth_short(values: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    for axis in range(3):
        values = _convolve_short(values, kernel, axis)
    return values


def _smooth_histogram(counts: np.ndarray) -> np.ndarray:
    kernel = _gaussian_kernel(2.)
    half = len(kernel) // 2
    result = np.empty(len(counts), np.float32)
    for bin_index in range(len(counts)):
        total = norm = np.float32(0)
        for kernel_index, weight in enumerate(kernel):
            position = bin_index + kernel_index - half
            if position < 0 or position >= len(counts):
                continue
            norm = np.float32(norm + weight)
            total = np.float32(total + np.float32(weight * counts[position]))
        result[bin_index] = np.float32(total / norm)
    return result


def _control_intensity_limits(intensities: np.ndarray) -> tuple[int, int, int]:
    counts = np.bincount(intensities, minlength=256).astype(np.float32)
    smooth = _smooth_histogram(counts)
    threshold = .15 * float(smooth.max())
    peak = next(bin_index for bin_index in range(255, -1, -1)
                if smooth[bin_index] > threshold and
                smooth[max(0, bin_index - 3):min(256, bin_index + 4)].max() <= smooth[bin_index])
    prev = float(smooth[peak])
    valley = -1
    for index in range(peak - 1, -1, -1):
        if smooth[index] > prev:
            valley = index
            break
        prev = float(smooth[index])
    start = peak - 1
    while start >= valley and smooth[start] >= .01 * smooth[peak]:
        start -= 1
    if start < valley:
        start = valley - 1
    prev = float(smooth[peak])
    valley = -1
    for index in range(peak + 1, 256):
        if smooth[index] > prev:
            valley = index
            break
        prev = float(smooth[index])
    stop = peak + 1
    while stop <= valley and smooth[stop] >= .01 * smooth[peak]:
        stop += 1
    if stop > valley:
        stop = 255
    return peak, max(start, 88), min(stop, 132)


def control_frame(gca: GCA, samples: AtlasSamples, voxel_lta: np.ndarray,
                  scaled: np.ndarray, nregions: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Select one pass's control labels and atlas means."""

    labels = np.zeros(scaled.shape, np.float32)
    means = np.zeros(scaled.shape, np.float32)
    details = []
    minimum = minimum_filter(scaled, size=3, mode="nearest")
    maximum = maximum_filter(scaled, size=3, mode="nearest")
    for label in NORMALIZATION_LABELS:
        p, s, variances = _prefilter_controls(gca, samples, scaled, label)
        original_source = s.copy()
        extent = samples.source_coordinates[samples.labels == label]
        lower_edge, upper_edge = extent.min(axis=0), extent.max(axis=0)
        selected_prior, selected_original = [], []
        for rx in range(nregions):
            for ry in range(nregions):
                for rz in range(nregions):
                    nsigma = 1.
                    while True:
                        region = nregions * (s - lower_edge) // (upper_edge - lower_edge + 1)
                        current_values = scaled[s[:, 0], s[:, 1], s[:, 2]]
                        in_region = np.all(region == (rx, ry, rz), axis=1) & \
                            (current_values >= MIN_CANDIDATE_INTENSITY) & \
                            (current_values <= MAX_CANDIDATE_INTENSITY)
                        indices = np.flatnonzero(in_region)
                        picked, moved = _uniform_controls(
                            s[indices], voxel_lta, scaled,
                            variances, minimum, maximum, nsigma)
                        chosen = indices[picked]
                        prior, original = p[chosen], original_source[chosen]
                        moved = moved[picked]
                        s[chosen] = moved
                        nsigma *= 1.1
                        if len(prior) >= 8 or nsigma >= 3:
                            break
                    if len(prior) < 8:
                        continue
                    moved_values = scaled[moved[:, 0], moved[:, 1], moved[:, 2]]
                    valid = moved_values > 0
                    prior, original, moved = prior[valid], original[valid], moved[valid]
                    histogram = np.bincount(scaled[moved[:, 0], moved[:, 1], moved[:, 2]],
                                            minlength=256).astype(np.float32)
                    image_peak = int(np.argmax(_smooth_histogram(histogram)[1:])) + 1
                    variance = variances[prior[:, 0] // 2, prior[:, 1] // 2, prior[:, 2] // 2]
                    observed = scaled[original[:, 0], original[:, 1], original[:, 2]].astype(np.float32)
                    squared = np.square(observed - np.float32(image_peak))
                    retained = np.float32(squared / variance) < 9
                    selected_prior.append(prior[retained])
                    selected_original.append(original[retained])
        if not selected_prior:
            details.append({"label": label, "region_samples": 0, "discarded": 0})
            continue
        prior = np.concatenate(selected_prior)
        original = np.concatenate(selected_original)
        observed = scaled[original[:, 0], original[:, 1], original[:, 2]]
        _, lower, upper = _control_intensity_limits(observed)
        acceptable = (observed >= lower) & (observed <= upper)
        node_means = _node_label_means(gca, label)
        values = node_means[prior[:, 0] // 2, prior[:, 1] // 2, prior[:, 2] // 2]
        coordinates = tuple(original.T)
        labels[coordinates] = np.where(acceptable, label, 0).astype(np.float32)
        means[coordinates] = values
        details.append({"label": label, "region_samples": len(original),
                        "lower": lower, "upper": upper,
                        "discarded": int(np.count_nonzero(~acceptable))})
    return labels, means, details


def bias_from_control_frame(scaled: np.ndarray, controls: np.ndarray,
                            means: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """One bias field using a previously selected native-style control frame.

    ``controls`` and ``means`` are corresponding frames from a 6-frame
    ``ctrl_pts.mgz``. This does not select the points from a GCA.
    """

    if scaled.dtype != np.uint8 or scaled.shape != controls.shape or controls.shape != means.shape:
        raise ValueError("Expected aligned 3D uint8 image and control/mean frames")
    selected = controls > 0
    bias = np.full(scaled.shape, 1000, np.int16)
    values = np.maximum(scaled[selected].astype(np.float64), 1)
    candidates = np.float32(1000 * (means[selected].astype(np.float64) / values))
    bias[selected] = np.floor(candidates + .5).astype(np.int16)
    at_controls = bias[selected].astype(np.float64)
    center = at_controls.mean()
    spread = np.sqrt(np.mean(at_controls * at_controls) - center * center)
    discarded = selected & (np.abs(bias.astype(np.float64) - center) > 4 * spread)
    selected[discarded] = False
    bias[discarded] = 1000
    bias = _propagate_short_bias(bias, selected)
    down = np.stack([np.stack([np.stack([
        bias[x::2, y::2, z::2] for z in range(2)
    ]) for y in range(2)]) for x in range(2)])
    down = np.floor(down.astype(np.float32).sum(axis=(0, 1, 2)) / 8 + .5).astype(np.int16)
    bias = np.repeat(np.repeat(np.repeat(_smooth_short(down, 16), 2, axis=0),
                               2, axis=1), 2, axis=2)
    bias = _smooth_short(bias, 2)
    return bias, {"control_count": int(np.count_nonzero(selected)),
                  "discarded": int(np.count_nonzero(discarded)),
                  "bias_mean": float(center / 1000), "bias_std": float(spread / 1000)}


def normalize_from_control_frame(scaled: np.ndarray, controls: np.ndarray,
                                 means: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Correct one image with a saved control frame; does not select controls."""

    bias, stats = bias_from_control_frame(scaled, controls, means)
    corrected_float = np.float32(scaled.astype(np.float64) *
                                 (bias.astype(np.float32) / 1000))
    corrected = np.clip(np.floor(corrected_float.astype(np.float64) + .5),
                        0, 255).astype(np.uint8)
    return corrected, stats


def run_ca_normalize(nu_path: str | Path, mask_path: str | Path,
                     gca_path: str | Path, lta_path: str | Path,
                     norm_path: str | Path, ctrl_path: str | Path) -> dict:
    """Write the fixed three-pass ``norm.mgz`` and ``ctrl_pts.mgz`` outputs."""

    start = perf_counter()
    gca = read_gca(gca_path)
    lta = read_voxel_lta(lta_path)
    masked = masked_input(nu_path, mask_path)
    atlas_peak = atlas_label_peak(gca, 2)
    image_peak, _, _ = estimate_image_white_matter_peak(gca, masked)
    image = scale_masked_input(masked, atlas_peak, image_peak)
    samples = atlas_samples(gca, lta)
    setup_seconds = perf_counter() - start

    controls = np.zeros((*image.shape, 6), np.float32)
    passes = []
    for frame in range(3):
        start = perf_counter()
        labels, means, details = control_frame(gca, samples, lta, image, frame + 1)
        selection_seconds = perf_counter() - start
        controls[..., frame] = labels
        controls[..., frame + 3] = means
        start = perf_counter()
        image, bias_stats = normalize_from_control_frame(image, labels, means)
        passes.append({"selection_seconds": selection_seconds,
                       "bias_seconds": perf_counter() - start,
                       "controls": details, "bias": bias_stats})

    start = perf_counter()
    source = nib.load(str(nu_path))
    nib.save(nib.MGHImage(image, source.affine, header=source.header.copy()), str(norm_path))
    ctrl_header = source.header.copy()
    ctrl_header.set_data_dtype(np.float32)
    nib.save(nib.MGHImage(controls, source.affine, header=ctrl_header), str(ctrl_path))
    write_seconds = perf_counter() - start
    return {"atlas_peak": atlas_peak, "image_peak": image_peak,
            "setup_seconds": setup_seconds, "passes": passes,
            "write_seconds": write_seconds,
            "total_seconds": setup_seconds + write_seconds + sum(
                stage["selection_seconds"] + stage["bias_seconds"] for stage in passes)}
