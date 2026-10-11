"""FreeSurfer 8.2 GCA reader and fixed-profile registration input preparation.

This module currently covers the input stages of ``mri_em_register``. It does
not calculate or emit a Talairach transform.
"""

from array import array
from dataclasses import dataclass
from pathlib import Path
import math
import struct

import nibabel as nib
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


_INT = struct.Struct(">i")
_PAIR = struct.Struct(">ii")
_CLASSIFIER = struct.Struct(">iff")
_PRIOR_LABEL = struct.Struct(">if")
_HEADER = struct.Struct(">fff8i")


@dataclass(frozen=True)
class GCA:
    """The node densities and priors needed by the registration search."""

    prior_spacing: float
    node_spacing: float
    prior_shape: tuple[int, int, int]
    node_shape: tuple[int, int, int]
    node_offsets: np.ndarray
    node_training: np.ndarray
    node_labels: np.ndarray
    node_means: np.ndarray
    node_variances: np.ndarray
    prior_offsets: np.ndarray
    prior_training: np.ndarray
    prior_labels: np.ndarray
    prior_values: np.ndarray
    atlas_type: int
    volume_shape: tuple[int, int, int]
    direction_cosines: tuple[float, ...]
    center_ras: tuple[float, float, float]
    voxel_sizes: tuple[float, float, float]


@dataclass(frozen=True)
class CovarianceCheck:
    average_variance: float
    minimum_determinant: float
    singular: int
    ill_conditioned: int
    variances: np.ndarray


@dataclass(frozen=True)
class StableSamples:
    coordinates: np.ndarray
    labels: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    priors: np.ndarray


class _VnlRandom:
    """The VXL Marsaglia-Zaman sequence used by ``setRandomSeed(-1)``."""

    def __init__(self):
        self.state = []
        value = -1
        for _ in range(37):
            value = (value * 1664525 + 1) & 0xFFFFFFFF
            self.state.append(value)
        self.position = 0
        self.borrow = 0
        for _ in range(1000):
            self._int32()
        self.uniform()  # OpenRan1 call within setRandomSeed

    def _int32(self) -> int:
        previous = self.state[(37 + self.position - 24) % 37]
        value = (previous - self.state[self.position] - self.borrow) & 0xFFFFFFFF
        if value < previous:
            self.borrow = 0
        if value > previous:
            self.borrow = 1
        self.state[self.position] = value
        self.position = (self.position + 1) % 37
        return value

    def uniform(self) -> float:
        first, second = self._int32(), self._int32()
        return float(np.float32(first / 0xFFFFFFFF + second / (0xFFFFFFFF ** 2)))


def read_gca(path: str | Path) -> GCA:
    """Read the one-channel v5 GCA used by the fixed T1 recon-all profile."""

    data = Path(path).read_bytes()
    version, prior_spacing, node_spacing, *header = _HEADER.unpack_from(data)
    if version != 5 or header[-2:] != [1, 0]:
        raise ValueError("Expected a v5, one-channel GCA with Gibbs priors")
    prior_shape = tuple(header[:3])
    node_shape = tuple(header[3:6])
    pos = _HEADER.size
    node_offsets = array("I", [0])
    node_training = array("i")
    node_labels = array("i")
    node_means = array("f")
    node_variances = array("f")
    for _ in range(int(np.prod(node_shape))):
        count, training = _PAIR.unpack_from(data, pos)
        pos += _PAIR.size
        node_training.append(training)
        for _ in range(count):
            label, mean, variance = _CLASSIFIER.unpack_from(data, pos)
            pos += _CLASSIFIER.size
            node_labels.append(label)
            node_means.append(mean)
            node_variances.append(variance)
            for _ in range(6):
                neighbors = _INT.unpack_from(data, pos)[0]
                pos += 4 + neighbors * 8
        node_offsets.append(len(node_labels))

    prior_offsets = array("I", [0])
    prior_training = array("i")
    prior_labels = array("i")
    prior_values = array("f")
    for _ in range(int(np.prod(prior_shape))):
        count, training = _PAIR.unpack_from(data, pos)
        pos += _PAIR.size
        prior_training.append(training)
        for _ in range(count):
            label, value = _PRIOR_LABEL.unpack_from(data, pos)
            pos += _PRIOR_LABEL.size
            prior_labels.append(label)
            prior_values.append(value)
        prior_offsets.append(len(prior_labels))

    if _INT.unpack_from(data, pos)[0] != 0xAB2C:
        raise ValueError("Invalid GCA tag section")
    pos += 4
    if _INT.unpack_from(data, pos)[0] != 2 or _INT.unpack_from(data, pos + 4)[0] != 1:
        raise ValueError("Missing GCA type tag")
    atlas_type = _INT.unpack_from(data, pos + 8)[0]
    if _INT.unpack_from(data, len(data) - 76)[0] != 3:
        raise ValueError("Missing final GCA direction cosine tag")
    geometry = struct.unpack_from(">12f3i3f", data, len(data) - 72)
    return GCA(
        prior_spacing, node_spacing, prior_shape, node_shape,
        np.asarray(node_offsets), np.asarray(node_training), np.asarray(node_labels),
        np.asarray(node_means), np.asarray(node_variances),
        np.asarray(prior_offsets), np.asarray(prior_training), np.asarray(prior_labels),
        np.asarray(prior_values), atlas_type, tuple(geometry[12:15]),
        tuple(geometry[:9]), tuple(geometry[9:12]), tuple(geometry[15:18]),
    )


def read_masked_input(nu_path: str | Path, mask_path: str | Path) -> np.ndarray:
    """Apply the five ``MRImask`` calls (mask values 0..4) for T1 input."""

    nu_image = nib.load(str(nu_path))
    mask_image = nib.load(str(mask_path))
    if nu_image.shape != mask_image.shape or not np.allclose(
        nu_image.affine, mask_image.affine, atol=1e-4, rtol=0
    ):
        raise ValueError("Registration input and mask geometry differ")
    data = np.asarray(nu_image.dataobj).copy()
    data[np.asarray(mask_image.dataobj) < 5] = 0
    return data


def scale_input_intensity(data: np.ndarray, atlas_peak: float, image_peak: float) -> np.ndarray:
    """The uint8 ``MRIscalarMulFrame`` kernel after peaks have been estimated."""

    if data.dtype != np.uint8:
        raise ValueError("Fixed single-T1 registration input must be uint8")
    scale = np.float32(atlas_peak / image_peak)
    return np.clip(data.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum, maximum = float(values.min()), float(values.max())
    count = int(np.floor(maximum - minimum + 1.5))
    width = (maximum - minimum) / (count - 1)
    index = np.clip(np.floor((values.ravel() - minimum) / width + .5).astype(np.int32),
                    0, count - 1)
    return (minimum + np.arange(count, dtype=np.float32) * np.float32(width),
            np.bincount(index, minlength=count).astype(np.float32))


def _smooth_histogram(counts: np.ndarray) -> np.ndarray:
    """固定sigma=2直方图平滑，逐项FP32核归一化和加权累积。"""
    kernel = np.empty(17, np.float32)
    total = np.float32(0)
    for index, offset in enumerate(range(-8, 9)):
        distance = abs(offset)
        if distance <= 4:
            value = np.float32(math.exp(-offset * offset / 8.0))
        else:
            coefficient = np.float32(1) / np.float32(16 * np.float32(math.e * math.e))
            value = np.float32(coefficient * np.float32((4 - distance / 2.0) ** 4))
        kernel[index] = value
        total = np.float32(total + value)
    kernel /= total
    result = np.empty_like(counts)
    for index in range(len(counts)):
        weight_sum = weighted_sum = np.float32(0)
        for offset in range(-8, 9):
            location = index + offset
            if 0 <= location < len(counts):
                weight = kernel[offset + 8]
                weight_sum = np.float32(weight_sum + weight)
                weighted_sum = np.float32(weighted_sum + np.float32(counts[location] * weight))
        result[index] = weighted_sum / weight_sum
    return result


def _robust_histogram_gaussian(bins: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """固定uint8 GCA直方图的峰中心/截尾Gaussian宽度，独立协议实现。

    bins为单位间距FP32；counts为已平滑非负FP32。再次sigma2平滑找峰，
    由峰右侧十bin后的谷或绝对距离CDF(.3)确定拟合窗，至少12bin，
    sigma从0.1逐次扫描、双精度SSE选择，保持边界和扫描顺序。
    不调用原软件，不把标准差直接替换为普通样本方差。
    """
    if len(bins) < 2 or not np.all(np.diff(bins) == np.float32(1)):
        raise ValueError('fixed GCA fit requires unit-spaced histogram bins')
    smoothed = _smooth_histogram(counts)
    center_index = int(np.argmax(smoothed))
    center = float(bins[center_index])
    centered = (bins - np.float32(center)).astype(np.float32)
    valley_start = center_index + 10
    valley = next((index for index in range(valley_start + 1, len(bins))
                   if smoothed[index] > smoothed[index-1]), -1)
    if valley_start >= len(bins):
        valley = valley_start
    if valley < 0:
        # 合并正负距离为有序FP32计数，再按double累积/FP32存储CDF。
        maximum_distance = int(np.max(np.abs(centered)))
        absolute_counts = np.zeros(maximum_distance + 1, np.float32)
        for distance, count in zip(np.abs(centered), counts):
            at = int(distance)
            absolute_counts[at] = np.float32(absolute_counts[at] + count)
        cumulative = np.zeros(len(bins), np.float32)
        for index, distance in enumerate(centered):
            if distance >= 0:
                cumulative[index] = np.float32(sum(float(x) for x in absolute_counts[:int(distance)+1]))
        cumulative /= float(sum(float(x) for x in absolute_counts[:int(centered[-1])+1]))
        threshold = float(centered[int(np.argmin(np.abs(cumulative - np.float32(.3))))])
    else:
        valley = min(valley, len(bins)-1)
        while valley > center_index + 1 and counts[valley] < float(counts[center_index]) * .3:
            valley -= 1
        threshold = float(bins[valley]-bins[center_index])
    low = int(-threshold) - int(centered[0])
    high = int(threshold) - int(centered[0])
    while low >= high:
        low = max(0, low-1)
        high = min(len(bins)-1, high+1)
    if high-low < 12:
        high += (12-(high-low)) // 2
        low -= (12-(high-low)) // 2
        low = max(0, low)
        high = min(len(bins)-1, high)
    maximum_distance = int(np.max(np.abs(centered)))
    upper_sigma = (maximum_distance + 1) / 2.0
    increment = float(np.float32(.1))
    sigma = increment
    best_error, best_sigma = math.inf, 0.0
    root_two_pi = math.sqrt(2*math.pi)
    while sigma <= upper_sigma:
        predictions = [math.exp(-float(value)**2/(2*sigma*sigma))/(sigma*root_two_pi)
                       for value in centered[low:high+1]]
        ratios = [float(counts[low+index])/prediction for index,prediction in enumerate(predictions)
                  if prediction > 1e-3]
        amplitude = sum(ratios) / len(ratios)
        error = sum((amplitude*prediction-float(counts[low+index]))**2
                    for index,prediction in enumerate(predictions))
        if error < best_error:
            best_error, best_sigma = error, sigma
        sigma += increment
    return center, best_sigma


def _histogram_peak(counts: np.ndarray, width: int, first: bool, fraction: float) -> int:
    minimum = fraction * float(counts.max())
    half = (width - 1) // 2
    candidates = range(len(counts)) if first else range(len(counts) - 1, -1, -1)
    for index in candidates:
        if counts[index] > minimum and not np.any(
            counts[max(0, index - half):min(len(counts), index + half + 1)] > counts[index]
        ):
            return index
    raise ValueError("No image-histogram peak")


def _skull_box(mean_image: np.ndarray, threshold: int) -> tuple[int, int, int, int, int, int]:
    """The directional dark-run box of ``MRIfindApproximateSkullBoundingBox``."""

    positive = mean_image > threshold
    weight = mean_image[positive].astype(np.float64)
    points = np.argwhere(positive)
    center = np.floor(np.sum(points * weight[:, None], axis=0) / weight.sum() + .5).astype(int)

    def edge(line: np.ndarray, start: int, direction: int, inclusive: bool,
             superior: bool = False) -> int:
        dark = light = longest = maximum_light = 0
        result = run_start = start
        for location in range(start, len(line) if direction > 0 else -1, direction):
            if line[location] < threshold:
                if superior and light > maximum_light:
                    maximum_light = light
                if dark == 0:
                    run_start = location
                dark += 1
                light = 0
            else:
                light += 1
                if light > 30:
                    longest = 0
                if dark > longest or (inclusive and dark >= longest):
                    longest, result = dark, run_start
                    if superior:
                        maximum_light = 0
                dark = 0
        if dark > longest or (superior and dark > longest / 2 and maximum_light > 15):
            longest, result = dark, run_start
        if longest < 10:
            return len(line) - 1 if direction > 0 else 0
        return result

    cx, cy, cz = center.tolist()
    left = edge(mean_image[:, cy, cz], cx, -1, False)
    right = edge(mean_image[:, cy, cz], cx, 1, True)
    if right - left + 1 <= 10:
        raise ValueError("Fixed T1 skull-box left/right search failed")
    top = edge(mean_image[max(0, cx - 20), :, cz], cy, -1, True, True)
    bottom = edge(mean_image[cx, :, cz], cy, 1, True)
    back = edge(mean_image[cx, cy, :], cz, -1, True)
    front = edge(mean_image[cx, cy, :], cz, 1, True)
    if front - back + 1 < 5:
        raise ValueError("Fixed T1 skull-box anterior/posterior search failed")
    return left, top, back, right - left + 1, bottom - top + 1, front - back + 1


def estimate_image_white_matter_peak(gca: GCA, masked: np.ndarray, *,
                                     profile: dict | None = None) -> tuple[int, float, tuple[int, ...]]:
    """固定GCA/同网格uint8 masked T1→强度峰、阈值和WM采样盒。

    gca为已有单T1 GCA；masked为conform强度，单位任意强度值；可选
    profile为调用者dict，追加稳健mean/sigma、宽度阈值和缩放禁用记录。
    返回(image_peak, skull_threshold, (x,y,z,dx,dy,dz))，盒坐标为体素。
    保持源过宽分布禁用缩放规则；内部仅Python/NumPy，不执行原程序。
    非单位bin的固定配方拟合失败明确传播。GCA与CA normalization
    共用本函数；真实宽分布修复与回归见GCA_COMPLETE_PYTHON.md。
    """

    image = masked.astype(np.float32)
    mean_image = uniform_filter(image, size=5, mode="constant")
    mean_image /= uniform_filter(np.ones_like(image), size=5, mode="constant")
    bins, counts = _histogram(mean_image)
    counts[0] = 0
    smooth = _smooth_histogram(counts)
    first_peak = _histogram_peak(smooth, 5, True, .1)
    next_valley = next((i for i in range(first_peak + 1, len(smooth))
                        if smooth[i] > smooth[i - 1]), len(smooth) - 1)
    threshold_count = .25 * smooth[first_peak]
    end = next((i for i in range(first_peak + 1, next_valley + 1)
                if smooth[i] < threshold_count), next_valley)
    if end - first_peak > 40:
        end = next_valley
    if end - first_peak > 40:
        end = first_peak + 39
    threshold = min(float(bins[end]), .25 * float(mean_image.max()))
    x, y, z, dx, dy, dz = _skull_box(mean_image, int(threshold))
    _, site_labels, _ = find_all_sample_sites(gca)
    left = int(np.count_nonzero(site_labels == 2))
    right = int(np.count_nonzero(site_labels == 41))
    x0 = x + (2 * dx // 3 if left > 2 * right else dx // 3)
    y0, z0 = y + dy // 3, z + dz // 2
    dx, dy, dz = dx // 4, dy // 4, dz // 4
    x, y, z = x0 - dx // 2, y0 - dy // 2, z0 - dz // 2
    region = masked[x:x + dx, y:y + dy, z:z + dz].copy()
    region[mean_image[x:x + dx, y:y + dy, z:z + dz] < threshold] = 0
    bins, counts = _histogram(region)
    counts[bins <= 50] = 0
    counts[bins > 240] = 0
    smooth = _smooth_histogram(counts)
    peak = _histogram_peak(smooth, 7, False, .15)
    robust_mean, robust_sigma = _robust_histogram_gaussian(bins, smooth)
    atlas_peak = atlas_label_peak(gca, 2)
    width_threshold = 10.0 * (robust_mean / atlas_peak)
    disabled = robust_sigma > width_threshold
    if disabled:
        # Disable scaling by returning the atlas value, even outside local bins.
        image_peak = int(atlas_peak)
    else:
        if not 0 <= peak < len(bins) or bins[peak] <= 0:
            raise ValueError('fixed GCA histogram scaling peak is invalid')
        image_peak = int(bins[peak])
    if profile is not None:
        profile.update(robust_mean=robust_mean,robust_sigma=robust_sigma,
                       width_threshold=width_threshold,scaling_disabled=disabled,
                       image_peak=image_peak,atlas_peak=atlas_peak)
    return image_peak, threshold, (x, y, z, dx, dy, dz)


def atlas_label_peak(gca: GCA, label: int) -> int:
    """Mode of the 256-bin node-mean histogram weighted by node priors."""

    histogram = np.zeros(256, np.float64)
    indices = np.flatnonzero(gca.node_labels == label)
    nodes = np.searchsorted(gca.node_offsets[1:], indices, side="right")
    ny, nz = gca.node_shape[1:]
    py, pz = gca.prior_shape[1:]
    ratio = int(gca.node_spacing / gca.prior_spacing)
    for classifier, node in zip(indices, nodes):
        x, yz = divmod(int(node), ny * nz)
        y, z = divmod(yz, nz)
        prior = ((x * ratio) * py + y * ratio) * pz + z * ratio
        begin, end = gca.prior_offsets[prior:prior + 2]
        candidates = gca.prior_labels[begin:end]
        matches = np.flatnonzero(candidates == label)
        if len(matches):
            weight = float(gca.prior_values[begin + matches[0]])
        else:
            total = int(gca.prior_training[prior])
            weight = .1 / total if total else 0.
        bin_index = int(np.clip(np.floor(float(gca.node_means[classifier]) + .5), 0, 255))
        histogram[bin_index] += weight
    return int(np.argmax(histogram))


def regularize_covariance(gca: GCA) -> CovarianceCheck:
    """Replicate the single-input branch of ``GCAfixSingularCovarianceMatrices``."""

    training = np.empty(len(gca.node_labels), dtype=np.int32)
    ny, nz = gca.node_shape[1:]
    py, pz = gca.prior_shape[1:]
    ratio = int(gca.node_spacing / gca.prior_spacing)
    for node in range(len(gca.node_training)):
        begin, end = gca.node_offsets[node:node + 2]
        if begin == end:
            continue
        x, yz = divmod(node, ny * nz)
        y, z = divmod(yz, nz)
        prior = ((x * ratio) * py + y * ratio) * pz + z * ratio
        pbegin, pend = gca.prior_offsets[prior:prior + 2]
        priors = dict(zip(gca.prior_labels[pbegin:pend], gca.prior_values[pbegin:pend]))
        total = int(gca.prior_training[prior])
        for sample in range(int(begin), int(end)):
            value = priors.get(gca.node_labels[sample], 0.1 / total if total else 0.0)
            training[sample] = int(np.float32(gca.node_training[node] * value))

    original = gca.node_variances
    eligible = ((training == 0) & (original > 1)) | (training > 5)
    average_variance = float(np.sum(original[eligible], dtype=np.float64) / np.count_nonzero(eligible))
    minimum_determinant = average_variance / 10.0
    singular = original <= 0
    ill_conditioned = ~singular & (((training < 4) & (original < 0.1)) | (original < minimum_determinant))
    variances = original.copy()
    variances[singular | ill_conditioned] += np.float32(average_variance)
    return CovarianceCheck(average_variance, minimum_determinant,
                           int(np.count_nonzero(singular)),
                           int(np.count_nonzero(ill_conditioned)), variances)


def find_stable_samples(gca: GCA) -> StableSamples:
    """Select the fixed T1 profile's 8-mm, ``-uns 3`` GCA samples."""

    if gca.prior_shape != (128, 128, 128) or gca.node_shape != (64, 64, 64):
        raise ValueError("Expected the fixed 2020 single-T1 GCA")
    corrected_variances = regularize_covariance(gca).variances
    max_label = int(max(gca.prior_labels.max(), gca.node_labels.max()))
    node_ids = np.repeat(np.arange(len(gca.node_training), dtype=np.int32),
                         np.diff(gca.node_offsets))
    classifier = np.full((len(gca.node_training), max_label + 1), -1, np.int32)
    classifier[node_ids, gca.node_labels] = np.arange(len(gca.node_labels), dtype=np.int32)

    cells = np.repeat(np.arange(len(gca.prior_training), dtype=np.int32),
                      np.diff(gca.prior_offsets))
    xp, yz = divmod(cells, 128 * 128)
    yp, zp = divmod(yz, 128)
    node = ((xp // 2 * 64) + yp // 2) * 64 + zp // 2
    classifier_at_prior = classifier[node, gca.prior_labels]
    valid = classifier_at_prior >= 0
    weights = np.where(valid, np.float32(gca.prior_values * gca.prior_training[cells]), 0)
    max_priors = np.zeros(max_label + 1, np.float32)
    np.maximum.at(max_priors, gca.prior_labels, gca.prior_values)

    max_values = np.full(len(gca.prior_training), -1, np.float32)
    np.maximum.at(max_values, cells, gca.prior_values)
    is_maximum = gca.prior_values == max_values[cells]
    last_maximum = np.full(len(gca.prior_training), -1, np.int32)
    np.maximum.at(last_maximum, cells[is_maximum], np.flatnonzero(is_maximum))
    max_labels = np.full(len(gca.prior_training), -1, np.int16)
    occupied = last_maximum >= 0
    max_labels[occupied] = gca.prior_labels[last_maximum[occupied]]
    has_nonunknown_neighbor = maximum_filter(
        (max_labels.reshape(gca.prior_shape) > 0).astype(np.uint8),
        size=7, mode="constant",
    )

    rng = _VnlRandom()
    label_counts = np.zeros(max_label + 1, np.int32)
    unknown_occupied = np.zeros((256, 256, 256), np.uint8)
    coordinates, labels, means, variances, priors = [], [], [], [], []
    for x in range(0, 128, 4):
        for y in range(0, 128, 4):
            for z in range(0, 128, 4):
                regional_weights = np.zeros(max_label + 1, np.float32)
                total_training = 0
                for xx in range(max(0, x - 2), min(128, x + 3)):
                    for yy in range(max(0, y - 2), min(128, y + 3)):
                        for zz in range(max(0, z - 2), min(128, z + 3)):
                            cell = (xx * 128 + yy) * 128 + zz
                            total_training += int(gca.prior_training[cell])
                            for k in range(int(gca.prior_offsets[cell]), int(gca.prior_offsets[cell + 1])):
                                label = int(gca.prior_labels[k])
                                regional_weights[label] = np.float32(regional_weights[label] + weights[k])
                regional_priors = regional_weights / np.float32(total_training)
                if (has_nonunknown_neighbor[x, y, z] and regional_priors[0] >= .5
                        and regional_priors[0] >= .5 * max_priors[0]):
                    best_label = 0
                else:
                    best_label = -1
                best_regional_prior = -1.
                for label in range(1, max_label + 1):
                    prior = float(regional_priors[label])
                    if ((prior < .5 and prior < .5 * float(max_priors[label]))
                            or prior < .25):
                        continue
                    if (best_label == 0 or prior > best_regional_prior or (
                            abs(prior - best_regional_prior) < 1e-5
                            and label_counts[best_label] > label_counts[label])):
                        best_label, best_regional_prior = label, prior
                if len(labels) and rng.uniform() < (
                        label_counts[best_label] / len(labels) if best_label >= 0 else 0):
                    continue
                if best_label < 0:
                    continue

                highest_prior = 0.
                chosen = None
                chosen_classifier = -1
                for xx in range(max(0, x - 2), min(128, x + 3)):
                    for yy in range(max(0, y - 2), min(128, y + 3)):
                        for zz in range(max(0, z - 2), min(128, z + 3)):
                            cell = (xx * 128 + yy) * 128 + zz
                            for k in range(int(gca.prior_offsets[cell]), int(gca.prior_offsets[cell + 1])):
                                if (gca.prior_labels[k] == best_label and valid[k] and
                                        float(gca.prior_values[k]) > highest_prior):
                                    highest_prior = float(gca.prior_values[k])
                                    chosen = (xx, yy, zz)
                                    chosen_classifier = int(classifier_at_prior[k])
                if chosen is None:
                    continue
                if best_label == 0:
                    xv, yv, zv = 2 * x, 2 * y, 2 * z
                    if unknown_occupied[xv, yv, zv]:
                        continue
                    unknown_occupied[max(0, xv - 2):min(256, xv + 3),
                                     max(0, yv - 2):min(256, yv + 3),
                                     max(0, zv - 2):min(256, zv + 3)] = 1
                label_counts[best_label] += 1
                coordinates.append(chosen)
                labels.append(best_label)
                means.append(gca.node_means[chosen_classifier])
                variances.append(corrected_variances[chosen_classifier])
                priors.append(highest_prior)
    return StableSamples(np.asarray(coordinates, np.int32), np.asarray(labels, np.int32),
                         np.asarray(means, np.float32), np.asarray(variances, np.float32),
                         np.asarray(priors, np.float32))


def find_all_sample_sites(gca: GCA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select atlas prior cells for the final EM stage with fixed ``-uns 3``."""

    cells = np.repeat(np.arange(len(gca.prior_training), dtype=np.int32),
                      np.diff(gca.prior_offsets))
    max_values = np.full(len(gca.prior_training), -1, np.float32)
    np.maximum.at(max_values, cells, gca.prior_values)
    is_last_maximum = gca.prior_values == max_values[cells]
    last_maximum = np.full(len(gca.prior_training), -1, np.int32)
    np.maximum.at(last_maximum, cells[is_last_maximum], np.flatnonzero(is_last_maximum))
    occupied = last_maximum >= 0
    labels = np.full(len(gca.prior_training), -1, np.int16)
    labels[occupied] = gca.prior_labels[last_maximum[occupied]]
    label_grid = labels.reshape(gca.prior_shape)
    has_other_than_unknown = maximum_filter(
        (label_grid > 0).astype(np.uint8), size=7, mode="constant")
    unknown = np.isin(label_grid, (0, 255, 190, 191))
    selected = (occupied.reshape(gca.prior_shape) &
                (~unknown | has_other_than_unknown.astype(bool)))
    coordinates = np.argwhere(selected).astype(np.int32)
    indices = last_maximum[selected.ravel()]
    return coordinates, gca.prior_labels[indices], gca.prior_values[indices]


def find_all_samples(gca: GCA) -> StableSamples:
    """Attach atlas means and corrected variances to every final EM site."""

    coordinates, labels, priors = find_all_sample_sites(gca)
    node_ids = np.repeat(np.arange(len(gca.node_training), dtype=np.int32),
                         np.diff(gca.node_offsets))
    classifier = np.full((len(gca.node_training), int(gca.node_labels.max()) + 1),
                         -1, np.int32)
    classifier[node_ids, gca.node_labels] = np.arange(len(gca.node_labels), dtype=np.int32)
    node_coordinates = coordinates // int(gca.node_spacing / gca.prior_spacing)
    node = np.ravel_multi_index(node_coordinates.T, gca.node_shape)
    chosen = classifier[node, labels]
    if np.any(chosen < 0):
        raise ValueError("Final GCA sample lacks a direct node classifier")
    variances = regularize_covariance(gca).variances
    return StableSamples(coordinates, labels, gca.node_means[chosen],
                         variances[chosen], priors)


def gca_mean_volume(gca: GCA) -> np.ndarray:
    """Rasterize ``GCAmri`` onto the 256³ uchar Talairach volume."""

    node_ids = np.repeat(np.arange(len(gca.node_training), dtype=np.int32),
                         np.diff(gca.node_offsets))
    classifier = np.full((len(gca.node_training), int(gca.prior_labels.max()) + 1), -1, np.int32)
    classifier[node_ids, gca.node_labels] = np.arange(len(gca.node_labels), dtype=np.int32)
    cells = np.repeat(np.arange(len(gca.prior_training), dtype=np.int32),
                      np.diff(gca.prior_offsets))
    xp, yz = divmod(cells, 128 * 128)
    yp, zp = divmod(yz, 128)
    node = ((xp // 2 * 64) + yp // 2) * 64 + zp // 2
    classifier_at_prior = classifier[node, gca.prior_labels]
    valid = classifier_at_prior >= 0
    weighted = np.zeros(len(gca.prior_training), np.float32)
    np.add.at(weighted, cells[valid], np.float32(
        gca.node_means[classifier_at_prior[valid]] * gca.prior_values[valid]))
    prior_grid = weighted.reshape(gca.prior_shape)
    nearest_prior = np.minimum((np.arange(256, dtype=np.int32) + 1) // 2, 127)
    values = prior_grid[np.ix_(nearest_prior, nearest_prior, nearest_prior)]
    return np.floor(values + np.float32(.5)).astype(np.uint8)


def gca_centroid(volume: np.ndarray) -> np.ndarray:
    """Intensity-weighted voxel center used for nine-parameter rotations."""

    weight = float(volume.sum(dtype=np.float64))
    coordinates = np.arange(256, dtype=np.float64)
    return np.array([
        np.dot(volume.sum(axis=(1, 2), dtype=np.float64), coordinates),
        np.dot(volume.sum(axis=(0, 2), dtype=np.float64), coordinates),
        np.dot(volume.sum(axis=(0, 1), dtype=np.float64), coordinates),
    ]) / weight


def _vnl_affine_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert the fixed 4x4 affine with VNL's float32 cofactor order."""

    matrix = np.asarray(matrix, np.float32)
    a, b, c = matrix[0, :3]
    d, e, f = matrix[1, :3]
    g, h, i = matrix[2, :3]
    t0, t1, t2 = matrix[:3, 3]
    cast = np.float32
    adjugate = np.array([
        [cast(e*i-f*h), cast(c*h-b*i), cast(b*f-c*e)],
        [cast(f*g-d*i), cast(a*i-c*g), cast(c*d-a*f)],
        [cast(d*h-e*g), cast(b*g-a*h), cast(a*e-b*d)],
    ], dtype=np.float32)
    determinant = cast(cast(cast(cast(cast(
        cast(a*e*i) - cast(a*f*h)) - cast(b*d*i)) +
        cast(b*f*g)) + cast(c*d*h)) - cast(c*e*g))
    reciprocal = cast(1 / determinant)
    translation = np.array([
        cast(-b*f*t2 + b*t1*i + e*c*t2 - e*t0*i - h*c*t1 + h*t0*f),
        cast(a*f*t2 - a*t1*i - d*c*t2 + d*t0*i + g*c*t1 - g*t0*f),
        cast(-a*e*t2 + a*t1*h + d*b*t2 - d*t0*h - g*b*t1 + g*t0*e),
    ], dtype=np.float32)
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = cast(adjugate * reciprocal)
    inverse[:3, 3] = cast(translation * reciprocal)
    return inverse


def log_sample_probability(samples: StableSamples, source: np.ndarray,
                           source_to_atlas: np.ndarray, *, clamp: float = 6.0) -> float:
    """Mean clamped GCA sample log density for a voxel-to-voxel transform."""

    inverse = _vnl_affine_inverse(source_to_atlas)
    prior_to_source = inverse @ np.diag(np.array([2, 2, 2, 1], dtype=np.float32))
    prior_voxels = samples.coordinates.astype(np.float32)
    voxel_float = np.zeros((len(prior_voxels), 3), dtype=np.float32)
    for axis in range(3):
        for source_axis in range(3):
            voxel_float[:, axis] = np.float32(
                voxel_float[:, axis] +
                np.float32(prior_to_source[axis, source_axis] * prior_voxels[:, source_axis])
            )
        voxel_float[:, axis] = np.float32(voxel_float[:, axis] + prior_to_source[axis, 3])
    voxels = np.where(voxel_float < 0, np.ceil(voxel_float - .5),
                      np.floor(voxel_float + .5)).astype(np.int32)
    inside = np.all((voxels >= 0) & (voxels < np.asarray(source.shape)), axis=1)
    log_values = np.full(len(samples.labels), -1000000., np.float64)
    selected = voxels[inside]
    intensity = source[selected[:, 0], selected[:, 1], selected[:, 2]].astype(np.float32)
    difference = intensity - samples.means[inside]
    squared = np.float32(difference * difference)
    mahalanobis = np.float32(squared / samples.variances[inside])
    likelihood = (-np.log(np.sqrt(samples.variances[inside].astype(np.float64)))
                  - .5 * mahalanobis.astype(np.float64)
                  + np.log(samples.priors[inside].astype(np.float64)))
    log_values[inside] = np.maximum(likelihood, -float(clamp))
    return float(np.float32(np.float32(np.sum(log_values, dtype=np.float64)) /
                            np.float32(len(log_values))))


def search_translation_grid(samples: StableSamples, source: np.ndarray,
                            base_transform: np.ndarray, lower: float,
                            upper: float, steps: int = 19) -> tuple[np.ndarray, float, tuple[float, float, float]]:
    """One FreeSurfer translation-search grid, preserving traversal and ties."""

    def candidates():
        value = lower
        delta = (upper - lower) / steps
        while value <= upper:
            yield value
            value += delta

    base = np.asarray(base_transform, np.float32)
    maximum = log_sample_probability(samples, source, base)
    best = (0., 0., 0.)
    for x in candidates():
        for y in candidates():
            for z in candidates():
                trial = base.copy()
                trial[:3, 3] += np.asarray((x, y, z), np.float32)
                score = log_sample_probability(samples, source, trial)
                if score > maximum:
                    maximum = score
                    best = (x, y, z)
    result = base.copy()
    result[:3, 3] += np.asarray(best, np.float32)
    return result, log_sample_probability(samples, source, result), best


def find_optimal_translation(samples: StableSamples, source: np.ndarray,
                             base_transform: np.ndarray) -> tuple[np.ndarray, list[tuple[float, tuple[float, float, float]]]]:
    """The eight progressively narrower translation grids in FreeSurfer 8.2."""

    matrix = np.asarray(base_transform, np.float32).copy()
    lower, upper = -200., 200.
    history = []
    for _ in range(8):
        matrix, score, offset = search_translation_grid(samples, source, matrix, lower, upper)
        history.append((score, offset))
        middle = (lower + upper) / 2
        quarter_width = (upper - lower) / 4
        lower, upper = middle - quarter_width, middle + quarter_width
    return matrix, history


def _grid_values(lower: np.float32, upper: np.float32, step: float) -> list[float]:
    values = []
    value = float(lower)
    while value <= upper:
        values.append(value)
        value += step
    return values


def _rotation(axis: int, angle: float) -> np.ndarray:
    angle = np.float32(angle)
    cosine = np.float32(math.cos(angle))
    sine = np.float32(math.sin(angle))
    matrix = np.eye(4, dtype=np.float32)
    if axis == 0:
        matrix[1, 1], matrix[1, 2] = cosine, sine
        matrix[2, 1], matrix[2, 2] = -sine, cosine
    elif axis == 1:
        matrix[0, 0], matrix[0, 2] = cosine, -sine
        matrix[2, 0], matrix[2, 2] = sine, cosine
    else:
        matrix[0, 0], matrix[0, 1] = cosine, sine
        matrix[1, 0], matrix[1, 1] = -sine, cosine
    return matrix


def _centered(matrix: np.ndarray, origin: np.ndarray) -> np.ndarray:
    forward = np.eye(4, dtype=np.float32)
    backward = np.eye(4, dtype=np.float32)
    forward[:3, 3] = origin
    backward[:3, 3] = -origin
    return (forward @ matrix @ backward).astype(np.float32)


def search_linear_iteration(samples: StableSamples, source: np.ndarray,
                            base_transform: np.ndarray, origin: np.ndarray,
                            search_scale: float, minimum_scale: float,
                            maximum_scale: float) -> tuple[np.ndarray, float]:
    """Two exhaustive nine-parameter reductions from FreeSurfer 8.2."""

    base = np.asarray(base_transform, np.float32).copy()
    origin = np.asarray(origin, np.float32)
    minimum_scale, maximum_scale = np.float32(minimum_scale), np.float32(maximum_scale)
    minimum_angle = np.float32(-np.pi / 6 * search_scale)
    maximum_angle = np.float32(np.pi / 6 * search_scale)
    minimum_translation = np.float32(-15 * search_scale)
    maximum_translation = np.float32(15 * search_scale)
    maximum = log_sample_probability(samples, source, base)
    for _ in range(2):
        scales = _grid_values(minimum_scale, maximum_scale,
                              float(np.float32((maximum_scale - minimum_scale) / np.float32(2))))
        angles = _grid_values(minimum_angle, maximum_angle,
                              float(np.float32((maximum_angle - minimum_angle) / np.float32(4))))
        translations = _grid_values(minimum_translation, maximum_translation,
                                    float(np.float32((maximum_translation - minimum_translation) / np.float32(2))))
        best = (1., 1., 1., 0., 0., 0., 0., 0., 0.)
        for sx in scales:
            for sy in scales:
                for sz in scales:
                    scale_matrix = np.eye(4, dtype=np.float32)
                    scale_matrix[0, 0], scale_matrix[1, 1], scale_matrix[2, 2] = sx, sy, sz
                    centered_scale = _centered(scale_matrix, origin)
                    for ax in angles:
                        x_rotation = _rotation(0, ax)
                        for ay in angles:
                            yx_rotation = (_rotation(1, ay) @ x_rotation).astype(np.float32)
                            for az in angles:
                                rotation = _centered((_rotation(2, az) @ yx_rotation).astype(np.float32), origin)
                                fixed = (centered_scale @ rotation @ base).astype(np.float32)
                                for tx in translations:
                                    for ty in translations:
                                        for tz in translations:
                                            trial = fixed.copy()
                                            trial[:3, 3] += np.asarray((tx, ty, tz), np.float32)
                                            score = log_sample_probability(samples, source, trial)
                                            if score > maximum:
                                                maximum = score
                                                best = (sx, sy, sz, ax, ay, az, tx, ty, tz)
        sx, sy, sz, ax, ay, az, tx, ty, tz = best
        scale_matrix = np.eye(4, dtype=np.float32)
        scale_matrix[0, 0], scale_matrix[1, 1], scale_matrix[2, 2] = sx, sy, sz
        rotation = (_rotation(2, az) @ _rotation(1, ay) @ _rotation(0, ax)).astype(np.float32)
        base = (_centered(scale_matrix, origin) @ _centered(rotation, origin) @ base).astype(np.float32)
        base[:3, 3] += np.asarray((tx, ty, tz), np.float32)
        scale_center = np.float32((maximum_scale + minimum_scale) / 2)
        scale_half_width = np.float32((maximum_scale - minimum_scale) / 4)
        minimum_scale, maximum_scale = (np.float32(scale_center - scale_half_width),
                                        np.float32(scale_center + scale_half_width))
        angle_center = np.float32((maximum_angle + minimum_angle) / 2)
        angle_half_width = np.float32((maximum_angle - minimum_angle) / 4)
        minimum_angle, maximum_angle = (np.float32(angle_center - angle_half_width),
                                        np.float32(angle_center + angle_half_width))
        translation_center = np.float32((maximum_translation + minimum_translation) / 2)
        translation_half_width = np.float32((maximum_translation - minimum_translation) / 4)
        minimum_translation, maximum_translation = (
            np.float32(translation_center - translation_half_width),
            np.float32(translation_center + translation_half_width),
        )
    return base, maximum


def find_optimal_linear_transform(
    samples: StableSamples, source: np.ndarray, base_transform: np.ndarray,
    origin: np.ndarray, initial_score: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Repeat nine-parameter grids until the native 0.025 search-scale stop."""

    matrix = np.asarray(base_transform, np.float32).copy()
    current_score = initial_score
    search_scale = 1.
    minimum_scale, maximum_scale = .85, 1.15
    scale_reductions = 0
    good_step = False
    done = False
    history = []
    while True:
        old_score = current_score
        matrix, current_score = search_linear_iteration(
            samples, source, matrix, origin, search_scale,
            minimum_scale, maximum_scale,
        )
        history.append((search_scale, current_score))
        if current_score < old_score + abs(.001 * old_score):
            search_scale *= .25
            if search_scale < .025:
                break
            half_width = (maximum_scale - minimum_scale) / 2
            minimum_scale = 1 - half_width * search_scale
            maximum_scale = 1 + half_width * search_scale
            done = not good_step
            good_step = False
            scale_reductions += 1
        else:
            good_step = True
        if scale_reductions >= 3 and done:
            break
    return matrix, history
