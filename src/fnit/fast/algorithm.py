"""Single-channel T1 tissue segmentation without native software calls.

The default tensor path uses synchronous spatial updates. The explicit fsl
path preserves the original FAST4 ordered scans and arithmetic conventions;
its CUDA scans use wavefronts with the same directed dependencies.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FASTConfig:
    """Single-channel T1 settings corresponding to FAST4 defaults."""

    init_iterations: int = 15
    bias_iterations: int = 4
    fixed_iterations: int = 4
    bias_fwhm_mm: float = 20.0
    init_mrf: float = 0.02
    mrf: float = 0.1
    mixel_mrf: float = 0.3
    pve_steps: int = 100
    mean_field_iterations: int = 5
    pve_chunk_size: int = 8
    variance_floor_fraction: float = 1e-6
    execution: str = "tensor"

    def __post_init__(self):
        if self.execution not in {"tensor", "fsl"}:
            raise ValueError("execution must be 'tensor' or 'fsl'")
        integer_fields = (
            "init_iterations", "bias_iterations", "fixed_iterations",
            "pve_steps", "mean_field_iterations", "pve_chunk_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or value < (1 if name in {
                    "pve_steps", "mean_field_iterations", "pve_chunk_size"} else 0):
                raise ValueError(f"{name} has an invalid value: {value}")
        for name in ("bias_fwhm_mm", "init_mrf", "mrf", "mixel_mrf"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.variance_floor_fraction) or self.variance_floor_fraction <= 0:
            raise ValueError("variance_floor_fraction must be finite and positive")


@dataclass
class FASTTensorResult:
    """Tensor outputs on the input voxel grid."""

    pve: torch.Tensor
    hard_segmentation: torch.Tensor
    pve_segmentation: torch.Tensor
    mixel_type: torch.Tensor
    bias_field: torch.Tensor
    restored: torch.Tensor
    tissue_means: torch.Tensor
    tissue_variances: torch.Tensor


def _moments(values, probabilities, mask, variance_floor):
    weights = probabilities * mask.unsqueeze(0)
    totals = weights.sum(dim=(1, 2, 3)).clamp_min(torch.finfo(values.dtype).tiny)
    means = (weights * values).sum(dim=(1, 2, 3)) / totals
    residual = values.unsqueeze(0) - means[:, None, None, None]
    variances = (weights * residual.square()).sum(dim=(1, 2, 3)) / totals
    return means, variances.clamp_min(variance_floor)


def _order_statistic(values, fraction):
    ordered = torch.sort(values.reshape(-1)).values
    index = min(int(math.floor(ordered.numel() * fraction)), ordered.numel() - 1)
    return ordered[index]


def _log_likelihood(values, means, variances):
    residual = values.unsqueeze(0) - means[:, None, None, None]
    return -0.5 * (
        math.log(2 * math.pi)
        + torch.log(variances)[:, None, None, None]
        + residual.square() / variances[:, None, None, None]
    )


def _neighbor_kernel(voxel_size, *, device, dtype):
    kernel = torch.zeros((3, 3, 3), device=device, dtype=dtype)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                nonzero = (i != 0) + (j != 0) + (k != 0)
                if nonzero not in (1, 2):
                    continue
                distance = math.sqrt(
                    (i * voxel_size[0]) ** 2
                    + (j * voxel_size[1]) ** 2
                    + (k * voxel_size[2]) ** 2
                )
                kernel[i + 1, j + 1, k + 1] = 1.0 / distance
    return kernel


def _neighbor_support(probabilities, kernel):
    channels = probabilities.shape[0]
    weight = kernel[None, None].expand(channels, 1, 3, 3, 3)
    return F.conv3d(probabilities.unsqueeze(0), weight, padding=1,
                    groups=channels).squeeze(0)


def _hmrf_probabilities(values, means, variances, mask, kernel, beta, iterations):
    likelihood = _log_likelihood(values, means, variances)
    probabilities = torch.softmax(likelihood, dim=0) * mask.unsqueeze(0)
    for _ in range(iterations):
        support = _neighbor_support(probabilities, kernel)
        probabilities = torch.softmax(likelihood + beta * support, dim=0)
        probabilities *= mask.unsqueeze(0)
    return probabilities


def _gaussian_kernel(sigma, *, device, dtype):
    if sigma < 1:
        return torch.ones(1, device=device, dtype=dtype)
    radius = 2 * int(sigma)
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    return kernel / kernel.sum()


def _separable_blur(volume, kernels):
    result = volume[None, None]
    for axis, kernel in enumerate(kernels):
        radius = kernel.numel() // 2
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = radius
        result = F.conv3d(result, kernel.reshape(shape), padding=tuple(padding))
    return result[0, 0]


def _estimate_bias(log_input, probabilities, means, variances, mask, kernels):
    precision = probabilities / variances[:, None, None, None]
    denominator = _separable_blur(precision.sum(dim=0), kernels)
    residual = log_input.unsqueeze(0) - means[:, None, None, None]
    numerator = _separable_blur((precision * residual).sum(dim=0), kernels)
    bias = numerator / denominator.clamp_min(torch.finfo(log_input.dtype).tiny)
    bias = torch.where(mask, bias - bias[mask].mean(), torch.zeros_like(bias))
    return bias


def _mixture_log_evidence(values, mean_a, variance_a, mean_b, variance_b,
                          fractions, chunk_size):
    total = torch.full_like(values, -torch.inf)
    for start in range(0, fractions.numel(), chunk_size):
        fraction = fractions[start:start + chunk_size, None, None, None]
        mean = fraction * mean_a + (1 - fraction) * mean_b
        variance = (fraction.square() * variance_a
                    + (1 - fraction).square() * variance_b)
        residual = values.unsqueeze(0) - mean
        log_probability = -0.5 * (
            math.log(2 * math.pi) + torch.log(variance)
            + residual.square() / variance
        )
        total = torch.logaddexp(total, torch.logsumexp(log_probability, dim=0))
    return total + math.log(1.0 / (fractions.numel() - 1))


def _mixel_labels(values, means, variances, mask, kernel, config):
    fractions = torch.linspace(0, 1, config.pve_steps + 1,
                               device=values.device, dtype=values.dtype)
    evidence_fractions = torch.linspace(0, 1, 101,
                                        device=values.device, dtype=values.dtype)
    pairs = ((0, 1), (0, 2), (1, 2))
    scores = [*_log_likelihood(values, means, variances)]
    scores.extend(_mixture_log_evidence(
        values, means[a], variances[a], means[b], variances[b],
        evidence_fractions, config.pve_chunk_size,
    ) for a, b in pairs)
    scores = torch.stack(scores)
    labels = scores.argmax(dim=0)

    one_hot = F.one_hot(labels, num_classes=6).movedim(-1, 0).to(values.dtype)
    one_hot *= mask.unsqueeze(0)
    support = _neighbor_support(one_hot, kernel)
    constituents = ({0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2})
    compatibility = torch.full((6, 6), -1.0, device=values.device,
                               dtype=values.dtype)
    for candidate in range(6):
        for neighbor in range(6):
            if candidate == neighbor:
                compatibility[candidate, neighbor] = 2
            elif ((len(constituents[candidate]) == 1) !=
                  (len(constituents[neighbor]) == 1)) and (
                    constituents[candidate] & constituents[neighbor]):
                compatibility[candidate, neighbor] = 1
    clique = torch.stack([
        sum(compatibility[candidate, neighbor] * support[neighbor]
            for neighbor in range(6))
        for candidate in range(6)
    ])
    labels = (scores + config.mixel_mrf * clique).argmax(dim=0)
    return torch.where(mask, labels, torch.zeros_like(labels)), fractions, pairs


def _partial_volumes(values, means, variances, mask, mixel, fractions, pairs,
                     chunk_size):
    pve = torch.zeros((3, *values.shape), device=values.device, dtype=values.dtype)
    for tissue in range(3):
        pve[tissue][mask & (mixel == tissue)] = 1

    for pair_index, (a, b) in enumerate(pairs, start=3):
        selected = mask & (mixel == pair_index)
        if not torch.any(selected):
            continue
        selected_values = values[selected]
        best_energy = torch.full_like(selected_values, torch.inf)
        best_fraction = torch.zeros_like(selected_values)
        for start in range(0, fractions.numel(), chunk_size):
            fraction = fractions[start:start + chunk_size, None]
            mean = fraction * means[a] + (1 - fraction) * means[b]
            variance = (fraction.square() * variances[a]
                        + (1 - fraction).square() * variances[b])
            energy = 0.5 * (
                (selected_values.unsqueeze(0) - mean).square() / variance
                + torch.log(variance)
            )
            chunk_energy, chunk_index = energy.min(dim=0)
            improve = chunk_energy < best_energy
            best_energy[improve] = chunk_energy[improve]
            best_fraction[improve] = fractions[start + chunk_index[improve]]
        pve[a][selected] = best_fraction
        pve[b][selected] = 1 - best_fraction
    return pve


def _fsl_fractions(steps, *, device):
    """The float32 loop variable, advanced by a double-precision step."""
    delta = np.float32(0)
    values = []
    while delta <= 1:
        values.append(delta)
        delta = np.float32(float(delta) + 1.0 / steps)
    return torch.tensor(np.asarray(values), device=device)


def _fsl_moments(values, probabilities, mask):
    """Float32 products, double reductions, then stored float32 moments."""
    weights = probabilities * mask[None]
    total = weights.sum(dim=(1, 2, 3), dtype=torch.float64)
    first = weights * values
    means = (first.sum(dim=(1, 2, 3), dtype=torch.float64) / total).float()
    second = first * values
    variances = (second.sum(dim=(1, 2, 3), dtype=torch.float64) / total
                 - (means * means).double()).float()
    if not torch.isfinite(means).all() or not torch.isfinite(variances).all() or (variances <= 0).any():
        raise RuntimeError("FAST execution='fsl' encountered a non-positive class variance")
    return means, variances


def _fsl_energy(values, means, variances):
    delta = (values.unsqueeze(0) - means[:, None, None, None]).float()
    variance = variances[:, None, None, None]
    scale = torch.log(torch.sqrt(2.0 * float(np.float32(math.pi)) * variance.double())).float()
    quadratic = (0.5 * delta.double() * delta.double() / variance.double()).float()
    return scale + quadratic


def _fsl_initial_probabilities(values, means, variances, mask):
    probability = torch.exp(-_fsl_energy(values, means, variances).double()).float()
    total = (probability[0] + probability[1]) + probability[2]
    return torch.where((mask & (total > 0))[None], probability / total[None], 0)


def _fsl_bias_kernels(config, voxel_size, device):
    kernels = []
    for spacing in voxel_size:
        sigma = np.float32(0.51 * float(np.float32(config.bias_fwhm_mm)) / spacing)
        radius = 2 * int(sigma)
        values = [np.float32(math.exp(-j * j / (2.0 * float(sigma) * float(sigma))))
                  if sigma > 1e-6 else np.float32(j == 0)
                  for j in range(-radius, radius + 1)]
        total = np.float32(0)
        for value in values:
            total = np.float32(total + value)
        kernels.append(torch.tensor(np.asarray(values, dtype=np.float64) / float(total), device=device))
    return tuple(kernels)


def _fsl_bias(log_input, probabilities, means, variances, mask, kernels):
    from ._fsl_scan import blur

    precision = probabilities / variances[:, None, None, None]
    denominator = (precision[0] + precision[1]) + precision[2]
    residual = log_input[None] - means[:, None, None, None]
    numerator = precision * residual
    numerator = (numerator[0] + numerator[1]) + numerator[2]
    numerator = blur(numerator, kernels)
    denominator = blur(denominator, kernels)
    bias = torch.where(mask, numerator / denominator, 0)
    offset = bias[mask].mean(dtype=torch.float64).float()
    return torch.where(mask, bias - offset, 0)


def _fsl_mixel_probabilities(values, means, variances, mask):
    pure = torch.exp(-_fsl_energy(values, means, variances).double()).float()
    fractions = _fsl_fractions(100, device=values.device)
    evidence = [pure[0], pure[1], pure[2]]
    if not values.is_cuda:
        from ._fsl_cpu import mixel_evidence

        pair_means, pair_variances = _fsl_pair_parameters(fractions, means, variances)
        scales = torch.log(torch.sqrt(
            2.0 * float(np.float32(math.pi)) * pair_variances.double())).float()
        mixed = mixel_evidence(values.contiguous().numpy().reshape(-1),
                               mask.contiguous().numpy().reshape(-1),
                               pair_means.numpy(), pair_variances.numpy(), scales.numpy())
        evidence.extend(torch.from_numpy(mixed.reshape(3, *values.shape)))
        return torch.stack(evidence) * mask[None]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        probability = torch.zeros_like(values)
        for fraction in fractions:
            mean = fraction * means[a] + (1 - fraction) * means[b]
            variance = fraction * fraction * variances[a] + (1 - fraction) * (1 - fraction) * variances[b]
            energy = _fsl_energy(values, mean[None], variance[None])[0]
            probability = (probability.double() + torch.exp(-energy.double()) * 0.01).float()
        evidence.append(probability)
    return torch.stack(evidence) * mask[None]


def _fsl_pair_parameters(fractions, means, variances):
    pair_means, pair_variances = [], []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pair_means.append(fractions * means[a] + (1 - fractions) * means[b])
        pair_variances.append(fractions * fractions * variances[a]
                              + (1 - fractions) * (1 - fractions) * variances[b])
    return torch.stack(pair_means), torch.stack(pair_variances)


def _fsl_partial_volumes(values, means, variances, mask, mixel, fractions, chunk_size):
    if not values.is_cuda:
        from ._fsl_cpu import partial_volumes

        pair_means, pair_variances = _fsl_pair_parameters(fractions, means, variances)
        output = partial_volumes(values.contiguous().numpy().reshape(-1),
                                 mask.contiguous().numpy().reshape(-1),
                                 mixel.contiguous().numpy().reshape(-1), fractions.numpy(),
                                 pair_means.numpy(), pair_variances.numpy(),
                                 torch.log(pair_variances).numpy())
        return torch.from_numpy(output.reshape(3, *values.shape))
    result = torch.zeros((3, *values.shape), dtype=torch.float32, device=values.device)
    for tissue in range(3):
        result[tissue][mask & (mixel == tissue)] = 1
    for code, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), start=3):
        selected = mask & (mixel == code)
        selected_values = values[selected]
        if not selected_values.numel():
            continue
        minimum = torch.full_like(selected_values, 1e13)
        best_fraction = torch.zeros_like(selected_values)
        assigned = torch.zeros_like(selected_values, dtype=torch.bool)
        for start in range(0, fractions.numel(), chunk_size):
            fraction = fractions[start:start + chunk_size, None]
            mean = fraction * means[a] + (1 - fraction) * means[b]
            variance = fraction * fraction * variances[a] + (1 - fraction) * (1 - fraction) * variances[b]
            delta = selected_values[None] - mean
            energy = (delta * delta / variance + torch.log(variance)) / 2
            chunk_energy, chunk_index = energy.min(dim=0)
            improved = chunk_energy < minimum
            minimum[improved] = chunk_energy[improved]
            best_fraction[improved] = fractions[start + chunk_index[improved]]
            assigned |= improved
        result[a][selected] = torch.where(assigned, best_fraction, 0)
        result[b][selected] = torch.where(assigned, 1 - best_fraction, 0)
    return result


def _segment_t1_fsl(original, mask, voxel_size, config):
    """Ordered single-channel three-tissue FAST defaults, without FSL calls."""
    from ._fsl_scan import GlibcRandom, icm, random_posteriors, schedule, tanaka

    log_input = torch.where(mask, torch.log(original.double() + 1).float(), 0)
    samples = torch.sort(log_input[mask]).values
    means = torch.stack([samples[min(int(math.floor(samples.numel() * fraction)), samples.numel() - 1)]
                         for fraction in (0.25, 0.5, 0.75)])
    if not torch.all(means[1:] > means[:-1]):
        raise ValueError("input does not contain three separable intensity ranges")
    nearest = (log_input[None] - means[:, None, None, None]).square().argmin(dim=0)
    probabilities = F.one_hot(nearest, num_classes=3).movedim(-1, 0).float() * mask[None]
    for _ in range(config.init_iterations + config.fixed_iterations):
        means, variances = _fsl_moments(log_input, probabilities, mask)
        probabilities = _fsl_initial_probabilities(log_input, means, variances, mask)
    if config.init_iterations + config.fixed_iterations == 0:
        means, variances = _fsl_moments(log_input, probabilities, mask)
    scan = schedule(mask, voxel_size)
    random = GlibcRandom(-1, compiled=not original.is_cuda)
    kernels = _fsl_bias_kernels(config, voxel_size, original.device)
    bias_enabled = config.bias_fwhm_mm > 0
    bias_log = _fsl_bias(log_input, probabilities, means, variances, mask, kernels) if bias_enabled else torch.zeros_like(original)
    corrected_log = log_input - bias_log
    for _ in range(config.bias_iterations):
        probabilities = tanaka(random_posteriors(scan, random),
                               _fsl_energy(corrected_log, means, variances), scan,
                               config.init_mrf, config.mean_field_iterations)
        if bias_enabled:
            bias_log = _fsl_bias(log_input, probabilities, means, variances, mask, kernels)
            corrected_log = log_input - bias_log
        means, variances = _fsl_moments(corrected_log, probabilities, mask)
    beta = config.init_mrf
    for _ in range(config.fixed_iterations):
        probabilities = tanaka(random_posteriors(scan, random),
                               _fsl_energy(corrected_log, means, variances), scan,
                               beta, config.mean_field_iterations)
        beta = config.mrf
        means, variances = _fsl_moments(corrected_log, probabilities, mask)
    order = torch.argsort(means, stable=True)
    probabilities, means, variances = probabilities[order], means[order], variances[order]
    hard = torch.where(mask, probabilities.argmax(dim=0) + 1, 0)
    corrected_linear = torch.where(mask, torch.exp(corrected_log), 0)
    linear_means, linear_variances = _fsl_moments(corrected_linear, probabilities, mask)
    mixel = icm(_fsl_mixel_probabilities(corrected_linear, linear_means, linear_variances, mask),
                mask, scan, config.mixel_mrf)
    fractions = _fsl_fractions(config.pve_steps, device=original.device)
    pve = _fsl_partial_volumes(corrected_linear, linear_means, linear_variances, mask,
                              mixel, fractions, config.pve_chunk_size)
    correction = torch.exp(-bias_log)
    bias_field = 1 / correction
    restored = torch.where(mask, original * correction, 0)
    if not all(torch.isfinite(value).all() for value in (pve, bias_field, restored)):
        raise RuntimeError("segmentation produced non-finite values")
    return FASTTensorResult(pve, hard, torch.where(mask, pve.argmax(dim=0) + 1, 0),
                            mixel, bias_field, restored, linear_means, linear_variances)


@torch.no_grad()
def segment_t1(image, mask=None, voxel_size=(1.0, 1.0, 1.0), config=None):
    """Segment one brain-extracted T1 tensor into CSF, GM and WM.

    ``image`` and the returned maps use ``(X, Y, Z)`` order. An explicit mask
    is intersected with positive input voxels; without one, positive voxels
    define the brain region, matching the input convention of FAST.
    """
    config = FASTConfig() if config is None else config
    if not isinstance(config, FASTConfig):
        raise TypeError("config must be FASTConfig")
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise ValueError("image must be a 3D torch.Tensor")
    image = image.detach().to(dtype=torch.float32)
    if not torch.isfinite(image).all():
        raise ValueError("image contains NaN or infinity")
    if len(voxel_size) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0
                                   for v in voxel_size):
        raise ValueError("voxel_size must contain three positive finite values")
    voxel_size = tuple(float(v) for v in voxel_size)
    if image.min() < 0:
        if _order_statistic(image, 0.02) < 0:
            image = image - image.min()
        else:
            image = image.clamp_min(0)
    positive = image > 0
    if mask is None:
        mask = positive
    else:
        if not isinstance(mask, torch.Tensor) or mask.shape != image.shape:
            raise ValueError("mask must be a tensor with the same shape as image")
        mask = mask.to(device=image.device, dtype=torch.bool) & positive
    if int(mask.sum()) < 3:
        raise ValueError("the positive brain mask contains fewer than three voxels")

    original = torch.where(mask, image, torch.zeros_like(image))
    if config.execution == "fsl":
        if not original.is_cuda:
            from ._fsl_cpu import thread_budget

            with thread_budget(torch.get_num_threads()):
                return _segment_t1_fsl(original, mask, voxel_size, config)
        return _segment_t1_fsl(original, mask, voxel_size, config)
    log_input = torch.where(mask, torch.log1p(original), torch.zeros_like(original))
    samples = log_input[mask]
    ordered_samples = torch.sort(samples).values
    means = torch.stack([
        ordered_samples[min(int(math.floor(ordered_samples.numel() * fraction)),
                            ordered_samples.numel() - 1)]
        for fraction in (0.25, 0.5, 0.75)
    ])
    if not torch.all(means[1:] > means[:-1]):
        raise ValueError("input does not contain three separable intensity ranges")
    global_variance = samples.var(correction=0)
    variance_floor = torch.clamp(
        global_variance * config.variance_floor_fraction,
        min=torch.finfo(image.dtype).eps,
    )
    nearest = (log_input.unsqueeze(0) - means[:, None, None, None]).abs().argmin(dim=0)
    probabilities = F.one_hot(nearest, num_classes=3).movedim(-1, 0).to(image.dtype)
    probabilities *= mask.unsqueeze(0)
    variances = global_variance.clamp_min(variance_floor).expand_as(means).clone()
    for _ in range(config.init_iterations + config.fixed_iterations):
        means, variances = _moments(log_input, probabilities, mask, variance_floor)
        probabilities = torch.softmax(_log_likelihood(log_input, means, variances), dim=0)
        probabilities *= mask.unsqueeze(0)

    spatial_kernel = _neighbor_kernel(voxel_size, device=image.device, dtype=image.dtype)
    blur_kernels = tuple(_gaussian_kernel(
        0.51 * config.bias_fwhm_mm / spacing,
        device=image.device, dtype=image.dtype,
    ) for spacing in voxel_size)
    bias_log = torch.zeros_like(image)
    corrected_log = log_input
    bias_enabled = config.bias_fwhm_mm > 0
    if bias_enabled:
        bias_log = _estimate_bias(log_input, probabilities, means, variances,
                                  mask, blur_kernels)
        corrected_log = log_input - bias_log
    for _ in range(config.bias_iterations):
        probabilities = _hmrf_probabilities(
            corrected_log, means, variances, mask, spatial_kernel,
            config.init_mrf, config.mean_field_iterations,
        )
        if bias_enabled:
            bias_log = _estimate_bias(log_input, probabilities, means, variances,
                                      mask, blur_kernels)
            corrected_log = log_input - bias_log
        means, variances = _moments(corrected_log, probabilities, mask,
                                    variance_floor)

    beta = config.init_mrf
    for _ in range(config.fixed_iterations):
        probabilities = _hmrf_probabilities(
            corrected_log, means, variances, mask, spatial_kernel, beta,
            config.mean_field_iterations,
        )
        beta = config.mrf
        means, variances = _moments(corrected_log, probabilities, mask, variance_floor)

    order = torch.argsort(means)
    probabilities, means, variances = probabilities[order], means[order], variances[order]
    hard = torch.where(mask, probabilities.argmax(dim=0) + 1,
                       torch.zeros_like(mask, dtype=torch.long))

    corrected_linear = torch.exp(corrected_log)
    linear_means, linear_variances = _moments(
        corrected_linear, probabilities, mask,
        torch.clamp(corrected_linear[mask].var(correction=0)
                    * config.variance_floor_fraction,
                    min=torch.finfo(image.dtype).eps),
    )
    mixel, fractions, pairs = _mixel_labels(
        corrected_linear, linear_means, linear_variances, mask,
        spatial_kernel, config,
    )
    pve = _partial_volumes(
        corrected_linear, linear_means, linear_variances, mask, mixel,
        fractions, pairs, config.pve_chunk_size,
    )
    pve_segmentation = torch.where(mask, pve.argmax(dim=0) + 1,
                                   torch.zeros_like(mask, dtype=torch.long))
    bias_field = torch.where(mask, torch.exp(bias_log), torch.ones_like(image))
    restored = torch.where(mask, original / bias_field, torch.zeros_like(image))
    outputs = (pve, bias_field, restored, linear_means, linear_variances)
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError("segmentation produced non-finite values")
    return FASTTensorResult(
        pve=pve,
        hard_segmentation=hard,
        pve_segmentation=pve_segmentation,
        mixel_type=mixel,
        bias_field=bias_field,
        restored=restored,
        tissue_means=linear_means,
        tissue_variances=linear_variances,
    )


__all__ = ["FASTConfig", "FASTTensorResult", "segment_t1"]
