"""GPU FOD tractography on independent diffusion and 5TT anatomy grids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from functools import partial

import torch
from torch.nn import functional as F

from .fod import real_sh, tracking_sh_precomputed


@dataclass
class Tractogram:
    """Accepted world-mm tracks and per-track quantities.

    ``paths`` contains N tensors ``[Pi,3]``; ``endpoints`` is ``[N,2,3]``;
    ``lengths_mm`` and optional ``mean_fa`` are ``[N]``; ``accepted_seeds`` is
    ``[N,3]``. Float tensors stay on the caller's CPU/CUDA device. The integer
    ``seeds_attempted`` is the requested number, before acceptance filtering.
    """

    paths: tuple[torch.Tensor, ...]
    endpoints: torch.Tensor
    lengths_mm: torch.Tensor
    mean_fa: torch.Tensor | None
    seeds_attempted: int
    accepted_seeds: torch.Tensor


@dataclass
class _VolumeSampler:
    """Call-scoped grid/affine views; input tensors must not change layout."""

    volume: torch.Tensor
    inverse_affine: torch.Tensor

    def __post_init__(self):
        self.linear = self.inverse_affine[:3, :3].double().T
        self.translation = self.inverse_affine[:3, 3].double()
        self.scale = self.inverse_affine.new_tensor(
            [max(size - 1, 1) for size in self.volume.shape[:3]], dtype=torch.float64)
        self.image = self.volume.permute(3, 2, 1, 0)[None]

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        voxel = points.double() @ self.linear + self.translation
        grid = (2 * voxel / self.scale - 1).float().reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.image, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=True).reshape(self.volume.shape[-1], -1).T


def _sample(volume: torch.Tensor, points: torch.Tensor, inverse_affine: torch.Tensor) -> torch.Tensor:
    """Sample float32 ``[X,Y,Z,C]`` at world-mm ``[N,3]``; return ``[N,C]``.

    ``inverse_affine`` is float64 RAS-mm to voxel-center ``[4,4]``. The
    local float64 coordinate calculation preserves the 0.001 mm 5TT normal
    finite difference under the globally enabled CUDA TF32 setting.
    """
    shape = volume.shape[:3]
    voxel = points.double() @ inverse_affine[:3, :3].double().T + inverse_affine[:3, 3].double()
    scale = voxel.new_tensor([max(size - 1, 1) for size in shape])
    grid = (2 * voxel / scale - 1).float().reshape(1, 1, 1, -1, 3)
    image = volume.permute(3, 2, 1, 0)[None]
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=True).reshape(volume.shape[-1], -1).T


def _initial_directions(
    coefficients: torch.Tensor, generator: torch.Generator, *, lmax: int, cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw MRtrix iFOD2 initial directions for real FOD coefficients ``[B,C]``.

    Directions are sampled uniformly from the unit ball, then normalised.
    The first direction with FOD amplitude strictly above ``cutoff`` is
    retained, up to 1000 attempts per seed. Returns float32 directions
    ``[B,3]``, a bool success mask ``[B]``, and int32 attempt counts ``[B]``.
    Failed seeds receive a harmless x-axis direction and must be filtered by
    the caller. MRtrix equivalents are ``MethodBase::random_direction()`` and
    ``iFOD2::init()`` in commit eeab681d3e0c.
    """
    width = (lmax + 1) * (lmax + 2) // 2
    if coefficients.ndim != 2 or coefficients.shape[1] != width or coefficients.dtype != torch.float32:
        raise ValueError("coefficients must be float32 [B,C] for the chosen lmax")
    batch = coefficients.shape[0]
    initial = coefficients.new_zeros((batch, 3))
    initial[:, 0] = 1
    valid = torch.zeros(batch, dtype=torch.bool, device=coefficients.device)
    attempts = torch.zeros(batch, dtype=torch.int32, device=coefficients.device)
    for offset in range(0, 1000, 16):
        pending = (~valid).nonzero(as_tuple=False).flatten()
        if pending.numel() == 0:
            break
        count = min(16, 1000 - offset)
        candidate = 2 * torch.rand((len(pending), count, 3), device=coefficients.device,
                                   generator=generator) - 1
        outside = candidate.square().sum(-1) > 1
        while bool(outside.any()):
            candidate[outside] = 2 * torch.rand((int(outside.sum()), 3),
                                                device=coefficients.device,
                                                generator=generator) - 1
            outside = candidate.square().sum(-1) > 1
        candidate = F.normalize(candidate, dim=-1)
        basis = tracking_sh_precomputed(candidate.reshape(-1, 3), lmax).reshape(
            len(pending), count, width,
        )
        amplitude = (coefficients[pending, None] * basis).sum(-1)
        accepted = torch.isfinite(amplitude) & (amplitude > cutoff)
        first = accepted.int().argmax(-1)
        success = accepted.any(-1)
        attempts[pending] += torch.where(success, first + 1, count).int()
        selected = pending[success]
        initial[selected] = candidate[success, first[success]]
        valid[selected] = True
    return initial, valid, attempts


def _five_tissue_values(five_tissue: torch.Tensor, points: torch.Tensor,
                        inverse_affine: torch.Tensor) -> torch.Tensor:
    """Sample 5TT ``[X,Y,Z,5]`` at world-mm ``[N,3]``; return ``[N,5]``."""
    return _five_tissue_mrtrix(five_tissue, points, inverse_affine)


def _gmwmi_gradient(five_tissue: torch.Tensor, points: torch.Tensor,
                    inverse_affine: torch.Tensor, *, wm_minus_gm: bool = False) -> torch.Tensor:
    """Return ``[N,3]`` GM-WM world gradient for ``[N,3]`` RAS-mm points.

    Uses MRtrix 3.0.3's ±0.0005 mm central difference on float32 5TT with
    a float64 coordinate map. ``wm_minus_gm`` flips the gradient sign.
    """
    offsets = torch.eye(3, dtype=points.dtype, device=points.device) * 0.0005
    plus_points = (points[:, None] + offsets[None]).reshape(-1, 3)
    minus_points = (points[:, None] - offsets[None]).reshape(-1, 3)
    plus = _five_tissue_values(five_tissue, plus_points, inverse_affine).reshape(-1, 3, 5)
    minus = _five_tissue_values(five_tissue, minus_points, inverse_affine).reshape(-1, 3, 5)
    difference = ((plus[..., 0] + plus[..., 1] - plus[..., 2]) -
                  (minus[..., 0] + minus[..., 1] - minus[..., 2])) / 0.001
    return -difference if wm_minus_gm else difference


def _find_gmwmi(five_tissue: torch.Tensor, points: torch.Tensor,
                inverse_affine: torch.Tensor, min_voxel_mm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``[N,3]`` RAS-mm points to the 5TT GM=WM boundary.

    Returns projected float32 ``[N,3]`` points and a bool ``[N]`` acceptance
    mask. MRtrix stops within a 0.01 tissue fraction difference on the GM side.
    """
    positions = points.clone()
    for iteration in range(10):
        values = _five_tissue_values(five_tissue, positions, inverse_affine)
        difference = values[:, 0] + values[:, 1] - values[:, 2]
        gradient = _gmwmi_gradient(five_tissue, positions, inverse_affine)
        norm_squared = gradient.square().sum(-1)
        step = -gradient * (difference / norm_squared.clamp_min(1e-12))[:, None]
        step *= (0.5 * min_voxel_mm / step.norm(dim=-1).clamp_min(1e-12)).clamp(max=1)[:, None]
        update = (norm_squared > 1e-12) & ((difference.abs() > .01) | (iteration == 0))
        positions = torch.where(update[:, None], positions + step, positions)
    values = _five_tissue_values(five_tissue, positions, inverse_affine)
    difference = values[:, 0] + values[:, 1] - values[:, 2]
    # MRtrix moves points that land on the WM side slightly into GM.
    gradient = _gmwmi_gradient(five_tissue, positions, inverse_affine)
    norm_squared = gradient.square().sum(-1)
    step = -gradient * (difference / norm_squared.clamp_min(1e-12))[:, None]
    step *= (0.5 * min_voxel_mm / step.norm(dim=-1).clamp_min(1e-12)).clamp(max=1)[:, None]
    need_gm = difference < 0
    step *= 1.5
    for _ in range(12):
        trial = positions + step
        trial_values = _five_tissue_values(five_tissue, trial, inverse_affine)
        trial_diff = trial_values[:, 0] + trial_values[:, 1] - trial_values[:, 2]
        accept = need_gm & (trial_diff >= 0) & (trial_diff < .01)
        positions = torch.where(accept[:, None], trial, positions)
        need_gm &= ~accept & (step.norm(dim=-1) < .5 * min_voxel_mm)
        step *= 1.5
    values = _five_tissue_values(five_tissue, positions, inverse_affine)
    difference = values[:, 0] + values[:, 1] - values[:, 2]
    valid = (values.sum(-1) >= .5) & (values[:, 2] > 0) & \
        (difference >= 0) & (difference <= .01) & \
        (values[:, 3] < values[:, :3].amax(-1)) & \
        (values[:, 4] < values[:, :3].amax(-1))
    return positions, valid


@torch.inference_mode()
def sample_gmwmi_seeds(gmwmi: torch.Tensor, five_tissue: torch.Tensor,
                       affine: torch.Tensor, n_seeds: int,
                       generator: torch.Generator) -> torch.Tensor:
    """Draw weighted GMWMI candidates and project them to the 5TT boundary.

    ``gmwmi`` is the float32 output of ``5tt2gmwmi`` with shape ``[X,Y,Z]``.
    ``five_tissue`` is the matching ``5ttgen`` float32 image ``[X,Y,Z,5]``
    (cortical GM, subcortical GM, WM, CSF, pathology), and ``affine`` maps
    voxel centers to RAS millimetres. All tensors reside on one CPU/CUDA
    device, with float64 allowed for ``affine``. ``generator`` controls
    PyTorch sampling; it cannot reproduce the MRtrix RNG sequence. Output is
    float32 ``[n_seeds,3]`` in RAS millimetres on the input device.

    MRtrix equivalent: ``tckgen -seed_gmwmi gmwmi.mif -act 5tt.mif ...``.
    The real-data comparison is
    ``tools/benchmark_connectome_tracking_act_seeds.py``.
    """
    if (gmwmi.ndim != 3 or five_tissue.shape != (*gmwmi.shape, 5) or
            affine.shape != (4, 4) or n_seeds < 1):
        raise ValueError('expected GMWMI [X,Y,Z], matching 5TT [X,Y,Z,5], affine [4,4]')
    if bool((gmwmi < 0).any()) or not bool((gmwmi > 0).any()):
        raise ValueError('GMWMI weights must be nonnegative and nonempty')
    device = gmwmi.device
    affine = affine.to(device=device, dtype=torch.float64)
    inverse = torch.linalg.inv(affine.double())
    min_voxel_mm = float(torch.linalg.vector_norm(affine[:3, :3], dim=0).min())
    nonzero = torch.nonzero(gmwmi.reshape(-1) > 0, as_tuple=False).flatten()
    weights = gmwmi.reshape(-1)[nonzero].float()
    cumulative = weights.double().cumsum(0)
    selected = []
    remaining = n_seeds
    for _ in range(20):
        count = max(remaining, min(n_seeds, 1024))
        draws = torch.rand(count, device=device, dtype=torch.float64, generator=generator) * cumulative[-1]
        flat = nonzero[torch.searchsorted(cumulative, draws, right=True).clamp_max(len(nonzero) - 1)]
        yz = gmwmi.shape[1] * gmwmi.shape[2]
        voxel = torch.stack((flat // yz, flat // gmwmi.shape[2] % gmwmi.shape[1],
                             flat % gmwmi.shape[2]), dim=-1).float()
        voxel += torch.rand((count, 3), device=device, generator=generator) - .5
        points = (voxel.double() @ affine[:3, :3].double().T + affine[:3, 3].double()).float()
        points, valid = _find_gmwmi(five_tissue, points, inverse, min_voxel_mm)
        normals = _gmwmi_gradient(five_tissue, points, inverse, wm_minus_gm=True)
        normals = F.normalize(normals, dim=-1)
        z_axis = torch.tensor([0., 0., 1.], device=device)
        y_axis = torch.tensor([0., 1., 0.], device=device)
        first = torch.linalg.cross(normals, z_axis.expand_as(normals))
        fallback = first.norm(dim=-1) < 1e-6
        first[fallback] = torch.linalg.cross(normals[fallback], y_axis.expand_as(normals[fallback]))
        first = F.normalize(first, dim=-1)
        second = F.normalize(torch.linalg.cross(normals, first), dim=-1)
        jitter = (torch.rand((count, 2), device=device, generator=generator) - .5) * (4 * min_voxel_mm)
        points += jitter[:, :1] * first + jitter[:, 1:] * second
        points, perturbed_valid = _find_gmwmi(five_tissue, points, inverse, min_voxel_mm)
        valid &= perturbed_valid & torch.isfinite(points).all(-1)
        accepted = points[valid][:remaining]
        selected.append(accepted)
        remaining -= len(accepted)
        if remaining == 0:
            return torch.cat(selected, dim=0)
    raise RuntimeError('GMWMI/5TT interface projection rejected too many candidates')



def _ifod2_calibration(
    lmax: int, step_mm: float, voxel_mm: float, max_angle_degrees: float,
    power: float, device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Return MRtrix iFOD2 calibration directions ``[K,3]`` and rejection ratio.

    ``step_mm`` and ``voxel_mm`` are positive RAS-mm lengths; the remaining
    inputs match ``tckgen -algorithm iFOD2 -angle ... -power ...``. The
    calibration uses an SH delta FOD and the synthetic 1-voxel gradient from
    ``calibrator.h``; it does not sample the subject's FOD.
    """
    if lmax < 0 or lmax % 2 or step_mm <= 0 or voxel_mm <= 0 or not 0 < max_angle_degrees < 180:
        raise ValueError('invalid iFOD2 calibration geometry')
    angle_limit = math.radians(max_angle_degrees)
    elevations = []
    elevation = 0.0
    while elevation < angle_limit:
        elevations.append(elevation)
        elevation = float(torch.tensor(elevation + .001, dtype=torch.float32))
    theta = torch.tensor(elevations, dtype=torch.float32)
    start = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
    delta = real_sh(start, lmax)[0]
    start_amplitude = (delta * delta).sum()
    middle_direction = torch.stack((torch.sin(theta / 2), torch.zeros_like(theta),
                                    torch.cos(theta / 2)), -1)
    end_direction = torch.stack((torch.sin(theta), torch.zeros_like(theta),
                                 torch.cos(theta)), -1)
    mid_x = torch.where(theta > 0, step_mm / theta.clamp_min(1e-20) *
                        (1 - torch.cos(theta / 2)), 0.)
    end_x = torch.where(theta > 0, step_mm / theta.clamp_min(1e-20) *
                        (1 - torch.cos(theta)), 0.)
    middle = (real_sh(middle_direction, lmax) * delta).sum(-1) * (1 - mid_x / voxel_mm)
    end = (real_sh(end_direction, lmax) * delta).sum(-1) * (1 - end_x / voxel_mm)
    amplitude = torch.where(
        (middle > 0) & (end > 0),
        torch.exp(power * (.5 * start_amplitude.log() +
                           middle.clamp_min(1e-20).log() + .5 * end.clamp_min(1e-20).log())),
        0.,
    )
    stop = next((i for i, value in enumerate(amplitude) if value <= 0), len(theta) - 1)

    def grid(maximum: float, spacing: float) -> list[tuple[float, float, float]]:
        extent = math.ceil(maximum / spacing)
        radius_sq = (maximum / spacing) ** 2
        points = []
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                x, y = i + .5 * j, math.sqrt(3) / 2 * j
                radial_sq = x * x + y * y
                if radial_sq > radius_sq:
                    continue
                arc = spacing * math.sqrt(radial_sq)
                scale = spacing * math.sin(arc) / arc if arc else spacing
                points.append((scale * x, scale * y, math.cos(arc)))
        return points

    best = (math.inf, None, None)
    for i in range(1, stop + 1):
        if amplitude[i] <= 0:
            continue
        ratio = float(amplitude[0] / amplitude[i])
        directions = grid(angle_limit + elevations[i], math.sqrt(3) * elevations[i])
        expected = angle_limit ** 2 * (1 + ratio) / (2 * elevations[stop] ** 2) + len(directions)
        if 0 < expected < best[0]:
            best = (expected, directions, ratio)
    if best[1] is None:
        raise RuntimeError('iFOD2 calibration found no positive path probabilities')
    return torch.tensor(best[1], device=device, dtype=torch.float32), best[2]


def _rotate_ifod2_directions(prior: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Rotate local cone directions ``[B,K,3]`` around ``prior [B,3]``.

    Returns world directions ``[B,K,3]`` by the MRtrix MethodBase rotation.
    """
    radius = prior[:, :2].norm(dim=-1)
    safe = radius.clamp_min(1e-20)
    basis = torch.stack((prior[:, 0] / safe, prior[:, 1] / safe,
                         torch.zeros_like(radius)), -1)
    tilted = torch.stack((prior[:, 2] * basis[:, 0],
                          prior[:, 2] * basis[:, 1], -radius), -1)
    alpha = local[..., 2]
    beta = (local[..., :2] * basis[:, None, :2]).sum(-1)
    rotated = local + alpha[..., None] * (prior[:, None] - prior.new_tensor([0., 0., 1.])) + \
        beta[..., None] * (tilted - basis)[:, None]
    straight = torch.where(prior[:, 2:3, None] < 0, -local, local)
    return torch.where((radius == 0)[:, None, None], straight, rotated)


def _ifod2_arc_probability(
    position: torch.Tensor, prior: torch.Tensor, directions: torch.Tensor,
    half_log_start: torch.Tensor, fod: torch.Tensor, five_tissue: torch.Tensor,
    fod_inverse: torch.Tensor, five_inverse: torch.Tensor, *,
    lmax: int, step_mm: float, cutoff: float, power: float,
    _fod_sampler=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate candidate arcs ``[B,K,3]`` against FOD and endpoint CSF.

    Returns probability, midpoint, endpoint, midpoint amplitude and endpoint
    amplitude, all ``[B,K]`` except positions ``[B,K,3]``. ``half_log_start``
    contains the state carried from the preceding accepted arc. Equivalent
    MRtrix operations: ``iFOD2::get_path`` and ``iFOD2::path_prob``.
    """
    batch, count = directions.shape[:2]
    sample_fod = (partial(_sample, fod, inverse_affine=fod_inverse)
                  if _fod_sampler is None else _fod_sampler)
    cosine = (prior[:, None] * directions).sum(-1).clamp(-1, 1)
    angle = cosine.acos()
    curvature = F.normalize(directions - cosine[..., None] * prior[:, None], dim=-1)
    radius = step_mm / angle.clamp_min(1e-6)
    half = .5 * angle
    midpoint = position[:, None] + radius[..., None] * (
        half.sin()[..., None] * prior[:, None] +
        (1 - half.cos())[..., None] * curvature)
    endpoint = position[:, None] + radius[..., None] * (
        angle.sin()[..., None] * prior[:, None] +
        (1 - cosine)[..., None] * curvature)
    mid_tangent = half.cos()[..., None] * prior[:, None] + \
        half.sin()[..., None] * curvature
    straight = angle < 1e-4
    midpoint = torch.where(straight[..., None],
                           position[:, None] + .5 * step_mm * prior[:, None], midpoint)
    endpoint = torch.where(straight[..., None],
                           position[:, None] + step_mm * prior[:, None], endpoint)
    mid_tangent = torch.where(straight[..., None], prior[:, None], mid_tangent)
    mid_amplitude = (sample_fod(midpoint.reshape(-1, 3)).reshape(
        batch, count, -1) * tracking_sh_precomputed(mid_tangent.reshape(-1, 3), lmax).reshape(
            batch, count, -1)).sum(-1)
    end_amplitude = (sample_fod(endpoint.reshape(-1, 3)).reshape(
        batch, count, -1) * tracking_sh_precomputed(directions.reshape(-1, 3), lmax).reshape(
            batch, count, -1)).sum(-1)
    csf = _five_tissue_mrtrix(five_tissue, endpoint.reshape(-1, 3), five_inverse)[
        :, 3].reshape(batch, count)
    valid = ((mid_amplitude >= cutoff) & (end_amplitude >= cutoff) & (csf < .5) &
             torch.isfinite(mid_amplitude) & torch.isfinite(end_amplitude))
    log_prob = half_log_start[:, None] + mid_amplitude.clamp_min(1e-20).log() + \
        .5 * end_amplitude.clamp_min(1e-20).log()
    probability = torch.where(valid, torch.exp(power * log_prob), 0.)
    return probability, midpoint, endpoint, mid_amplitude, end_amplitude


def _grow(
    seeds: torch.Tensor,
    tangents: torch.Tensor,
    fod: torch.Tensor,
    five_tissue: torch.Tensor,
    fod_inverse: torch.Tensor,
    five_inverse: torch.Tensor,
    generator: torch.Generator,
    *,
    lmax: int,
    proposals_per_step: int,
    step_mm: float,
    max_steps: int,
    max_angle_degrees: float,
    cutoff: float,
    power: float,
    calibration_local: torch.Tensor | None = None,
    calibration_ratio: float | None = None,
    seed_to_wm_initial: torch.Tensor | None = None,
    arc_probability_fn=None,
    _fod_sampler=None,
):
    """Propagate iFOD2 arcs with calibrated batched rejection sampling.

    Input seeds/tangents are float32 ``[B,3]``, FOD is ``[XF,YF,ZF,C]``, 5TT
    is ``[XA,YA,ZA,5]``, and both inverse affines are float64 ``[4,4]``.
    Output is padded paths ``[B,max_steps+1,3]``, counts ``[B]``, ACT exit
    flags ``[B]``, lengths ``[B]`` in millimetres, and seed-to-WM flags ``[B]``.

    ``-samples 3`` gives a midpoint and endpoint on each arc. For each active
    track, calibrate the maximum path probability, then accept the first cone
    proposal whose uniform draw is below path probability / calibrated max.
    ``proposals_per_step`` controls only the GPU proposal block size; up to
    1000 proposals per arc are attempted, as in MRtrix iFOD2.
    """
    if calibration_local is None or calibration_ratio is None:
        calibration_local, calibration_ratio = (
            _ifod2_calibration(lmax, step_mm, 2 * step_mm, max_angle_degrees,
                               power, seeds.device) if max_angle_degrees > 0 else
            (seeds.new_tensor([[0., 0., 1.]]), 1.)
        )
    if arc_probability_fn is None:
        arc_probability_fn = _ifod2_arc_probability
    sample_fod = (partial(_sample, fod, inverse_affine=fod_inverse)
                  if _fod_sampler is None else _fod_sampler)
    batch = seeds.shape[0]
    paths = seeds.new_zeros((batch, max_steps + 1, 3))
    paths[:, 0] = seeds
    counts = torch.ones(batch, dtype=torch.long, device=seeds.device)
    start_amp = (sample_fod(seeds) * tracking_sh_precomputed(tangents, lmax)).sum(-1)
    active = torch.isfinite(start_amp) & (start_amp > cutoff)
    half_log_start = .5 * start_amp.clamp_min(1e-20).log()
    ended_in_gm = torch.zeros_like(active)
    seed_tissue = _five_tissue_mrtrix(five_tissue, seeds, five_inverse)
    seed_in_sgm = ((seed_tissue[:, 1] > seed_tissue[:, 0]) &
                   (seed_tissue[:, 1] >= seed_tissue[:, 2]) &
                   (seed_tissue[:, 1] > seed_tissue[:, 3]) &
                   (seed_tissue[:, 1] > seed_tissue[:, 4]))
    seed_to_wm = (torch.zeros_like(active) if seed_to_wm_initial is None else
                  seed_to_wm_initial.clone())
    sgm_depth = torch.zeros(batch, dtype=torch.int32, device=seeds.device)
    best_sgm_metric = seeds.new_full((batch,), float('inf'))
    best_sgm_position = seeds.clone()
    best_sgm_count = counts.clone()
    best_sgm_length = seeds.new_zeros(batch)
    lengths = seeds.new_zeros(batch)
    positions = seeds.clone()
    prior = tangents.clone()
    cosine_limit = math.cos(math.radians(max_angle_degrees))
    rows = torch.arange(batch, device=seeds.device)
    for step in range(max_steps):
        if not bool(active.any()):
            break
        calibrated = _rotate_ifod2_directions(
            prior, calibration_local[None].expand(batch, -1, -1))
        calibration_probability, _, _, _, _ = arc_probability_fn(
            positions, prior, calibrated, half_log_start, fod, five_tissue,
            fod_inverse, five_inverse, lmax=lmax, step_mm=step_mm,
            cutoff=cutoff, power=power,
        )
        maximum = calibration_probability.amax(-1) * calibration_ratio
        pending = active & (maximum > 0)
        direction = prior.clone()
        chosen_mid = positions.clone()
        chosen_end = positions.clone()
        mid_metric = seeds.new_zeros(batch)
        end_metric = seeds.new_zeros(batch)
        moving = torch.zeros_like(active)
        for first_trial in range(0, 1000, proposals_per_step):
            indices = pending.nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                break
            width = min(proposals_per_step, 1000 - first_trial)
            cosine = 1 - torch.rand((len(indices), width), device=seeds.device,
                                    generator=generator) * (1 - cosine_limit)
            azimuth = 2 * math.pi * torch.rand((len(indices), width),
                                               device=seeds.device, generator=generator)
            local = torch.stack(((1 - cosine.square()).sqrt() * azimuth.cos(),
                                 (1 - cosine.square()).sqrt() * azimuth.sin(), cosine), -1)
            proposals = _rotate_ifod2_directions(prior[indices], local)
            probability, midpoint, endpoint, mid_amp, end_amp = arc_probability_fn(
                positions[indices], prior[indices], proposals, half_log_start[indices],
                fod, five_tissue, fod_inverse, five_inverse, lmax=lmax,
                step_mm=step_mm, cutoff=cutoff, power=power,
            )
            draws = torch.rand((len(indices), width), device=seeds.device,
                               generator=generator)
            accepted = draws < probability / maximum[indices, None]
            chosen = accepted.int().argmax(-1)
            success = accepted.any(-1)
            selected = indices[success]
            choice = chosen[success]
            direction[selected] = proposals[success, choice]
            chosen_mid[selected] = midpoint[success, choice]
            chosen_end[selected] = endpoint[success, choice]
            mid_metric[selected] = mid_amp[success, choice]
            end_metric[selected] = end_amp[success, choice]
            moving[selected] = True
            pending[selected] = False
        half_log_start = torch.where(moving, .5 * end_metric.clamp_min(1e-20).log(),
                                     half_log_start)
        mid_tissue = _five_tissue_mrtrix(five_tissue, chosen_mid, five_inverse)
        chosen_end_tissue = _five_tissue_mrtrix(five_tissue, chosen_end, five_inverse)
        ended_in_gm |= active & ~moving & (sgm_depth > 0)
        previous_seed_to_wm = seed_to_wm
        mid_term, new_depth, new_to_wm, mid_sgm = _act_structural_step(
            mid_tissue, sgm_depth, seed_in_sgm, seed_to_wm)
        sgm_depth = torch.where(moving, new_depth, sgm_depth)
        seed_to_wm = torch.where(moving, new_to_wm, seed_to_wm)
        mid_cgm = moving & (mid_term == 1)
        mid_exit = moving & (mid_term == 9)
        mid_image = moving & (mid_term == 3)
        live = moving & (mid_term == 0)
        best_sgm_metric = torch.where(
            ~previous_seed_to_wm & seed_to_wm,
            torch.full_like(best_sgm_metric, float('inf')), best_sgm_metric,
        )
        better_mid = moving & mid_sgm & (mid_metric < best_sgm_metric)
        best_sgm_metric = torch.where(better_mid, mid_metric, best_sgm_metric)
        best_sgm_position = torch.where(better_mid[:, None], chosen_mid, best_sgm_position)
        best_sgm_count = torch.where(better_mid, counts + 1, best_sgm_count)
        best_sgm_length = torch.where(
            better_mid, lengths + (chosen_mid - positions).norm(dim=-1), best_sgm_length,
        )
        previous_seed_to_wm = seed_to_wm
        end_active = live
        end_term, new_depth, new_to_wm, end_sgm = _act_structural_step(
            chosen_end_tissue, sgm_depth, seed_in_sgm, seed_to_wm)
        sgm_depth = torch.where(live, new_depth, sgm_depth)
        seed_to_wm = torch.where(live, new_to_wm, seed_to_wm)
        end_cgm = live & (end_term == 1)
        end_exit = live & (end_term == 9)
        end_image = live & (end_term == 3)
        live = live & (end_term == 0)
        best_sgm_metric = torch.where(
            ~previous_seed_to_wm & seed_to_wm,
            torch.full_like(best_sgm_metric, float('inf')), best_sgm_metric,
        )
        better_end = (end_active & end_sgm & (end_term == 0) &
                      (end_metric < best_sgm_metric))
        best_sgm_metric = torch.where(better_end, end_metric, best_sgm_metric)
        best_sgm_position = torch.where(better_end[:, None], chosen_end, best_sgm_position)
        best_sgm_count = torch.where(better_end, counts + 1, best_sgm_count)
        best_sgm_length = torch.where(
            better_end, lengths + (chosen_end - positions).norm(dim=-1), best_sgm_length,
        )
        append = mid_cgm | end_cgm | live
        proposed = torch.where(mid_cgm[:, None], chosen_mid, chosen_end)
        paths[append, step + 1] = proposed[append]
        counts[append] += 1
        lengths += torch.where(append, (proposed - positions).norm(dim=-1), 0.)
        sgm_exit = mid_exit | end_exit
        exit_rows = rows[sgm_exit]
        paths[exit_rows, best_sgm_count[sgm_exit] - 1] = best_sgm_position[sgm_exit]
        counts = torch.where(sgm_exit, best_sgm_count, counts)
        lengths = torch.where(sgm_exit, best_sgm_length, lengths)
        ended_in_gm |= mid_cgm | end_cgm | sgm_exit | mid_image | end_image
        active = live
        positions = torch.where(append[:, None], proposed, positions)
        prior = torch.where(append[:, None], direction, prior)
    return paths, counts, ended_in_gm, lengths, (~seed_in_sgm | seed_to_wm)


def _valid_act_tissue(tissue: torch.Tensor) -> torch.Tensor:
    """Return bool ``[...]`` validity for 5TT fractions ``[...,5]``.

    Valid tissue has total fraction at least 0.5 with no dominant CSF or
    pathology channel, following the MRtrix ACT tissue rule.
    """
    dominant = tissue[..., :3].amax(-1)
    return ((tissue.sum(-1) >= .5) & (tissue[..., 3] < dominant) &
            (tissue[..., 4] < dominant))


def _act_structural_step(
    tissue: torch.Tensor, depth: torch.Tensor, seed_in_sgm: torch.Tensor,
    seed_to_wm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MRtrix ACT check_structural to 5TT ``[B,5]`` and integer depth.

    Inputs ``depth`` int32 ``[B]`` and two bool seed flags ``[B]`` carry across
    sampled points and through direction reversal. Returns termination codes
    ``[B]`` (0 continue, 1 cGM, 3 image exit, 4 CSF, 9 sGM exit), updated
    depth, updated seed-to-WM flag, and bool current sGM. The tissue is already
    sampled with ``_five_tissue_mrtrix``. Official equivalent:
    ``ACT/method.h::check_structural`` at MRtrix3 eeab681d3e0c.
    """
    valid = tissue.sum(-1) >= .5
    cgm, sgm, wm, csf, path = tissue.unbind(-1)
    is_csf = (csf >= cgm) & (csf >= sgm) & (csf >= wm) & (csf >= path)
    is_gm = ((cgm + sgm >= wm) & (cgm + sgm > csf) & (cgm + sgm > path))
    cortical = valid & ~is_csf & is_gm & (cgm >= sgm)
    subcortical = valid & ~is_csf & is_gm & (sgm > cgm)
    exit_sgm = valid & ~is_csf & ~is_gm & (depth > 0)
    seed_wm_exit = exit_sgm & seed_in_sgm & ~seed_to_wm
    term = torch.zeros_like(depth, dtype=torch.int32)
    term = torch.where(~valid, 3, term)
    term = torch.where(valid & is_csf, torch.where(depth > 0, 9, 4), term)
    term = torch.where(cortical, 1, term)
    term = torch.where(exit_sgm & ~seed_wm_exit, 9, term)
    depth = torch.where(seed_wm_exit, 0, depth + subcortical.int())
    return term, depth, seed_to_wm | seed_wm_exit, subcortical


def _five_tissue_mrtrix(
    five_tissue: torch.Tensor, points: torch.Tensor, inverse_affine: torch.Tensor,
) -> torch.Tensor:
    """Sample float32 5TT at world-mm ``[B,3]`` using MRtrix masked linear rules.

    ``inverse_affine`` must match the reference 5TT world transform; a precise
    NIfTI-2 affine can be passed without spacing correction. Return clamped
    tissue fractions ``[B,5]``; invalid
    nearest voxels and positions outside the image return zeros.
    """
    voxel = points.double() @ inverse_affine[:3, :3].double().T + inverse_affine[:3, 3].double()
    shape = five_tissue.shape[:3]
    inside = ((voxel > -.5) & (voxel < voxel.new_tensor(shape) - .5)).all(-1)
    nearest = torch.floor(voxel + .5).long()
    nearest = torch.stack(tuple(nearest[:, axis].clamp(0, shape[axis] - 1) for axis in range(3)), -1)
    nonzero = (five_tissue[nearest[:, 0], nearest[:, 1], nearest[:, 2]] != 0).any(-1)
    lower = torch.floor(voxel).long()
    fraction = (voxel - lower).float()
    # Reuse axis indices and weights; accumulate the same eight corners in
    # dz -> dy -> dx order with the original FP32 multiplication grouping.
    indices = tuple(tuple((lower[:, axis] + offset).clamp(0, shape[axis] - 1)
                          for offset in (0, 1)) for axis in range(3))
    weights = tuple((1 - fraction[:, axis], fraction[:, axis]) for axis in range(3))
    result = five_tissue.new_zeros((len(points), 5))
    for dz in (0, 1):
        iz = indices[2][dz]
        wz = weights[2][dz]
        for dy in (0, 1):
            iy = indices[1][dy]
            wy = weights[1][dy]
            partial = wy * wz
            for dx in (0, 1):
                ix = indices[0][dx]
                wx = weights[0][dx]
                weight = wx * partial
                result += five_tissue[ix, iy, iz] * torch.where(weight < 1e-6, 0., weight)[:, None]
    return torch.where((inside & nonzero)[:, None], result.clamp(0, 1), 0.)


def _act_seed_direction(
    five_tissue: torch.Tensor, seeds: torch.Tensor, directions: torch.Tensor,
    inverse_affine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MRtrix ACT check_seed and seed_is_unidirectional on ``[B,3]`` seeds.

    Input 5TT is float32 ``[X,Y,Z,5]`` and ``inverse_affine`` maps RAS-mm
    positions to MRtrix voxel centers. Returns bool valid ``[B]``, bool one-way
    ``[B]``, and oriented float32 directions ``[B,3]``. Cortical GM-side
    seeds are one-way; the direction is flipped toward WM using a 0.001 mm
    central sample. Equivalent source: ACT/method.h in MRtrix3 eeab681d3e0c.
    """
    tissue = _five_tissue_mrtrix(five_tissue, seeds, inverse_affine)
    cgm, sgm, wm, csf, path = tissue.unbind(-1)
    is_sgm = (sgm > cgm) & (sgm >= wm) & (sgm > csf) & (sgm > path)
    is_csf = (csf >= cgm) & (csf >= sgm) & (csf >= wm) & (csf >= path)
    valid = _valid_act_tissue(tissue) & (
        is_sgm | (~is_csf & (wm > 0) & ((cgm + sgm - wm) < .01))
    )
    one_way = valid & ~is_sgm & (wm < cgm + sgm) & (sgm < cgm)
    plus = _five_tissue_mrtrix(five_tissue, seeds + .001 * directions, inverse_affine)
    minus = _five_tissue_mrtrix(five_tissue, seeds - .001 * directions, inverse_affine)
    gradient = (plus[:, 0] + plus[:, 1] - plus[:, 2]) - (
        minus[:, 0] + minus[:, 1] - minus[:, 2]
    )
    flip = one_way & (gradient > 0)
    oriented = torch.where(flip[:, None], -directions, directions)
    return valid, one_way, oriented


def _collect_tracks(
    forward: torch.Tensor,
    backward: torch.Tensor,
    forward_counts: torch.Tensor,
    backward_counts: torch.Tensor,
    one_way: torch.Tensor,
    keep: torch.Tensor,
    total_lengths: torch.Tensor,
    seeds: torch.Tensor,
    *,
    fa_image: torch.Tensor | None = None,
    fod_inverse: torch.Tensor | None = None,
) -> Tractogram:
    """Collect accepted paths without a CUDA scalar transfer for each path.

    Padded forward/backward buffers and counts come directly from ``_grow``;
    ``keep`` preserves the accepted seed order. Only integer slice metadata
    is copied to CPU, once per batch. A one-way path remains a forward-buffer
    view; a two-way path uses the same backward flip and concatenation as
    before. FA sampling and its per-path reduction retain their original
    order. Endpoint, length and seed coordinates are gathered in batches.
    """
    if keep.numel() == 0:
        return Tractogram(
            paths=(), endpoints=forward.new_empty((0, 2, 3)),
            lengths_mm=forward.new_empty(0),
            mean_fa=forward.new_empty(0) if fa_image is not None else None,
            seeds_attempted=seeds.shape[0],
            accepted_seeds=seeds.new_empty((0, 3)),
        )
    kept_forward_counts = forward_counts[keep]
    kept_backward_counts = backward_counts[keep]
    kept_one_way = one_way[keep]
    metadata = torch.stack(
        (keep, kept_forward_counts, kept_backward_counts, kept_one_way.long()),
        dim=-1,
    ).cpu().tolist()
    paths = []
    fa_means = []
    for index, forward_count, backward_count, is_one_way in metadata:
        path = (forward[index, :forward_count] if is_one_way else
                torch.cat((backward[index, :backward_count].flip(0),
                           forward[index, 1:forward_count]), dim=0))
        paths.append(path)
        if fa_image is not None:
            fa_means.append(_sample(fa_image, path, fod_inverse).mean())
    starts = torch.where(
        kept_one_way[:, None], forward[keep, 0],
        backward[keep, kept_backward_counts - 1],
    )
    ends = forward[keep, kept_forward_counts - 1]
    return Tractogram(
        paths=tuple(paths), endpoints=torch.stack((starts, ends), dim=1),
        lengths_mm=total_lengths[keep],
        mean_fa=torch.stack(fa_means) if fa_image is not None else None,
        seeds_attempted=seeds.shape[0], accepted_seeds=seeds[keep],
    )


@torch.inference_mode()
def probabilistic_tractography(
    wm_sh: torch.Tensor,
    fod_affine: torch.Tensor,
    five_tissue: torch.Tensor,
    five_tissue_affine: torch.Tensor,
    gmwmi: torch.Tensor,
    *,
    n_seeds: int,
    lmax: int = 8,
    five_tissue_spacing_mm: tuple[float, float, float] | None = None,
    fa: torch.Tensor | None = None,
    seed: int = 0,
    batch_size: int = 8192,
    arc_proposals: int = 16,
    max_length_mm: float = 250.,
    min_length_mm: float | None = None,
    step_mm: float | None = None,
    max_angle_degrees: float = 45.,
    cutoff: float = 0.1,
    power: float = 0.5,
    compile_arc: bool = False,
) -> Tractogram:
    """Generate FOD streamlines from 5TT GMWMI seeds on a CUDA or CPU device.

    Inputs: float32 WM SH ``[XF,YF,ZF,C]`` and its voxel-to-RAS ``[4,4]``
    affine; float32 5TT ``[XA,YA,ZA,5]`` in cGM/sGM/WM/CSF/path order and
    its ``[4,4]`` affine; and matching float32 GMWMI ``[XA,YA,ZA]``. Anatomy
    and FOD may have different grids; optional float32 ``fa`` is on the FOD
    grid. Inputs and ``Tractogram`` float tensors use one CPU/CUDA device.
    ``paths`` is N variable-length ``[Pi,3]`` tensors in RAS millimetres;
    ``endpoints`` is ``[N,2,3]``; ``lengths_mm``, optional ``mean_fa`` are
    ``[N]``; ``accepted_seeds`` is ``[N,3]``; and ``seeds_attempted`` is N's
    prefilter denominator. Use ``seed`` for reproducible PyTorch runs.

    MRtrix equivalent: ``tckgen -algorithm iFOD2 -seed_gmwmi gmwmi.mif
    -act 5tt.mif -seeds N -select 0 -maxlength 250 -cutoff 0.1 -samples 3
    -power 0.5 wm_fod_norm.mif tracks.tck``. Initial directions use the
    MRtrix continuous sphere rule and 1000 attempts. ``five_tissue_spacing_mm``
    is the 5TT header voxel spacing; if absent, affine column norms are used.
    Cortical GM-side ACT seeds are oriented toward WM and tracked one-way.
    iFOD2 propagation uses calibrated rejection sampling. ACT seed and
    per-point structural states were checked on real inputs; full-track
    truncation and image-exit parity are still under validation.
    ``compile_arc=True`` compiles only the iFOD2 arc probability kernel with
    PyTorch Inductor on CUDA. It preserves float32/TF32 and spends additional
    time compiling on first use; choose it for high seed counts.
    """
    if (wm_sh.ndim != 4 or fod_affine.shape != (4, 4) or
            five_tissue.ndim != 4 or five_tissue.shape[-1] != 5 or
            five_tissue_affine.shape != (4, 4) or
            gmwmi.shape != five_tissue.shape[:3]):
        raise ValueError('expected WM SH [X,Y,Z,C], 5TT [X,Y,Z,5], matching GMWMI and two affines')
    if fa is not None and fa.shape != wm_sh.shape[:3]:
        raise ValueError('FA must match the FOD grid')
    if n_seeds < 1 or batch_size < 1 or arc_proposals < 1:
        raise ValueError('n_seeds, batch_size and arc_proposals must be positive')
    device = wm_sh.device
    if compile_arc and device.type != 'cuda':
        raise ValueError('compile_arc requires CUDA')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    fod_affine = fod_affine.to(device=device, dtype=torch.float64)
    five_tissue_affine = five_tissue_affine.to(device=device, dtype=torch.float64)
    fod_inverse = torch.linalg.inv(fod_affine)
    five_spacing = (torch.linalg.vector_norm(five_tissue_affine[:3, :3], dim=0)
                    if five_tissue_spacing_mm is None else
                    torch.as_tensor(five_tissue_spacing_mm, device=device, dtype=torch.float64))
    if five_spacing.shape != (3,) or not bool(torch.isfinite(five_spacing).all()) or bool((five_spacing <= 0).any()):
        raise ValueError('five_tissue_spacing_mm must contain three positive finite voxel sizes')
    act_affine = five_tissue_affine.clone()
    act_affine[:3, :3] *= five_spacing / torch.linalg.vector_norm(act_affine[:3, :3], dim=0)
    act_inverse = torch.linalg.inv(act_affine)
    wm_sh = wm_sh.to(torch.float32)
    five_tissue = five_tissue.to(device=device, dtype=torch.float32)
    gmwmi = gmwmi.to(device=device, dtype=torch.float32)
    voxel_mm = torch.linalg.vector_norm(fod_affine[:3, :3], dim=0)
    voxel_size_mm = float(voxel_mm.prod().pow(1 / 3))
    step_mm = voxel_size_mm / 2 if step_mm is None else float(step_mm)
    if step_mm <= 0 or max_length_mm <= step_mm:
        raise ValueError('step and maximum length must be positive')
    max_steps = math.ceil(max_length_mm / step_mm)
    min_length_mm = 2 * voxel_size_mm if min_length_mm is None else min_length_mm
    if min_length_mm < 0 or min_length_mm > max_length_mm:
        raise ValueError('minimum length must lie between zero and maximum length')
    if lmax < 0 or lmax % 2 or wm_sh.shape[-1] != (lmax + 1) * (lmax + 2) // 2:
        raise ValueError('WM SH coefficient count does not match lmax')
    calibration_local, calibration_ratio = (
        _ifod2_calibration(lmax, step_mm, voxel_size_mm, max_angle_degrees, power, device)
        if max_angle_degrees > 0 else
        (wm_sh.new_tensor([[0., 0., 1.]]), 1.)
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    arc_probability_fn = (torch.compile(_ifod2_arc_probability, fullgraph=True,
                                        dynamic=True) if compile_arc else None)
    fod_sampler = _VolumeSampler(wm_sh, fod_inverse)
    if not compile_arc:
        arc_probability_fn = partial(_ifod2_arc_probability, _fod_sampler=fod_sampler)
    seeds = sample_gmwmi_seeds(gmwmi, five_tissue, five_tissue_affine,
                               n_seeds, generator)
    collected = []
    endpoints = []
    lengths = []
    fa_means = []
    accepted_seeds = []
    fa_image = fa.to(device=device, dtype=torch.float32)[..., None] if fa is not None else None
    for first in range(0, n_seeds, batch_size):
        batch_seeds = seeds[first:first + batch_size]
        initial, valid_seed, _ = _initial_directions(
            fod_sampler(batch_seeds), generator, lmax=lmax, cutoff=cutoff,
        )
        valid_act, one_way, initial = _act_seed_direction(
            five_tissue, batch_seeds, initial, act_inverse,
        )
        valid_seed &= valid_act
        forward, nf, gf, lf, wf = _grow(
            batch_seeds, initial, wm_sh, five_tissue, fod_inverse, act_inverse,
            generator, lmax=lmax, proposals_per_step=arc_proposals,
            calibration_local=calibration_local, calibration_ratio=calibration_ratio,
            arc_probability_fn=arc_probability_fn, _fod_sampler=fod_sampler,
            step_mm=step_mm, max_steps=max_steps,
            max_angle_degrees=max_angle_degrees, cutoff=cutoff, power=power,
        )
        backward, nb, gb, lb, wb = _grow(
            batch_seeds, -initial, wm_sh, five_tissue, fod_inverse, act_inverse,
            generator, lmax=lmax, proposals_per_step=arc_proposals,
            calibration_local=calibration_local, calibration_ratio=calibration_ratio,
            arc_probability_fn=arc_probability_fn, _fod_sampler=fod_sampler,
            seed_to_wm_initial=wf,
            step_mm=step_mm, max_steps=max_steps,
            max_angle_degrees=max_angle_degrees, cutoff=cutoff, power=power,
        )
        total = torch.where(one_way, lf, lf + lb)
        keep = torch.nonzero(valid_seed & gf & (one_way | gb) & (wf | wb) &
                             (total >= min_length_mm) &
                             (total <= max_length_mm), as_tuple=False).flatten()
        batch_tracks = _collect_tracks(
            forward, backward, nf, nb, one_way, keep, total, batch_seeds,
            fa_image=fa_image, fod_inverse=fod_inverse,
        )
        if batch_tracks.paths:
            collected.extend(batch_tracks.paths)
            endpoints.append(batch_tracks.endpoints)
            lengths.append(batch_tracks.lengths_mm)
            accepted_seeds.append(batch_tracks.accepted_seeds)
            if fa_image is not None:
                fa_means.append(batch_tracks.mean_fa)
    empty_endpoints = wm_sh.new_empty((0, 2, 3))
    return Tractogram(
        paths=tuple(collected),
        endpoints=torch.cat(endpoints) if endpoints else empty_endpoints,
        lengths_mm=torch.cat(lengths) if lengths else wm_sh.new_empty(0),
        mean_fa=torch.cat(fa_means) if fa_means else (wm_sh.new_empty(0) if fa is not None else None),
        seeds_attempted=n_seeds,
        accepted_seeds=torch.cat(accepted_seeds) if accepted_seeds else wm_sh.new_empty((0, 3)),
    )
