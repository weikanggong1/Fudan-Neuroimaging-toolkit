"""Rigid EPI-to-T1 boundary registration using the FLIRT BBR cost.

FSL's no-fieldmap BBR samples EPI intensities 2 mm either side of the T1
white-matter boundary. Matrices here map EPI to T1 in FLIRT scaled-mm space.
"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..flirt import TorchFLIRT
from ..flirt.coordinates import (flirt_to_world_affine, voxel_to_fsl_scaled_mm,
                                  world_to_flirt_affine)
from ..flirt.core import (
    _clamp_like_fsl, _flip_to_radiological, _centre_of_gravity,
    _initial_bound, _next_point, _extrapolated_point,
    fsl_parameters_from_affine, fsl_affine_from_parameters,
    fsl_coordinate_optimize, _resample_output, _output_image, _blur,
)
from ..flirt.affine_batch import fsl_affine_from_parameters_batch


def _image(value, name):
    image = nib.load(str(value)) if isinstance(value, (str, Path)) else value
    if not isinstance(image, nib.spatialimages.SpatialImage) or image.ndim != 3:
        raise ValueError(f"{name} must be a 3D NIfTI image or path")
    return image


def _smooth_wm(data, spacing):
    """NEWIMAGE smooth(2mm): double kernels, float accumulator per tap."""
    result = data
    for axis, size in enumerate(spacing):
        sigma = np.float32(2.0 / np.float32(size))
        radius = int(float(sigma) - 0.001) * 2 + 3
        values = [np.float32(np.exp(-(j * j) / (2.0 * float(sigma) ** 2)))
                  for j in range(-radius, radius + 1)]
        total = np.float32(0)
        for value in values:
            total = np.float32(total + value)
        coefficients = np.asarray(values, dtype=np.float64) / float(total)
        if (result.device.type == "cpu" and result.dtype == torch.float32
                and not result.requires_grad and bool(torch.isfinite(result).all())):
            from ._bbr_cpu import smooth_axis
            result = torch.from_numpy(smooth_axis(result.numpy(), coefficients, axis))
            continue
        pad_shape = list(result.shape)
        pad_shape[axis] = radius
        padded = torch.cat((result.new_zeros(pad_shape), result,
                            result.new_zeros(pad_shape)), dim=axis)
        output = torch.zeros_like(result)
        for index, coefficient in enumerate(coefficients):
            output = (output.to(torch.float64)
                      + padded.narrow(axis, index, result.shape[axis]).to(torch.float64)
                      * float(coefficient)).to(torch.float32)
        result = output
    return result


def _boundary(wm, reference, *, device="cpu", reference_sampling=None):
    segmentation = np.asarray(wm.dataobj, dtype=np.float32)
    if segmentation.shape != reference.shape or not np.allclose(
        wm.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise ValueError("wmseg must have the same voxel grid and affine as t1")
    # NEWIMAGE reads radiological storage and traverses x fastest. In particular,
    # bbrstep=200 must select the same boundary subset in both implementations.
    if not np.isfinite(segmentation).all():
        raise ValueError("wmseg must contain finite values")
    segmentation = _flip_to_radiological(segmentation, reference.affine)
    image = torch.as_tensor(segmentation, device=device)
    padded = F.pad(image[None, None], (1,1,1,1,1,1))[0,0]
    neighbours = torch.zeros_like(image)
    for z, y, x in product(range(3), repeat=3):
        neighbours += padded[x:x+image.shape[0], y:y+image.shape[1], z:z+image.shape[2]]
    boundary = (image > 0.5) & (neighbours < 26.5)
    points = torch.nonzero(boundary.permute(2, 1, 0), as_tuple=False).flip(1)
    if len(points) == 0:
        raise ValueError("wmseg has no white-matter boundary points")
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=np.float32)
    smoothed = _smooth_wm(image, spacing)
    gradients = torch.zeros((len(points), 3), dtype=torch.float32, device=device)
    shape = torch.tensor(image.shape, device=device)
    # NEWIMAGE's gradient uses these 27 taps, rather than central differences.
    for z, y, x in product((-1, 0, 1), repeat=3):
        indices = points + points.new_tensor((x, y, z))
        valid = ((indices >= 0) & (indices < shape)).all(1)
        safe = indices.clamp_min(0)
        safe = torch.minimum(safe, shape - 1)
        values = smoothed[tuple(safe.T)] * valid
        weights = [np.float32(x * 3.0 ** (1 - abs(y) - abs(z))),
                   np.float32(y * 3.0 ** (1 - abs(x) - abs(z))),
                   np.float32(z * 3.0 ** (1 - abs(x) - abs(y)))]
        for axis, weight in enumerate(weights):
            gradients[:, axis] += values * float(weight)
    normal_spacing = spacing if reference_sampling is None else reference_sampling
    normal = (-gradients.to(torch.float64)
              / torch.as_tensor(normal_spacing, dtype=torch.float64, device=device)).to(torch.float32)
    square = ((normal[:, 0].square() + normal[:, 1].square())
              + normal[:, 2].square())
    length = torch.sqrt(square.to(torch.float64)).to(torch.float32)
    if not bool((length > 0).all()):
        raise ValueError("wmseg has a boundary point with an undefined normal")
    normal = (normal.to(torch.float64) / length[:, None].to(torch.float64)).to(torch.float32)
    surface = points.to(torch.float64) * torch.as_tensor(spacing, device=device)
    grey = (surface + (2.0 * normal).to(torch.float64)).to(torch.float32)
    white = (surface - (2.0 * normal).to(torch.float64)).to(torch.float32)
    return grey.cpu().numpy(), white.cpu().numpy(), surface.mean(0).cpu().numpy()


def _sample_zero(image, coordinates):
    """NEWIMAGE float trilinear interpolation, including zero-valued corners."""
    coordinates = coordinates.to(torch.float32)
    lower = torch.floor(coordinates).to(torch.long)
    delta = coordinates - lower.to(torch.float32)
    shape = coordinates.new_tensor(image.shape, dtype=torch.long)
    corners = []
    for x, y, z in product((0, 1), repeat=3):
        indices = lower + lower.new_tensor((x, y, z))
        valid = ((indices >= 0) & (indices < shape)).all(-1)
        safe = torch.minimum(indices.clamp_min(0), shape - 1)
        corners.append(image[safe[..., 0], safe[..., 1], safe[..., 2]] * valid)
    v000, v001, v010, v011, v100, v101, v110, v111 = corners
    dx, dy, dz = delta.unbind(-1)
    a = (v100 - v000) * dx + v000
    b = (v101 - v001) * dx + v001
    c = (v110 - v010) * dx + v010
    d = (v111 - v011) * dx + v011
    e, f = (c - a) * dy + a, (d - b) * dy + b
    return (f - e) * dz + e


class _BBRCost:
    def __init__(self, epi, grey_points, white_points, device, *, max_batch_size=128, use_fused=True):
        self.device = torch.device(device)
        data = _flip_to_radiological(_clamp_like_fsl(np.asarray(epi.dataobj)), epi.affine)
        cpu_image = _blur(torch.from_numpy(data), 1., epi.header.get_zooms()[:3])
        # NIfTI ArrayProxy reads are often Fortran ordered; fused sampling uses
        # C-order xyz addressing. Materialize this layout once, not per cost.
        self.image = cpu_image.to(self.device).contiguous()
        self.shape = data.shape
        self.centre = _centre_of_gravity(cpu_image,
            np.diag([*epi.header.get_zooms()[:3], 1.]))
        self.input_sampling_inverse = np.diag(
            [*(1.0 / np.asarray(epi.header.get_zooms()[:3], dtype=np.float64)), 1.0]
        )
        # NEWMAT transforms float coordinates using double affine coefficients.
        self.points = torch.as_tensor(np.stack((grey_points, white_points)),
                                      dtype=torch.float64, device=self.device)
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        self.evaluations = 0
        self.host_result_transfers = 0
        self._point_cache = {}
        self.use_fused = bool(use_fused)
        self._cpu_updates = None
        if (self.device.type == "cpu" and self.use_fused
                and self.image.dtype == torch.float32 and self.points.dtype == torch.float64
                and not self.image.requires_grad and not self.points.requires_grad
                and bool(torch.isfinite(self.image).all())
                and bool(torch.isfinite(self.points).all())
                and float(self.image.abs().max()) < np.finfo(np.float32).max / 16):
            from ._bbr_cpu import cost_updates
            self._cpu_updates = cost_updates
        self._cpu_transform_limit = None
        if self._cpu_updates is not None:
            self._cpu_transform_limit = (2.0**60) / (4 * (3 * float(self.points.abs().max()) + 1))

    def evaluate(self, matrices, step=2):
        """Return ordered float32 costs without a GPU-to-host result transfer."""
        matrices = np.asarray(matrices, dtype=np.float64)
        if matrices.ndim == 2:
            matrices = matrices[None]
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError("BBR matrices must have shape (B,4,4)")
        if int(step) != step or step < 1:
            raise ValueError("BBR vertex step must be a positive integer")
        step = int(step)
        if step not in self._point_cache:
            self._point_cache[step] = self.points[:, ::step].contiguous()
        points = self._point_cache[step]
        transforms = self.input_sampling_inverse @ np.linalg.inv(matrices)
        costs = []
        # Coordinates and all eight sampled corners fit comfortably below 20GB.
        batch_size = min(self.max_batch_size, max(1, int(16 * 2**30 / (points.numel() * 96))))
        for start in range(0, len(matrices), batch_size):
            current_transforms = transforms[start:start + batch_size]
            if (self._cpu_updates is not None
                    and self.image.dtype == torch.float32 and points.dtype == torch.float64
                    and not self.image.requires_grad and not points.requires_grad
                    and np.isfinite(current_transforms).all()
                    and np.max(np.abs(current_transforms)) < self._cpu_transform_limit):
                updates = self._cpu_updates(self.image.numpy(), points.numpy(),
                    transforms[start:start + batch_size])
                # Keep the existing PyTorch float64 reduction order and
                # final float32 cost, including batched/reference searches.
                costs.append(torch.from_numpy(updates).mean(-1).to(torch.float32))
                continue
            # Pageable CPU input is not mutated; Torch stages its asynchronous
            # copy on the same stream as the following cost kernels.
            transform = torch.from_numpy(transforms[start:start + batch_size]).to(
                self.device, non_blocking=True)
            kernel = None
            if self.use_fused and self.device.type == "cuda":
                try:
                    from ._bbr_cuda import _cost_kernel
                    kernel = _cost_kernel
                except ImportError:
                    pass
            if kernel is not None:
                updates = torch.empty((len(transform), points.shape[1]),
                                      dtype=torch.float64, device=self.device)
                with torch.cuda.device(self.device):
                    kernel[((points.shape[1] + 255) // 256, len(transform))](
                        self.image, points, transform.contiguous(), updates,
                        points.shape[1], *self.image.shape, 256,
                        enable_fp_fusion=False,
                    )
                costs.append(updates.mean(-1).to(torch.float32))
                continue
            coordinates = (points[None, ..., 0:1] * transform[:, None, None, :3, 0]
                           + points[None, ..., 1:2] * transform[:, None, None, :3, 1])
            coordinates = coordinates + points[None, ..., 2:3] * transform[:, None, None, :3, 2]
            coordinates = coordinates + transform[:, None, None, :3, 3]
            values = _sample_zero(self.image, coordinates).to(torch.float64)
            grey, white = values.unbind(1)
            total = grey + white
            safe = torch.where(total.abs() > 1e-6, total, torch.ones_like(total))
            difference = torch.where(total.abs() > 1e-6, 200.0 * (grey - white) / safe, 0.0)
            costs.append((1.0 + torch.tanh(-0.5 * difference)).mean(-1).to(torch.float32))
        self.evaluations += len(matrices)
        return torch.cat(costs) if costs else self.image.new_empty((0,))

    def __call__(self, matrix, step=2):
        self.host_result_transfers += 1
        return float(self.evaluate(matrix, step=step)[0])


def _resample(epi, t1, matrix, device):
    moving_fsl = voxel_to_fsl_scaled_mm(epi.affine, epi.shape, epi.header.get_zooms()[:3])
    fixed_fsl = voxel_to_fsl_scaled_mm(t1.affine, t1.shape, t1.header.get_zooms()[:3])
    data = _resample_output(
        np.asarray(epi.dataobj, dtype=np.float32), t1.shape, moving_fsl, fixed_fsl,
        matrix, epi.header.get_zooms()[:3], t1.header.get_zooms()[:3], device,
    ).cpu().numpy()
    pull_voxel = np.linalg.inv(moving_fsl) @ np.linalg.inv(matrix) @ fixed_fsl
    # BBR's Python API retains its float32 image while sharing FLIRT's header rules.
    floating = nib.Nifti1Image(np.asarray(epi.dataobj, dtype=np.float32), epi.affine, epi.header)
    floating.header.set_data_dtype(np.float32)
    return _output_image(data, floating, t1, pull_voxel)


def _powell_line(point, direction, tolerance, function, initial_value):
    """MISCMATHS line search, including double RHS accumulation for non-axis directions."""
    f32 = np.float32
    unit = direction / np.linalg.norm(direction)
    dir_tol = f32(0)
    for index in range(len(tolerance)):
        if abs(tolerance[index]) > 1e-15:
            dir_tol = f32(float(dir_tol) + abs(unit[index] / tolerance[index]))
    unit_tol = f32(abs(f32(1) / dir_tol))
    middle, x1 = f32(0), unit_tol  # bbr.sch boundguess=1
    y_middle = f32(initial_value if initial_value != 0 else function(point))
    y1 = f32(function(float(x1) * unit + point))
    x1, middle, x2, y1, y_middle, y2 = _initial_bound(
        x1, middle, y1, y_middle, function, unit, point)
    min_dist = f32(.1 * float(unit_tol))
    for _ in range(100):
        if abs((x2 - x1) / unit_tol) <= 1:
            break
        new = _next_point(x1, middle, x2, y1, y_middle, y2)
        sign = -1. if x2 < x1 else 1.
        if abs(new - x1) < min_dist:
            new = f32(x1 + f32(sign * min_dist))
        if abs(new - x2) < min_dist:
            new = f32(x2 - f32(sign * min_dist))
        if abs(new - middle) < min_dist:
            new = _extrapolated_point(x1, middle, x2)
        if abs(middle - x1) < .4 * unit_tol:
            new = f32(middle + f32(sign * .5 * unit_tol))
        if abs(middle - x2) < .4 * unit_tol:
            new = f32(middle - f32(sign * .5 * unit_tol))
        new = f32(new)
        value = f32(function(float(new) * unit + point))
        if f32(new - middle) * f32(x2 - middle) > 0:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if value < y_middle:
            x2, y2, middle, y_middle = middle, y_middle, new, value
        else:
            x1, y1 = new, value
    return float(middle) * unit + point, float(y_middle)


def _powell_optimize(point, tolerance, function, *, maximum_iterations=8, numopt=6):
    """FLIRT's MISCMATHS Powell direction updates; no SciPy optimizer or bounds."""
    f32 = np.float32
    point = np.asarray(point, dtype=np.float64).copy()
    tolerance = np.asarray(tolerance, dtype=np.float64)
    inverse = np.where(abs(tolerance) > 1e-15, abs(1. / tolerance), 0.) / len(tolerance)
    directions = np.eye(len(point))
    changes = np.zeros(len(point), dtype=np.float32)
    value = f32(0)
    for _ in range(maximum_iterations):
        initial = point.copy()
        for index in range(numopt):
            # optimise1d updates its init_value reference when the sentinel is zero.
            if value == 0:
                value = f32(function(point))
            point, new_value = _powell_line(point, directions[:, index], tolerance, function, value)
            new_value = f32(new_value)
            changes[index] = f32(new_value - value)
            if index == 0:
                first = value
            value = new_value
        if f32(np.abs((initial - point) * inverse).sum()) < 1:
            break
        best = int(np.argmin(changes[:numopt]))
        end = value
        extrapolated = f32(function(initial + 2 * (point - initial)))
        change = f32(abs(changes[best]))
        a = f32(f32(first - f32(2 * end)) + extrapolated)
        b = f32(f32(first - end) - change)
        c = f32(first - extrapolated)
        lhs = f32(f32(f32(f32(2 * a) * b) * b))
        rhs = f32(f32(c * c) * change)
        if lhs < rhs and extrapolated < first:
            direction = point - initial
            point, new_value = _powell_line(point, direction, tolerance, function, value)
            value = f32(new_value)
            # The source stores the displacement after the additional line search.
            directions[:, best] = point - initial
    return point, float(value)


def _grid_perturbations(rotation, translation):
    """bbr.sch gridmeasurecost order: first parameter fastest, increment before use."""
    # Schedule numbers pass through setscalarvariable(float), then NEWMAT double.
    rotation, translation = float(np.float32(rotation)), float(np.float32(translation))
    lower = np.array([-rotation] * 3 + [-translation] * 3)
    step = np.array([rotation] * 3 + [translation] * 3)
    indices = np.array([[(index // 3**axis) % 3 for axis in range(6)]
                        for index in range(1, 3**6 + 1)])
    output = np.zeros((3**6, 12), dtype=np.float64)
    output[:, :6] = lower + indices * step
    return output


class _BBRSchedule:
    def __init__(self, cost, centre, initial, qsform, execution):
        self.cost = cost
        self.centre = centre
        self.initial = initial
        self.qsform = qsform @ np.linalg.inv(initial)
        self.execution = execution
        self.cache = {}
        self.phase_evaluations = {}
        self.active_phase = "local_bbr"

    def matrices(self, parameters, dof):
        if self.execution == "batched":
            return fsl_affine_from_parameters_batch(np.asarray(parameters), self.centre, dof).numpy()
        return np.stack([fsl_affine_from_parameters(torch.as_tensor(p),
                         torch.as_tensor(self.centre), dof).numpy() for p in parameters])

    def evaluate(self, matrices, step):
        missing = {}
        keys = [(np.asarray(matrix).tobytes(), step) for matrix in matrices]
        for key, matrix in zip(keys, matrices):
            if key not in self.cache:
                missing.setdefault(key, matrix)
        if missing:
            actual = np.stack([matrix @ self.initial for matrix in missing.values()])
            if self.execution == "batched":
                values = self.cost.evaluate(actual, step=step).cpu().numpy()
                self.cost.host_result_transfers += 1
            else:
                values = [self.cost(matrix, step=step) for matrix in actual]
            self.phase_evaluations[self.active_phase] = self.phase_evaluations.get(self.active_phase, 0) + len(actual)
            self.cache.update(zip(missing, map(float, values)))
        return [self.cache[key] for key in keys]

    def measure(self, matrices, step):
        parameters = [fsl_parameters_from_affine(matrix, self.centre) + np.zeros(12)
                      for matrix in matrices]
        rebuilt = self.matrices(parameters, 12)
        return list(zip(self.evaluate(rebuilt, step), rebuilt))

    def grid(self, matrices, rotation, translation, step):
        perturbations = _grid_perturbations(rotation, translation)
        parameters = np.concatenate([fsl_parameters_from_affine(matrix, self.centre)
                                     + perturbations for matrix in matrices])
        matrices = self.matrices(parameters, 12)
        return list(zip(self.evaluate(matrices, step), matrices))

    def optimize(self, matrix, step, tolerance, optimizer, iterations):
        # usroptimise: decompose12, zero perturbation, compose12, decompose again.
        params = fsl_parameters_from_affine(matrix, self.centre) + np.zeros(12)
        matrix = self.matrices([params], 12)[0]
        params = fsl_parameters_from_affine(matrix, self.centre)
        def objective(parameters):
            return self.evaluate(self.matrices([parameters], 6), step)[0]
        if optimizer == "powell":
            fitted, value = _powell_optimize(params, tolerance, objective,
                                             maximum_iterations=iterations)
        else:
            fitted, value = fsl_coordinate_optimize(params, tolerance, objective,
                maximum_iterations=iterations, bound_guess=(1.,), numopt=6)
        return value, self.matrices([fitted], 6)[0]

    def refinement(self, candidate, step, tolerance):
        brent = self.optimize(candidate, step, tolerance, "brent", 8)
        powell = self.optimize(brent[1], step, tolerance, "powell", 8)
        final = self.optimize(powell[1], step, tolerance, "brent", 4)
        return min((brent, powell, final), key=lambda item: item[0])[1]



@dataclass(frozen=True)
class BBRResult:
    moved: nib.Nifti1Image
    matrix: np.ndarray
    moving_to_fixed_world: np.ndarray
    initial_cost: float
    final_cost: float
    boundary_points: int
    runtime_seconds: float
    phase_timings: dict = field(default_factory=dict)
    cost_evaluations: int = 0
    host_result_transfers: int = 0
    phase_cost_evaluations: dict = field(default_factory=dict)

    def save(self, output=None, omat=None):
        if output is not None:
            nib.save(self.moved, str(output))
        if omat is not None:
            np.savetxt(omat, self.matrix, fmt="%.12g")
        return self


def register_bbr(epi, t1, wmseg, *, init=None, device=None, grid_search=True,
                 execution="batched", candidate_batch_size=128):
    """Rigid no-fieldmap EPI-to-T1 BBR using the complete FSL bbr.sch schedule.

    ``init`` maps EPI to T1 in FLIRT scaled-mm coordinates. If absent, FNIT's
    6-DOF normmi FLIRT initializes the registration. ``execution="reference"``
    retains serial candidate evaluation with the same boundaries and optimizer;
    ``"batched"`` batches each independent 729-candidate search. ``grid_search``
    controls both coarse and fine candidate grids. White-matter segmentation
    must be on the T1 voxel grid. No fieldmap or distortion correction is used.
    """
    start = time.perf_counter()
    if execution not in ("batched", "reference"):
        raise ValueError("execution must be 'batched' or 'reference'")
    epi_input, t1_input = epi, t1
    epi, t1, wmseg = _image(epi, "epi"), _image(t1, "t1"), _image(wmseg, "wmseg")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timings = {}
    stage = time.perf_counter()
    if init is None:
        if not isinstance(epi_input, (str, Path)) or not isinstance(t1_input, (str, Path)):
            raise ValueError("file paths are required for automatic FLIRT initialization")
        initial = TorchFLIRT(device=str(device), dof=6, cost="normmi", execution=execution,
                             candidate_batch_size=candidate_batch_size)(epi_input, t1_input).matrix
    elif isinstance(init, (str, Path)):
        initial = np.loadtxt(init)
    else:
        initial = np.asarray(init, dtype=np.float64)
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("init must be a finite 4x4 FLIRT matrix")
    if not np.allclose(initial[3], (0.,0.,0.,1.)) or abs(np.linalg.det(initial[:3,:3])) < 1e-10:
        raise ValueError("init must be an invertible homogeneous FLIRT matrix")
    timings["initial_flirt"] = time.perf_counter() - stage
    stage = time.perf_counter()
    grey, white, _ = _boundary(wmseg, t1, device=device, reference_sampling=(1.,1.,1.))
    # The schedule's reference mode controls candidate ordering, independently
    # of the CPU scalar kernel. CUDA reference keeps its existing tensor path.
    cost = _BBRCost(epi, grey, white, device, max_batch_size=candidate_batch_size,
                    use_fused=device.type == "cpu" or execution == "batched")
    centre = cost.centre
    qsform = world_to_flirt_affine(np.eye(4), epi.affine, t1.affine, epi.shape, t1.shape,
                                  epi.header.get_zooms()[:3], t1.header.get_zooms()[:3])
    schedule = _BBRSchedule(cost, centre, initial, qsform, execution)
    initial_cost = cost(initial)
    timings["boundary_preparation"] = time.perf_counter() - stage
    coarse_tol = np.array([.0005]*3 + [.02]*3 + [.002]*3 + [.001]*3,
                          dtype=np.float32).astype(np.float64)
    fine_tol = np.array([.0002]*3 + [.02]*3 + [.002]*3 + [.001]*3,
                        dtype=np.float32).astype(np.float64)
    stage = time.perf_counter()
    schedule.active_phase = "coarse_bbr"
    seeds = [schedule.qsform, np.eye(4)] if grid_search else [np.eye(4)]
    candidates = schedule.measure(seeds, 200)
    if grid_search:
        candidates += schedule.grid(seeds, .07, 4., 200)
    matrix = min(candidates, key=lambda item: item[0])[1]
    matrix = schedule.refinement(matrix, 200, coarse_tol)
    timings["coarse_bbr"] = time.perf_counter() - stage
    stage = time.perf_counter()
    schedule.active_phase = "local_bbr"
    if grid_search:
        matrix = min(schedule.grid([matrix], .0017, .1, 2), key=lambda item: item[0])[1]
    residual = schedule.refinement(matrix, 2, fine_tol)
    matrix = residual @ initial
    final_cost = cost(matrix)
    timings["local_bbr"] = time.perf_counter() - stage
    stage = time.perf_counter()
    moved = _resample(epi, t1, matrix, device)
    world = flirt_to_world_affine(matrix, epi.affine, t1.affine, epi.shape, t1.shape,
                                 epi.header.get_zooms()[:3], t1.header.get_zooms()[:3])
    timings["final_resampling"] = time.perf_counter() - stage
    return BBRResult(
        moved=moved, matrix=matrix, moving_to_fixed_world=world,
        initial_cost=initial_cost, final_cost=final_cost, boundary_points=len(grey),
        runtime_seconds=time.perf_counter()-start, phase_timings=timings,
        cost_evaluations=cost.evaluations, host_result_transfers=cost.host_result_transfers,
        phase_cost_evaluations=schedule.phase_evaluations,
    )


__all__ = ["BBRResult", "register_bbr"]
