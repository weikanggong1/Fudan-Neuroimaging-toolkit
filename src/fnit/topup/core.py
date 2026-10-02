"""PyTorch port of the FSL TOPUP ``b02b0.cnf`` path used by UK Biobank."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..fnirt.spline import (
    BendingOperator,
    adjoint_field,
    cubic_bspline_basis,
    design_diagonal,
    expand_coefficients,
    fit_field_coefficients,
    fsl_control_shape,
    spline_bases,
)
from ..fnirt.optimizer import (
    preconditioned_conjugate_gradient,
    scaled_conjugate_gradient,
)
from .io import (
    FSL_TOPUP_FIELD,
    image_path,
    make_output_image,
    make_topup_coefficient_image,
    make_topup_jacobian_image,
)


FSL_TOPUP_VERSION = "2203.2"
FSL_TOPUP_COMMIT = "3e2cb9104e834ce18c10e4b7edddbd500d0c459c"


@dataclass(frozen=True)
class TOPUPConfig:
    """The nine-level schedule in FSL 6.0.7.4 ``b02b0.cnf``."""

    warp_resolution_mm: tuple[float, ...] = (20, 16, 14, 12, 10, 6, 4, 4, 4)
    subsampling: tuple[int, ...] = (2, 2, 2, 2, 2, 1, 1, 1, 1)
    fwhm_mm: tuple[float, ...] = (8, 6, 4, 3, 3, 2, 1, 0, 0)
    maximum_iterations: tuple[int, ...] = (5, 5, 5, 5, 5, 10, 10, 20, 20)
    regularization: tuple[float, ...] = (
        0.005,
        0.001,
        0.0001,
        0.000015,
        0.000005,
        0.0000005,
        0.00000005,
        0.0000000005,
        0.00000000001,
    )

    def __post_init__(self):
        lengths = {
            len(self.warp_resolution_mm),
            len(self.subsampling),
            len(self.fwhm_mm),
            len(self.maximum_iterations),
            len(self.regularization),
        }
        if lengths != {9}:
            raise ValueError("TOPUP b02b0 schedule must contain nine levels")
        if any(value <= 0 or not math.isfinite(value) for value in self.warp_resolution_mm):
            raise ValueError("warp resolutions must be finite and positive")
        if any(not isinstance(value, int) or value < 1 for value in self.subsampling):
            raise ValueError("subsampling factors must be positive integers")
        if any(not isinstance(value, int) or value < 0 for value in self.maximum_iterations):
            raise ValueError("maximum iterations must be non-negative integers")
        if any(not math.isfinite(value) or value < 0 for value in self.fwhm_mm + self.regularization):
            raise ValueError("smoothing and regularization must be finite and non-negative")


@dataclass(frozen=True)
class TOPUPResult:
    field_hz: nib.Nifti1Image
    corrected: nib.Nifti1Image
    corrected_mean: nib.Nifti1Image
    jacobians: tuple[nib.Nifti1Image, ...]
    coefficients: nib.Nifti1Image
    movement_parameters: np.ndarray
    qc: dict

    def save(self, *, out, fout=None, iout=None, jacout=None, overwrite=False):
        root = Path(out).expanduser()
        if root.suffix:
            raise ValueError("out must be an extensionless FSL TOPUP basename")
        outputs = {
            image_path(root.with_name(root.name + "_fieldcoef")): self.coefficients,
            root.with_name(root.name + "_movpar.txt"): self.movement_parameters,
        }
        if fout is not None:
            outputs[image_path(fout)] = self.field_hz
        if iout is not None:
            target = image_path(iout)
            outputs[target] = self.corrected
        if jacout is not None:
            target = image_path(jacout)
            stem = target.name[:-7] if target.name.endswith(".nii.gz") else target.stem
            suffix = ".nii.gz" if target.name.endswith(".nii.gz") else ".nii"
            for index, image in enumerate(self.jacobians, 1):
                outputs[target.with_name(f"{stem}_{index:02d}{suffix}")] = image
        existing = [path for path in outputs if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, np.ndarray):
                np.savetxt(path, value, fmt="%.10g")
            else:
                nib.save(value, str(path))
        return tuple(outputs)


def _load_inputs(imain, datain):
    image = nib.load(os.fspath(imain))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    if data.ndim != 4 or data.shape[3] != 2:
        raise ValueError("the b02b0 path requires exactly two 3D volumes")
    if not np.isfinite(data).all():
        raise ValueError("imain contains non-finite values")
    acquisition = np.loadtxt(os.fspath(datain), dtype=np.float64, ndmin=2)
    if acquisition.shape != (data.shape[3], 4):
        raise ValueError("datain must have one four-column row per input volume")
    if not np.isfinite(acquisition).all():
        raise ValueError("datain contains non-finite values")
    phase = acquisition[:, :3]
    norms = np.linalg.norm(phase, axis=1)
    if not np.allclose(norms, 1.0, atol=0.01, rtol=0):
        raise ValueError("phase-encoding vectors must have unit length")
    axes = np.flatnonzero(np.any(np.abs(phase) > 1e-6, axis=0))
    if len(axes) != 1 or axes[0] not in (0, 1):
        raise NotImplementedError("the b02b0 path currently supports one i or j PE axis")
    if not np.allclose(phase[0], -phase[1], atol=1e-6, rtol=0):
        raise ValueError("the b02b0 path requires opposite phase-encoding vectors")
    if np.any(acquisition[:, 3] <= 0):
        raise ValueError("total readout times must be positive")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(not math.isfinite(value) or value <= 0 for value in zooms):
        raise ValueError("input voxel sizes must be finite and positive")
    return image, data, acquisition, int(axes[0]), zooms


def _average_pool(images, factor):
    if factor == 1:
        return images
    shape = tuple(size // factor for size in images.shape[1:])
    output = images.new_zeros((images.shape[0], *shape))
    # TopupScan adds x inside y inside z in a float volume.
    for z in range(factor):
        for y in range(factor):
            for x in range(factor):
                output += images[:, x::factor, y::factor, z::factor]
    return output / float(factor ** 3)


def _gaussian_blur(images, fwhm_mm, voxel_sizes):
    """NEWIMAGE periodic smoothing with float kernel initialization."""
    if fwhm_mm <= 0:
        return images
    output = images
    sigma_mm = np.float32(float(fwhm_mm) / math.sqrt(8.0 * math.log(2.0)))
    for axis, voxel_size in enumerate(voxel_sizes, 1):
        sigma = np.float32(sigma_mm / np.float32(voxel_size))
        radius = int(float(sigma) - 0.001) * 2 + 3
        raw, total = [], np.float32(0)
        for offset in range(-radius, radius + 1):
            value = np.float32(math.exp(-(offset * offset) / (2.0 * float(sigma) * float(sigma)))) if sigma > 1e-6 else np.float32(offset == 0)
            raw.append(float(value))
            total = np.float32(total + value)
        inverse_total = 1.0 / float(total)
        kernel = tuple(value * inverse_total for value in raw)
        accumulated = torch.zeros_like(output)
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            # source and result are float, the ColumnVector kernel is double.
            # Each compound += rounds its double product/sum into float.
            accumulated = (accumulated.double() + torch.roll(output, -offset, dims=axis).double() * weight).to(output.dtype)
        output = accumulated
    return output


def _grid(shape, *, device, dtype):
    axes = tuple(torch.arange(size, device=device, dtype=dtype) for size in shape)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


def _rigid_coordinates(grid, parameters, voxel_sizes):
    """Apply FSL TOPUP's inverse rigid pull in scaled-mm coordinates."""
    sizes = torch.as_tensor(
        voxel_sizes, device=grid.device, dtype=grid.dtype
    )
    translation = parameters[:3]
    rx, ry, rz = parameters[3:]
    one = torch.ones((), device=grid.device, dtype=grid.dtype)
    zero = torch.zeros((), device=grid.device, dtype=grid.dtype)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    mx = torch.stack((one, zero, zero, zero, cx, sx, zero, -sx, cx)).reshape(3, 3)
    my = torch.stack((cy, zero, -sy, zero, one, zero, sy, zero, cy)).reshape(3, 3)
    mz = torch.stack((cz, sz, zero, -sz, cz, zero, zero, zero, one)).reshape(3, 3)
    rotation = mx @ my @ mz
    centre = (
        torch.as_tensor(grid.shape[1:], device=grid.device, dtype=grid.dtype) - 1
    ) * sizes / 2
    offset = centre - rotation @ centre + translation
    target_mm = grid.reshape(3, -1) * sizes[:, None]
    source_mm = rotation.T @ (target_mm - offset[:, None])
    return (source_mm / sizes[:, None]).reshape_as(grid)


def _sample(volume, coordinates):
    shape = coordinates.shape[1:]
    valid = torch.ones(shape, dtype=torch.bool, device=volume.device)
    normalized = []
    for coordinate, size in zip(coordinates, volume.shape):
        valid &= (coordinate >= 0) & (coordinate <= size - 1)
        normalized.append(2 * coordinate / max(size - 1, 1) - 1)
    sampling_grid = torch.stack(
        (normalized[0], normalized[1], normalized[2]), dim=-1
    )[None]
    source = volume.permute(2, 1, 0)[None, None]
    sampled = F.grid_sample(
        source,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    return sampled, valid


def _cubic_spline_coefficients(images):
    """Periodic cubic prefilter with FSL's 1e-8 initialization and rounding.

    Each axis is a double recurrence followed by a float image write. Columns
    are evaluated together on the CPU; input/output images stay float32.
    """
    values = images.detach().cpu().numpy().astype(np.float32, copy=True)
    pole = math.sqrt(3.0) - 2.0
    horizon = int(math.log(1e-8) / math.log(abs(pole)) + 1.5)
    for axis in (1, 2, 3):
        size = values.shape[axis]
        if size == 1:
            continue
        columns = np.moveaxis(values, axis, 0).astype(np.float64, copy=True)
        count = min(size, horizon)
        initial = columns[0].copy()
        power = pole
        for index in range(1, count):
            initial += power * columns[size - index]
            power *= pole
        columns[0] = initial
        for index in range(1, size):
            columns[index] += pole * columns[index - 1]
        initial = pole * columns[-1]
        power = pole * pole
        for index in range(1, count):
            initial += power * columns[index - 1]
            power *= pole
        columns[-1] = initial / (power - 1.0)
        for index in range(size - 2, -1, -1):
            columns[index] = pole * (columns[index + 1] - columns[index])
        values = np.moveaxis((columns * 6).astype(np.float32), 0, axis)
    return torch.as_tensor(values.copy(), dtype=images.dtype, device=images.device)


def _fsl_frame_mask(valid, phase_encode_axis):
    """Apply TopupScan's geometric validity and non-PE one-voxel frame."""
    mask = valid.clone()
    mask[:, :, (0, -1)] = False
    if phase_encode_axis == 0:
        mask[:, (0, -1), :] = False
    else:
        mask[(0, -1), :, :] = False
    return mask


def _sample_cubic_with_derivatives(coefficients, coordinates, phase_encode_axis):
    """FSL spline interpolation: float coordinates, double taps, float outputs.

    The image coefficients are fixed. NEWIMAGE's Splinterpolator accumulates
    the x taps inside y inside z; this differs from the legacy float sampler.
    """
    if coefficients.is_cuda:
        from ._sampling_cuda import sample_cubic_with_derivatives_cuda
        return sample_cubic_with_derivatives_cuda(
            coefficients, coordinates, phase_encode_axis, official_precision=True
        )
    shape = coordinates.shape[1:]
    flat = coordinates.reshape(3, -1)
    valid = torch.ones(flat.shape[1], dtype=torch.bool, device=coefficients.device)
    indices, weights, derivatives = [], [], []
    offsets = torch.arange(4, device=coefficients.device, dtype=torch.long)
    for axis, (coordinate, size) in enumerate(zip(flat, coefficients.shape)):
        if axis != phase_encode_axis:
            valid &= (coordinate.double() + 1e-8 >= 0) & (coordinate.double() <= size - 1 + 1e-8)
        coordinate = coordinate.to(torch.float64)
        nearest = torch.trunc(coordinate + 0.5).long()
        first = torch.where(nearest < coordinate, nearest - 1, nearest - 2)
        taps = first[:, None] + offsets
        distance = coordinate[:, None] - taps
        absolute = distance.abs()
        remaining = 2 - absolute
        weights.append(torch.where(absolute < 1,
            2 / 3 + 0.5 * absolute * absolute * (absolute - 2),
            torch.where(absolute < 2, (1 / 6) * (remaining * remaining * remaining), 0.0)))
        sign = torch.where(distance < 0, -1.0, 1.0)
        derivatives.append(torch.where(absolute < 1,
            sign * (1.5 * absolute * absolute - 2 * absolute),
            torch.where(absolute < 2, (sign * -0.5) * remaining * remaining, 0.0)))
        indices.append(torch.remainder(taps, size))
    linear = coefficients.reshape(-1)
    sy, sz = coefficients.shape[1:]
    output = torch.zeros(flat.shape[1], dtype=torch.float64, device=coefficients.device)
    gradient = torch.zeros((3, flat.shape[1]), dtype=torch.float64, device=coefficients.device)
    for iz in range(4):
        for iy in range(4):
            yz = weights[2][:, iz] * weights[1][:, iy]
            dyz = weights[2][:, iz] * derivatives[1][:, iy]
            ydz = derivatives[2][:, iz] * weights[1][:, iy]
            for ix in range(4):
                address = indices[0][:, ix] * (sy * sz) + indices[1][:, iy] * sz + indices[2][:, iz]
                value = linear[address].to(torch.float64)
                wx = weights[0][:, ix]
                output += value * wx * yz
                gradient[0] += value * derivatives[0][:, ix] * yz
                gradient[1] += value * wx * dyz
                gradient[2] += value * wx * ydz
    return (output.to(coefficients.dtype).reshape(shape),
            gradient.to(coefficients.dtype).reshape_as(coordinates), valid.reshape(shape))


def _regrid_images(images, voxel_sizes, maximum_subsampling, phase_encode_axis):
    """FSL default --regrid=1: enlarge the source grid by max(subsamp).

    The output/field Target retains its original grid. ReGrid's source voxel
    sizes are stored as floats, and interpolation receives float coordinates.
    """
    original_shape = tuple(int(value) for value in images.shape[1:])
    shape = tuple(value + int(maximum_subsampling) for value in original_shape)
    voxels = tuple(float(np.float32((float(np.float32((size - 1) * np.float32(voxel))) - 1e-6) / new_size))
                   for size, voxel, new_size in zip(original_shape, voxel_sizes, shape))
    grid = _grid(shape, device=images.device, dtype=torch.float32)
    # ReGrid stores xfac after a float division, then i*xfac in double.
    # Its y/z expressions multiply and divide in float before assignment.
    xfactor = np.float32(np.float32(voxels[0]) / np.float32(voxel_sizes[0]))
    coordinates = torch.stack((
        (grid[0].double() * float(xfactor)).float(),
        (grid[1] * np.float32(voxels[1])) / np.float32(voxel_sizes[1]),
        (grid[2] * np.float32(voxels[2])) / np.float32(voxel_sizes[2]),
    ))
    coefficients = _cubic_spline_coefficients(images)
    resampled = []
    for image in coefficients:
        if images.is_cuda:
            from ._sampling_cuda import sample_cubic_cuda
            value, _ = sample_cubic_cuda(image, coordinates, phase_encode_axis, official_precision=True)
        else:
            value, _, _ = _sample_cubic_with_derivatives(image, coordinates, phase_encode_axis)
        resampled.append(value)
    return torch.stack(resampled), voxels


def _rigid_pull_matrix(parameters, shape, voxel_sizes, *, target_voxel_sizes=None):
    """Double inverse rigid matrix in voxel coordinates, centred on this level."""
    # Unlike the legacy helper, this explicitly forms the level's matrix
    # before the image sampler casts its entries to float.
    sizes = parameters.new_tensor(voxel_sizes)
    target_sizes = sizes if target_voxel_sizes is None else parameters.new_tensor(target_voxel_sizes)
    rx, ry, rz = parameters[3:]
    one, zero = parameters.new_ones(()), parameters.new_zeros(())
    def rotation_terms(angle):
        # MISCMATHS make_rot stores theta=norm(angle) in float, while the
        # single-axis basis is angle/double(theta). Preserve that rounding.
        theta = angle.abs().float().double()
        small = theta < 1e-8
        axis = angle / theta.clamp_min(1e-8)
        cosine = torch.where(small, one, torch.cos(theta))
        sine = torch.where(small, zero, torch.sin(theta) * torch.where(angle < 0, -one, one))
        axial = torch.where(small, one, axis.square())
        return cosine, sine, axial
    cx, sx, ax = rotation_terms(rx)
    cy, sy, ay = rotation_terms(ry)
    cz, sz, az = rotation_terms(rz)
    mx = torch.stack((ax, zero, zero, zero, cx, sx, zero, -sx, cx)).reshape(3, 3)
    my = torch.stack((cy, zero, -sy, zero, ay, zero, sy, zero, cy)).reshape(3, 3)
    mz = torch.stack((cz, sz, zero, -sz, cz, zero, zero, zero, az)).reshape(3, 3)
    rotation = mx @ my @ mz
    centre = ((parameters.new_tensor(shape).float() - 1) * sizes.float()).double() / 2
    offset = centre - rotation @ centre + parameters[:3]
    inverse = torch.linalg.inv(rotation)
    matrix = parameters.new_zeros((3, 4))
    matrix[:, :3] = inverse * target_sizes[None] / sizes[:, None]
    matrix[:, 3] = -(inverse @ offset) / sizes
    return matrix


def _matrix_coordinates(grid, matrix):
    # general_transform casts matrix entries to float, then performs these
    # explicit x, y, z contractions; avoid a reassociated GEMM/TF32 here.
    return torch.stack(tuple(
        ((grid[0] * row[0] + grid[1] * row[1]) + grid[2] * row[2]) + row[3]
        for row in matrix
    ))


def _transfer_field(coefficients, old_shape, old_spacing, new_shape,
                    new_spacing, new_voxels, *, step=1.0):
    """Sample the old spline directly and apply splinefield::Set's fit.

    SubSample uses start=(new_ss/old_ss-1)/2. SetWarpResolution uses integer
    positions. Neither operation interpolates an already sampled dense field.
    """
    start = (float(step) - 1.0) / 2.0
    positions = tuple(
        torch.arange(size, device=coefficients.device, dtype=torch.float64) * step + start
        for size in new_shape
    )
    bases = tuple(cubic_bspline_basis(position, spacing, controls)
                  for position, spacing, controls in zip(positions, old_spacing, coefficients.shape))
    dense = expand_coefficients(coefficients[None], bases).to(torch.float32)
    return fit_field_coefficients(dense, new_spacing, new_voxels, dtype=torch.float64)[0]


class _TOPUPLevel:
    """Cached b02b0 geometry with analytical gradient and matrix-free GN."""

    def __init__(self, interpolation_images, shape, spacing, voxel_sizes,
                 acquisition, pe_axis, factor, regularization, movement,
                 *, sampling_voxel_sizes=None):
        self.images = interpolation_images
        self.shape, self.spacing = tuple(shape), tuple(spacing)
        self.voxel_sizes, self.pe_axis = tuple(voxel_sizes), int(pe_axis)
        self.source_shape = tuple(int(value) for value in interpolation_images.shape[1:])
        self.source_voxel_sizes = tuple(voxel_sizes if sampling_voxel_sizes is None else sampling_voxel_sizes)
        self.factor, self.regularization = int(factor), float(regularization)
        self.device = interpolation_images.device
        self.acquisition = np.asarray(acquisition, dtype=np.float64)
        self.movement = movement.clone().to(torch.float64)
        self.movement_indices = tuple(index for index in range(6) if index != pe_axis)
        self.control_shape = fsl_control_shape(shape, spacing)
        self.coefficient_count = math.prod(self.control_shape)
        self.grid = _grid(shape, device=self.device, dtype=torch.float32)
        self.grid_double = self.grid.to(torch.float64)
        self.bases = spline_bases(shape, spacing, (1, 1, 1), device=self.device, dtype=torch.float64)
        derivatives = tuple(1 if axis == pe_axis else 0 for axis in range(3))
        self.derivative_bases = spline_bases(shape, spacing, (1, 1, 1),
                                           device=self.device, dtype=torch.float64, derivatives=derivatives)
        self.bending = BendingOperator(shape, spacing, voxel_sizes, device=self.device,
                                       dtype=torch.float64, execution="optimized")
        self.latest_ssd = 0.0
        self._cached_parameters = None
        self._cached_state = None

    def parameters(self, coefficients, estimate_movement):
        values = coefficients.reshape(-1)
        if estimate_movement:
            values = torch.cat((values, self.movement[1, list(self.movement_indices)]))
        return values.clone()

    def decode(self, parameters):
        coefficients = parameters[:self.coefficient_count].reshape(self.control_shape)
        movement = self.movement.clone()
        if parameters.numel() > self.coefficient_count:
            movement[1, list(self.movement_indices)] = parameters[self.coefficient_count:]
        return coefficients, movement

    def state(self, parameters):
        if parameters is self._cached_parameters:
            return self._cached_state
        coefficients, movement = self.decode(parameters)
        field = expand_coefficients(coefficients[None], self.bases)[0].to(torch.float32)
        derivative = expand_coefficients(coefficients[None], self.derivative_bases)[0]
        corrected, jacobians, masks, alphas, betas = [], [], [], [], []
        spatial, sampling_coordinates = [], []
        for scan in range(2):
            row = self.acquisition[scan]
            sign_readout = float(row[3] * row[self.pe_axis])
            matrix = _rigid_pull_matrix(movement[scan], self.source_shape, self.source_voxel_sizes,
                                        target_voxel_sizes=self.voxel_sizes)
            # iA maps output voxel coordinates into physical mm. iM maps
            # those mm (after displacement) into source sampling voxels.
            physical_matrix = matrix * matrix.new_tensor(self.source_voxel_sizes)[:, None]
            coordinates_mm = _matrix_coordinates(self.grid, physical_matrix.to(torch.float32))
            # The displacement is first rounded in physical mm and then
            # transformed into the level's sampling voxel coordinates.
            original_voxel = self.voxel_sizes[self.pe_axis] / self.factor
            displacement_mm = field * np.float32(sign_readout * original_voxel)
            coordinates_mm[self.pe_axis] += displacement_mm
            inverse_voxels = coordinates_mm.new_tensor(tuple(np.float32(1 / value) for value in self.source_voxel_sizes))
            coordinates = coordinates_mm * inverse_voxels[:, None, None, None]
            sampling_coordinates.append(coordinates)
            sampled, gradient, valid = _sample_cubic_with_derivatives(self.images[scan], coordinates, self.pe_axis)
            jacobian = (1 + derivative * (sign_readout / self.factor)).to(torch.float32)
            corrected.append(sampled * jacobian)
            jacobians.append(jacobian)
            masks.append(_fsl_frame_mask(valid, self.pe_axis))
            alpha_scale = np.float32(sign_readout * original_voxel / self.source_voxel_sizes[self.pe_axis])
            beta_scale = np.float32(sign_readout / self.factor)
            alphas.append((gradient[self.pe_axis] * alpha_scale) * jacobian)
            betas.append(sampled * beta_scale)
            spatial.append(gradient)
        corrected = torch.stack(corrected)
        mask = masks[0] & masks[1]
        count = int(mask.sum())
        if count == 0:
            raise ValueError("TOPUP has no geometrically valid common voxels")
        mean = corrected.mean(0)
        diff = mean[None] - corrected
        ssd = diff[:, mask].square().sum(dtype=torch.float64) / count
        state = {"field": field, "derivative": derivative, "corrected": corrected,
                 "jacobians": torch.stack(jacobians), "mask": mask, "voxels": count,
                 "diff": diff, "ssd": ssd, "alpha": torch.stack(alphas), "beta": torch.stack(betas),
                 "spatial": spatial, "movement": movement, "coefficients": coefficients}
        state["coordinates"] = tuple(sampling_coordinates)
        self._cached_parameters, self._cached_state = parameters, state
        return state

    def regularization_weight(self):
        return self.regularization * self.latest_ssd

    def energy(self, coefficients):
        return (coefficients * self.bending.normal(coefficients[None])[0]).sum()

    def cost(self, parameters):
        state = self.state(parameters)
        self.latest_ssd = float(state["ssd"])
        return state["ssd"] + self.regularization_weight() * self.energy(state["coefficients"])

    def movement_derivatives(self, state):
        result = []
        movement = state["movement"][1]
        base = _rigid_pull_matrix(movement, self.source_shape, self.source_voxel_sizes,
                                  target_voxel_sizes=self.voxel_sizes)
        base_coordinates = _matrix_coordinates(self.grid_double, base)
        for index in self.movement_indices:
            tiny = 1e-5 if index >= 3 else 1e-4
            perturbed = movement.clone()
            perturbed[index] += tiny
            alternative = _matrix_coordinates(self.grid_double,
                                               _rigid_pull_matrix(perturbed, self.source_shape, self.source_voxel_sizes,
                                                                   target_voxel_sizes=self.voxel_sizes))
            # Upstream first casts the summed spatial product to float,
            # divides by tiny, and finally multiplies the Jacobian.
            delta = alternative - base_coordinates
            value = (delta * state["spatial"][1].to(torch.float64)).sum(0).to(torch.float32)
            result.append((value / tiny) * state["jacobians"][1])
        return torch.stack(result)

    def gradient(self, parameters):
        state = self.state(parameters)
        alpha_diff = state["alpha"].mean(0)[None] - state["alpha"]
        beta_diff = state["beta"].mean(0)[None] - state["beta"]
        mask = state["mask"]
        weight0 = (alpha_diff * state["diff"]).sum(0) * mask
        weight1 = (beta_diff * state["diff"]).sum(0) * mask
        scale = 2.0 / state["voxels"]
        value = scale * (adjoint_field(weight0.double()[None], self.bases)[0] +
                         adjoint_field(weight1.double()[None], self.derivative_bases)[0])
        value += 2 * self.regularization_weight() * self.bending.normal(state["coefficients"][None])[0]
        value = value.reshape(-1)
        if parameters.numel() > self.coefficient_count:
            motion = self.movement_derivatives(state)
            gradient = -scale * ((motion * state["diff"][1])[:, mask]).sum(1, dtype=torch.float64)
            value = torch.cat((value, gradient))
        return value

    def hessian(self, parameters):
        state = self.state(parameters)
        alpha_diff = state["alpha"].mean(0)[None] - state["alpha"]
        beta_diff = state["beta"].mean(0)[None] - state["beta"]
        mask = state["mask"]
        aa = ((alpha_diff * alpha_diff).sum(0) * mask).double()
        ab = ((alpha_diff * beta_diff).sum(0) * mask).double()
        bb = ((beta_diff * beta_diff).sum(0) * mask).double()
        scale, weight = 2.0 / state["voxels"], self.regularization_weight()
        # diag(B.T diag(ab) D) uses products of the two 1-D bases.
        cross_bases = tuple(b * d for b, d in zip(self.bases, self.derivative_bases))
        diagonal = scale * (design_diagonal(aa, self.bases) +
                            2 * adjoint_field(ab[None], cross_bases)[0] +
                            design_diagonal(bb, self.derivative_bases))
        diagonal += 2 * weight * self.bending.diagonal()
        interaction, motion_hessian = None, None
        if parameters.numel() > self.coefficient_count:
            motion = self.movement_derivatives(state)
            columns = []
            for image in motion:
                w0 = ((image * alpha_diff[1]) * mask).double()
                w1 = ((image * beta_diff[1]) * mask).double()
                columns.append(-scale * (adjoint_field(w0[None], self.bases)[0] +
                                          adjoint_field(w1[None], self.derivative_bases)[0]).reshape(-1))
            interaction = torch.stack(columns, dim=1)
            motion_hessian = parameters.new_empty((5, 5))
            for row in range(5):
                for column in range(row, 5):
                    value = ((motion[row] * motion[column])[mask]).sum(dtype=torch.float64) / state["voxels"]
                    motion_hessian[row, column] = motion_hessian[column, row] = value
            diagonal = torch.cat((diagonal.reshape(-1), motion_hessian.diagonal()))
        else:
            diagonal = diagonal.reshape(-1)

        def matvec(vector):
            coefficients = vector[:self.coefficient_count].reshape(self.control_shape)
            field = expand_coefficients(coefficients[None], self.bases)[0]
            derivative = expand_coefficients(coefficients[None], self.derivative_bases)[0]
            value = scale * (adjoint_field((aa * field + ab * derivative)[None], self.bases)[0] +
                             adjoint_field((ab * field + bb * derivative)[None], self.derivative_bases)[0])
            value += 2 * weight * self.bending.normal(coefficients[None])[0]
            value = value.reshape(-1)
            if interaction is not None:
                motion = vector[self.coefficient_count:]
                value += interaction @ motion
                value = torch.cat((value, interaction.T @ vector[:self.coefficient_count] + motion_hessian @ motion))
            return value
        return matvec, diagonal


def _levenberg_marquardt(problem, initial, *, max_iterations):
    """FSL LM: joint field/movement GN, accepted-step budget, diagonal damping."""
    parameters = initial.clone()
    cost = float(problem.cost(parameters))
    damping, accepted, attempts, converged = 0.1, 0, 0, False
    termination = "maximum_accepted_iterations"
    history = []
    success = True
    while accepted < max_iterations:
        if success:
            gradient = problem.gradient(parameters)
            matvec, diagonal = problem.hessian(parameters)
        current_damping = damping
        step, report = preconditioned_conjugate_gradient(
            lambda vector: matvec(vector) + current_damping * diagonal * vector,
            -gradient, diagonal=(1 + current_damping) * diagonal,
            tolerance=1e-3, max_iterations=500,
        )
        attempts += 1
        candidate = parameters + step
        # SpMat::SolveForx warns but returns its approximate CG solution
        # when the tolerance was not attained. LM still evaluates that step.
        candidate_cost = float("inf")
        if bool(torch.isfinite(candidate).all()):
            try:
                candidate_cost = float(problem.cost(candidate))
            except ValueError as error:
                if str(error) != "TOPUP has no geometrically valid common voxels":
                    raise
        success = math.isfinite(candidate_cost) and candidate_cost < cost
        history.append({"attempt": attempts, "accepted": success,
                        "cost": candidate_cost if math.isfinite(candidate_cost) else None,
                        "lambda": damping, "pcg_iterations": report.iterations,
                        "pcg_converged": report.converged,
                        "pcg_relative_residual": report.relative_residual})
        if success:
            old_cost, parameters, cost = cost, candidate, candidate_cost
            accepted += 1
            damping /= 10
            if 2 * abs(old_cost - cost) <= 1e-8 * (abs(old_cost) + abs(cost) + 1e-16):
                converged = True
                termination = "fractional_cost_change"
                break
        else:
            damping *= 10
            if damping > 1e20:
                converged = True
                termination = "lambda_limit"
                break
    return parameters, {"optimizer": "joint LM", "iterations": accepted,
                        "accepted_iterations": accepted, "attempts": attempts,
                        "converged": converged, "cost": cost,
                        "termination": termination,
                        "lambda_final": damping, "history": history}


def _decode_fsl_coefficient_image(image, *, device):
    """Decode intent-2016 geometry without treating its qform as an affine."""
    if int(image.header["intent_code"]) != 2016 or len(image.shape) != 3:
        raise ValueError("expected a 3D TOPUP cubic coefficient image (intent 2016)")
    shape_values = tuple(float(image.header[name]) for name in ("qoffset_x", "qoffset_y", "qoffset_z"))
    spacing_values = tuple(float(value) for value in image.header["pixdim"][1:4])
    if any(not math.isfinite(value) or value < 1 or abs(value-round(value)) > 1e-5
           for value in (*shape_values, *spacing_values)):
        raise ValueError("invalid TOPUP field shape or knot spacing in header")
    shape, spacing = tuple(round(v) for v in shape_values), tuple(round(v) for v in spacing_values)
    voxels = tuple(float(image.header[name]) for name in ("intent_p1", "intent_p2", "intent_p3"))
    if any(not math.isfinite(value) or value <= 0 for value in voxels):
        raise ValueError("invalid TOPUP field voxel sizes in header")
    values = np.asarray(image.dataobj, dtype=np.float64)
    if values.shape != fsl_control_shape(shape, spacing) or not np.isfinite(values).all():
        raise ValueError("TOPUP coefficient grid does not match its encoded geometry")
    return torch.as_tensor(values.copy(), dtype=torch.float64, device=device), shape, spacing, voxels


def _render_fixed_topup(imain, datain, fieldcoef, movpar, *, device="cpu"):
    """Private same-input oracle diagnostic; estimate no field or movement.

    Returns arrays in input storage order for field/iout, canonical FSL order
    for Analyze-style Jacobians, and masks plus numerical cost diagnostics.
    It never writes images or calls an external executable.
    """
    reference, values, acquisition, pe_axis, voxels = _load_inputs(imain, datain)
    coefficients, shape, spacing, encoded_voxels = _decode_fsl_coefficient_image(nib.load(os.fspath(fieldcoef)), device=device)
    if shape != values.shape[:3] or not np.allclose(voxels, encoded_voxels, atol=1e-5, rtol=0):
        raise ValueError("fixed coefficients and input must have identical field geometry")
    movement = np.loadtxt(os.fspath(movpar), ndmin=2)
    if movement.shape != (2, 6) or not np.isfinite(movement).all():
        raise ValueError("fixed movpar must contain two finite six-column rows")
    flipped = np.linalg.det(reference.affine[:3, :3]) > 0
    if flipped:
        values = values[::-1].copy()
    means = values.mean(axis=(0, 1, 2), dtype=np.float64)
    if np.any(np.abs(means) < np.finfo(np.float32).tiny):
        raise ValueError("an input volume has zero mean intensity")
    images = torch.as_tensor(np.moveaxis(values * (100.0 / means).astype(np.float32)[None, None, None, :], -1, 0).copy(),
                             dtype=torch.float32, device=device)
    regridded, source_voxels = _regrid_images(images, voxels, 2, pe_axis)
    problem = _TOPUPLevel(_cubic_spline_coefficients(regridded), shape, spacing, voxels,
                          acquisition, pe_axis, 1, 0, torch.as_tensor(movement, dtype=torch.float64, device=device),
                          sampling_voxel_sizes=source_voxels)
    parameters = problem.parameters(coefficients, False)
    cost = float(problem.cost(parameters))
    state = problem.state(parameters)
    corrected = state["corrected"].clone()
    corrected[:, ~state["mask"]] = 0
    corrected *= float(means.mean()) / 100
    arrays = {"field_hz": state["field"].cpu().numpy(),
              "corrected": np.moveaxis(corrected.cpu().numpy(), 0, -1),
              "jacobians": np.moveaxis(state["jacobians"].cpu().numpy(), 0, -1),
              "common_mask": state["mask"].cpu().numpy(),
              "ssd": cost, "valid_voxels": state["voxels"], "canonical_x_flip": bool(flipped),
              "source_shape": list(regridded.shape[1:]), "source_voxel_sizes": list(source_voxels)}
    if flipped:
        for key in ("field_hz", "corrected", "common_mask"):
            arrays[key] = arrays[key][::-1].copy()
    return arrays


class TorchTOPUP:
    """Estimate one TOPUP field and apply it to a b02b0 acquisition pair."""

    def __init__(self, device=None, *, config=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.config = TOPUPConfig() if config is None else config

    def __call__(self, imain, datain):
        reference, values, acquisition, pe_axis, voxel_sizes = _load_inputs(imain, datain)
        # FSL NEWIMAGE estimates in radiological storage coordinates.
        canonical_x_flip = bool(np.linalg.det(reference.affine[:3, :3]) > 0)
        if canonical_x_flip:
            values = values[::-1].copy()
        if any(size % factor for factor in self.config.subsampling for size in values.shape[:3]):
            raise ValueError("TOPUP subsampling factors must divide each spatial input dimension")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        means = values.mean(axis=(0, 1, 2), dtype=np.float64)
        if np.any(np.abs(means) < np.finfo(np.float32).tiny):
            raise ValueError("an input volume has zero mean intensity")
        global_mean = float(means.mean())
        normalized = values * (100.0 / means).astype(np.float32)[None, None, None, :]
        images = torch.as_tensor(np.moveaxis(normalized, -1, 0).copy(),
                                 dtype=torch.float32, device=self.device)
        regridded_images, source_voxel_sizes = _regrid_images(
            images, voxel_sizes, max(self.config.subsampling), pe_axis)
        movement = torch.zeros((2, 6), dtype=torch.float64, device=self.device)
        coefficients = None
        previous_factor, previous_warp, previous_shape, previous_spacing = None, None, None, None
        interpolation_cache = {}
        level_reports = []
        for level, (warp_mm, factor, fwhm, iterations, regularization) in enumerate(zip(
                self.config.warp_resolution_mm, self.config.subsampling,
                self.config.fwhm_mm, self.config.maximum_iterations, self.config.regularization), 1):
            level_started = time.perf_counter()
            level_voxels = tuple(value * factor for value in voxel_sizes)
            level_source_voxels = tuple(float(np.float32(value * factor)) for value in source_voxel_sizes)
            shape = tuple(int(size // factor) for size in images.shape[1:])
            spacing = tuple(max(1, int(math.floor(warp_mm / size + 0.5))) for size in level_voxels)
            if coefficients is None:
                coefficients = torch.zeros(fsl_control_shape(shape, spacing),
                                           dtype=torch.float64, device=self.device)
            else:
                # Match official order: SubSample first using the OLD warp
                # resolution, then SetWarpResolution. Same grids are not refit.
                if factor != previous_factor:
                    interim_spacing = tuple(max(1, int(math.floor(previous_warp / size + 0.5))) for size in level_voxels)
                    coefficients = _transfer_field(coefficients, previous_shape, previous_spacing,
                                                   shape, interim_spacing, level_voxels,
                                                   step=factor / previous_factor)
                    previous_shape, previous_spacing = shape, interim_spacing
                if warp_mm != previous_warp:
                    coefficients = _transfer_field(coefficients, previous_shape, previous_spacing,
                                                   shape, spacing, level_voxels)
            cache_key = (int(factor), float(fwhm))
            if cache_key not in interpolation_cache:
                level_images = _gaussian_blur(_average_pool(regridded_images, factor), fwhm, level_source_voxels)
                interpolation_cache[cache_key] = _cubic_spline_coefficients(level_images)
            problem = _TOPUPLevel(interpolation_cache[cache_key], shape, spacing, level_voxels,
                                  acquisition, pe_axis, factor, regularization, movement,
                                  sampling_voxel_sizes=level_source_voxels)
            parameters = problem.parameters(coefficients, estimate_movement=level <= 5)
            if level <= 5:
                parameters, optimizer_report = _levenberg_marquardt(problem, parameters,
                                                                    max_iterations=int(iterations))
            else:
                parameters, report = scaled_conjugate_gradient(problem.cost, problem.gradient, parameters,
                                                               max_iterations=int(iterations))
                optimizer_report = {"optimizer": "field SCG", "iterations": report.iterations,
                                    "accepted_iterations": report.accepted_iterations,
                                    "converged": report.converged, "cost": report.cost,
                                    "lambda_final": report.lambda_final, "history": list(report.history)}
            coefficients, movement = problem.decode(parameters)
            # Report the accepted state, not the last rejected/curvature probe.
            state = problem.state(parameters)
            energy = float(problem.energy(coefficients))
            level_reports.append({"level": level, "shape": shape, "knot_spacing": spacing,
                                  "ssd": float(state["ssd"]), "bending": energy,
                                  "voxels": state["voxels"], "regularization": float(regularization),
                                  "elapsed_seconds": time.perf_counter() - level_started, **optimizer_report})
            previous_factor, previous_warp = factor, float(warp_mm)
            previous_shape, previous_spacing = shape, spacing
            del problem
        full_shape = tuple(int(value) for value in images.shape[1:])
        final_spacing = tuple(max(1, int(math.floor(self.config.warp_resolution_mm[-1] / size + 0.5))) for size in voxel_sizes)
        if previous_factor != 1:
            coefficients = _transfer_field(coefficients, previous_shape, previous_spacing,
                                           full_shape, final_spacing, voxel_sizes,
                                           step=1.0 / previous_factor)
        if (1, 0.0) not in interpolation_cache:
            interpolation_cache[(1, 0.0)] = _cubic_spline_coefficients(regridded_images)
        final_problem = _TOPUPLevel(interpolation_cache[(1, 0.0)], full_shape, final_spacing,
                                    voxel_sizes, acquisition, pe_axis, 1, 0, movement,
                                    sampling_voxel_sizes=source_voxel_sizes)
        final_state = final_problem.state(final_problem.parameters(coefficients, False))
        field = final_state["field"]
        corrected = final_state["corrected"].clone()
        corrected[:, ~final_state["mask"]] = 0
        corrected *= global_mean / 100.0
        jacobians = final_state["jacobians"]
        field_np = field.cpu().numpy()
        corrected_np = np.moveaxis(corrected.cpu().numpy(), 0, -1)
        jacobian_np = jacobians.cpu().numpy()
        coefficient_np = coefficients.cpu().numpy().astype(np.float32)
        if canonical_x_flip:
            field_np, corrected_np = field_np[::-1].copy(), corrected_np[::-1].copy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None
        return TOPUPResult(
            field_hz=make_output_image(field_np, reference, intent=FSL_TOPUP_FIELD),
            corrected=make_output_image(corrected_np, reference),
            corrected_mean=make_output_image(corrected_np.mean(axis=3), reference),
            jacobians=tuple(make_topup_jacobian_image(value, reference) for value in jacobian_np),
            coefficients=make_topup_coefficient_image(coefficient_np, full_shape, voxel_sizes, final_spacing),
            movement_parameters=movement.cpu().numpy(),
            qc={"device": str(self.device), "dtype": "float32 images / float64 coefficients and solvers",
                "tf32": bool(self.device.type == "cuda"), "phase_encode_axis": pe_axis,
                "canonical_x_flip": canonical_x_flip,
                "regrid": True, "source_shape": list(regridded_images.shape[1:]),
                "source_voxel_sizes": list(source_voxel_sizes),
                "reference_implementation": f"FSL TOPUP {FSL_TOPUP_VERSION}",
                "optimizer": "joint field and 5-DOF motion LM (levels 1-5), field SCG (levels 6-9)",
                "spline_sampling": "float coordinates, double spline weights and accumulation, float outputs",
                "fsl_output_contract": True, "fsl_numerically_equivalent": False,
                "bitwise_equivalent": False, "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": peak_memory, "levels": level_reports})

    def run(
        self,
        imain,
        datain,
        *,
        out,
        fout=None,
        iout=None,
        jacout=None,
        overwrite=False,
    ):
        result = self(imain, datain)
        result.save(
            out=out,
            fout=fout,
            iout=iout,
            jacout=jacout,
            overwrite=overwrite,
        )
        return result


__all__ = ["FSL_TOPUP_COMMIT", "FSL_TOPUP_VERSION", "TOPUPConfig", "TOPUPResult", "TorchTOPUP"]
