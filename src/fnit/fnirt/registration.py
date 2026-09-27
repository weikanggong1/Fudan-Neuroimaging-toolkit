"""PyTorch implementation of the UKB/FSL GM FNIRT optimisation path.

This module ports the algorithmic structure of FSL FNIRT 2203.0: cubic
B-spline displacement fields, the four-level GM schedule, global-linear
reference intensity scaling, SSD-weighted bending regularisation and
matrix-free Gauss-Newton/Levenberg-Marquardt updates.  It does not claim FSL
numerical equivalence until the remaining end-to-end gates pass against the
FSL 6.0.7.4 binary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
import surfa as sf
import torch

from ..flirt.coordinates import voxel_to_fsl_scaled_mm, world_to_flirt_affine
from ..fast_vbm.synthmorph_backend import (
    _world_affine,
    pull_jacobian_determinants,
)
from .optimizer import preconditioned_conjugate_gradient
from .io import make_fsl_coefficient_image
from .spline import (
    BendingOperator,
    adjoint_field,
    design_diagonal,
    expand_coefficients,
    fit_field_coefficients,
    fsl_control_shape,
    spline_bases,
    zoom_coefficients,
)
from .topology import constrain_topology


FSL_SOURCE_VERSIONS = {
    "fnirt": "2203.0 (27f514a182b5972094e30d8ea79f4fad89cbf03d)",
    "basisfield": "2203.1 (9588bbe8eb8aa0939ddefd00df756aeb80d2305b)",
    "miscmaths": "2203.2 (7824d74cdfa9fb65de178f642c3c05e57c8c8868)",
    "newimage": "2203.11 (19e3ddd10138d8ea1394fd522fb0770435c61ddd)",
    "warpfns": "2203.0 (50ea45cb0b9661adba7844444cb38649ae44892b)",
}


_GOOD_FFT_SIZES = (
    4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 60, 64, 75,
    80, 90, 98, 100, 104, 108, 112, 117, 121, 125, 128, 135, 144, 145,
    150, 153, 160, 162, 169, 175, 180, 189, 192, 196, 200, 208, 216,
    225, 240, 245, 250, 256, 270, 272, 275, 288, 289, 294, 300, 304,
    315, 320, 325, 336, 338, 343, 350, 360, 363, 375, 384, 392, 400,
    405, 416, 420, 432, 441, 448, 450, 459, 475, 480, 484, 490, 500,
    504, 507, 512, 525, 550, 600, 650, 700, 750, 800, 850, 900, 950,
    1000, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120,
    6144, 7168, 8192,
)


def _good_fft_size(size):
    return next((value for value in _GOOD_FFT_SIZES if value >= size), int(size))


@dataclass(frozen=True)
class GMFNIRTConfig:
    """Parameters in FSL ``GM_2_MNI152GM_2mm.cnf``."""

    subsampling: tuple[int, ...] = (4, 2, 1, 1)
    maximum_iterations: tuple[int, ...] = (5, 5, 10, 5)
    input_fwhm_mm: tuple[float, ...] = (6.0, 4.0, 2.0, 2.0)
    reference_fwhm_mm: tuple[float, ...] = (4.0, 2.0, 0.0, 0.0)
    regularization: tuple[float, ...] = (150.0, 75.0, 50.0, 30.0)
    estimate_intensity: tuple[bool, ...] = (True, True, True, False)
    apply_reference_mask: tuple[bool, ...] = (False, False, False, True)
    warp_resolution_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    warp_resolution_schedule_mm: tuple[tuple[float, float, float], ...] | None = None
    jacobian_range: tuple[float, float] = (0.2, 5.0)
    ssd_weighted_lambda: bool = True

    def __post_init__(self):
        count = len(self.subsampling)
        schedules = (
            self.maximum_iterations,
            self.input_fwhm_mm,
            self.reference_fwhm_mm,
            self.regularization,
            self.estimate_intensity,
            self.apply_reference_mask,
        )
        if count == 0 or any(len(schedule) != count for schedule in schedules):
            raise ValueError("all FNIRT schedules must have the same non-zero length")
        if any(value not in (1, 2, 4, 8, 16) for value in self.subsampling):
            raise ValueError("FNIRT subsampling factors must be powers of two")
        if any(value < 0 for value in self.maximum_iterations):
            raise ValueError("maximum iterations must be non-negative")
        if any(value < 0 for value in self.regularization):
            raise ValueError("regularization must be non-negative")
        if self.warp_resolution_schedule_mm is not None:
            if len(self.warp_resolution_schedule_mm) != count:
                raise ValueError("warp resolution schedule must match the level count")
            if any(
                len(value) != 3 or any(item <= 0 for item in value)
                for value in self.warp_resolution_schedule_mm
            ):
                raise ValueError("each warp resolution must contain three positive values")


@dataclass
class TorchFNIRTResult:
    moved: sf.Volume
    pull_transform: sf.Warp
    full_pull_jacobian: sf.Volume
    nonlinear_jacobian: sf.Volume
    modulated_gm: sf.Volume
    affine_pull_determinant: float
    coefficients: np.ndarray
    coefficient_image: object
    qc: dict


def spm_like_mean(data):
    """Mean used by FSL FNIRT before both images are scaled to 100."""
    values = np.asarray(data, dtype=np.float32)
    # NEWIMAGE traverses x fastest and adds each float into a double scalar.
    # NumPy's reduction is pairwise and C-order, which changes the subsequent
    # float intensity scaling enough to perturb FNIRT's truncated CG path.
    ordered = np.ravel(values, order="F")
    first = float(np.add.accumulate(ordered, dtype=np.float64)[-1]) / ordered.size
    selected = ordered[ordered > 0.125 * first]
    if selected.size == 0:
        raise ValueError("FNIRT intensity normalization selected no voxels")
    return float(np.add.accumulate(selected, dtype=np.float64)[-1]) / selected.size


def _fsl_gaussian_blur(volume, fwhm_mm, voxel_sizes):
    """Separable zero-padded Gaussian used by ``newimage::smooth``."""
    if fwhm_mm <= 0:
        return volume
    result = volume
    sigma_mm = np.float32(
        float(fwhm_mm) / math.sqrt(8.0 * math.log(2.0))
    )
    for axis, voxel_size in enumerate(voxel_sizes):
        sigma = np.float32(sigma_mm / np.float32(voxel_size))
        radius = int(np.float32(sigma - np.float32(0.001))) * 2 + 3
        values = []
        total = np.float32(0.0)
        for offset in range(-radius, radius + 1):
            if sigma > np.float32(1e-6):
                value = np.float32(
                    math.exp(
                        -(offset * offset)
                        / (2.0 * float(sigma) * float(sigma))
                    )
                )
            else:
                value = np.float32(1.0 if offset == 0 else 0.0)
            values.append(value)
            total = np.float32(total + value)
        kernel = tuple(float(value) * (1.0 / float(total)) for value in values)

        convolved = torch.zeros_like(result)
        dimension = axis + 2
        length = result.shape[dimension]
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            source_start = max(0, offset)
            source_stop = min(length, length + offset)
            if source_start >= source_stop:
                continue
            target_start = source_start - offset
            target_stop = source_stop - offset
            source_slice = [slice(None)] * result.ndim
            target_slice = [slice(None)] * result.ndim
            source_slice[dimension] = slice(source_start, source_stop)
            target_slice[dimension] = slice(target_start, target_stop)
            target = tuple(target_slice)
            product = result[tuple(source_slice)].to(torch.float64) * weight
            convolved[target] = (
                convolved[target].to(torch.float64) + product
            ).to(result.dtype)
        result = convolved
    return result


def _subsampled_size(size, factor):
    result = int(size)
    remaining = int(factor)
    while remaining > 1:
        result = result // 2 + 1
        remaining //= 2
    return result


def _level_positions(shape, stride, *, device, dtype):
    return tuple(
        torch.arange(
            _subsampled_size(size, stride), device=device, dtype=dtype
        )
        * int(stride)
        for size in shape
    )


def _take_integer_grid(volume, positions):
    shape = tuple(axis.numel() for axis in positions)
    output = volume.new_zeros(shape)
    valid_axes = tuple(axis < size for axis, size in zip(positions, volume.shape))
    counts = tuple(int(valid.sum()) for valid in valid_axes)
    if all(counts):
        source = tuple(axis[valid].to(torch.long) for axis, valid in zip(positions, valid_axes))
        output[: counts[0], : counts[1], : counts[2]] = volume[
            source[0][:, None, None], source[1][None, :, None], source[2][None, None, :]
        ]
    return output


def _trilinear_sample(volume, coordinates):
    """Trilinear values and exact piecewise voxel-coordinate derivatives."""
    if volume.ndim != 3 or coordinates.shape[0] != 3:
        raise ValueError("invalid trilinear input shapes")
    spatial_shape = coordinates.shape[1:]
    flat = coordinates.reshape(3, -1)
    valid = torch.ones(flat.shape[1], dtype=torch.bool, device=volume.device)
    lower = []
    upper = []
    fraction = []
    for coordinate, size in zip(flat, volume.shape):
        valid &= (coordinate >= 0) & (coordinate <= size - 1)
        lo = torch.floor(coordinate).to(torch.long).clamp(0, size - 1)
        hi = lo + 1
        lower.append(lo)
        upper.append(hi)
        fraction.append((coordinate - lo.to(coordinate.dtype)).clamp(0, 1))

    sx, sy, sz = volume.shape
    linear = volume.reshape(-1)

    def take(ix, iy, iz):
        inside = (
            (ix >= 0)
            & (ix < sx)
            & (iy >= 0)
            & (iy < sy)
            & (iz >= 0)
            & (iz < sz)
        )
        values = linear[
            ix.clamp(0, sx - 1) * (sy * sz)
            + iy.clamp(0, sy - 1) * sz
            + iz.clamp(0, sz - 1)
        ]
        return values * inside.to(volume.dtype)

    x0, y0, z0 = lower
    x1, y1, z1 = upper
    wx, wy, wz = fraction
    v000 = take(x0, y0, z0)
    v001 = take(x0, y0, z1)
    v010 = take(x0, y1, z0)
    v011 = take(x0, y1, z1)
    v100 = take(x1, y0, z0)
    v101 = take(x1, y0, z1)
    v110 = take(x1, y1, z0)
    v111 = take(x1, y1, z1)

    # Keep the operation order of newimage::volume<float>::interp3partial.
    # The intermediate float roundings affect FNIRT's truncated PCG path.
    one_minus_z = 1.0 - wz
    one_minus_y = 1.0 - wy
    tmp11 = one_minus_z * v000 + wz * v001
    tmp12 = one_minus_z * v010 + wz * v011
    tmp13 = one_minus_z * v100 + wz * v101
    tmp14 = one_minus_z * v110 + wz * v111
    derivative_x = one_minus_y * (tmp13 - tmp11) + wy * (tmp14 - tmp12)
    derivative_y = (1.0 - wx) * (tmp12 - tmp11) + wx * (tmp14 - tmp13)
    tmp11 = one_minus_y * v000 + wy * v010
    tmp12 = one_minus_y * v001 + wy * v011
    tmp13 = one_minus_y * v100 + wy * v110
    tmp14 = one_minus_y * v101 + wy * v111
    tmp21 = (1.0 - wx) * tmp11 + wx * tmp13
    tmp22 = (1.0 - wx) * tmp12 + wx * tmp14
    derivative_z = tmp22 - tmp21
    sampled = one_minus_z * tmp21 + wz * tmp22
    valid_float = valid.to(volume.dtype)
    sampled = (sampled * valid_float).reshape(spatial_shape)
    gradient = torch.stack(
        tuple(
            (value * valid_float).reshape(spatial_shape)
            for value in (derivative_x, derivative_y, derivative_z)
        )
    )
    return sampled, valid.reshape(spatial_shape), gradient


def _coordinate_grid(affine, positions):
    voxels = torch.stack(torch.meshgrid(*positions, indexing="ij"))
    return (
        torch.einsum("ij,jxyz->ixyz", affine[:3, :3], voxels)
        + affine[:3, 3, None, None, None]
    )


def _fsl_affine_grid(affine, shape):
    """Apply an affine in the scalar-float order used by warpfns."""
    matrix = affine.to(dtype=torch.float32)
    axes = tuple(
        torch.arange(size, device=matrix.device, dtype=torch.float32)
        for size in shape
    )
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    rows = []
    for row in range(3):
        value = x * matrix[row, 0]
        value = value + y * matrix[row, 1]
        value = value + z * matrix[row, 2]
        value = value + matrix[row, 3]
        rows.append(value)
    return torch.stack(rows)


def _fsl_displacement_coordinates(field, coordinate_affine, mm_to_voxel):
    """Coordinates from ``warpfns::displacements_no_iT``.

    ``coordinate_affine`` is ``inverse(FLIRT) @ target.sampling_mat`` in
    double precision.  warpfns casts its entries and those of
    ``source.sampling_mat().i()`` to float before evaluating the expressions.
    """
    source_mm = _fsl_affine_grid(coordinate_affine, field.shape[1:])
    source_mm = torch.stack(
        tuple(source_mm[axis] + field[axis] for axis in range(3))
    )
    matrix = mm_to_voxel.to(dtype=torch.float32)
    source_voxels = []
    for row in range(3):
        value = source_mm[0] * matrix[row, 0]
        value = value + source_mm[1] * matrix[row, 1]
        value = value + source_mm[2] * matrix[row, 2]
        value = value + matrix[row, 3]
        source_voxels.append(value)
    return torch.stack(source_voxels)


def _pack(coefficients, scale=None):
    # NEWMAT/FSL coefficient vectors run x fastest, followed by y and z.
    vector = coefficients.permute(0, 3, 2, 1).reshape(-1)
    if scale is not None:
        vector = torch.cat((vector, scale.reshape(1)))
    return vector


def _unpack(vector, coefficient_shape, includes_scale):
    count = math.prod(coefficient_shape)
    coefficients = vector[: 3 * count].reshape(
        3, coefficient_shape[2], coefficient_shape[1], coefficient_shape[0]
    ).permute(0, 3, 2, 1)
    scale = vector[3 * count] if includes_scale else None
    return coefficients, scale


def _spline_jacobian(
    coefficients,
    shape,
    knot_spacing,
    voxel_sizes,
    *,
    affine_pull_linear=None,
):
    """Evaluate FSL's analytic spline-field Jacobian determinant."""
    derivative_columns = []
    for axis in range(3):
        derivative = [0, 0, 0]
        derivative[axis] = 1
        derivative_bases = spline_bases(
            shape,
            knot_spacing,
            voxel_sizes,
            device=coefficients.device,
            dtype=coefficients.dtype,
            derivatives=tuple(derivative),
        )
        derivative_columns.append(
            expand_coefficients(coefficients, derivative_bases)
        )
    matrix = torch.stack(derivative_columns, dim=-1).movedim(0, -2)
    if affine_pull_linear is None:
        base = torch.eye(
            3, device=coefficients.device, dtype=coefficients.dtype
        )
    else:
        base = affine_pull_linear.to(
            device=coefficients.device, dtype=coefficients.dtype
        )
    return torch.linalg.det(matrix + base[None, None, None])


def _force_jacobian_range(
    coefficients,
    shape,
    knot_spacing,
    voxel_sizes,
    affine_pull,
    minimum,
    maximum,
    max_tries,
):
    """Apply FNIRT ``ForceJacobianRange`` and refit spline coefficients."""
    shape = tuple(int(value) for value in shape)
    voxel_sizes = tuple(float(value) for value in voxel_sizes)
    wide_shape = tuple(_good_fft_size(value) for value in shape)
    offsets = tuple((wide - size) // 2 for wide, size in zip(wide_shape, shape))
    device, dtype = coefficients.device, coefficients.dtype
    # FNIRT stores spline parameters in double precision, but materialises the
    # dense warp passed to warpfns as float volumes.
    field_dtype = torch.float32
    lower, upper = float(minimum), float(maximum)

    full_jacobian = _spline_jacobian(
        coefficients,
        shape,
        knot_spacing,
        voxel_sizes,
        affine_pull_linear=affine_pull[:3, :3],
    )
    jacobian_range = (float(full_jacobian.min()), float(full_jacobian.max()))
    last_range = jacobian_range
    calls = []
    while (
        (jacobian_range[0] < lower or jacobian_range[1] > upper)
        and len(calls) < int(max_tries)
    ):
        wide_positions = tuple(
            torch.arange(size, device=device, dtype=dtype) - offset
            for size, offset in zip(wide_shape, offsets)
        )
        wide_bases = spline_bases(
            shape,
            knot_spacing,
            voxel_sizes,
            device=device,
            dtype=dtype,
            positions=wide_positions,
        )
        residual = expand_coefficients(coefficients, wide_bases).to(field_dtype)

        axes = tuple(
            torch.arange(size, device=device, dtype=dtype) * voxel_size
            for size, voxel_size in zip(wide_shape, voxel_sizes)
        )
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
        affine_relative = (
            torch.einsum("ij,jxyz->ixyz", affine_pull[:3, :3], grid)
            - grid
            + affine_pull[:3, 3, None, None, None]
        )
        # FSL adds the double-precision affine term to each float defvol
        # element, then convertwarp_rel2abs adds float coordinates.  Keep
        # those two assignment roundings separate.
        relative_with_affine = (residual.to(dtype) + affine_relative).to(
            field_dtype
        )
        absolute = (relative_with_affine + grid.to(field_dtype)).to(field_dtype)
        constrained, inner_qc = constrain_topology(
            absolute, voxel_sizes, lower, upper
        )
        relative_with_affine = (constrained - grid.to(field_dtype)).to(
            field_dtype
        )
        # remove_affine_part again performs a double expression followed by
        # assignment to the float cvol passed to splinefield::Set.
        constrained_residual = (
            relative_with_affine.to(dtype) - affine_relative
        ).to(field_dtype)
        crop = tuple(
            slice(offset, offset + size) for offset, size in zip(offsets, shape)
        )
        coefficients = fit_field_coefficients(
            constrained_residual[(slice(None), *crop)],
            knot_spacing,
            voxel_sizes,
            dtype=dtype,
        )
        full_jacobian = _spline_jacobian(
            coefficients,
            shape,
            knot_spacing,
            voxel_sizes,
            affine_pull_linear=affine_pull[:3, :3],
        )
        jacobian_range = (
            float(full_jacobian.min()),
            float(full_jacobian.max()),
        )
        calls.append(
            {
                "wide_shape": list(wide_shape),
                "offsets": list(offsets),
                "range_before": list(last_range),
                "range_after": list(jacobian_range),
                "constrain_topology": inner_qc,
            }
        )
        if (
            abs(last_range[0] - jacobian_range[0]) < 1e-6
            or abs(last_range[1] - jacobian_range[1]) < 1e-6
        ):
            break
        last_range = jacobian_range

    return coefficients, full_jacobian, {
        "required": bool(calls),
        "calls": calls,
        "range": list(jacobian_range),
        "succeeded": jacobian_range[0] >= lower and jacobian_range[1] <= upper,
    }


class _LevelSystem:
    def __init__(
        self,
        moving,
        fixed,
        reference_mask,
        moving_fsl2vox,
        target_fsl,
        affine_pull,
        bases,
        bending,
        regularization,
        ssd_weighted_lambda,
        estimate_scale,
        coordinate_affine=None,
    ):
        self.moving = moving
        self.fixed = fixed
        self.reference_mask = reference_mask
        self.moving_fsl2vox = moving_fsl2vox
        self.target_fsl = target_fsl
        self.affine_pull = affine_pull
        self.coordinate_affine = coordinate_affine
        self.bases = bases
        self.bending = bending
        self.regularization = float(regularization)
        self.ssd_weighted_lambda = bool(ssd_weighted_lambda)
        self.estimate_scale = bool(estimate_scale)

    def evaluate(self, coefficients, scale, *, derivatives=False):
        # basisfield coefficients and spline arithmetic are double precision;
        # AsVolume then rounds the dense displacement to float before warping.
        field = expand_coefficients(coefficients, self.bases).to(
            self.moving.dtype
        )
        if self.coordinate_affine is None:
            source_fsl = (
                torch.einsum(
                    "ij,jxyz->ixyz", self.affine_pull[:3, :3], self.target_fsl
                )
                + self.affine_pull[:3, 3, None, None, None]
                + field
            )
            source_voxels = (
                torch.einsum(
                    "ij,jxyz->ixyz", self.moving_fsl2vox[:3, :3], source_fsl
                )
                + self.moving_fsl2vox[:3, 3, None, None, None]
            )
        else:
            source_voxels = _fsl_displacement_coordinates(
                field, self.coordinate_affine, self.moving_fsl2vox
            )
        warped, valid, gradient_voxels = _trilinear_sample(
            self.moving, source_voxels
        )
        mask = valid
        if self.reference_mask is not None:
            mask = mask & self.reference_mask
        count = int(mask.sum())
        if count < 8:
            raise RuntimeError("FNIRT mask has fewer than eight voxels")
        scaled_fixed = (scale * self.fixed.to(scale.dtype)).to(self.fixed.dtype)
        residual = warped - scaled_fixed
        weight = mask.to(residual.dtype)
        # Upstream squares float residuals and accumulates them into a double.
        ssd = (residual.square() * weight).sum(dtype=torch.float64) / count
        effective_lambda = self.regularization
        if self.ssd_weighted_lambda:
            effective_lambda *= float(ssd.detach())
        bend = self.bending.energy(coefficients)
        cost = ssd + effective_lambda * bend / count
        state = {
            "field": field,
            "warped": warped,
            "mask": mask,
            "count": count,
            "residual": residual,
            "ssd": ssd,
            "bending_energy": bend,
            "effective_lambda": effective_lambda,
            "cost": cost,
        }
        if derivatives:
            state["gradient_fsl"] = torch.einsum(
                "ixyz,ij->jxyz",
                gradient_voxels,
                self.moving_fsl2vox[:3, :3],
            )
        return state

    def linearize(self, coefficients, scale):
        state = self.evaluate(coefficients, scale, derivatives=True)
        mask = state["mask"].to(state["residual"].dtype)
        count = state["count"]
        normalizer = math.sqrt(count)
        gradient_fsl = state["gradient_fsl"]
        coefficient_shape = tuple(coefficients.shape[1:])
        bend_factor = state["effective_lambda"] / count

        def adjoint(value):
            weighted = value * mask / normalizer
            dense = gradient_fsl * weighted[None]
            coefficient_gradient = adjoint_field(
                dense.to(coefficients.dtype), self.bases
            )
            scale_gradient = None
            if self.estimate_scale:
                scale_gradient = -(
                    self.fixed * weighted
                ).sum(dtype=coefficients.dtype)
            return _pack(coefficient_gradient, scale_gradient)

        def bend_normal(vector):
            delta_coefficients, _ = _unpack(
                vector, coefficient_shape, self.estimate_scale
            )
            result = bend_factor * self.bending.normal(delta_coefficients)
            scale_zero = coefficients.new_zeros(()) if self.estimate_scale else None
            return _pack(result, scale_zero)

        residual = state["residual"] * mask / normalizer
        gradient = adjoint(residual) + bend_normal(
            _pack(coefficients, scale if self.estimate_scale else None)
        )

        # FSL materialises every image pair passed to splinefield::JtJ as a
        # float volume before assembling its double sparse Hessian.  Form the
        # same float Hadamard products here; composing forward/adjoint in
        # double would silently use different weights after the first warp.
        spatial_weights = tuple(
            tuple(mask * gradient_fsl[row] * gradient_fsl[column] for column in range(3))
            for row in range(3)
        )
        cross_weights = None
        scale_weight = None
        if self.estimate_scale:
            cross_weights = tuple(
                -(mask * gradient_fsl[axis] * self.fixed) for axis in range(3)
            )
            scale_weight = (mask * self.fixed * self.fixed).sum(
                dtype=coefficients.dtype
            ) / count

        def data_normal(vector):
            delta_coefficients, delta_scale = _unpack(
                vector, coefficient_shape, self.estimate_scale
            )
            delta_field = expand_coefficients(delta_coefficients, self.bases)
            dense = torch.zeros_like(delta_field)
            for row in range(3):
                for column in range(3):
                    dense[row] = dense[row] + spatial_weights[row][column].to(
                        coefficients.dtype
                    ) * delta_field[column]
                if delta_scale is not None:
                    dense[row] = dense[row] + cross_weights[row].to(
                        coefficients.dtype
                    ) * delta_scale
            coefficient_result = adjoint_field(dense / count, self.bases)
            scale_result = None
            if delta_scale is not None:
                scale_result = scale_weight * delta_scale
                for column in range(3):
                    scale_result = scale_result + (
                        cross_weights[column].to(coefficients.dtype)
                        * delta_field[column]
                    ).sum() / count
            return _pack(coefficient_result, scale_result)

        def matvec(vector):
            return data_normal(vector) + bend_normal(vector)

        diagonal_parts = []
        for axis in range(3):
            diagonal_parts.append(
                design_diagonal(
                    (gradient_fsl[axis].square() * mask).to(
                        coefficients.dtype
                    ) / count,
                    self.bases,
                )
            )
        diagonal_coefficients = torch.stack(diagonal_parts)
        diagonal_coefficients = (
            diagonal_coefficients + bend_factor * self.bending.diagonal()[None]
        )
        diagonal_scale = None
        if self.estimate_scale:
            diagonal_scale = (self.fixed.square() * mask).sum(
                dtype=coefficients.dtype
            ) / count
        diagonal = _pack(diagonal_coefficients, diagonal_scale)
        return state, gradient, matvec, diagonal


class TorchFNIRT:
    """Matrix-free PyTorch FNIRT for the FSL GM registration schedule.

    The optimiser is genuine Gauss-Newton/Levenberg-Marquardt with analytic
    B-spline forward/adjoint operators and PCG.  The class is deliberately not
    named ``Exact``: the ported ``constrain_topology`` projection and the full
    optimisation trajectory still require external FSL 6.0.7.4 numerical gates.
    """

    def __init__(
        self,
        *,
        device="cpu",
        config: GMFNIRTConfig | None = None,
        reference_mask=None,
        pcg_tolerance=1e-3,
        pcg_max_iterations=500,
        cost_tolerance=1e-8,
        initial_lm_lambda=0.1,
        strict_topology=False,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.config = GMFNIRTConfig() if config is None else config
        self.reference_mask = reference_mask
        self.pcg_tolerance = float(pcg_tolerance)
        self.pcg_max_iterations = int(pcg_max_iterations)
        self.cost_tolerance = float(cost_tolerance)
        self.initial_lm_lambda = float(initial_lm_lambda)
        self.strict_topology = bool(strict_topology)
        if self.pcg_tolerance <= 0 or self.pcg_max_iterations < 1:
            raise ValueError("invalid PCG options")
        if self.cost_tolerance <= 0 or self.initial_lm_lambda <= 0:
            raise ValueError("invalid LM options")

    def __call__(self, moving, fixed, moving_to_fixed, *, reference_mask=None):
        if not isinstance(moving, sf.Volume) or not isinstance(fixed, sf.Volume):
            raise TypeError("moving and fixed must be surfa.Volume objects")
        initial = _world_affine(moving_to_fixed, moving, fixed)
        moving_data = np.asarray(moving.data, dtype=np.float32).squeeze()
        fixed_data = np.asarray(fixed.data, dtype=np.float32).squeeze()
        if moving_data.ndim != 3 or fixed_data.ndim != 3:
            raise ValueError("moving and fixed must each contain one 3D frame")
        if not np.isfinite(moving_data).all() or not np.isfinite(fixed_data).all():
            raise ValueError("moving and fixed must contain only finite values")

        selected_mask = self.reference_mask if reference_mask is None else reference_mask
        mask_data = None
        if selected_mask is not None:
            if not isinstance(selected_mask, sf.Volume):
                selected_mask = sf.load_volume(selected_mask)
            if selected_mask.shape[:3] != fixed.shape[:3] or not np.allclose(
                selected_mask.geom.vox2world.matrix,
                fixed.geom.vox2world.matrix,
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError("reference mask must be on the fixed image grid")
            mask_data = np.asarray(selected_mask.data).squeeze() > 0.5

        device = self.device
        image_dtype = torch.float32
        dtype = torch.float64
        moving_raw = torch.from_numpy(moving_data.copy()).to(device)
        moving_mean = spm_like_mean(moving_data)
        fixed_mean = spm_like_mean(fixed_data)
        moving_scale = torch.tensor(
            np.float32(100.0 / moving_mean), device=device, dtype=image_dtype
        )
        fixed_scale = torch.tensor(
            np.float32(100.0 / fixed_mean), device=device, dtype=image_dtype
        )
        moving_tensor = moving_raw * moving_scale
        fixed_tensor = torch.from_numpy(fixed_data.copy()).to(device) * fixed_scale
        mask_tensor = (
            None if mask_data is None else torch.from_numpy(mask_data.copy()).to(device)
        )

        moving_fsl_array = voxel_to_fsl_scaled_mm(
            moving.geom.vox2world.matrix, moving_data.shape, moving.geom.voxsize
        )
        fixed_fsl_array = voxel_to_fsl_scaled_mm(
            fixed.geom.vox2world.matrix, fixed_data.shape, fixed.geom.voxsize
        )
        forward_array = world_to_flirt_affine(
            initial.matrix,
            moving.geom.vox2world.matrix,
            fixed.geom.vox2world.matrix,
            moving_data.shape,
            fixed_data.shape,
            moving.geom.voxsize,
            fixed.geom.voxsize,
        )
        moving_fsl_exact = torch.as_tensor(
            moving_fsl_array, device=device, dtype=dtype
        )
        fixed_fsl_exact = torch.as_tensor(
            fixed_fsl_array, device=device, dtype=dtype
        )
        fixed_fsl = fixed_fsl_exact.to(image_dtype)
        moving_fsl2vox = torch.linalg.inv(moving_fsl_exact).to(image_dtype)
        affine_pull_exact = torch.linalg.inv(
            torch.as_tensor(forward_array, device=device, dtype=dtype)
        )
        affine_pull = affine_pull_exact.to(image_dtype)

        fixed_shape = tuple(int(value) for value in fixed_data.shape)
        fixed_voxel_sizes = tuple(float(value) for value in fixed.geom.voxsize)
        moving_voxel_sizes = tuple(float(value) for value in moving.geom.voxsize)
        resolution_schedule = self.config.warp_resolution_schedule_mm
        if resolution_schedule is None:
            resolution_schedule = (self.config.warp_resolution_mm,) * len(
                self.config.subsampling
            )
        knot_spacing = None
        coefficients = None
        previous_stride = None
        previous_level_voxel_sizes = None
        previous_bases = None
        scale = torch.ones((), device=device, dtype=dtype)
        levels = []

        for level, (
            stride,
            maximum_iterations,
            input_fwhm,
            reference_fwhm,
            regularization,
            estimate_intensity,
            apply_reference_mask,
            warp_resolution,
        ) in enumerate(
            zip(
                self.config.subsampling,
                self.config.maximum_iterations,
                self.config.input_fwhm_mm,
                self.config.reference_fwhm_mm,
                self.config.regularization,
                self.config.estimate_intensity,
                self.config.apply_reference_mask,
                resolution_schedule,
            ),
            start=1,
        ):
            full_positions = _level_positions(
                fixed_shape, stride, device=device, dtype=image_dtype
            )
            level_shape = tuple(axis.numel() for axis in full_positions)
            level_voxel_sizes = tuple(
                size * stride for size in fixed_voxel_sizes
            )
            level_positions = tuple(
                torch.arange(size, device=device, dtype=dtype)
                for size in level_shape
            )
            new_knot_spacing = tuple(
                max(1, int(math.floor(mm / voxel + 0.5)))
                for mm, voxel in zip(warp_resolution, fixed_voxel_sizes)
            )
            if coefficients is None:
                knot_spacing = new_knot_spacing
                coefficient_shape = fsl_control_shape(level_shape, knot_spacing)
                coefficients = torch.zeros(
                    (3, *coefficient_shape), device=device, dtype=dtype
                )
            else:
                if stride != previous_stride:
                    coefficients = zoom_coefficients(
                        coefficients,
                        level_shape,
                        knot_spacing,
                        previous_level_voxel_sizes,
                        level_voxel_sizes,
                    )
                if new_knot_spacing != knot_spacing:
                    old_bases = spline_bases(
                        level_shape,
                        knot_spacing,
                        level_voxel_sizes,
                        device=device,
                        dtype=dtype,
                        positions=level_positions,
                    )
                    dense_field = expand_coefficients(coefficients, old_bases)
                    coefficients = fit_field_coefficients(
                        dense_field,
                        new_knot_spacing,
                        level_voxel_sizes,
                        dtype=dtype,
                    )
                    knot_spacing = new_knot_spacing
                coefficient_shape = fsl_control_shape(level_shape, knot_spacing)
            bases = spline_bases(
                level_shape,
                knot_spacing,
                level_voxel_sizes,
                device=device,
                dtype=dtype,
                positions=level_positions,
            )

            moving_level = _fsl_gaussian_blur(
                moving_tensor[None, None], input_fwhm, moving_voxel_sizes
            )[0, 0]
            fixed_smoothed = _fsl_gaussian_blur(
                fixed_tensor[None, None], reference_fwhm, fixed_voxel_sizes
            )[0, 0]
            fixed_level = _take_integer_grid(fixed_smoothed, full_positions)
            reference_mask_level = None
            if apply_reference_mask and mask_tensor is not None:
                reference_mask_level = _take_integer_grid(
                    mask_tensor.to(dtype), full_positions
                ) > 0.99
            target_fsl = _coordinate_grid(fixed_fsl, full_positions)
            level_to_full = torch.diag(
                torch.tensor(
                    [stride, stride, stride, 1.0],
                    device=device,
                    dtype=dtype,
                )
            )
            coordinate_affine = (
                affine_pull_exact @ fixed_fsl_exact @ level_to_full
            )
            bending = BendingOperator(
                level_shape,
                knot_spacing,
                level_voxel_sizes,
                device=device,
                dtype=dtype,
            )
            system = _LevelSystem(
                moving_level,
                fixed_level,
                reference_mask_level,
                moving_fsl2vox,
                target_fsl,
                affine_pull,
                bases,
                bending,
                regularization,
                self.config.ssd_weighted_lambda,
                estimate_intensity,
                coordinate_affine,
            )

            lm_lambda = self.initial_lm_lambda
            accepted = 0
            attempts = 0
            converged = False
            pcg_reports = []
            state = system.evaluate(coefficients, scale)
            while accepted < maximum_iterations and attempts < 10 * max(
                1, maximum_iterations
            ):
                attempts += 1
                state, gradient, matvec, diagonal = system.linearize(
                    coefficients, scale
                )
                damping_diagonal = diagonal.clamp_min(
                    torch.finfo(dtype).eps * diagonal.abs().mean().clamp_min(1)
                )

                def damped(value):
                    return matvec(value) + lm_lambda * damping_diagonal * value

                step, pcg = preconditioned_conjugate_gradient(
                    damped,
                    -gradient,
                    diagonal=(1 + lm_lambda) * damping_diagonal,
                    tolerance=self.pcg_tolerance,
                    max_iterations=self.pcg_max_iterations,
                )
                pcg_reports.append(
                    {
                        "iterations": pcg.iterations,
                        "converged": pcg.converged,
                        "relative_residual": pcg.relative_residual,
                    }
                )
                delta_coefficients, delta_scale = _unpack(
                    step, coefficient_shape, estimate_intensity
                )
                candidate_coefficients = coefficients + delta_coefficients
                candidate_scale = (
                    scale + delta_scale if delta_scale is not None else scale
                )
                candidate = system.evaluate(
                    candidate_coefficients, candidate_scale
                )
                if bool(torch.isfinite(candidate["cost"])) and float(
                    candidate["cost"]
                ) < float(state["cost"]):
                    old_cost = float(state["cost"])
                    new_cost = float(candidate["cost"])
                    coefficients = candidate_coefficients
                    scale = candidate_scale
                    state = candidate
                    accepted += 1
                    lm_lambda /= 10.0
                    converged = (
                        2 * abs(old_cost - new_cost)
                        <= self.cost_tolerance
                        * (abs(old_cost) + abs(new_cost) + torch.finfo(dtype).eps)
                    )
                    if converged:
                        break
                else:
                    lm_lambda *= 10.0
                    if lm_lambda > 1e20:
                        break

            full_jacobian = _spline_jacobian(
                coefficients,
                level_shape,
                knot_spacing,
                level_voxel_sizes,
                affine_pull_linear=affine_pull_exact[:3, :3],
            )
            jacobian_min = float(full_jacobian.min())
            jacobian_max = float(full_jacobian.max())
            lower, upper = self.config.jacobian_range
            topology_projection_required = (
                jacobian_min < lower or jacobian_max > upper
            )
            coefficients, full_jacobian, topology_qc = _force_jacobian_range(
                coefficients,
                level_shape,
                knot_spacing,
                level_voxel_sizes,
                affine_pull_exact,
                lower,
                upper,
                5 if level == 1 else 10,
            )
            if topology_qc["required"]:
                state = system.evaluate(coefficients, scale)
            if not topology_qc["succeeded"]:
                message = (
                    "FSL ForceJacobianRange did not reach the requested range; "
                    f"Jacobian range was {topology_qc['range'][0]:.6g}--"
                    f"{topology_qc['range'][1]:.6g} "
                    f"and the requested range is {lower:.6g}--{upper:.6g}"
                )
                if self.strict_topology:
                    raise RuntimeError(message)
                warnings.warn(
                    message + "; continuing as FSL FNIRT does",
                    RuntimeWarning,
                    stacklevel=2,
                )

            levels.append(
                {
                    "level": level,
                    "stride": stride,
                    "matrix_size": list(level_shape),
                    "control_grid_shape": list(coefficient_shape),
                    "warp_resolution_mm": list(warp_resolution),
                    "knot_spacing_voxels": list(knot_spacing),
                    "maximum_iterations": maximum_iterations,
                    "accepted_iterations": accepted,
                    "attempts": attempts,
                    "converged": converged,
                    "input_fwhm_mm": input_fwhm,
                    "reference_fwhm_mm": reference_fwhm,
                    "base_lambda": regularization,
                    "effective_lambda": state["effective_lambda"],
                    "ssd": float(state["ssd"]),
                    "bending_energy": float(state["bending_energy"]),
                    "cost": float(state["cost"]),
                    "intensity_scale": float(scale),
                    "estimate_intensity": estimate_intensity,
                    "apply_reference_mask": apply_reference_mask,
                    "reference_mask_available": mask_tensor is not None,
                    "mask_voxels": state["count"],
                    "lm_lambda_final": lm_lambda,
                    "pcg": pcg_reports,
                    "full_jacobian_min_before_projection": jacobian_min,
                    "full_jacobian_max_before_projection": jacobian_max,
                    "topology_projection_required": topology_projection_required,
                    "topology_projection": topology_qc,
                }
            )
            previous_stride = stride
            previous_level_voxel_sizes = level_voxel_sizes
            previous_bases = bases

        if previous_stride != 1:
            raise ValueError("the final FNIRT level must use full resolution")
        with torch.no_grad():
            field = expand_coefficients(coefficients, previous_bases).to(
                image_dtype
            )[None]
            constrained_coefficients = coefficients
            full_positions = tuple(
                torch.arange(size, device=device, dtype=image_dtype)
                for size in fixed_shape
            )
            target_fsl = _coordinate_grid(fixed_fsl, full_positions)
            source_fsl = (
                torch.einsum(
                    "ij,jxyz->ixyz", affine_pull[:3, :3], target_fsl
                )
                + affine_pull[:3, 3, None, None, None]
                + field[0]
            )
            source_voxels = (
                torch.einsum(
                    "ij,jxyz->ixyz", moving_fsl2vox[:3, :3], source_fsl
                )
                + moving_fsl2vox[:3, 3, None, None, None]
            )
            moved, _, _ = _trilinear_sample(moving_raw, source_voxels)

            nonlinear_jacobian = _spline_jacobian(
                constrained_coefficients,
                fixed_shape,
                knot_spacing,
                fixed_voxel_sizes,
            )

            fixed_world = torch.as_tensor(
                np.array(fixed.geom.vox2world.matrix, dtype=np.float32, copy=True),
                device=device,
            )
            moving_world = torch.as_tensor(
                np.array(moving.geom.vox2world.matrix, dtype=np.float32, copy=True),
                device=device,
            )
            target_world = _coordinate_grid(fixed_world, full_positions)
            source_world = (
                torch.einsum(
                    "ij,jxyz->ixyz", moving_world[:3, :3], source_voxels
                )
                + moving_world[:3, 3, None, None, None]
            )
            displacement = (source_world - target_world).movedim(0, -1)

        displacement_array = displacement.cpu().numpy().astype(np.float32)
        pull_transform = sf.Warp(
            displacement_array,
            source=moving,
            target=fixed,
            format=sf.Warp.Format.disp_ras,
        )
        full_world, _, affine_pull_determinant = pull_jacobian_determinants(
            displacement_array,
            fixed.geom.vox2world.matrix,
            initial.matrix,
            device=device,
        )
        moved_array = moved.cpu().numpy().astype(np.float32)
        nonlinear_array = nonlinear_jacobian.cpu().numpy().astype(np.float32)
        full_array = full_world.cpu().numpy().astype(np.float32)
        coefficient_array = (
            constrained_coefficients.movedim(0, -1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        coefficient_image = make_fsl_coefficient_image(
            coefficient_array,
            fixed_shape,
            fixed_voxel_sizes,
            knot_spacing,
            forward_array,
        )
        qc = {
            "backend": "pytorch-fnirt-matrix-free-lm",
            "device": str(self.device),
            "tf32": {
                "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "reduced_precision_tensor_dtype": False,
            },
            "fsl_fnirt_numerically_equivalent": False,
            "equivalence_status": "external FSL 6.0.7.4 numerical gate not passed",
            "fsl_source_versions": FSL_SOURCE_VERSIONS,
            "optimizer": "Gauss-Newton/Levenberg-Marquardt with PCG",
            "hessian": "analytic matrix-free B-spline JtJ plus bending Hessian",
            "global_intensity_model": "multiplicative scale of reference",
            "ssd_weighted_lambda": self.config.ssd_weighted_lambda,
            "mask_schedule_matches_gm_config": mask_tensor is not None,
            "intensity_schedule_matches_gm_config": True,
            "topology_projection_matches_fsl": False,
            "topology_projection": (
                "FSL 2203.0 algorithm ported; external numerical gate pending"
            ),
            "strict_topology": self.strict_topology,
            "topology_projection_required": any(
                item["topology_projection_required"] for item in levels
            ),
            "accepts_fsl_coefficient_file": False,
            "exports_fsl_coefficient_file": True,
            "reference_mask_available": mask_tensor is not None,
            "knot_spacing_voxels": list(knot_spacing),
            "control_grid_shape": list(coefficients.shape[1:]),
            "levels": levels,
            "final_intensity_scale": float(scale),
            "nonlinear_jacobian_min": float(nonlinear_jacobian.min()),
            "nonlinear_jacobian_max": float(nonlinear_jacobian.max()),
            "full_pull_jacobian_min": float(full_world.min()),
            "full_pull_jacobian_max": float(full_world.max()),
        }
        return TorchFNIRTResult(
            moved=fixed.new(moved_array),
            pull_transform=pull_transform,
            full_pull_jacobian=fixed.new(full_array),
            nonlinear_jacobian=fixed.new(nonlinear_array),
            modulated_gm=fixed.new(moved_array * nonlinear_array),
            affine_pull_determinant=float(affine_pull_determinant),
            coefficients=coefficient_array,
            coefficient_image=coefficient_image,
            qc=qc,
        )


__all__ = [
    "FSL_SOURCE_VERSIONS",
    "GMFNIRTConfig",
    "TorchFNIRT",
    "TorchFNIRTResult",
    "spm_like_mean",
]
