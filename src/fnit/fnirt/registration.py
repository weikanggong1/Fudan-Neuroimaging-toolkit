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
import torch

from .._nib import FNITNifti1Image, load_image, new_image
from .._transforms import AffineTransform, DenseWarp, same_geometry
from ..flirt.coordinates import (
    flirt_to_world_affine,
    voxel_to_fsl_scaled_mm,
    world_to_flirt_affine,
)
from .optimizer import (
    preconditioned_conjugate_gradient,
    scaled_conjugate_gradient,
)
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


def _field_determinant(matrix):
    return (
        matrix[0, 0]
        * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1]
        * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2]
        * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def _pull_jacobian_determinants(
    displacement_ras, target_vox2world, moving_to_fixed_world, *, device
):
    field = torch.as_tensor(
        np.array(displacement_ras, dtype=np.float32, copy=True), device=device
    )
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("displacement_ras must have shape (X, Y, Z, 3)")
    target = torch.as_tensor(
        target_vox2world, dtype=torch.float32, device=device
    )[:3, :3]
    affine = torch.as_tensor(
        moving_to_fixed_world, dtype=torch.float32, device=device
    )[:3, :3]
    target_determinant = torch.linalg.det(target)
    affine_determinant = torch.linalg.det(affine)
    if abs(float(target_determinant)) < 1e-8:
        raise ValueError("target_vox2world must be invertible")
    if abs(float(affine_determinant)) < 1e-8:
        raise ValueError("moving_to_fixed_world must be invertible")
    field = field.movedim(-1, 0)
    voxel_gradient = torch.stack(
        torch.gradient(field, dim=(1, 2, 3), edge_order=1), dim=1
    )
    world_gradient = torch.einsum(
        "abxyz,bc->acxyz", voxel_gradient, torch.linalg.inv(target)
    )
    pull_gradient = world_gradient + torch.eye(
        3, dtype=field.dtype, device=device
    )[:, :, None, None, None]
    full = _field_determinant(pull_gradient)
    affine_pull_determinant = affine_determinant.reciprocal()
    return full, full / affine_pull_determinant, affine_pull_determinant


def _world_affine(value, moving, fixed):
    if isinstance(value, AffineTransform):
        if not same_geometry(value.source, moving) or not same_geometry(
            value.target, fixed
        ):
            raise ValueError("moving_to_fixed geometry does not match the images")
        matrix = value.convert(space="world").matrix
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("moving_to_fixed must contain a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("moving_to_fixed must be a homogeneous affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-8:
        raise ValueError("moving_to_fixed must be invertible")
    return np.array(matrix, dtype=np.float64, copy=True)


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
class FNIRTConfig:
    """Supported settings from FSL FNIRT without a configuration file."""

    subsampling: tuple[int, ...] = (4, 2, 1, 1)
    maximum_iterations: tuple[int, ...] = (5, 5, 5, 5)
    input_fwhm_mm: tuple[float, ...] = (6.0, 4.0, 2.0, 2.0)
    reference_fwhm_mm: tuple[float, ...] = (4.0, 2.0, 0.0, 0.0)
    regularization: tuple[float, ...] = (120.0, 60.0, 30.0, 30.0)
    estimate_intensity: tuple[bool, ...] = (True, True, True, False)
    apply_reference_mask: tuple[bool, ...] = (True,) * 4
    minimization_methods: tuple[str, ...] | None = None
    process_stages: tuple[int, ...] | None = None
    implicit_reference_mask: bool = True
    implicit_input_mask: bool = True
    warp_resolution_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    warp_resolution_schedule_mm: tuple[tuple[float, float, float], ...] | None = None
    jacobian_range: tuple[float, float] = (0.01, 100.0)
    ssd_weighted_lambda: bool = True
    intensity_model: str = "global_non_linear_with_bias"
    intensity_order: int = 5
    bias_resolution_mm: tuple[float, float, float] = (50.0, 50.0, 50.0)
    bias_regularization: float = 10000.0

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
        optional_schedules = (
            self.minimization_methods,
            self.process_stages,
        )
        if count == 0 or any(len(schedule) != count for schedule in schedules):
            raise ValueError("all FNIRT schedules must have the same non-zero length")
        if any(
            schedule is not None and len(schedule) != count
            for schedule in optional_schedules
        ):
            raise ValueError("all FNIRT schedules must have the same non-zero length")
        if any(value not in (1, 2, 4, 8, 16) for value in self.subsampling):
            raise ValueError("FNIRT subsampling factors must be powers of two")
        if any(value < 0 for value in self.maximum_iterations):
            raise ValueError("maximum iterations must be non-negative")
        if any(value < 0 for value in self.regularization):
            raise ValueError("regularization must be non-negative")
        if any(value < 0 for value in (*self.input_fwhm_mm, *self.reference_fwhm_mm)):
            raise ValueError("FWHM values must be non-negative")
        if len(self.warp_resolution_mm) != 3 or any(
            value <= 0 for value in self.warp_resolution_mm
        ):
            raise ValueError("warp resolution must contain three positive values")
        if len(self.jacobian_range) != 2 or not 0 < self.jacobian_range[0] < self.jacobian_range[1]:
            raise ValueError("Jacobian range must contain two increasing positive values")
        if self.minimization_methods is not None and any(
            value not in ("lm", "scg") for value in self.minimization_methods
        ):
            raise ValueError("minimization methods must be lm or scg")
        if self.process_stages is not None:
            if any(value < 1 for value in self.process_stages):
                raise ValueError("process stage identifiers must be positive")
            if any(
                current < previous
                for previous, current in zip(
                    self.process_stages, self.process_stages[1:]
                )
            ):
                raise ValueError("process stage identifiers must be non-decreasing")
        if self.warp_resolution_schedule_mm is not None:
            if len(self.warp_resolution_schedule_mm) != count:
                raise ValueError("warp resolution schedule must match the level count")
            if any(
                len(value) != 3 or any(item <= 0 for item in value)
                for value in self.warp_resolution_schedule_mm
            ):
                raise ValueError("each warp resolution must contain three positive values")
        if self.intensity_model not in ("global_linear", "global_non_linear_with_bias"):
            raise ValueError("unsupported FNIRT intensity model")
        if self.intensity_model == "global_non_linear_with_bias" and not 2 <= self.intensity_order <= 5:
            raise ValueError("nonlinear intensity order must be between 2 and 5")
        if any(value <= 0 for value in self.bias_resolution_mm) or self.bias_regularization < 0:
            raise ValueError("invalid bias field resolution or regularization")


@dataclass(frozen=True)
class GMFNIRTConfig(FNIRTConfig):
    """FSL ``GM_2_MNI152GM_2mm.cnf`` preset for grey matter maps."""

    maximum_iterations: tuple[int, ...] = (5, 5, 10, 5)
    regularization: tuple[float, ...] = (150.0, 75.0, 50.0, 30.0)
    apply_reference_mask: tuple[bool, ...] = (False, False, False, True)
    implicit_reference_mask: bool = False
    implicit_input_mask: bool = False
    jacobian_range: tuple[float, float] = (0.2, 5.0)
    intensity_model: str = "global_linear"


@dataclass(frozen=True)
class T1FNIRTConfig(FNIRTConfig):
    """FSL ``T1_2_MNI152_2mm.cnf`` geometry and T1 intensity settings."""

    subsampling: tuple[int, ...] = (4, 4, 2, 2, 1, 1)
    maximum_iterations: tuple[int, ...] = (5, 5, 5, 5, 5, 10)
    input_fwhm_mm: tuple[float, ...] = (8.0, 6.0, 5.0, 4.5, 3.0, 2.0)
    reference_fwhm_mm: tuple[float, ...] = (8.0, 6.0, 5.0, 4.0, 2.0, 0.0)
    regularization: tuple[float, ...] = (300.0, 150.0, 100.0, 50.0, 40.0, 30.0)
    estimate_intensity: tuple[bool, ...] = (True, True, True, True, True, False)
    apply_reference_mask: tuple[bool, ...] = (True,) * 6
    intensity_model: str = "global_non_linear_with_bias"
    intensity_order: int = 5


@dataclass(frozen=True)
class TBSSFNIRTConfig(FNIRTConfig):
    """UK Biobank ``oxford_s1/s2/s3.cnf`` FA registration preset."""

    subsampling: tuple[int, ...] = (8, 4, 2, 2, 1, 1)
    maximum_iterations: tuple[int, ...] = (5, 5, 5, 5, 50, 25)
    input_fwhm_mm: tuple[float, ...] = (12.0, 8.0, 4.0, 4.0, 1.0, 1.0)
    reference_fwhm_mm: tuple[float, ...] = (12.0, 8.0, 4.0, 4.0, 1.0, 1.0)
    regularization: tuple[float, ...] = (300.0, 75.0, 50.0, 40.0, 100.0, 30.0)
    estimate_intensity: tuple[bool, ...] = (True, True, True, False, False, False)
    apply_reference_mask: tuple[bool, ...] = (False,) * 6
    minimization_methods: tuple[str, ...] = ("lm", "lm", "lm", "lm", "scg", "scg")
    process_stages: tuple[int, ...] = (1, 1, 1, 1, 2, 3)
    warp_resolution_schedule_mm: tuple[tuple[float, float, float], ...] = (
        (10.0, 10.0, 10.0),
        (10.0, 10.0, 10.0),
        (10.0, 10.0, 10.0),
        (10.0, 10.0, 10.0),
        (2.0, 2.0, 2.0),
        (2.0, 2.0, 2.0),
    )
    intensity_model: str = "global_linear"


def resolve_fnirt_config(value=None, *, default="default"):
    """Return a named FNIRT preset or an explicitly supplied configuration."""
    if isinstance(value, FNIRTConfig):
        return value
    name = default if value is None else value
    presets = {
        "default": FNIRTConfig,
        "gm": GMFNIRTConfig,
        "t1": T1FNIRTConfig,
        "tbss": TBSSFNIRTConfig,
    }
    if not isinstance(name, str) or name.lower() not in presets:
        raise ValueError("FNIRT config must be default, gm, t1, tbss, or FNIRTConfig")
    return presets[name.lower()]()


@dataclass
class TorchFNIRTResult:
    moved: FNITNifti1Image
    pull_transform: DenseWarp
    full_pull_jacobian: FNITNifti1Image
    nonlinear_jacobian: FNITNifti1Image
    modulated_gm: FNITNifti1Image
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
    if volume.is_cuda and volume.dtype == torch.float32 and fwhm_mm > 0:
        try:
            from ._smoothing_triton import gaussian_blur_cuda
        except ImportError:
            pass
        else:
            return gaussian_blur_cuda(volume, _fsl_gaussian_kernels(fwhm_mm, voxel_sizes))
    if volume.device.type == "cpu" and not volume.requires_grad:
        return _fsl_gaussian_blur_cpu(volume, fwhm_mm, voxel_sizes)
    return _fsl_gaussian_blur_reference(volume, fwhm_mm, voxel_sizes)


def _fsl_gaussian_kernels(fwhm_mm, voxel_sizes):
    """Generate the float Gaussian entries and double normalisation in FSL."""
    kernels = []
    sigma_mm = np.float32(float(fwhm_mm) / math.sqrt(8.0 * math.log(2.0)))
    for voxel_size in voxel_sizes:
        sigma = np.float32(sigma_mm / np.float32(voxel_size))
        radius = int(np.float32(sigma - np.float32(0.001))) * 2 + 3
        values = []
        total = np.float32(0.0)
        for offset in range(-radius, radius + 1):
            value = (
                np.float32(math.exp(-(offset * offset) / (2.0 * float(sigma) * float(sigma))))
                if sigma > np.float32(1e-6)
                else np.float32(1.0 if offset == 0 else 0.0)
            )
            values.append(value)
            total = np.float32(total + value)
        kernels.append(tuple(float(value) * (1.0 / float(total)) for value in values))
    return tuple(kernels)


def _fsl_gaussian_blur_reference(volume, fwhm_mm, voxel_sizes):
    """Offset-serial implementation, also used without optional Triton."""
    if fwhm_mm <= 0:
        return volume
    result = volume
    for axis, kernel in enumerate(_fsl_gaussian_kernels(fwhm_mm, voxel_sizes)):
        radius = len(kernel) // 2
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


def _fsl_gaussian_blur_cpu(volume, fwhm_mm, voxel_sizes):
    """Reuse CPU scratch while retaining every offset's float assignment.

    FSL multiplies a float sample by a double kernel entry, adds in double,
    then stores back to the image dtype after *each* offset.  A conventional
    convolution changes that rounding order.  ``add(..., out=target)`` uses
    the double scratch's promoted dtype and casts only its result, exactly
    as the reference expression does.  No thread settings are changed.
    """
    if fwhm_mm <= 0:
        return volume
    if volume.ndim == 5 and volume.dtype in (torch.float32, torch.float64) and not volume.requires_grad:
        from ._smoothing_cpu import gaussian_blur_cpu
        return gaussian_blur_cpu(volume, _fsl_gaussian_kernels(fwhm_mm, voxel_sizes))
    scratch = torch.empty_like(volume, dtype=torch.float64)
    result = volume
    for axis, kernel in enumerate(_fsl_gaussian_kernels(fwhm_mm, voxel_sizes)):
        radius = len(kernel) // 2
        convolved = torch.zeros_like(result)
        dimension = axis + 2
        length = result.shape[dimension]
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            source_start = max(0, offset)
            source_stop = min(length, length + offset)
            count = source_stop - source_start
            if count <= 0:
                continue
            source = result.narrow(dimension, source_start, count)
            target = convolved.narrow(dimension, source_start - offset, count)
            product = scratch.narrow(dimension, 0, count)
            product.copy_(source)
            product.mul_(weight)
            torch.add(product, target, out=target)
        result = convolved
    return result


def _fsl_masked_gaussian_blur(volume, fwhm_mm, voxel_sizes, mask, *, execution="optimized"):
    """FSL ``fnirt_CF::masked_smoothing`` for an input image."""
    blur = _fsl_gaussian_blur if execution == "optimized" else _fsl_gaussian_blur_reference
    if fwhm_mm <= 0 or mask is None:
        return blur(volume, fwhm_mm, voxel_sizes)
    if volume.ndim != 5 or mask.ndim != 3 or volume.shape[2:] != mask.shape:
        raise ValueError("masked smoothing expects [N,C,X,Y,Z] and [X,Y,Z]")
    mask_image = mask.to(dtype=volume.dtype)[None, None]
    numerator = blur(
        volume * mask_image, fwhm_mm, voxel_sizes
    )
    denominator = blur(mask_image, fwhm_mm, voxel_sizes)
    return torch.where(
        mask_image > 0,
        numerator / denominator.clamp_min(torch.finfo(volume.dtype).tiny),
        torch.zeros_like(numerator),
    )


def _subsampled_size(size, factor):
    result = int(size)
    remaining = int(factor)
    while remaining > 1:
        result = result // 2 + 1
        remaining //= 2
    return result


def _process_knot_spacing_schedule(
    resolution_schedule, process_stages, subsampling, voxel_sizes
):
    """Convert FNIRT warpres values to each process' full-grid spacing.

    FSL constructs a process at warpres / reference_voxel_size knots and then
    divides that spacing by the process' final subsampling factor in
    fnirt_clp::FullResKsp. The same full-grid spacing is retained while that
    process moves between resolution levels.
    """
    result = [None] * len(process_stages)
    start = 0
    while start < len(process_stages):
        stage = process_stages[start]
        stop = start + 1
        while stop < len(process_stages) and process_stages[stop] == stage:
            stop += 1
        requested = tuple(float(value) for value in resolution_schedule[start])
        if any(
            tuple(float(value) for value in resolution_schedule[index]) != requested
            for index in range(start + 1, stop)
        ):
            raise ValueError("warp resolution must be constant within a FNIRT process")
        final_subsampling = int(subsampling[stop - 1])
        spacing = []
        for resolution, voxel_size in zip(requested, voxel_sizes):
            native_spacing = max(
                1, int(math.floor(resolution / float(voxel_size) + 0.5))
            )
            full_spacing = native_spacing // final_subsampling
            if full_spacing < 1:
                raise ValueError(
                    "warp resolution is incompatible with the process' final "
                    "subsampling factor"
                )
            spacing.append(full_spacing)
        for index in range(start, stop):
            result[index] = tuple(spacing)
        start = stop
    return tuple(result)


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


def _trilinear_sample(volume, coordinates, *, derivatives=True):
    """Trilinear values and optional piecewise voxel-coordinate derivatives.

    Value-only calls retain the interpolation arithmetic and return ``None``
    for the unused gradient. The default keeps the existing derivative API.
    """
    if volume.ndim != 3 or coordinates.shape[0] != 3:
        raise ValueError("invalid trilinear input shapes")
    if (volume.device.type == "cpu" and coordinates.device.type == "cpu"
            and volume.dtype == torch.float32 and coordinates.dtype == torch.float32
            and not volume.requires_grad and not coordinates.requires_grad):
        # CUDA retains its tensor operations and never imports this CPU module.
        # The helper returns None for unsupported/exceptional tensor semantics.
        from ._sampling_cpu import try_sample_cpu
        result = try_sample_cpu(volume, coordinates, derivatives=derivatives)
        if result is not None:
            return result
    spatial_shape = coordinates.shape[1:]
    flat = coordinates.reshape(3, -1)
    valid = torch.ones(flat.shape[1], dtype=torch.bool, device=volume.device)
    lower = []
    upper = []
    fraction = []
    for coordinate, size in zip(flat, volume.shape):
        # newimage::volume::valid(float, float, float) uses a 1e-8 tolerance.
        # Keep extrapolated samples within that tolerance in the data mask;
        # interpolation itself remains zero padded, as in newimage.
        valid &= (coordinate + 1e-8 >= 0) & (coordinate <= size - 1 + 1e-8)
        lo = torch.floor(coordinate).to(torch.long)
        hi = lo + 1
        lower.append(lo)
        upper.append(hi)
        fraction.append(coordinate - lo.to(coordinate.dtype))

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
    if derivatives:
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
    if derivatives:
        derivative_z = tmp22 - tmp21
    sampled = one_minus_z * tmp21 + wz * tmp22
    valid_float = valid.to(volume.dtype)
    sampled = (sampled * valid_float).reshape(spatial_shape)
    gradient = None
    if derivatives:
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


def _fsl_displacement_coordinates(field, coordinate_affine, mm_to_voxel, *, affine_grid=None):
    """Coordinates from ``warpfns::displacements_no_iT``.

    ``coordinate_affine`` is ``inverse(FLIRT) @ target.sampling_mat`` in
    double precision.  warpfns casts its entries and those of
    ``source.sampling_mat().i()`` to float before evaluating the expressions.
    """
    source_mm = (
        _fsl_affine_grid(coordinate_affine, field.shape[1:])
        if affine_grid is None else affine_grid
    )
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
        moving_mask,
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
        self.moving_mask = moving_mask
        self.moving_fsl2vox = moving_fsl2vox
        self.target_fsl = target_fsl
        self.affine_pull = affine_pull
        self.coordinate_affine = coordinate_affine
        self.bases = bases
        self.bending = bending
        self.regularization = float(regularization)
        self.ssd_weighted_lambda = bool(ssd_weighted_lambda)
        self.estimate_scale = bool(estimate_scale)
        self.affine_grid = (
            None if coordinate_affine is None
            else _fsl_affine_grid(coordinate_affine, fixed.shape)
        )

    def evaluate(self, coefficients, scale, *, derivatives=False, _compute_cost=True):
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
                field, self.coordinate_affine, self.moving_fsl2vox,
                affine_grid=self.affine_grid,
            )
        warped, valid, gradient_voxels = _trilinear_sample(
            self.moving, source_voxels, derivatives=derivatives
        )
        mask = valid
        if self.moving_mask is not None:
            warped_mask, _, _ = _trilinear_sample(
                self.moving_mask, source_voxels, derivatives=False
            )
            # ``robjmask`` is a ``volume<char>`` upstream. warpfns casts the
            # trilinear value to char while resampling, before ``Mask()``
            # applies its >0.5 test. For a binary mask this admits only
            # samples whose interpolated float value survives truncation to 1.
            mask = mask & (warped_mask.to(torch.int8) > 0)
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
        if _compute_cost:
            bend = self.bending.energy(coefficients)
            cost = ssd + effective_lambda * bend / count
        else:
            if not derivatives or coefficients.device.type != "cpu":
                raise ValueError("discarded SCG cost is CPU derivative-only")
            bend, cost = None, None
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

    def gradient(self, coefficients, scale, *, effective_lambda=None, _skip_cost=False):
        """Return the FSL SSD gradient and its evaluated state.

        The override reproduces stateful latest_ssd behavior in SCG because a
        finite-difference gradient does not itself update that value.
        """
        skip_cost = bool(_skip_cost and coefficients.device.type == "cpu")
        if skip_cost:
            values = (coefficients, scale, self.moving, self.fixed, self.moving_mask,
                self.reference_mask, self.moving_fsl2vox, self.target_fsl,
                self.affine_pull, self.coordinate_affine, *self.bases,
                *(basis for bases, _ in self.bending.operators for basis in bases))
            for value in values:
                if value is None:
                    continue
                if (value.requires_grad
                        or torch._C._functorch.is_functorch_wrapped_tensor(value)
                        or torch.autograd.forward_ad.unpack_dual(value).tangent is not None):
                    skip_cost = False
                    break
        state = self.evaluate(coefficients, scale, derivatives=True, _compute_cost=not skip_cost)
        mask = state["mask"].to(state["residual"].dtype)
        count = state["count"]
        weighted = state["residual"] * mask / count
        coefficient_gradient = adjoint_field(
            (state["gradient_fsl"] * weighted[None]).to(coefficients.dtype),
            self.bases,
        )
        scale_gradient = None
        if self.estimate_scale:
            scale_gradient = -(
                self.fixed * weighted
            ).sum(dtype=coefficients.dtype)
        if effective_lambda is None:
            effective_lambda = state["effective_lambda"]
        bend_factor = effective_lambda / count
        coefficient_gradient = coefficient_gradient + (
            bend_factor * self.bending.normal(coefficients)
        )
        # FSL applies a factor of two to both the mean-SSD and bending-energy
        # gradients. It cancels from the LM normal equations but is required by
        # the finite-difference curvature and lambda updates in SCG.
        return state, 2.0 * _pack(coefficient_gradient, scale_gradient)

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
            tuple((mask * gradient_fsl[row] * gradient_fsl[column]).to(coefficients.dtype) for column in range(3))
            for row in range(3)
        )
        cross_weights = None
        scale_weight = None
        if self.estimate_scale:
            cross_weights = tuple(
                (-(mask * gradient_fsl[axis] * self.fixed)).to(coefficients.dtype) for axis in range(3)
            )
            scale_weight = (mask * self.fixed * self.fixed).sum(
                dtype=coefficients.dtype
            ) / count

        cpu_normal = None
        if (coefficients.device.type == "cpu" and coefficients.dtype == torch.float64
                and self.bending.execution == "optimized" and not coefficients.requires_grad
                and all(not value.requires_grad for row in spatial_weights for value in row)
                and (cross_weights is None or all(not value.requires_grad for value in cross_weights))):
            from ._normal_cpu import SpatialNormalCPU
            cpu_normal = SpatialNormalCPU(spatial_weights, cross_weights, count)

        def data_normal(vector, *, delta_field=None):
            delta_coefficients, delta_scale = _unpack(
                vector, coefficient_shape, self.estimate_scale
            )
            if delta_field is None:
                delta_field = expand_coefficients(delta_coefficients, self.bases)
            if cpu_normal is not None and not delta_field.requires_grad and (delta_scale is None or not delta_scale.requires_grad):
                # Final division follows the original FP64 additions. CUDA and
                # differentiable/reference CPU calls keep the existing path.
                dense = cpu_normal(delta_field, delta_scale)
                coefficient_result = adjoint_field(dense, self.bases)
            else:
                dense = torch.zeros_like(delta_field)
                for row in range(3):
                    for column in range(3):
                        dense[row] = dense[row] + spatial_weights[row][column] * delta_field[column]
                    if delta_scale is not None:
                        dense[row] = dense[row] + cross_weights[row] * delta_scale
                coefficient_result = adjoint_field(dense / count, self.bases)
            scale_result = None
            if delta_scale is not None:
                scale_result = scale_weight * delta_scale
                for column in range(3):
                    scale_result = scale_result + (
                        cross_weights[column] * delta_field[column]
                    ).sum() / count
            return _pack(coefficient_result, scale_result)

        def matvec(vector, *, delta_field=None):
            return data_normal(vector, delta_field=delta_field) + bend_normal(vector)

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


def _pack_t1(coefficients, bias_coefficients, polynomial):
    return torch.cat((
        _pack(coefficients),
        bias_coefficients.permute(0, 3, 2, 1).reshape(-1),
        polynomial.reshape(-1),
    ))


def _unpack_t1(vector, coefficient_shape, bias_shape):
    def_size = 3 * math.prod(coefficient_shape)
    bias_size = math.prod(bias_shape)
    coefficients, _ = _unpack(vector[:def_size], coefficient_shape, False)
    bias = vector[def_size:def_size + bias_size].reshape(
        1, bias_shape[2], bias_shape[1], bias_shape[0]
    ).permute(0, 3, 2, 1)
    return coefficients, bias, vector[def_size + bias_size:]


class _JointT1System:
    """Joint LM system for deformation, global polynomial and spline bias."""

    def __init__(self, deformation, bias_bases, bias_bending, bias_lambda, polynomial_order):
        self.deformation = deformation
        self.bias_bases = bias_bases
        self.bias_bending = bias_bending
        self.bias_lambda = float(bias_lambda)
        reference = deformation.fixed
        x = reference / 100.0
        self.powers = torch.stack(
            [100.0 * x.pow(degree) for degree in range(polynomial_order)],
            dim=0,
        )

    def _mapping(self, polynomial, bias_coefficients):
        bias = expand_coefficients(bias_coefficients, self.bias_bases)[0].to(
            self.powers.dtype
        )
        global_map = torch.einsum(
            "ixyz,i->xyz", self.powers, polynomial.to(self.powers.dtype)
        )
        return global_map * bias, global_map, bias

    def evaluate(self, coefficients, polynomial, bias_coefficients, *, derivatives=False):
        mapped, global_map, bias = self._mapping(polynomial, bias_coefficients)
        self.deformation.fixed = mapped
        state = self.deformation.evaluate(
            coefficients, coefficients.new_ones(()), derivatives=derivatives
        )
        bias_energy = self.bias_bending.energy(bias_coefficients)
        state["cost"] = state["cost"] + self.bias_lambda * bias_energy / state["count"]
        state["bias_energy"] = bias_energy
        state["global_map"] = global_map
        state["bias"] = bias
        return state

    def linearize(self, coefficients, polynomial, bias_coefficients, *, fit_intensity):
        mapped, global_map, bias = self._mapping(polynomial, bias_coefficients)
        self.deformation.fixed = mapped
        state, deformation_gradient, deformation_matvec, deformation_diagonal = (
            self.deformation.linearize(coefficients, coefficients.new_ones(()))
        )
        bias_energy = self.bias_bending.energy(bias_coefficients)
        state["cost"] = state["cost"] + self.bias_lambda * bias_energy / state["count"]
        state["bias_energy"] = bias_energy
        if not fit_intensity:
            return state, deformation_gradient, deformation_matvec, deformation_diagonal

        mask = state["mask"].to(self.powers.dtype)
        count = state["count"]
        gradient_fsl = state["gradient_fsl"]
        polynomial_images = self.powers * bias[None]
        residual = state["residual"] * mask / count
        bias_factor = self.bias_lambda / count
        bias_gradient = -adjoint_field(
            (global_map * residual)[None].to(coefficients.dtype), self.bias_bases
        ) + bias_factor * self.bias_bending.normal(bias_coefficients)
        polynomial_gradient = -torch.einsum(
            "ixyz,xyz->i", polynomial_images, residual
        ).to(coefficients.dtype)
        def_shape = tuple(coefficients.shape[1:])
        bias_shape = tuple(bias_coefficients.shape[1:])
        def matvec(vector):
            def_coefficients, bias_part, polynomial_part = _unpack_t1(
                vector, def_shape, bias_shape
            )
            def_part = _pack(def_coefficients)
            deformation_field = expand_coefficients(
                def_coefficients, self.deformation.bases
            )
            # Reuse the original double expansion in the deformation normal
            # equation. Its float image copy is only for the intensity block.
            delta_field = deformation_field.to(self.powers.dtype)
            delta_map = (
                torch.einsum("ixyz,i->xyz", polynomial_images, polynomial_part.to(self.powers.dtype))
                + global_map * expand_coefficients(bias_part, self.bias_bases)[0].to(self.powers.dtype)
            )
            total = (gradient_fsl * delta_field).sum(0) - delta_map
            weighted = total * mask / count
            def_result = deformation_matvec(def_part, delta_field=deformation_field) - _pack(adjoint_field(
                (gradient_fsl * (delta_map * mask / count)[None]).to(coefficients.dtype),
                self.deformation.bases,
            ))
            bias_result = -adjoint_field(
                (global_map * weighted)[None].to(coefficients.dtype), self.bias_bases
            ) + bias_factor * self.bias_bending.normal(bias_part)
            polynomial_result = -torch.einsum(
                "ixyz,xyz->i", polynomial_images, weighted
            ).to(coefficients.dtype)
            return torch.cat((
                def_result,
                bias_result.permute(0, 3, 2, 1).reshape(-1),
                polynomial_result,
            ))

        bias_diagonal = design_diagonal(
            (global_map.square() * mask / count).to(coefficients.dtype),
            self.bias_bases,
        )[None] + bias_factor * self.bias_bending.diagonal()[None]
        polynomial_diagonal = (
            polynomial_images.square() * mask[None]
        ).flatten(1).sum(1, dtype=coefficients.dtype) / count
        gradient = torch.cat((
            deformation_gradient,
            bias_gradient.permute(0, 3, 2, 1).reshape(-1),
            polynomial_gradient,
        ))
        diagonal = torch.cat((
            deformation_diagonal,
            bias_diagonal.permute(0, 3, 2, 1).reshape(-1),
            polynomial_diagonal,
        ))
        return state, gradient, matvec, diagonal


class TorchFNIRT:
    """Matrix-free PyTorch FNIRT for supported GM and T1 schedules.

    The optimiser is genuine Gauss-Newton/Levenberg-Marquardt with analytic
    B-spline forward/adjoint operators and PCG. T1 additionally fits intensity
    and bias at configured levels. Its trajectory is not numerically
    identical to FSL FNIRT.
    """

    def __init__(
        self,
        *,
        device="cpu",
        config: FNIRTConfig | None = None,
        reference_mask=None,
        pcg_tolerance=1e-3,
        pcg_max_iterations=500,
        cost_tolerance=1e-8,
        initial_lm_lambda=0.1,
        strict_topology=False,
        execution="optimized",
    ):
        if execution not in ("reference", "optimized"):
            raise ValueError("execution must be reference or optimized")
        self.execution = execution
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.config = resolve_fnirt_config(config)
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
        moving = load_image(moving, "moving")
        fixed = load_image(fixed, "fixed")
        initial = _world_affine(moving_to_fixed, moving, fixed)
        moving_data = np.asarray(moving.dataobj, dtype=np.float32).squeeze()
        fixed_data = np.asarray(fixed.dataobj, dtype=np.float32).squeeze()
        if moving_data.ndim != 3 or fixed_data.ndim != 3:
            raise ValueError("moving and fixed must each contain one 3D frame")
        if not np.isfinite(moving_data).all() or not np.isfinite(fixed_data).all():
            raise ValueError("moving and fixed must contain only finite values")

        selected_mask = self.reference_mask if reference_mask is None else reference_mask
        mask_data = None
        if selected_mask is not None:
            selected_mask = load_image(selected_mask, "reference_mask")
            if selected_mask.shape[:3] != fixed.shape[:3] or not np.allclose(
                selected_mask.affine,
                fixed.affine,
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError("reference mask must be on the fixed image grid")
            mask_data = np.asanyarray(selected_mask.dataobj).squeeze() > 0.5

        device = self.device
        # newimage reads sampling_mat and physical resolutions from pixdim;
        # a sheared sform's column norms are a different quantity.
        moving_voxel_sizes = tuple(float(value) for value in moving.header.get_zooms()[:3])
        fixed_voxel_sizes = tuple(float(value) for value in fixed.header.get_zooms()[:3])
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
        explicit_reference_mask = (
            None
            if mask_data is None
            else torch.from_numpy(mask_data.copy()).to(device)
        )
        implicit_reference_mask = (
            torch.from_numpy(
                (np.abs(fixed_data.astype(np.float64)) >= 1e-16).copy()
            ).to(device)
            if self.config.implicit_reference_mask
            else None
        )
        # FNIRT builds the input implicit mask after in-place mean scaling,
        # while it builds the reference implicit mask before scaling.
        implicit_input_mask = (
            moving_tensor.abs() >= 1e-16
            if self.config.implicit_input_mask
            else None
        )

        moving_fsl_array = voxel_to_fsl_scaled_mm(
            moving.affine, moving_data.shape, moving_voxel_sizes
        )
        fixed_fsl_array = voxel_to_fsl_scaled_mm(
            fixed.affine, fixed_data.shape, fixed_voxel_sizes
        )
        forward_array = world_to_flirt_affine(
            initial,
            moving.affine,
            fixed.affine,
            moving_data.shape,
            fixed_data.shape,
            moving_voxel_sizes,
            fixed_voxel_sizes,
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
        stage_forward_array = np.asarray(forward_array, dtype=np.float64)

        fixed_shape = tuple(int(value) for value in fixed_data.shape)
        resolution_schedule = self.config.warp_resolution_schedule_mm
        if resolution_schedule is None:
            resolution_schedule = (self.config.warp_resolution_mm,) * len(
                self.config.subsampling
            )
        minimization_methods = self.config.minimization_methods
        if minimization_methods is None:
            minimization_methods = ("lm",) * len(self.config.subsampling)
        process_stages = self.config.process_stages
        if process_stages is None:
            process_stages = (1,) * len(self.config.subsampling)
        knot_spacing_schedule = _process_knot_spacing_schedule(
            resolution_schedule,
            process_stages,
            self.config.subsampling,
            fixed_voxel_sizes,
        )
        knot_spacing = None
        coefficients = None
        previous_stride = None
        previous_level_voxel_sizes = None
        previous_bases = None
        previous_stage = None
        scale = torch.ones((), device=device, dtype=dtype)
        t1_polynomial = None
        t1_bias_coefficients = None
        t1_bias_spacing = None
        levels = []

        for level, (
            stride,
            maximum_iterations,
            input_fwhm,
            reference_fwhm,
            regularization,
            estimate_intensity,
            apply_reference_mask,
            minimization_method,
            process_stage,
            warp_resolution,
            full_knot_spacing,
        ) in enumerate(
            zip(
                self.config.subsampling,
                self.config.maximum_iterations,
                self.config.input_fwhm_mm,
                self.config.reference_fwhm_mm,
                self.config.regularization,
                self.config.estimate_intensity,
                self.config.apply_reference_mask,
                minimization_methods,
                process_stages,
                resolution_schedule,
                knot_spacing_schedule,
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
            new_knot_spacing = tuple(full_knot_spacing)
            stage_boundary = (
                coefficients is not None and process_stage != previous_stage
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
                        old_knot_spacing=knot_spacing,
                    )
                if stage_boundary:
                    # ``--cout`` stores float coefficients and ``--intout``
                    # writes global parameters with MISCMATHS precision 10.
                    coefficients = coefficients.to(image_dtype).to(dtype)
                    stage_forward_array = np.asarray(
                        stage_forward_array, dtype=np.float32
                    ).astype(np.float64)
                    affine_pull_exact = torch.linalg.inv(
                        torch.as_tensor(
                            stage_forward_array, device=device, dtype=dtype
                        )
                    )
                    affine_pull = affine_pull_exact.to(image_dtype)
                    scale = torch.tensor(
                        float(format(float(scale), ".10g")),
                        device=device,
                        dtype=dtype,
                    )
                if new_knot_spacing != knot_spacing:
                    coefficients = zoom_coefficients(
                        coefficients,
                        level_shape,
                        new_knot_spacing,
                        level_voxel_sizes,
                        level_voxel_sizes,
                        old_knot_spacing=knot_spacing,
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

            if self.config.intensity_model == "global_non_linear_with_bias":
                initial_stride = self.config.subsampling[0]
                initial_bias_spacing = tuple(
                    max(1, int(math.floor(resolution / (voxel * initial_stride) + 0.5)))
                    for resolution, voxel in zip(
                        self.config.bias_resolution_mm, fixed_voxel_sizes
                    )
                )
                next_bias_spacing = tuple(
                    (initial_stride // stride) * spacing
                    for spacing in initial_bias_spacing
                )
                if t1_bias_coefficients is None:
                    t1_polynomial = torch.zeros(
                        self.config.intensity_order, device=device, dtype=dtype
                    )
                    t1_polynomial[1] = 1.0
                    t1_bias_coefficients = torch.ones(
                        (1, *fsl_control_shape(level_shape, next_bias_spacing)),
                        device=device, dtype=dtype,
                    )
                elif stride != previous_stride:
                    t1_bias_coefficients = zoom_coefficients(
                        t1_bias_coefficients,
                        level_shape,
                        t1_bias_spacing,
                        previous_level_voxel_sizes,
                        level_voxel_sizes,
                    )
                    if next_bias_spacing != t1_bias_spacing:
                        t1_bias_coefficients = zoom_coefficients(
                            t1_bias_coefficients,
                            level_shape,
                            next_bias_spacing,
                            level_voxel_sizes,
                            level_voxel_sizes,
                            old_knot_spacing=t1_bias_spacing,
                        )
                t1_bias_spacing = next_bias_spacing
                bias_bases = spline_bases(
                    level_shape, t1_bias_spacing, level_voxel_sizes,
                    device=device, dtype=dtype, positions=level_positions,
                )
                bias_bending = BendingOperator(
                    level_shape, t1_bias_spacing, level_voxel_sizes,
                    device=device, dtype=dtype,
                    execution=self.execution,
                )

            moving_level = _fsl_masked_gaussian_blur(
                moving_tensor[None, None],
                input_fwhm,
                moving_voxel_sizes,
                implicit_input_mask,
                execution=self.execution,
            )[0, 0]
            blur = _fsl_gaussian_blur if self.execution == "optimized" else _fsl_gaussian_blur_reference
            fixed_smoothed = blur(
                fixed_tensor[None, None], reference_fwhm, fixed_voxel_sizes
            )[0, 0]
            fixed_level = _take_integer_grid(fixed_smoothed, full_positions)
            combined_reference_mask = implicit_reference_mask
            if apply_reference_mask and explicit_reference_mask is not None:
                combined_reference_mask = (
                    explicit_reference_mask
                    if combined_reference_mask is None
                    else combined_reference_mask & explicit_reference_mask
                )
            reference_mask_level = None
            if combined_reference_mask is not None:
                reference_mask_level = _take_integer_grid(
                    combined_reference_mask.to(dtype), full_positions
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
                execution=self.execution,
            )
            system = _LevelSystem(
                moving_level,
                fixed_level,
                reference_mask_level,
                (
                    None
                    if implicit_input_mask is None
                    else implicit_input_mask.to(image_dtype)
                ),
                moving_fsl2vox,
                target_fsl,
                affine_pull,
                bases,
                bending,
                regularization,
                self.config.ssd_weighted_lambda,
                estimate_intensity and self.config.intensity_model == "global_linear",
                coordinate_affine,
            )
            joint_system = None
            if self.config.intensity_model == "global_non_linear_with_bias":
                joint_system = _JointT1System(
                    system, bias_bases, bias_bending,
                    self.config.bias_regularization,
                    self.config.intensity_order,
                )
                scale = torch.ones((), device=device, dtype=dtype)
            optimize_scale = estimate_intensity and self.config.intensity_model == "global_linear"

            lm_lambda = self.initial_lm_lambda
            accepted = 0
            attempts = 0
            converged = False
            pcg_reports = []
            scg_history = []
            state = (
                joint_system.evaluate(coefficients, t1_polynomial, t1_bias_coefficients)
                if joint_system is not None else system.evaluate(coefficients, scale)
            )
            if minimization_method == "scg":
                if joint_system is not None and estimate_intensity:
                    raise ValueError("FSL does not estimate intensity with SCG")
                initial_vector = _pack(
                    coefficients, scale if optimize_scale else None
                )
                fixed_scale = scale
                latest_ssd = None

                def unpack_parameters(vector):
                    current_coefficients, current_scale = _unpack(
                        vector, coefficient_shape, optimize_scale
                    )
                    if current_scale is None:
                        current_scale = fixed_scale
                    return current_coefficients, current_scale

                def scg_cost(vector):
                    nonlocal latest_ssd
                    current_coefficients, current_scale = unpack_parameters(vector)
                    current_state = system.evaluate(
                        current_coefficients, current_scale
                    )
                    latest_ssd = float(current_state["ssd"])
                    return current_state["cost"]

                def scg_gradient(vector):
                    current_coefficients, current_scale = unpack_parameters(vector)
                    effective_lambda = regularization
                    if self.config.ssd_weighted_lambda:
                        if latest_ssd is None:
                            raise RuntimeError("SCG cost must be evaluated first")
                        effective_lambda *= latest_ssd
                    _, value = system.gradient(
                        current_coefficients,
                        current_scale,
                        effective_lambda=effective_lambda,
                        _skip_cost=self.device.type == "cpu",
                    )
                    return value

                result_vector, scg = scaled_conjugate_gradient(
                    scg_cost,
                    scg_gradient,
                    initial_vector,
                    max_iterations=maximum_iterations,
                    initial_lambda=self.initial_lm_lambda,
                )
                coefficients, updated_scale = unpack_parameters(result_vector)
                scale = updated_scale
                state = system.evaluate(coefficients, scale)
                attempts = scg.iterations
                accepted = scg.accepted_iterations
                converged = scg.converged
                lm_lambda = scg.lambda_final
                scg_history = list(scg.history)
            else:
                while accepted < maximum_iterations and attempts < 10 * max(
                    1, maximum_iterations
                ):
                    attempts += 1
                    if joint_system is None:
                        state, gradient, matvec, diagonal = system.linearize(
                            coefficients, scale
                        )
                    else:
                        state, gradient, matvec, diagonal = joint_system.linearize(
                            coefficients, t1_polynomial, t1_bias_coefficients,
                            fit_intensity=estimate_intensity,
                        )
                    damping_diagonal = diagonal.clamp_min(
                        torch.finfo(dtype).eps
                        * diagonal.abs().mean().clamp_min(1)
                    )

                    def damped(value):
                        return (
                            matvec(value)
                            + lm_lambda * damping_diagonal * value
                        )

                    step, pcg = preconditioned_conjugate_gradient(
                        damped,
                        -gradient,
                        diagonal=(1 + lm_lambda) * damping_diagonal,
                        tolerance=self.pcg_tolerance,
                        max_iterations=self.pcg_max_iterations,
                        execution=self.execution,
                    )
                    pcg_reports.append(
                        {
                            "iterations": pcg.iterations,
                            "converged": pcg.converged,
                            "relative_residual": pcg.relative_residual,
                        }
                    )
                    delta_bias = None
                    delta_polynomial = None
                    if joint_system is not None and estimate_intensity:
                        delta_coefficients, delta_bias, delta_polynomial = _unpack_t1(
                            step, coefficient_shape,
                            tuple(t1_bias_coefficients.shape[1:]),
                        )
                        delta_scale = None
                    else:
                        delta_coefficients, delta_scale = _unpack(
                            step, coefficient_shape, optimize_scale
                        )
                    candidate_coefficients = coefficients + delta_coefficients
                    candidate_scale = (
                        scale + delta_scale if delta_scale is not None else scale
                    )
                    candidate_polynomial = (
                        t1_polynomial + delta_polynomial
                        if delta_polynomial is not None else t1_polynomial
                    )
                    candidate_bias = (
                        t1_bias_coefficients + delta_bias
                        if delta_bias is not None else t1_bias_coefficients
                    )
                    candidate = (
                        joint_system.evaluate(
                            candidate_coefficients, candidate_polynomial,
                            candidate_bias,
                        ) if joint_system is not None else system.evaluate(
                            candidate_coefficients, candidate_scale
                        )
                    )
                    if bool(torch.isfinite(candidate["cost"])) and float(
                        candidate["cost"]
                    ) < float(state["cost"]):
                        old_cost = float(state["cost"])
                        new_cost = float(candidate["cost"])
                        coefficients = candidate_coefficients
                        scale = candidate_scale
                        t1_polynomial = candidate_polynomial
                        t1_bias_coefficients = candidate_bias
                        state = candidate
                        accepted += 1
                        lm_lambda /= 10.0
                        converged = (
                            2 * abs(old_cost - new_cost)
                            <= self.cost_tolerance
                            * (
                                abs(old_cost)
                                + abs(new_cost)
                                + torch.finfo(dtype).eps
                            )
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
                state = (
                    joint_system.evaluate(
                        coefficients, t1_polynomial, t1_bias_coefficients
                    ) if joint_system is not None else system.evaluate(coefficients, scale)
                )
            t1_bias = (
                expand_coefficients(t1_bias_coefficients, bias_bases)[0]
                if joint_system is not None else None
            )
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
                    "t1_polynomial": (
                        None if t1_polynomial is None
                        else [float(value) for value in t1_polynomial]
                    ),
                    "t1_bias_range": (
                        None if t1_bias is None
                        else [float(t1_bias.min()), float(t1_bias.max())]
                    ),
                    "t1_bias_bending_energy": (
                        None if joint_system is None
                        else float(state["bias_energy"])
                    ),
                    "apply_reference_mask": apply_reference_mask,
                    "explicit_reference_mask_available": (
                        explicit_reference_mask is not None
                    ),
                    "implicit_reference_mask": (
                        implicit_reference_mask is not None
                    ),
                    "implicit_input_mask": implicit_input_mask is not None,
                    "mask_voxels": state["count"],
                    "minimization_method": minimization_method,
                    "process_stage": process_stage,
                    "process_boundary_handoff": stage_boundary,
                    "lm_or_scg_lambda_final": lm_lambda,
                    "pcg": pcg_reports,
                    "scg": scg_history,
                    "full_jacobian_min_before_projection": jacobian_min,
                    "full_jacobian_max_before_projection": jacobian_max,
                    "topology_projection_required": topology_projection_required,
                    "topology_projection": topology_qc,
                }
            )
            previous_stride = stride
            previous_level_voxel_sizes = level_voxel_sizes
            previous_bases = bases
            previous_stage = process_stage

        final_output_upsampled = previous_stride != 1
        if final_output_upsampled:
            coefficients = zoom_coefficients(
                coefficients,
                fixed_shape,
                knot_spacing,
                previous_level_voxel_sizes,
                fixed_voxel_sizes,
                old_knot_spacing=knot_spacing,
            )
            previous_bases = spline_bases(
                fixed_shape,
                knot_spacing,
                fixed_voxel_sizes,
                device=device,
                dtype=dtype,
            )
            previous_stride = 1
            previous_level_voxel_sizes = fixed_voxel_sizes
        with torch.no_grad():
            field = expand_coefficients(coefficients, previous_bases).to(
                image_dtype
            )[None]
            constrained_coefficients = coefficients
            full_positions = tuple(
                torch.arange(size, device=device, dtype=image_dtype)
                for size in fixed_shape
            )
            source_voxels = _fsl_displacement_coordinates(
                field[0], affine_pull_exact @ fixed_fsl_exact,
                moving_fsl2vox,
            )
            moved, _, _ = _trilinear_sample(moving_raw, source_voxels)

            nonlinear_jacobian = _spline_jacobian(
                constrained_coefficients,
                fixed_shape,
                knot_spacing,
                fixed_voxel_sizes,
            )

            fixed_world = torch.as_tensor(
                np.array(fixed.affine, dtype=np.float32, copy=True),
                device=device,
            )
            moving_world = torch.as_tensor(
                np.array(moving.affine, dtype=np.float32, copy=True),
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
        pull_transform = DenseWarp(
            displacement_array, source=moving, target=fixed
        )
        pull_transform.header["pixdim"][1:4] = fixed_voxel_sizes
        full_world, _, affine_pull_determinant = _pull_jacobian_determinants(
            displacement_array,
            fixed.affine,
            flirt_to_world_affine(
                stage_forward_array,
                moving.affine,
                fixed.affine,
                moving_data.shape,
                fixed_data.shape,
                moving_voxel_sizes,
                fixed_voxel_sizes,
            ),
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
            stage_forward_array,
        )
        qc = {
            "backend": "pytorch-fnirt",
            "execution": self.execution,
            "bending_normal": "coefficient-space Gram" if self.execution == "optimized" else "dense forward-adjoint",
            "device": str(self.device),
            "tf32": {
                "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "reduced_precision_tensor_dtype": False,
            },
            "fsl_fnirt_numerically_equivalent": False,
            "equivalence_status": "external FSL 6.0.7.4 numerical gate not passed",
            "fsl_source_versions": FSL_SOURCE_VERSIONS,
            "optimizer": list(minimization_methods),
            "hessian": "analytic matrix-free B-spline JtJ plus bending Hessian",
            "global_intensity_model": self.config.intensity_model,
            "t1_intensity_fitting": (
                "joint LM polynomial, cubic bias and deformation estimation"
                if self.config.intensity_model == "global_non_linear_with_bias"
                else None
            ),
            "ssd_weighted_lambda": self.config.ssd_weighted_lambda,
            "mask_schedule_matches_gm_config": (
                explicit_reference_mask is not None
                or not any(self.config.apply_reference_mask)
            ),
            "implicit_reference_mask": self.config.implicit_reference_mask,
            "implicit_input_mask": self.config.implicit_input_mask,
            "masked_input_smoothing": self.config.implicit_input_mask,
            "process_stages": list(process_stages),
            "process_full_resolution_knot_spacing_voxels": [
                list(value) for value in knot_spacing_schedule
            ],
            "process_handoff_float32_coefficients": True,
            "process_handoff_float32_affine_header": True,
            "process_handoff_intensity_precision": 10,
            "final_output_upsampled_to_reference_grid": final_output_upsampled,
            "intensity_schedule_matches_gm_config": (
                self.config.intensity_model == "global_linear"
            ),
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
            "reference_mask_available": (
                explicit_reference_mask is not None
                or implicit_reference_mask is not None
            ),
            "knot_spacing_voxels": list(knot_spacing),
            "control_grid_shape": list(coefficients.shape[1:]),
            "levels": levels,
            "final_intensity_scale": float(scale),
            "nonlinear_jacobian_min": float(nonlinear_jacobian.min()),
            "nonlinear_jacobian_max": float(nonlinear_jacobian.max()),
            "full_pull_jacobian_min": float(full_world.min()),
            "full_pull_jacobian_max": float(full_world.max()),
        }
        def output_image(array):
            image = new_image(array, fixed)
            image.header["pixdim"][1:4] = fixed_voxel_sizes
            return image

        return TorchFNIRTResult(
            moved=output_image(moved_array),
            pull_transform=pull_transform,
            full_pull_jacobian=output_image(full_array),
            nonlinear_jacobian=output_image(nonlinear_array),
            modulated_gm=output_image(moved_array * nonlinear_array),
            affine_pull_determinant=float(affine_pull_determinant),
            coefficients=coefficient_array,
            coefficient_image=coefficient_image,
            qc=qc,
        )


__all__ = [
    "FSL_SOURCE_VERSIONS",
    "FNIRTConfig",
    "GMFNIRTConfig",
    "T1FNIRTConfig",
    "TBSSFNIRTConfig",
    "TorchFNIRT",
    "TorchFNIRTResult",
    "resolve_fnirt_config",
    "spm_like_mean",
]
