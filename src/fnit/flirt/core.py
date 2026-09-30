"""PyTorch ports of FSL FLIRT 2111.2 affine and rigid registration profiles.

The implementation in this module follows the code path used by ``fsl_reg``
for UK Biobank VBM: 12 degrees of freedom, correlation-ratio cost, the default
8/4/2/1 mm schedule, and the MISCMATHS Brent-style coordinate optimiser.
The connectome rigid path uses the same schedule with 6 degrees of freedom and
normalized mutual information.

FSL stores affine matrices in scaled-mm coordinates.  All matrices used by the
optimizer below are in that coordinate system and map input to reference.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
import math
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .._nib import new_image
from .coordinates import flirt_to_world_affine, voxel_to_fsl_scaled_mm, world_to_flirt_affine
from .types import FLIRTResult, _load_volume, _single_frame


FSL_FLIRT_VERSION = "2111.2"
FSL_FLIRT_COMMIT = "5036b4620ea97db0050f2dc132fbb331dbba060c"
FSL_NEWIMAGE_VERSION = "2203.11"
FSL_MISCMATHS_VERSION = "2203.2"

_BASE_TOLERANCE = np.array(
    [0.005, 0.005, 0.005, 0.2, 0.2, 0.2,
     0.002, 0.002, 0.002, 0.001, 0.001, 0.001],
    dtype=np.float64,
)
_IDENTITY_PARAMETERS = np.array(
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64
)


def _homogeneous(linear, offset, *, dtype, device):
    matrix = torch.eye(4, dtype=dtype, device=device)
    matrix[:3, :3] = linear
    matrix[:3, 3] = offset
    return matrix


def _axis_rotation(angle, axis, centre):
    """Return FSL's signed Euler rotation about ``centre``."""
    angles = torch.zeros(3, dtype=angle.dtype, device=angle.device)
    angles[axis] = angle
    # MISCMATHS::make_rot stores theta, sin(theta), and cos(theta) as float
    # while NEWMAT vectors and matrices use double precision.
    theta = torch.linalg.vector_norm(angles).to(torch.float32)
    if float(theta) < 1e-8:
        return torch.eye(4, dtype=angle.dtype, device=angle.device)
    direction = angles / theta.to(angle.dtype)
    x1 = direction
    x2 = torch.stack((-direction[1], direction[0], direction.new_zeros(())))
    if float(torch.linalg.vector_norm(x2)) <= 0:
        x2 = torch.tensor([1.0, 0.0, 0.0], dtype=angle.dtype, device=angle.device)
    x2 = x2 / torch.linalg.vector_norm(x2)
    x3 = torch.linalg.cross(x1, x2)
    x3 = x3 / torch.linalg.vector_norm(x3)
    basis = torch.stack((x2, x3, x1), dim=1)
    cosine = torch.cos(theta).to(angle.dtype)
    sine = torch.sin(theta).to(angle.dtype)
    core = torch.eye(3, dtype=angle.dtype, device=angle.device)
    core[0, 0] = core[1, 1] = cosine
    core[0, 1] = sine
    core[1, 0] = -sine
    linear = basis @ core @ basis.T
    return _homogeneous(
        linear, centre - linear @ centre, dtype=angle.dtype, device=angle.device
    )


def fsl_affine_from_parameters(parameters, centre, dof=12):
    """Compose the affine used by FLIRT's Euler 12-parameter model.

    Parameters are rotations, translations, scales, then xy/xz/yz skews.
    Seven degrees of freedom uses one common scale; eight degrees of freedom
    is not a supported FLIRT model and is rejected.
    """
    parameters = torch.as_tensor(parameters)
    centre = torch.as_tensor(
        centre, dtype=parameters.dtype, device=parameters.device
    )
    if parameters.shape != (12,) or centre.shape != (3,):
        raise ValueError("parameters and centre must have shapes (12,) and (3,)")
    if dof not in (6, 7, 9, 10, 11, 12):
        raise ValueError("dof must be one of 6, 7, 9, 10, 11, or 12")

    affine = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    for axis in range(3):
        affine = affine @ _axis_rotation(parameters[axis], axis, centre)
    affine[:3, 3] += parameters[3:6]
    if dof == 6:
        return affine

    scales = torch.ones(3, dtype=parameters.dtype, device=parameters.device)
    scales[0] = parameters[6]
    scales[1] = parameters[7] if dof >= 9 else parameters[6]
    scales[2] = parameters[8] if dof >= 9 else parameters[6]
    scale_linear = torch.diag(scales)
    scale = _homogeneous(
        scale_linear,
        centre - scale_linear @ centre,
        dtype=parameters.dtype,
        device=parameters.device,
    )

    skew_linear = torch.eye(3, dtype=parameters.dtype, device=parameters.device)
    if dof >= 10:
        skew_linear[0, 1] = parameters[9]
    if dof >= 11:
        skew_linear[0, 2] = parameters[10]
    if dof >= 12:
        skew_linear[1, 2] = parameters[11]
    skew = _homogeneous(
        skew_linear,
        centre - skew_linear @ centre,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    return affine @ skew @ scale


def fsl_parameters_from_affine(affine, centre):
    """Decompose an affine using FLIRT's rotation-skew-scale convention."""
    affine = np.asarray(affine, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    if affine.shape != (4, 4) or centre.shape != (3,):
        raise ValueError("affine and centre must have shapes (4, 4) and (3,)")
    linear = affine[:3, :3]
    x, y, z = linear[:, 0], linear[:, 1], linear[:, 2]
    # MISCMATHS ``decompose_aff`` stores these six intermediates as float,
    # even though NEWMAT's input vectors and subsequent matrices are double.
    # The casts therefore belong at assignment, not at function entry.
    sx = np.float32(np.linalg.norm(x))
    if sx <= 1e-12:
        raise ValueError("affine has a singular x scale")
    xy = float(x @ y)
    sx_squared = np.float32(sx * sx)
    sy_sq = float(y @ y - xy**2 / float(sx_squared))
    sy = np.float32(math.sqrt(max(sy_sq, 0.0)))
    if sy <= 1e-12:
        raise ValueError("affine has a singular y scale")
    a = np.float32(xy / float(np.float32(sx * sy)))
    x0 = x / float(sx)
    y0 = y / float(sy) - float(a) * x0
    sz_sq = float(z @ z - (x0 @ z) ** 2 - (y0 @ z) ** 2)
    sz = np.float32(math.sqrt(max(sz_sq, 0.0)))
    if sz <= 1e-12:
        raise ValueError("affine has a singular z scale")
    b = np.float32(float(x0 @ z) / float(sz))
    c = np.float32(float(y0 @ z) / float(sz))
    scale = np.diag([float(sx), float(sy), float(sz)])
    skew = np.array(
        [[1, float(a), float(b)], [0, 1, float(c)], [0, 0, 1]],
        dtype=np.float64,
    )
    rotation = linear @ np.linalg.inv(scale) @ np.linalg.inv(skew)

    # ``rotmat2euler`` likewise narrows each named scalar to float before the
    # branch and atan2 calls.
    cy = np.float32(
        math.sqrt(rotation[0, 0] ** 2 + rotation[0, 1] ** 2)
    )
    if cy < 1e-4:
        cos_x = np.float32(rotation[1, 1])
        sin_x = np.float32(-rotation[2, 1])
        sin_y = np.float32(-rotation[0, 2])
        rx = np.float32(math.atan2(float(sin_x), float(cos_x)))
        ry = np.float32(math.atan2(float(sin_y), 0.0))
        rz = np.float32(0.0)
    else:
        cos_z = np.float32(rotation[0, 0] / float(cy))
        sin_z = np.float32(rotation[0, 1] / float(cy))
        cos_x = np.float32(rotation[2, 2] / float(cy))
        sin_x = np.float32(rotation[1, 2] / float(cy))
        sin_y = np.float32(-rotation[0, 2])
        rx = np.float32(math.atan2(float(sin_x), float(cos_x)))
        ry = np.float32(math.atan2(float(sin_y), float(cy)))
        rz = np.float32(math.atan2(float(sin_z), float(cos_z)))
    translation = linear @ centre + affine[:3, 3] - centre
    return np.array(
        [rx, ry, rz, *translation, sx, sy, sz, a, b, c],
        dtype=np.float64,
    )


def _fsl_round(value):
    return int(math.floor(float(value) + 0.5))


def _robust_limits(data):
    """Port of NEWIMAGE ``find_thresholds`` used by FLIRT clamping."""
    values = np.asarray(data, dtype=np.float32).reshape(-1)
    minimum = np.float32(values.min())
    maximum = np.float32(values.max())
    original_minimum, original_maximum = minimum, maximum
    bins = 1000
    top_bin = bottom_bin = 0
    lowest_bin, highest_bin = 0, bins - 1
    pass_index = 1
    threshold_2 = threshold_98 = np.float32(0)
    while pass_index == 1 or float(threshold_98 - threshold_2) < float(
        maximum - minimum
    ) / 10.0:
        if pass_index > 1:
            bottom_bin = max(bottom_bin - 1, 0)
            top_bin = min(top_bin + 1, bins - 1)
            old_minimum = minimum
            minimum = np.float32(
                old_minimum + bottom_bin / bins * (maximum - old_minimum)
            )
            maximum = np.float32(
                old_minimum + (top_bin + 1) / bins * (maximum - old_minimum)
            )
        if pass_index == 10 or minimum == maximum:
            minimum, maximum = original_minimum, original_maximum
        if maximum <= minimum:
            return float(minimum), float(maximum)

        factor_a = bins / float(maximum - minimum)
        factor_b = bins * -float(minimum) / float(maximum - minimum)
        indices = np.trunc(values.astype(np.float64) * factor_a + factor_b)
        indices = np.clip(indices.astype(np.int64), 0, bins - 1)
        histogram = np.bincount(indices, minlength=bins)
        valid_size = int(values.size)
        if pass_index == 10:
            valid_size -= int(histogram[lowest_bin] + histogram[highest_bin])
            lowest_bin += 1
            highest_bin -= 1
        if valid_size < 0:
            threshold_2 = threshold_98 = minimum
            break
        target = valid_size // 50
        count = 0
        bottom_bin = lowest_bin
        while count < target:
            count += int(histogram[bottom_bin])
            bottom_bin += 1
        bottom_bin -= 1
        bin_width = float(maximum - minimum) / bins
        threshold_2 = np.float32(float(minimum) + bottom_bin * bin_width)
        count = 0
        top_bin = highest_bin
        while count < target:
            count += int(histogram[top_bin])
            top_bin -= 1
        top_bin += 1
        threshold_98 = np.float32(
            float(minimum) + (top_bin + 1) * bin_width
        )
        if pass_index == 10:
            break
        pass_index += 1
    return float(threshold_2), float(threshold_98)


def _clamp_like_fsl(data):
    lower, upper = _robust_limits(data)
    return np.clip(np.asarray(data, dtype=np.float32), lower, upper)


def _blur_kernel(final_size, initial_size, *, device):
    ratio = float(final_size) / float(initial_size)
    if ratio < 1.1:
        return torch.ones(1, dtype=torch.float32, device=device)
    sigma = 0.85 * ratio / 2.0
    if sigma < 0.5:
        return torch.ones(1, dtype=torch.float32, device=device)
    length = int(sigma - 0.001) * 2 + 3
    coordinate = torch.arange(
        length, dtype=torch.float32, device=device
    ) - length // 2
    kernel = torch.exp(-coordinate.square() / (sigma * sigma * 4.0))
    return kernel / kernel.sum()


def _convolve_axis(data, kernel, axis, *, boundary="zero"):
    if kernel.numel() == 1:
        return data
    radius = int(kernel.numel()) // 2
    pad_shape = list(data.shape)
    pad_shape[axis] = radius
    if boundary == "replicate":
        left = data.narrow(axis, 0, 1).expand(pad_shape)
        right = data.narrow(axis, data.shape[axis] - 1, 1).expand(pad_shape)
    else:
        left = right = data.new_zeros(pad_shape)
    padded = torch.cat((left, data, right), dim=axis)
    result = torch.zeros_like(data)
    for index, coefficient in enumerate(kernel):
        result.add_(padded.narrow(axis, index, data.shape[axis]) * coefficient)
    return result


def _blur(data, final_size, voxel_sizes, *, boundary="zero"):
    result = data
    for axis in range(3):
        result = _convolve_axis(
            result,
            _blur_kernel(final_size, voxel_sizes[axis], device=data.device),
            axis,
            boundary=boundary,
        )
    return result


def _filter_weight(weight, transform):
    """Port FLIRT filter_weight for a blur or subsampling transform."""
    binary = (weight > 0.01).to(torch.float32)
    support = transform(binary) > 0.9
    return transform(weight) * support


def _filter_image_with_weight(image, weight, transform):
    """Port FLIRT weighted image filtering with its safe division."""
    binary = (weight > 0.01).to(torch.float32)
    denominator = transform(binary)
    numerator = transform(image * binary)
    return torch.where(denominator != 0, numerator / denominator, denominator)


def _subsample_by_two(data):
    """NEWIMAGE's centred 3-D half-sampling filter."""
    device = data.device
    kernel = torch.tensor(
        [0.25, 0.5, 0.25], dtype=torch.float32, device=device
    )
    # NEWIMAGE uses rounded constants for edge and corner weights.
    exact = _convolve_axis(_convolve_axis(_convolve_axis(data, kernel, 0), kernel, 1), kernel, 2)
    correction = torch.zeros_like(data)
    # The source constants are 0.0312 and 0.0156 rather than 1/32 and 1/64.
    # Correcting the full 3-D kernel preserves that source-level arithmetic.
    padded = F.pad(data[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
    for dx in (-1, 1):
        for dy in (-1, 1):
            correction += -0.00005 * padded[
                1 + dx:1 + dx + data.shape[0],
                1 + dy:1 + dy + data.shape[1],
                1:1 + data.shape[2],
            ]
            for dz in (-1, 1):
                correction += -0.000025 * padded[
                    1 + dx:1 + dx + data.shape[0],
                    1 + dy:1 + dy + data.shape[1],
                    1 + dz:1 + dz + data.shape[2],
                ]
    for dx in (-1, 1):
        for dz in (-1, 1):
            correction += -0.00005 * padded[
                1 + dx:1 + dx + data.shape[0],
                1:1 + data.shape[1],
                1 + dz:1 + dz + data.shape[2],
            ]
    for dy in (-1, 1):
        for dz in (-1, 1):
            correction += -0.00005 * padded[
                1:1 + data.shape[0],
                1 + dy:1 + dy + data.shape[1],
                1 + dz:1 + dz + data.shape[2],
            ]
    return (exact + correction)[::2, ::2, ::2].contiguous()


def _manual_trilinear(data, coordinates):
    """FSL-order trilinear interpolation for in-bounds coordinates."""
    lower = torch.floor(coordinates).to(torch.long)
    maximum = torch.tensor(data.shape, device=data.device)[:, None] - 2
    lower = torch.minimum(lower, maximum)
    lower = torch.clamp(lower, min=0)
    delta = coordinates - lower.to(coordinates.dtype)
    ix, iy, iz = lower
    dx, dy, dz = delta
    v000 = data[ix, iy, iz]
    v001 = data[ix, iy, iz + 1]
    v010 = data[ix, iy + 1, iz]
    v011 = data[ix, iy + 1, iz + 1]
    v100 = data[ix + 1, iy, iz]
    v101 = data[ix + 1, iy, iz + 1]
    v110 = data[ix + 1, iy + 1, iz]
    v111 = data[ix + 1, iy + 1, iz + 1]
    temp1 = (v100 - v000) * dx + v000
    temp2 = (v101 - v001) * dx + v001
    temp3 = (v110 - v010) * dx + v010
    temp4 = (v111 - v011) * dx + v011
    temp5 = (temp3 - temp1) * dy + temp1
    temp6 = (temp4 - temp2) * dy + temp2
    return (temp6 - temp5) * dz + temp5


def _voxel_grid(shape, *, device):
    axes = torch.meshgrid(
        *[torch.arange(size, dtype=torch.float32, device=device) for size in shape],
        indexing="ij",
    )
    return torch.stack(axes).reshape(3, -1)


def _fsl_pull_coefficients(
    moving_to_reference,
    moving_voxel_sizes,
    reference_voxel_sizes,
    *,
    device,
):
    """Return the 12 float coefficients used by NEWIMAGE cost functions."""
    affine = np.asarray(moving_to_reference, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("moving_to_reference must have shape (4, 4)")
    moving_sampling_inverse = np.diag(
        [
            1 / float(moving_voxel_sizes[0]),
            1 / float(moving_voxel_sizes[1]),
            1 / float(moving_voxel_sizes[2]),
            1,
        ]
    )
    reference_sampling = np.diag(
        [
            float(reference_voxel_sizes[0]),
            float(reference_voxel_sizes[1]),
            float(reference_voxel_sizes[2]),
            1,
        ]
    )
    # NEWIMAGE forms iaffbig with NEWMAT Real (double), then assigns its first
    # three rows to twelve local float variables a11 ... a34.
    pull = moving_sampling_inverse @ np.linalg.inv(affine) @ reference_sampling
    coefficients = np.asarray(pull[:3, :], dtype=np.float32)
    return torch.as_tensor(coefficients, dtype=torch.float32, device=device)


def _coordinates_from_fsl_coefficients(coefficients, grid):
    """Evaluate FSL float affine coefficients without a TF32 matrix multiply."""
    x, y, z = grid.unbind(0)
    rows = []
    for row in coefficients:
        # Match the grouping at x=0 in p_corr_ratio_smoothed, followed by the
        # x contribution.  Scalar elementwise operations do not use TF32.
        coordinate = torch.add(torch.mul(y, row[1]), torch.mul(z, row[2]))
        coordinate = torch.add(coordinate, row[3])
        coordinate = torch.add(coordinate, torch.mul(x, row[0]))
        rows.append(coordinate)
    return torch.stack(rows)


class FSLCorrelationRatio:
    """FSL 2111.2 correlation-ratio cost on a PyTorch device."""

    def __init__(
        self,
        reference,
        moving,
        reference_voxel_sizes,
        moving_voxel_sizes,
        *,
        bins,
        smooth_size=1.0,
        reference_weight=None,
        moving_weight=None,
    ):
        self.reference = reference.contiguous().to(dtype=torch.float32)
        self.moving = moving.contiguous().to(dtype=torch.float32)
        self.device = self.reference.device
        self.reference_voxel_sizes = tuple(float(v) for v in reference_voxel_sizes)
        self.moving_voxel_sizes = tuple(float(v) for v in moving_voxel_sizes)
        self.bins = int(bins)
        self.smooth_size = float(smooth_size)
        self.weighted = reference_weight is not None or moving_weight is not None
        if self.weighted:
            self.reference_weight = (
                torch.ones_like(self.reference)
                if reference_weight is None
                else reference_weight.contiguous().to(dtype=torch.float32)
            )
            self.moving_weight = (
                torch.ones_like(self.moving)
                if moving_weight is None
                else moving_weight.contiguous().to(dtype=torch.float32)
            )
            if self.reference_weight.shape != self.reference.shape:
                raise ValueError("reference weight must match the reference grid")
            if self.moving_weight.shape != self.moving.shape:
                raise ValueError("input weight must match the input grid")
        if self.bins < 2:
            raise ValueError("bins must be at least two")
        # NEWIMAGE traverses x fastest, then y, then z.  Keeping that order
        # also reduces the float32 accumulation difference on CPU.
        z, y, x = torch.meshgrid(
            torch.arange(self.reference.shape[2], dtype=torch.float32, device=self.device),
            torch.arange(self.reference.shape[1], dtype=torch.float32, device=self.device),
            torch.arange(self.reference.shape[0], dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        self.grid = torch.stack((x, y, z)).reshape(3, -1)
        self.reference_values = self.reference.permute(2, 1, 0).reshape(-1)
        if self.weighted:
            self.reference_weight_values = (
                self.reference_weight.permute(2, 1, 0).reshape(-1)
            )
        ref_min = self.reference_values.min()
        ref_max = self.reference_values.max()
        if float(ref_max - ref_min) == 0:
            ref_max = ref_max + 1
        bin_index = torch.trunc(
            self.reference_values * (self.bins / (ref_max - ref_min))
            - ref_min * (self.bins / (ref_max - ref_min))
        ).to(torch.long)
        self.bin_index = bin_index.clamp(0, self.bins - 1)
        self.bin_sort_order = torch.argsort(self.bin_index, stable=True)
        self.bin_lengths = torch.bincount(
            self.bin_index, minlength=self.bins
        )

    def __call__(self, moving_to_reference):
        coefficients = _fsl_pull_coefficients(
            moving_to_reference,
            self.moving_voxel_sizes,
            self.reference_voxel_sizes,
            device=self.device,
        )
        coordinates = _coordinates_from_fsl_coefficients(
            coefficients, self.grid
        )
        upper = torch.tensor(
            [size - 1.0001 for size in self.moving.shape],
            dtype=torch.float32,
            device=self.device,
        )[:, None]
        valid = ((coordinates >= 0) & (coordinates <= upper)).all(dim=0)
        if not bool(valid.any()):
            return 1.0
        # Keep every reference voxel so that the precomputed stable bin order
        # can be reduced without CUDA atomics.  Invalid voxels receive zero
        # weight after interpolation at clamped coordinates.
        interpolation_coordinates = coordinates.clone()
        interpolation_coordinates.clamp_min_(0)
        interpolation_coordinates = torch.minimum(
            interpolation_coordinates, upper
        )
        values = _manual_trilinear(self.moving, interpolation_coordinates)
        if self.smooth_size > 0:
            smooth = torch.tensor(
                [
                    self.smooth_size / self.moving_voxel_sizes[0],
                    self.smooth_size / self.moving_voxel_sizes[1],
                    self.smooth_size / self.moving_voxel_sizes[2],
                ],
                dtype=torch.float32,
                device=self.device,
            )[:, None]
            selected = coordinates
            far_distance = upper - selected
            weight_per_axis = torch.where(
                selected < smooth,
                selected / smooth,
                torch.where(far_distance < smooth, far_distance / smooth, 1.0),
            )
            weights = weight_per_axis.prod(dim=0).clamp_min_(0)
        else:
            weights = torch.ones_like(values)
        if self.weighted:
            moving_weights = _manual_trilinear(
                self.moving_weight, interpolation_coordinates
            )
            weights = (
                weights * moving_weights * self.reference_weight_values
            ).clamp_min_(0)
        weights = weights * valid
        order = self.bin_sort_order
        lengths = self.bin_lengths
        sorted_weights = weights[order]
        sorted_values = values[order]
        counts = torch.segment_reduce(
            sorted_weights, "sum", lengths=lengths, initial=0
        )
        sums = torch.segment_reduce(
            sorted_weights * sorted_values, "sum", lengths=lengths, initial=0
        )
        sums2 = torch.segment_reduce(
            sorted_weights * sorted_values.square(),
            "sum",
            lengths=lengths,
            initial=0,
        )
        keep = counts > 2
        if not bool(keep.any()):
            return 1.0
        n = counts[keep].to(torch.float32)
        y = sums[keep]
        y2 = sums2[keep]
        within = (y2 - y.square() / n) / (n - 1)
        total_n = n.sum()
        total_sum = y.sum()
        total_sum2 = y2.sum()
        if float(total_n) <= 1:
            return 1.0
        total_variance = (
            total_sum2 - total_sum.square() / total_n
        ) / (total_n - 1)
        if float(total_variance) <= 0:
            return 1.0
        cost = (within * n).sum() / total_n / total_variance
        return float(cost)



class FSLNormalizedMutualInformation(FSLCorrelationRatio):
    """PyTorch port of NEWIMAGE's smoothed NMI histogram cost."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_min = self.moving.min()
        test_range = self.moving.max() - self.test_min
        if float(test_range) == 0:
            test_range = test_range + 1
        self.test_factor = self.bins / test_range

    def __call__(self, moving_to_reference):
        coefficients = _fsl_pull_coefficients(
            moving_to_reference, self.moving_voxel_sizes,
            self.reference_voxel_sizes, device=self.device,
        )
        coordinates = _coordinates_from_fsl_coefficients(coefficients, self.grid)
        upper = torch.tensor(
            [size - 1.0001 for size in self.moving.shape],
            dtype=torch.float32, device=self.device,
        )[:, None]
        valid = ((coordinates >= 0) & (coordinates <= upper)).all(dim=0)
        if not bool(valid.any()):
            return -1.0
        values = _manual_trilinear(
            self.moving, torch.minimum(coordinates.clamp_min(0), upper)
        )
        smooth = torch.tensor(
            [self.smooth_size / value for value in self.moving_voxel_sizes],
            dtype=torch.float32, device=self.device,
        )[:, None]
        edge_weight = torch.where(
            coordinates < smooth, coordinates / smooth,
            torch.where(upper - coordinates < smooth,
                        (upper - coordinates) / smooth, 1.0),
        ).prod(dim=0).clamp_min_(0)
        weight = edge_weight * valid
        bin_float = (values - self.test_min) * self.test_factor
        truncated = torch.trunc(bin_float)
        centre = truncated.long().clamp(0, self.bins - 1)
        minus = (centre - 1).clamp_min(0)
        plus = (centre + 1).clamp_max(self.bins - 1)
        fractional = (bin_float - truncated).abs()
        centre_weight = torch.where(
            fractional < 0.5, 0.5 + fractional,
            torch.where(fractional > 0.5, 1.5 - fractional, 1.0),
        ).clamp(0, 1)
        minus_weight = torch.where(fractional < 0.5, 1 - centre_weight, 0)
        plus_weight = torch.where(fractional > 0.5, 1 - centre_weight, 0)
        stride = self.bins + 1
        # Accumulate atomics in float64, then keep FSL's float32 entropy path.
        joint = torch.zeros(stride * stride, dtype=torch.float64, device=self.device)
        for bin_id, bin_weight in (
            (centre, centre_weight), (minus, minus_weight), (plus, plus_weight)
        ):
            joint.scatter_add_(0, self.bin_index * stride + bin_id,
                               (weight * bin_weight).double())
        joint = joint.reshape(stride, stride).float()
        first = joint.sum(1)
        second = joint.sum(0)
        total = second.sum()
        if float(total) <= 0:
            return -1.0

        def entropy(histogram):
            probabilities = histogram / total
            selected = probabilities > 0
            return -(probabilities[selected] * probabilities[selected].log()).sum()

        joint_entropy = entropy(joint)
        if float(joint_entropy) <= 0:
            return -1.0
        return float(-(entropy(first) + entropy(second)) / joint_entropy)



def _quadratic_minimum(x1, middle, x2, y1, y_middle, y2):
    f32 = np.float32
    x1, middle, x2 = f32(x1), f32(middle), f32(x2)
    y1, y_middle, y2 = f32(y1), f32(y_middle), f32(y2)
    a = f32(
        f32(f32(middle - x2) * f32(y_middle - y1))
        - f32(f32(middle - x1) * f32(y_middle - y2))
    )
    b = f32(
        -f32(f32(f32(middle * middle) - f32(x2 * x2)) * f32(y_middle - y1))
        + f32(f32(f32(middle * middle) - f32(x1 * x1)) * f32(y_middle - y2))
    )
    determinant = f32(
        f32(f32(middle - x2) * f32(x2 - x1)) * f32(x1 - middle)
    )
    if abs(float(determinant)) > 1e-15 and f32(a / determinant) < 0:
        return None
    if abs(float(a)) <= 1e-15:
        return None
    return float(f32(-b / f32(f32(2.0) * a)))


def _extrapolated_point(x1, middle, x2):
    f32 = np.float32
    x1, middle, x2 = f32(x1), f32(middle), f32(x2)
    ratio = f32(0.3819660)
    if abs(x2 - middle) > abs(x1 - middle):
        return float(f32(f32(ratio * x2) + f32(f32(1.0 - ratio) * middle)))
    return float(f32(f32(ratio * x1) + f32(f32(1.0 - ratio) * middle)))


def _next_point(x1, middle, x2, y1, y_middle, y2):
    value = _quadratic_minimum(x1, middle, x2, y1, y_middle, y2)
    if value is None or value < min(x1, x2) or value > max(x1, x2):
        return _extrapolated_point(x1, middle, x2)
    return value


def _initial_bound(x1, middle, y1, y_middle, function, direction, point):
    f32 = np.float32
    factor = f32(1.6)
    x1, middle = f32(x1), f32(middle)
    y1, y_middle = f32(y1), f32(y_middle)
    if y1 == 0:
        y1 = f32(function(float(x1) * direction + point))
    if y_middle == 0:
        y_middle = f32(function(float(middle) * direction + point))
    if y1 < y_middle:
        x1, middle = middle, x1
        y1, y_middle = y_middle, y1
    sign = f32(-1.0 if middle < x1 else 1.0)
    x2 = f32(middle + f32(factor * f32(middle - x1)))
    y2 = f32(function(float(x2) * direction + point))
    for _ in range(256):
        if y_middle <= y2:
            return x1, middle, x2, y1, y_middle, y2
        maximum = f32(middle + f32(f32(factor * f32(2.0)) * f32(x2 - middle)))
        new = _quadratic_minimum(x1, middle, x2, y1, y_middle, y2)
        if (
            new is None
            or (new - x1) * sign < 0
            or (new - maximum) * sign > 0
        ):
            new = f32(middle + f32(factor * f32(x2 - x1)))
        else:
            new = f32(new)
        y_new = f32(function(float(new) * direction + point))
        if (new - middle) * (new - x1) < 0:
            if y_new < y_middle:
                return x1, new, middle, y1, y_new, y_middle
            x1, y1 = new, y_new
        elif y_new > y_middle:
            return x1, middle, new, y1, y_middle, y_new
        elif (new - x2) * sign < 0:
            x1, y1, middle, y_middle = middle, y_middle, new, y_new
        else:
            x1, y1, middle, y_middle, x2, y2 = (
                middle, y_middle, x2, y2, new, y_new
            )
    raise RuntimeError("FLIRT line search did not bracket a minimum")


def _optimize_one_dimension(
    point, direction, tolerance, function, maximum_iterations, initial_value,
    bound_guess,
):
    f32 = np.float32
    unit = direction / np.linalg.norm(direction)
    direction_tolerance = f32(0.0)
    for index in range(len(tolerance)):
        if abs(tolerance[index]) > 1e-15:
            direction_tolerance = f32(
                direction_tolerance + f32(abs(unit[index] / tolerance[index]))
            )
    unit_tolerance = f32(abs(f32(1.0) / direction_tolerance))
    middle, x1 = f32(0.0), f32(f32(bound_guess) * unit_tolerance)
    y_middle = f32(initial_value if initial_value != 0 else function(point))
    y1 = f32(function(float(x1) * unit + point))
    x1, middle, x2, y1, y_middle, y2 = _initial_bound(
        x1, middle, y1, y_middle, function, unit, point
    )
    minimum_distance = f32(f32(0.1) * unit_tolerance)
    iteration = 0
    while (
        iteration < maximum_iterations
        and abs((x2 - x1) / unit_tolerance) > 1.0
    ):
        iteration += 1
        new = _next_point(x1, middle, x2, y1, y_middle, y2)
        sign = -1.0 if x2 < x1 else 1.0
        if abs(new - x1) < minimum_distance:
            new = f32(x1 + f32(sign * minimum_distance))
        if abs(new - x2) < minimum_distance:
            new = f32(x2 - f32(sign * minimum_distance))
        if abs(new - middle) < minimum_distance:
            new = _extrapolated_point(x1, middle, x2)
        if abs(middle - x1) < 0.4 * unit_tolerance:
            new = f32(middle + f32(f32(sign * f32(0.5)) * unit_tolerance))
        if abs(middle - x2) < 0.4 * unit_tolerance:
            new = f32(middle - f32(f32(sign * f32(0.5)) * unit_tolerance))
        new = f32(new)
        y_new = f32(function(float(new) * unit + point))
        if (new - middle) * (x2 - middle) > 0:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if y_new < y_middle:
            x2, y2, middle, y_middle = middle, y_middle, new, y_new
        else:
            x1, y1 = new, y_new
    return float(middle) * unit + point, float(y_middle)


def fsl_coordinate_optimize(
    point,
    tolerance,
    function,
    *,
    maximum_iterations=4,
    bound_guess=(10.0, 1.0),
    numopt=None,
):
    """Port of ``MISCMATHS::optimise`` with FLIRT's default Brent mode."""
    point = np.asarray(point, dtype=np.float64).copy()
    tolerance = np.asarray(tolerance, dtype=np.float64)
    if numopt is None:
        numopt = len(point)
    if not 0 < numopt <= len(point):
        raise ValueError("numopt must be between one and the point dimension")
    inverse_tolerance = np.zeros_like(tolerance)
    nonzero = np.abs(tolerance) > 1e-15
    inverse_tolerance[nonzero] = np.abs(1 / tolerance[nonzero])
    inverse_tolerance /= len(tolerance)
    value = 0.0
    for major in range(maximum_iterations):
        initial = point.copy()
        guess = bound_guess[min(major, len(bound_guess) - 1)]
        for index in range(numopt):
            direction = np.zeros_like(point)
            direction[index] = 1
            point, value = _optimize_one_dimension(
                point, direction, tolerance, function, 100, value, guess
            )
        average_tolerance = np.abs((initial - point) * inverse_tolerance).sum()
        if average_tolerance < 1.0:
            break
    return point, float(value)


def _serial_float64_sum(values):
    """Sum one NEWIMAGE COG block in scalar double order."""
    if values.size == 0:
        return 0.0
    return float(np.add.accumulate(values, dtype=np.float64)[-1])


def _centre_of_gravity(data, sampling):
    if data.device.type == "cpu":
        # NEWIMAGE calc_cog traverses x fastest, accumulates double scalars,
        # and flushes each block after n becomes greater than nlim.
        array = data.detach().numpy()
        minimum = np.float32(array.min())
        ordered = np.transpose(array, (2, 1, 0)).reshape(-1)
        values = np.asarray(ordered - minimum, dtype=np.float64)
        nlim = max(int(math.sqrt(values.size)), 1000)
        block_size = nlim + 1
        xsize, ysize, _ = array.shape
        total = vx = vy = vz = 0.0
        for start in range(0, values.size, block_size):
            stop = min(start + block_size, values.size)
            block = values[start:stop]
            indices = np.arange(start, stop, dtype=np.int64)
            x = indices % xsize
            yz = indices // xsize
            y = yz % ysize
            z = yz // ysize
            total += _serial_float64_sum(block)
            vx += _serial_float64_sum(block * x)
            vy += _serial_float64_sum(block * y)
            vz += _serial_float64_sum(block * z)
        if abs(total) < 1e-5:
            total = 1.0
        voxel = np.array([vx / total, vy / total, vz / total])
        sampling = np.asarray(sampling, dtype=np.float64)
        return sampling[:3, :3] @ voxel + sampling[:3, 3]

    values = data.to(torch.float64) - data.min().to(torch.float64)
    total = values.sum()
    if abs(float(total)) < 1e-5:
        total = total.new_tensor(1.0)
    axes = []
    for axis, size in enumerate(data.shape):
        marginal = values.sum(tuple(index for index in range(3) if index != axis))
        coordinate = torch.arange(size, dtype=torch.float64, device=data.device)
        axes.append((marginal * coordinate).sum() / total)
    voxel = torch.stack(axes)
    sampling = torch.as_tensor(sampling, dtype=torch.float64, device=data.device)
    return (sampling[:3, :3] @ voxel + sampling[:3, 3]).cpu().numpy()


def _rms_deviation(first, second, radius=80.0):
    """MISCMATHS affine RMS deviation used by FLIRT candidate pruning."""
    difference = np.asarray(first) @ np.linalg.inv(np.asarray(second)) - np.eye(4)
    linear = difference[:3, :3]
    translation = difference[:3, 3]
    return float(
        np.sqrt(
            translation @ translation
            + (float(radius) ** 2 / 5.0) * np.trace(linear.T @ linear)
        )
    )


def _find_cost_minima(costs):
    """Return minima in FLIRT's z/y/x scan order."""
    shape = costs.shape
    best = (0, 0, 0)
    minima = []
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                index = (x, y, z)
                if costs[index] < costs[best]:
                    best = index
                if x + 1 >= shape[0] or y + 1 >= shape[1] or z + 1 >= shape[2]:
                    continue
                neighbourhood = costs[
                    max(0, x - 1):min(shape[0], x + 2),
                    max(0, y - 1):min(shape[1], y + 2),
                    max(0, z - 1):min(shape[2], z + 2),
                ]
                if not bool(np.any(neighbourhood < costs[index])):
                    minima.append(index)
    return minima or [best]


def _newimage_percentile(values, probability):
    """NEWIMAGE order-statistic percentile for a dense volume."""
    ordered = np.sort(np.asarray(values).reshape(-1))
    if ordered.size == 0:
        raise ValueError("percentile input must not be empty")
    index = min(int(ordered.size * float(probability)), ordered.size - 1)
    return float(ordered[index])


@dataclass
class _Level:
    reference: torch.Tensor
    moving: torch.Tensor
    reference_sizes: tuple
    moving_sizes: tuple
    cost: FSLCorrelationRatio
    centre: np.ndarray


def _flip_to_radiological(data, vox2world):
    if np.linalg.det(np.asarray(vox2world)[:3, :3]) > 0:
        return np.flip(data, axis=0).copy()
    return np.array(data, copy=True)


def _isotropic_resample(data, voxel_sizes, scale):
    step = np.asarray([scale / value for value in voxel_sizes], dtype=np.float64)
    shape = tuple(max(1, int(data.shape[i] / step[i])) for i in range(3))
    grid = _voxel_grid(shape, device=data.device)
    coordinates = grid * torch.as_tensor(
        step, dtype=torch.float32, device=data.device
    )[:, None]
    upper = torch.tensor(data.shape, device=data.device)[:, None] - 1
    valid = ((coordinates >= 0) & (coordinates <= upper)).all(0)
    output = torch.zeros(int(np.prod(shape)), dtype=torch.float32, device=data.device)
    output[valid] = _manual_trilinear(data, coordinates[:, valid])
    return output.reshape(shape), (float(scale),) * 3


class _DefaultFLIRTEngine:
    def __init__(
        self,
        moving,
        reference,
        moving_vox2world,
        reference_vox2world,
        moving_voxel_sizes,
        reference_voxel_sizes,
        *,
        device,
        angular_search=True,
        initial_matrix=None,
        moving_weight=None,
        reference_weight=None,
        execution="reference",
        candidate_batch_size=128,
        memory_budget_gb=20.0,
    ):
        self.device = torch.device(device)
        self.execution = execution
        self.candidate_batch_size = candidate_batch_size
        self.memory_budget_gb = memory_budget_gb
        self.host_result_transfers = 0
        self.batch_evaluations = 0
        self.phase_timings = {}
        self.phase_cost_evaluations = {}
        self._active_phase = "preparation"
        self._batch_source = None
        self._batch_evaluator = None
        moving = _flip_to_radiological(
            _clamp_like_fsl(moving), moving_vox2world
        )
        reference = _flip_to_radiological(
            _clamp_like_fsl(reference), reference_vox2world
        )
        self.moving_original = torch.as_tensor(
            moving.copy(), dtype=torch.float32, device=self.device
        )
        self.reference_original = torch.as_tensor(
            reference.copy(), dtype=torch.float32, device=self.device
        )
        self.weighted = moving_weight is not None or reference_weight is not None
        if self.weighted:
            if moving_weight is None:
                moving_weight = np.ones_like(moving, dtype=np.float32)
            else:
                moving_weight = _flip_to_radiological(
                    moving_weight, moving_vox2world
                )
            if reference_weight is None:
                reference_weight = np.ones_like(reference, dtype=np.float32)
            else:
                reference_weight = _flip_to_radiological(
                    reference_weight, reference_vox2world
                )
            self.moving_weight_original = torch.as_tensor(
                np.asarray(moving_weight, dtype=np.float32).copy(),
                device=self.device,
            )
            self.reference_weight_original = torch.as_tensor(
                np.asarray(reference_weight, dtype=np.float32).copy(),
                device=self.device,
            )
        self.moving_sizes = tuple(float(v) for v in moving_voxel_sizes)
        self.reference_sizes = tuple(float(v) for v in reference_voxel_sizes)
        self.minimum_sampling = float(
            math.ceil(max(min(self.moving_sizes), min(self.reference_sizes)))
        )
        self.angular_search = bool(angular_search)
        if initial_matrix is None:
            initial_matrix = np.eye(4, dtype=np.float64)
        self.initial_matrix = np.asarray(initial_matrix, dtype=np.float64)
        if self.initial_matrix.shape != (4, 4):
            raise ValueError("initial_matrix must have shape (4, 4)")
        if not np.isfinite(self.initial_matrix).all():
            raise ValueError("initial_matrix must contain only finite values")
        self.bound_guess = (10.0, 1.0)
        self.requested_scale = 8.0
        self.level = None
        self.cost_evaluations = 0
        self._cache = {}
        with self.phase("preparation"):
            self._prepare_reference_pyramid()
            self.set_scale(8.0, force=True)

    def _prepare_reference_pyramid(self):
        def resample(value):
            blurred = _blur(
                value, self.minimum_sampling, self.reference_sizes
            )
            return _isotropic_resample(
                blurred, self.reference_sizes, self.minimum_sampling
            )[0]

        if self.weighted:
            reference = _filter_image_with_weight(
                self.reference_original, self.reference_weight_original, resample
            )
            reference_weight = _filter_weight(
                self.reference_weight_original, resample
            )
        else:
            reference = resample(self.reference_original)
            reference_weight = None
        sizes = (self.minimum_sampling,) * 3
        self.references = {1: (reference, sizes)}
        self.reference_weights = {1: reference_weight}
        current, current_sizes = reference, sizes
        current_weight = reference_weight
        for scale, threshold in ((2, 1.9), (4, 3.9), (8, 7.9)):
            if self.minimum_sampling < threshold:
                if self.weighted:
                    previous_weight = current_weight
                    current = _filter_image_with_weight(
                        current, previous_weight, _subsample_by_two
                    )
                    current_weight = _filter_weight(
                        previous_weight, _subsample_by_two
                    )
                else:
                    current = _subsample_by_two(current)
                current_sizes = tuple(value * 2 for value in current_sizes)
            self.references[scale] = (current, current_sizes)
            self.reference_weights[scale] = current_weight

    def set_scale(self, scale, *, force=False):
        self.requested_scale = float(scale)
        if not force and self.minimum_sampling > 1.25 * scale:
            if self.level is not None:
                # ``usrsetscale`` always calls get_testvol before deciding
                # whether the requested scale is available.  Costfn stores a
                # reference to that volume, so a skipped finer scale observes
                # the refreshed, unblurred input while retaining the previous
                # reference pyramid, binning, and centre of rotation.
                self.level.moving = self.moving_original
                self.level.cost.moving = self.moving_original
                if self.weighted:
                    self.level.cost.moving_weight = self.moving_weight_original
                self.level.cost.smooth_size = float(scale)
                self._cache.clear()
            return
        key = int(scale) if int(scale) in self.references else 1
        reference, reference_sizes = self.references[key]
        if self.weighted:
            transform = lambda value: _blur(
                value, float(scale), self.moving_sizes
            )
            moving = _filter_image_with_weight(
                self.moving_original, self.moving_weight_original, transform
            )
            moving_weight = _filter_weight(
                self.moving_weight_original, transform
            )
            reference_weight = self.reference_weights[key]
        else:
            moving = _blur(self.moving_original, float(scale), self.moving_sizes)
            moving_weight = reference_weight = None
        bins = max(2, int(256 / float(scale)))
        cost = FSLCorrelationRatio(
            reference,
            moving,
            reference_sizes,
            self.moving_sizes,
            bins=bins,
            smooth_size=float(scale),
            reference_weight=reference_weight,
            moving_weight=moving_weight,
        )
        sampling = np.diag([*self.moving_sizes, 1.0])
        centre = _centre_of_gravity(moving, sampling)
        self.level = _Level(
            reference, moving, reference_sizes, self.moving_sizes, cost, centre
        )
        self._cache.clear()

    @contextmanager
    def phase(self, name):
        previous = self._active_phase
        self._active_phase = name
        start = time.perf_counter()
        with torch.profiler.record_function("flirt." + name):
            try:
                yield
            finally:
                self.phase_timings[name] = (
                    self.phase_timings.get(name, 0.0) + time.perf_counter() - start
                )
                self._active_phase = previous

    def cost_many(self, affines):
        """Evaluate independent matrices in order, sharing one host result copy."""
        affines = [np.asarray(affine, dtype=np.float64) for affine in affines]
        keys = [affine.tobytes(order="C") for affine in affines]
        missing = {}
        for key, affine in zip(keys, affines):
            if key not in self._cache:
                missing.setdefault(key, affine)
        if missing:
            from .batched import BatchedAffineCost

            source = (id(self.level.cost), id(self.level.cost.moving),
                      self.level.cost.smooth_size)
            if source != self._batch_source:
                self._batch_evaluator = BatchedAffineCost(
                    self.level.cost,
                    max_batch_size=self.candidate_batch_size,
                    memory_budget_gb=self.memory_budget_gb,
                )
                self._batch_source = source
            matrices = np.stack(list(missing.values())) @ self.initial_matrix
            values = self._batch_evaluator(matrices).detach().cpu().numpy()
            self.host_result_transfers += 1
            self.batch_evaluations += 1
            self.cost_evaluations += len(missing)
            name = self._active_phase
            self.phase_cost_evaluations[name] = (
                self.phase_cost_evaluations.get(name, 0) + len(missing)
            )
            self._cache.update(zip(missing, (float(value) for value in values)))
        return [self._cache[key] for key in keys]

    def cost(self, affine):
        if getattr(self, "execution", "reference") == "batched":
            return self.cost_many([affine])[0]
        affine = np.asarray(affine, dtype=np.float64)
        key = affine.tobytes(order="C")
        if key not in self._cache:
            self._cache[key] = self.level.cost(affine @ self.initial_matrix)
            self.cost_evaluations += 1
            if hasattr(self, "phase_cost_evaluations"):
                name = self._active_phase
                self.phase_cost_evaluations[name] = (
                    self.phase_cost_evaluations.get(name, 0) + 1
                )
        return self._cache[key]

    def _parameters_to_matrix(self, parameters, dof):
        return fsl_affine_from_parameters(
            torch.as_tensor(parameters, dtype=torch.float64),
            torch.as_tensor(self.level.centre, dtype=torch.float64),
            dof,
        ).numpy()

    def _parameter_matrices(self, parameters, dof):
        if not len(parameters):
            return np.empty((0, 4, 4), dtype=np.float64)
        if self.execution == "batched":
            from .affine_batch import fsl_affine_from_parameters_batch

            return fsl_affine_from_parameters_batch(
                np.asarray(parameters, dtype=np.float64), self.level.centre, dof
            ).numpy()
        return np.stack([self._parameters_to_matrix(values, dof)
                         for values in parameters])

    def optimize_matrix(self, matrix, dof, maximum_iterations=4):
        parameters = fsl_parameters_from_affine(matrix, self.level.centre)
        tolerance = _BASE_TOLERANCE * self.requested_scale

        def objective(values):
            return self.cost(self._parameters_to_matrix(values, dof))

        fitted, value = fsl_coordinate_optimize(
            parameters,
            tolerance,
            objective,
            maximum_iterations=maximum_iterations,
            bound_guess=self.bound_guess,
            numopt=dof,
        )
        return self._parameters_to_matrix(fitted, dof), value

    def _optimize_search_subset(self, reference_parameters, maximum_iterations=4):
        basis = np.zeros((12, 4), dtype=np.float64)
        basis[6:9, 0] = 1
        basis[3, 1] = basis[4, 2] = basis[5, 3] = 1
        pseudo = np.linalg.pinv(basis)
        tolerance = pseudo @ (_BASE_TOLERANCE * self.requested_scale)

        def objective(values):
            parameters = reference_parameters + basis @ values
            return self.cost(self._parameters_to_matrix(parameters, 12))

        fitted, value = fsl_coordinate_optimize(
            np.zeros(4),
            tolerance,
            objective,
            maximum_iterations=maximum_iterations,
            bound_guess=self.bound_guess,
        )
        parameters = reference_parameters + basis @ fitted
        return parameters, value

    def _optimize_search_many(self, reference_parameters, maximum_iterations=4):
        from .search import coordinate_trials, evaluate_trials, map_trials

        basis = np.zeros((12, 4), dtype=np.float64)
        basis[6:9, 0] = 1
        basis[3, 1] = basis[4, 2] = basis[5, 3] = 1
        tolerance = np.linalg.pinv(basis) @ (
            _BASE_TOLERANCE * self.requested_scale
        )
        searches = []
        for reference in reference_parameters:
            trials = coordinate_trials(
                np.zeros(4), tolerance,
                maximum_iterations=maximum_iterations,
                bound_guess=self.bound_guess,
            )
            searches.append(map_trials(
                trials, lambda values, reference=reference: reference + basis @ values,
            ))
        fitted = evaluate_trials(
            searches, lambda values: self.cost_many(self._parameter_matrices(values, 12))
        )
        return [(reference + basis @ values, cost)
                for reference, (values, cost) in zip(reference_parameters, fitted)]

    @staticmethod
    def _angles(lower=-math.pi / 2, upper=math.pi / 2, delta=math.pi / 3):
        count = _fsl_round((upper - lower) / delta) + 1
        lower, upper = np.float32(lower), np.float32(upper)
        if count == 1:
            return np.array([np.float32((upper + lower) / np.float32(2))])
        span = np.float32(upper - lower)
        return np.array(
            [np.float32(lower + np.float32(np.float32(n) * span / np.float32(count - 1)))
             for n in range(count)],
            dtype=np.float32,
        )

    @staticmethod
    def _interpolate_coarse(volume, coordinate):
        x, y, z = (np.float32(value) for value in coordinate)
        x0, y0, z0 = int(x), int(y), int(z)
        x1 = min(x0 + 1, volume.shape[0] - 1)
        y1 = min(y0 + 1, volume.shape[1] - 1)
        z1 = min(z0 + 1, volume.shape[2] - 1)
        dx, dy, dz = np.float32(x - x0), np.float32(y - y0), np.float32(z - z0)
        v000, v001 = volume[x0, y0, z0], volume[x0, y0, z1]
        v010, v011 = volume[x0, y1, z0], volume[x0, y1, z1]
        v100, v101 = volume[x1, y0, z0], volume[x1, y0, z1]
        v110, v111 = volume[x1, y1, z0], volume[x1, y1, z1]
        t1 = np.float32(np.float32(v100 - v000) * dx + v000)
        t2 = np.float32(np.float32(v101 - v001) * dx + v001)
        t3 = np.float32(np.float32(v110 - v010) * dx + v010)
        t4 = np.float32(np.float32(v111 - v011) * dx + v011)
        t5 = np.float32(np.float32(t3 - t1) * dy + t1)
        t6 = np.float32(np.float32(t4 - t2) * dy + t2)
        return float(np.float32(np.float32(t6 - t5) * dz + t5))

    def angular_candidates(self, dof=12):
        if not self.angular_search:
            coarse = fine = np.array([0.0])
        else:
            coarse = self._angles(delta=math.pi / 3)
            fine = self._angles(delta=math.pi / 10)
        shape = (len(coarse),) * 3
        tx = np.zeros(shape, dtype=np.float32)
        ty = np.zeros(shape, dtype=np.float32)
        tz = np.zeros(shape, dtype=np.float32)
        scales = np.ones(shape, dtype=np.float32)
        moving_centre = _centre_of_gravity(
            self.level.moving, np.diag([*self.moving_sizes, 1.0])
        )
        reference_centre = _centre_of_gravity(
            self.level.reference,
            np.diag([*self.level.reference_sizes, 1.0]),
        )
        initial_moving_centre = self.initial_matrix @ np.r_[moving_centre, 1.0]
        translation = reference_centre - initial_moving_centre[:3]
        coarse_parameters = []
        for rx in coarse:
            for ry in coarse:
                for rz in coarse:
                    parameters = _IDENTITY_PARAMETERS.copy()
                    parameters[:3] = (rx, ry, rz)
                    parameters[3:6] = translation
                    coarse_parameters.append(parameters)
        with self.phase("angular_coarse"):
            if self.execution == "batched":
                fitted_coarse = self._optimize_search_many(coarse_parameters)
            else:
                fitted_coarse = [self._optimize_search_subset(parameters)
                                 for parameters in coarse_parameters]
        for index, (fitted, _) in zip(np.ndindex(shape), fitted_coarse):
            tx[index], ty[index], tz[index] = fitted[3:6]
            scales[index] = fitted[6]
        # NEWIMAGE's volume percentile is an order statistic at floor(N*p),
        # rather than NumPy's interpolated percentile.
        median_scale = _newimage_percentile(scales, 0.5)
        scales.fill(median_scale)

        costs = np.empty((len(fine),) * 3, dtype=np.float32)
        fine_parameters = {}
        factor = np.float32((len(coarse) - 1) / max(1, len(fine) - 1))
        for ix, rx in enumerate(fine):
            for iy, ry in enumerate(fine):
                for iz, rz in enumerate(fine):
                    coordinate = tuple(np.float32(np.float32(index) * factor)
                                       for index in (ix, iy, iz))
                    parameters = _IDENTITY_PARAMETERS.copy()
                    parameters[:3] = (rx, ry, rz)
                    parameters[3] = self._interpolate_coarse(tx, coordinate)
                    parameters[4] = self._interpolate_coarse(ty, coordinate)
                    parameters[5] = self._interpolate_coarse(tz, coordinate)
                    scale = self._interpolate_coarse(scales, coordinate)
                    if not 0.5 <= scale <= 2:
                        scale = 1.0
                    parameters[6:9] = scale
                    fine_parameters[(ix, iy, iz)] = parameters
        with self.phase("angular_fine_measure"):
            matrices = self._parameter_matrices(list(fine_parameters.values()), 12)
            if self.execution == "batched":
                values = self.cost_many(matrices)
            else:
                values = [self.cost(matrix) for matrix in matrices]
            costs[:] = np.asarray(values, dtype=np.float32).reshape(costs.shape)

        minimum, maximum = float(costs.min()), float(costs.max())
        threshold = min(
            minimum + 0.2 * (maximum - minimum),
            _newimage_percentile(costs, 0.2),
        )
        if threshold <= minimum:
            threshold = max(minimum * 1.0001, minimum * 0.9999)
        selected = [index for index in np.ndindex(costs.shape)
                    if costs[index] < threshold]
        with self.phase("angular_fine_refinement"):
            if self.execution == "batched":
                fitted_fine = self._optimize_search_many(
                    [fine_parameters[index] for index in selected]
                )
            else:
                fitted_fine = [self._optimize_search_subset(fine_parameters[index])
                               for index in selected]
        for index, (_, value) in zip(selected, fitted_fine):
            costs[index] = value

        candidates = [fine_parameters[index] for index in _find_cost_minima(costs)]

        pairs = []
        free_dof = min(dof, 7)
        rms_minimum = min(self.level.reference_sizes)
        if self.execution == "batched":
            matrices = self._parameter_matrices(candidates, free_dof)
            with self.phase("angular_candidate_refinement"):
                preoptimized_candidates = self._measure(
                    [(0.0, matrix) for matrix in matrices]
                )
                optimized_candidates = self._optimize(
                    preoptimized_candidates, free_dof, 4
                )
        else:
            preoptimized_candidates = []
            optimized_candidates = []
            with self.phase("angular_candidate_refinement"):
                for parameters in candidates:
                    matrix = self._parameters_to_matrix(parameters, free_dof)
                    preoptimized_candidates.append((self.cost(matrix), matrix.copy()))
                    optimized_matrix, value = self.optimize_matrix(matrix, free_dof)
                    optimized_candidates.append((value, optimized_matrix))
        for optimized, preoptimized in zip(
            optimized_candidates, preoptimized_candidates
        ):
            value, optimized_matrix = optimized
            candidate = (optimized, preoptimized)
            discard = False
            for index, current in enumerate(pairs):
                if _rms_deviation(optimized_matrix, current[0][1]) < rms_minimum:
                    if value < current[0][0]:
                        pairs.pop(index)
                    else:
                        discard = True
                    break
            if discard:
                continue
            position = next(
                (index for index, current in enumerate(pairs) if value < current[0][0]),
                len(pairs),
            )
            pairs.insert(position, candidate)
        return [item[0] for item in pairs], [item[1] for item in pairs]

    @staticmethod
    def _sort(candidates):
        return sorted(candidates, key=lambda item: item[0])

    def _measure(self, candidates):
        if self.execution == "batched":
            values = self.cost_many([matrix for _, matrix in candidates])
            return [(value, matrix.copy())
                    for value, (_, matrix) in zip(values, candidates)]
        return [(self.cost(matrix), matrix.copy()) for _, matrix in candidates]

    def _optimize(self, candidates, dof, maximum_iterations, perturbation=None):
        if self.execution == "batched":
            from .search import coordinate_trials, evaluate_trials

            trials = []
            for _, matrix in candidates:
                if perturbation is not None:
                    parameters = fsl_parameters_from_affine(matrix, self.level.centre)
                    parameters += perturbation
                    matrix = self._parameters_to_matrix(parameters, 12)
                trials.append(coordinate_trials(
                    fsl_parameters_from_affine(matrix, self.level.centre),
                    _BASE_TOLERANCE * self.requested_scale,
                    maximum_iterations=maximum_iterations,
                    bound_guess=self.bound_guess, numopt=dof,
                ))
            fitted = evaluate_trials(
                trials, lambda values: self.cost_many(self._parameter_matrices(values, dof))
            )
            matrices = self._parameter_matrices([parameters for parameters, _ in fitted], dof)
            return [(value, matrix) for (_, value), matrix in zip(fitted, matrices)]
        output = []
        for _, matrix in candidates:
            if perturbation is not None:
                parameters = fsl_parameters_from_affine(matrix, self.level.centre)
                parameters += perturbation
                matrix = self._parameters_to_matrix(parameters, 12)
            matrix, value = self.optimize_matrix(matrix, dof, maximum_iterations)
            output.append((value, matrix))
        return output

    def run(self, qsform_matrix, dof=12):
        qsform_matrix = np.asarray(qsform_matrix) @ np.linalg.inv(
            self.initial_matrix
        )
        with self.phase("level_8mm"):
            self.set_scale(8)
        search, presearch = self.angular_candidates(dof)

        with self.phase("level_4mm"):
            self.set_scale(4)
        with self.phase("measure_4mm"):
            search_costs = self._measure(search)
            presearch_costs = self._measure(presearch)
        paired = sorted(zip(search_costs, presearch_costs), key=lambda item: item[0][0])
        search_costs = [item[0] for item in paired]
        presearch_costs = [item[1] for item in paired]
        with self.phase("local_4mm_initial"):
            candidates = self._optimize(
                search_costs[:3] + presearch_costs[:3] + [(0.0, np.eye(4))],
                min(dof, 7), 4,
            )
        best = self._sort(candidates)
        candidates = best[:4]
        fine_step = math.pi / 10
        perturbations = []
        for axis in range(3):
            for sign in (1, -1):
                value = np.zeros(12)
                value[axis] = sign * fine_step / 2
                perturbations.append(value)
        for scale in (0.1, -0.1, 0.2, -0.2):
            value = np.zeros(12)
            value[6] = scale if dof >= 7 else 0.0
            perturbations.append(value)
        with self.phase("local_4mm_perturbations"):
            if self.execution == "batched":
                perturbed_parameters = []
                for perturbation in perturbations:
                    for _, matrix in best[:4]:
                        parameters = fsl_parameters_from_affine(matrix, self.level.centre)
                        parameters += perturbation
                        perturbed_parameters.append(parameters)
                perturbed = [(0.0, matrix) for matrix in
                             self._parameter_matrices(perturbed_parameters, 12)]
                candidates += self._optimize(perturbed, min(dof, 7), 4)
            else:
                for perturbation in perturbations:
                    candidates += self._optimize(best[:4], min(dof, 7), 4, perturbation)
        best = self._sort(candidates)

        with self.phase("level_2mm"):
            self.set_scale(2)
        with self.phase("measure_2mm"):
            measured = self._sort(self._measure(best))
        with self.phase("local_2mm"):
            best = self._optimize(measured[:1], min(dof, 7), 4)
            self.bound_guess = (1.0,)
            if dof > 7:
                best = self._optimize(best[:1], 9, 1)
            if dof > 9:
                best = self._optimize(best[:1], 12, 2)
        best = self._sort(best)

        with self.phase("level_1mm"):
            self.set_scale(1)
        with self.phase("local_1mm"):
            final = self._optimize((best + [(0.0, qsform_matrix)])[:2], dof, 1)
            final.append((self.cost(qsform_matrix), qsform_matrix.copy()))
        cost, residual = self._sort(final)[0]
        return cost, residual @ self.initial_matrix


class _RigidNMIEngine(_DefaultFLIRTEngine):
    """Use CorrRatio for angular search, then NMI for final 6-DOF stages."""

    def set_scale(self, scale, *, force=False):
        previous = self.level
        super().set_scale(scale, force=force)
        # FSL keeps the default searchcost (CorrRatio) at 8 mm when only
        # -cost normmi is supplied, then switches to the main cost below 8 mm.
        if self.level is not previous and float(scale) < 8.0:
            self.level.cost = FSLNormalizedMutualInformation(
                self.level.reference, self.level.moving,
                self.level.reference_sizes, self.moving_sizes,
                bins=max(2, int(256 / float(scale))), smooth_size=float(scale),
            )
        self._cache.clear()


def _resample_output(
    moving,
    fixed_shape,
    moving_fsl,
    fixed_fsl,
    matrix,
    moving_voxel_sizes,
    fixed_voxel_sizes,
    device,
    *,
    blur_boundary="zero",
):
    """Apply FLIRT's default trilinear output path on the reference grid."""
    moving = torch.as_tensor(
        np.asarray(moving, dtype=np.float32).copy(), device=device
    )
    # FLIRT enables ``interpblur`` by default.  Before trilinear resampling it
    # low-pass filters the input to the smallest reference sampling distance.
    # This step is deliberately absent from FSL applywarp and is material when
    # a sub-millimetre input is written on a 2 mm template grid.
    moving = _blur(moving, min(fixed_voxel_sizes), moving_voxel_sizes,
                   boundary=blur_boundary)
    grid = _voxel_grid(fixed_shape, device=device)
    moving_fsl = torch.as_tensor(moving_fsl, dtype=torch.float32, device=device)
    fixed_fsl = torch.as_tensor(fixed_fsl, dtype=torch.float32, device=device)
    matrix = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    pull = torch.linalg.inv(moving_fsl) @ torch.linalg.inv(matrix) @ fixed_fsl
    coordinates = pull[:3, :3] @ grid + pull[:3, 3:4]
    upper = torch.tensor(moving.shape, device=device)[:, None] - 1
    valid = ((coordinates >= 0) & (coordinates <= upper)).all(0)
    output = torch.zeros(grid.shape[1], dtype=torch.float32, device=device)
    output[valid] = _manual_trilinear(moving, coordinates[:, valid])
    return output.reshape(fixed_shape)


def _load_fsl_matrix(init):
    if isinstance(init, (str, os.PathLike)):
        matrix = np.loadtxt(os.fspath(init), dtype=np.float64)
    else:
        matrix = np.asarray(init, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("init must be a finite FSL 4x4 matrix or matrix path")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8):
        raise ValueError("init must be a homogeneous affine matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
        raise ValueError("init must be invertible")
    return matrix


class TorchFLIRT:
    """Source-derived PyTorch implementation of the supported FLIRT path.

    This class ports the supported correlation-ratio and normalized-mutual-
    information coordinate-search paths. External validation is reported as
    reference-suite evidence and is
    never treated as a per-input comparison with FSL.
    """

    def __init__(
        self, device=None, *, angular_search=True, dof=12, cost="corratio",
        execution="auto", candidate_batch_size=128, memory_budget_gb=20.0,
    ):
        if (dof, cost) not in ((12, "corratio"), (6, "normmi")):
            raise ValueError("supported FLIRT profiles are 12/corratio and 6/normmi")
        self.dof = dof
        self.cost_name = cost
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if execution not in ("auto", "reference", "batched"):
            raise ValueError("execution must be 'auto', 'reference', or 'batched'")
        if not isinstance(candidate_batch_size, int) or candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be a positive integer")
        if not math.isfinite(memory_budget_gb) or memory_budget_gb <= 0:
            raise ValueError("memory_budget_gb must be finite and positive")
        self.execution = (
            "batched" if self.device.type == "cuda" else "reference"
        ) if execution == "auto" else execution
        if (self.execution == "batched" and self.device.type == "cuda"
                and cost == "normmi" and find_spec("triton") is None):
            if execution == "auto":
                self.execution = "reference"
            else:
                raise ImportError(
                    "CUDA batched normmi requires Triton from FNIT's Conda "
                    "environment; alternatively use execution='reference'"
                )
        self.candidate_batch_size = candidate_batch_size
        self.memory_budget_gb = float(memory_budget_gb)
        self.angular_search = bool(angular_search)

    def __call__(self, moving, fixed, *, init=None, inweight=None, refweight=None):
        if self.dof == 6 and (inweight is not None or refweight is not None):
            raise NotImplementedError("weight images are supported only for 12-DOF corratio")
        moving = _load_volume(moving, "moving")
        fixed = _load_volume(fixed, "fixed")
        moving_data = _single_frame(moving, "moving")
        fixed_data = _single_frame(fixed, "fixed")
        moving_world = np.asarray(moving.affine, dtype=np.float64)
        fixed_world = np.asarray(fixed.affine, dtype=np.float64)
        moving_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(moving.affine))
        fixed_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(fixed.affine))

        def load_weight(value, image, data, name):
            if value is None:
                return None
            volume = _load_volume(value, name)
            weight = _single_frame(volume, name)
            if weight.shape != data.shape or not np.allclose(
                volume.affine,
                image.affine,
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError(f"{name} must match its image grid")
            if not np.isfinite(weight).all():
                raise ValueError(f"{name} must contain only finite values")
            return np.asarray(weight, dtype=np.float32)

        moving_weight = load_weight(inweight, moving, moving_data, "inweight")
        reference_weight = load_weight(
            refweight, fixed, fixed_data, "refweight"
        )
        moving_fsl = voxel_to_fsl_scaled_mm(
            moving_world, moving_data.shape, moving_sizes
        )
        fixed_fsl = voxel_to_fsl_scaled_mm(
            fixed_world, fixed_data.shape, fixed_sizes
        )
        if init is None:
            initial_matrix = np.eye(4, dtype=np.float64)
        else:
            initial_matrix = _load_fsl_matrix(init)
        qsform = (
            fixed_fsl
            @ np.linalg.inv(fixed_world)
            @ moving_world
            @ np.linalg.inv(moving_fsl)
        )
        engine_class = _RigidNMIEngine if self.dof == 6 else _DefaultFLIRTEngine
        engine = engine_class(
            moving_data,
            fixed_data,
            moving_world,
            fixed_world,
            moving_sizes,
            fixed_sizes,
            device=self.device,
            angular_search=self.angular_search,
            initial_matrix=initial_matrix,
            moving_weight=moving_weight,
            reference_weight=reference_weight,
            execution=self.execution,
            candidate_batch_size=self.candidate_batch_size,
            memory_budget_gb=self.memory_budget_gb,
        )
        cost, matrix = engine.run(qsform, dof=self.dof)
        with engine.phase("output_resampling"):
            moved_data = _resample_output(
                moving_data,
                fixed_data.shape,
                moving_fsl,
                fixed_fsl,
                matrix,
                moving_sizes,
                fixed_sizes,
                self.device,
            ).cpu().numpy()
        moved = new_image(moved_data.astype(np.float32, copy=False), fixed)
        forward_world = flirt_to_world_affine(
            matrix,
            moving_world,
            fixed_world,
            moving_data.shape,
            fixed_data.shape,
            moving_sizes,
            fixed_sizes,
        )
        pull_world = np.linalg.inv(forward_world)
        pull_world[3] = (0, 0, 0, 1)
        qc = {
            "backend": "pytorch-fsl-flirt-2111.2-source-derived",
            "device": str(self.device),
            "tf32": {
                "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "reduced_precision_tensor_dtype": False,
            },
            "cost": "FSL normalized mutual information" if self.dof == 6 else "FSL correlation ratio",
            "search_cost": "FSL correlation ratio",
            "optimizer": "MISCMATHS Brent coordinate search",
            "schedule": "FSL default 8/4/2/1 mm",
            "degrees_of_freedom": self.dof,
            "angular_search": self.angular_search,
            "cost_value": float(cost),
            "cost_evaluations": engine.cost_evaluations,
            "execution": self.execution,
            "candidate_batch_size": self.candidate_batch_size,
            "memory_budget_gb": self.memory_budget_gb,
            "phase_timings_seconds": engine.phase_timings,
            "phase_timing_scope": (
                "host wall scopes; asynchronous CUDA work can complete in the "
                "next scope; optimization scopes include host cost-result waits"
            ),
            "phase_cost_evaluations": engine.phase_cost_evaluations,
            "batched_cost_calls": engine.batch_evaluations,
            "batched_host_result_transfers": engine.host_result_transfers,
            "matrix_coordinate_system": "FSL scaled-mm",
            "matrix_direction": "moving/input-to-fixed/reference",
            "source_versions": {
                "flirt": FSL_FLIRT_VERSION,
                "newimage": FSL_NEWIMAGE_VERSION,
                "miscmaths": FSL_MISCMATHS_VERSION,
            },
            "source_commit": FSL_FLIRT_COMMIT,
            "initial_matrix_used": init is not None,
            "input_weight_used": inweight is not None,
            "reference_weight_used": refweight is not None,
            "reference_validation_report": "validation/flirt/report.public.json",
            "current_input_compared_with_fsl": False,
            "validated_fsl_equivalent": False,
            "validation_scope": (
                "release benchmark evidence is external to runtime QC and does not "
                "establish equivalence for the current input"
            ),
            "bitwise_identity_claimed": False,
            "complete_numerical_equivalence_claimed": False,
        }
        return FLIRTResult(
            moved=moved,
            matrix=matrix,
            moving_to_fixed_world=forward_world,
            fixed_to_moving_world=pull_world,
            qc=qc,
        )

    def applyxfm(self, moving, fixed, *, init=None, usesqform=False):
        """Apply a known FSL matrix or the images' qform/sform on a 3D grid."""
        if (init is None and not usesqform) or (init is not None and usesqform):
            raise ValueError("provide exactly one of init or usesqform=True")
        moving = _load_volume(moving, "moving")
        fixed = _load_volume(fixed, "fixed")
        moving_data = _single_frame(moving, "moving")
        fixed_data = _single_frame(fixed, "fixed")
        moving_world = np.asarray(moving.affine, dtype=np.float64)
        fixed_world = np.asarray(fixed.affine, dtype=np.float64)
        moving_sizes = tuple(float(value) for value in moving.header.get_zooms()[:3])
        fixed_sizes = tuple(float(value) for value in fixed.header.get_zooms()[:3])
        moving_fsl = voxel_to_fsl_scaled_mm(moving_world, moving_data.shape, moving_sizes)
        fixed_fsl = voxel_to_fsl_scaled_mm(fixed_world, fixed_data.shape, fixed_sizes)
        if usesqform:
            if any(int(image.header["qform_code"]) + int(image.header["sform_code"]) == 0
                   for image in (moving, fixed)):
                raise ValueError("usesqform requires qform or sform in both images")
            matrix = world_to_flirt_affine(
                np.eye(4), moving_world, fixed_world,
                moving_data.shape, fixed_data.shape, moving_sizes, fixed_sizes,
            )
        else:
            matrix = _load_fsl_matrix(init)
        moved_data = _resample_output(
            moving_data, fixed_data.shape, moving_fsl, fixed_fsl, matrix,
            moving_sizes, fixed_sizes, self.device, blur_boundary="replicate",
        ).cpu().numpy()
        moved_data = moved_data.astype(moving.get_data_dtype(), copy=False)
        forward_world = flirt_to_world_affine(
            matrix, moving_world, fixed_world,
            moving_data.shape, fixed_data.shape, moving_sizes, fixed_sizes,
        )
        pull_world = np.linalg.inv(forward_world)
        pull_world[3] = (0, 0, 0, 1)
        return FLIRTResult(
            moved=new_image(moved_data, fixed),
            matrix=matrix,
            moving_to_fixed_world=forward_world,
            fixed_to_moving_world=pull_world,
            qc={
                "backend": "pytorch-fsl-flirt-2111.2-source-derived",
                "mode": "applyxfm",
                "transform_source": "qform/sform" if usesqform else "init",
                "interpolation": "trilinear with default FLIRT blur",
                "device": str(self.device),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "matrix_coordinate_system": "FSL scaled-mm",
                "current_input_compared_with_fsl": False,
            },
        )

    def run(
        self,
        input,
        reference,
        *,
        output=None,
        omat=None,
        init=None,
        inweight=None,
        refweight=None,
        overwrite=False,
    ):
        """Register paths or volumes and atomically write FSL-style outputs."""
        from .standalone import (
            _image_output_path,
            _preflight_outputs,
            _write_outputs_atomic,
        )

        selected = _preflight_outputs(
            _image_output_path(output),
            Path(omat).expanduser() if omat is not None else None,
            (input, reference, init, inweight, refweight),
            overwrite,
        )
        result = self(
            input,
            reference,
            init=init,
            inweight=inweight,
            refweight=refweight,
        )
        _write_outputs_atomic(result, selected, overwrite)
        return result


__all__ = [
    "FSLCorrelationRatio",
    "TorchFLIRT",
    "FSL_FLIRT_COMMIT",
    "FSL_FLIRT_VERSION",
    "fsl_affine_from_parameters",
    "fsl_coordinate_optimize",
    "fsl_parameters_from_affine",
]
