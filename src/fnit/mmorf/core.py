"""PyTorch implementation of the MMORF scalar-plus-tensor registration path.

The public transform follows MMORF 0.3.2: the warp is sampled on the common
reference grid and stores millimetre displacement along the reference image
axes. Each input affine is an FSL scaled-mm matrix mapping that input into the
common space. This convention differs from FNIRT/applywarp displacement fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import spline_filter

from .._dmri import configure_device, image_like
from ..flirt import flirt_to_world_affine
from ..fnirt.spline import BendingOperator


MMORF_VERSION = "0.3.2"
MMORF_COMMIT = "1c1c13b8368f05e1a79a6dafe919d6b61df36bd6"
MMORF_WARP_UNITS = "reference_axis_mm"


@dataclass(frozen=True)
class MMORFConfig:
    """The five-level MMORF 0.3.2 registration schedule.

    ``warp_resolution_mm``, smoothing, regularisation and iteration counts are
    the defaults documented by MMORF.  The deformation uses MMORF's cubic
    B-spline lattice, symmetric scalar/tensor costs, local finite-strain tensor
    reorientation and the later-level SPRED penalty. The parameter update uses
    limited-memory BFGS with a strong-Wolfe line search; it does not reproduce
    MMORF's sparse CUDA LM/MM solve.
    """

    warp_resolution_mm: tuple[float, ...] = (32.0, 32.0, 16.0, 8.0, 4.0)
    smoothing_mm: tuple[float, ...] = (8.0, 8.0, 4.0, 2.0, 1.0)
    regularization: tuple[float, ...] = (4.0e5, 3.7e-1, 3.1e-1, 2.6e-1, 2.2e-1)
    iterations: tuple[int, ...] = (5, 5, 5, 5, 5)
    scalar_weight: float = 1.0
    tensor_weight: float = 1.0
    learning_rate: float = 1.0

    def __post_init__(self):
        count = len(self.warp_resolution_mm)
        if count == 0 or any(
            len(value) != count
            for value in (
                self.smoothing_mm,
                self.regularization,
                self.iterations,
            )
        ):
            raise ValueError("all MMORF schedules must have the same non-zero length")
        if any(value <= 0 for value in self.warp_resolution_mm):
            raise ValueError("warp resolutions must be positive")
        if any(value < 0 for value in self.smoothing_mm + self.regularization):
            raise ValueError("smoothing and regularisation must be non-negative")
        if any(value < 1 for value in self.iterations):
            raise ValueError("iteration counts must be positive")


@dataclass(frozen=True)
class MMORFResult:
    warp: nib.Nifti1Image
    jacobian: nib.Nifti1Image
    warped_scalar: nib.Nifti1Image
    warped_tensor: nib.Nifti1Image
    qc: dict

    def save(self, output_dir, *, overwrite=False):
        output_dir = Path(output_dir).expanduser()
        paths = {
            output_dir / "mmorf_warp.nii.gz": self.warp,
            output_dir / "mmorf_jacobian.nii.gz": self.jacobian,
            output_dir / "mmorf_warped_scalar.nii.gz": self.warped_scalar,
            output_dir / "mmorf_warped_tensor.nii.gz": self.warped_tensor,
        }
        report = output_dir / "mmorf_report.json"
        existing = [path for path in (*paths, report) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"output exists: {existing[0]}; pass overwrite=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, image in paths.items():
            nib.save(image, str(path))
        report.write_text(json.dumps(self.qc, indent=2) + "\n", encoding="utf-8")
        return {**{path.name: path for path in paths}, report.name: report}


def _load_image(value, name, *, frames=None):
    image = nib.load(os.fspath(value)) if isinstance(value, (str, os.PathLike)) else value
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError(f"{name} must be a NIfTI path or image")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if frames is None and data.ndim != 3:
        raise ValueError(f"{name} must be a 3D image")
    if frames is not None and (data.ndim != 4 or data.shape[3] != frames):
        raise ValueError(f"{name} must be a 4D image with {frames} frames")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return image, data


def _load_matrix(value):
    if value is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.loadtxt(os.fspath(value)) if isinstance(value, (str, os.PathLike)) else value
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("an MMORF affine must be one finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("an MMORF affine must be homogeneous")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
        raise ValueError("an MMORF affine must be invertible")
    return matrix


def _world_forward(matrix, image, common):
    return flirt_to_world_affine(
        _load_matrix(matrix),
        image.affine,
        common.affine,
        image.shape[:3],
        common.shape[:3],
        image.header.get_zooms()[:3],
        common.header.get_zooms()[:3],
    )


def _grid(shape, *, device, dtype=torch.float32):
    return torch.stack(
        torch.meshgrid(
            *(torch.arange(size, device=device, dtype=dtype) for size in shape),
            indexing="ij",
        )
    )


def _normalised_grid(coordinates, shape):
    values = []
    for axis, size in enumerate(shape):
        values.append(
            torch.zeros_like(coordinates[axis])
            if size == 1
            else 2 * coordinates[axis] / (size - 1) - 1
        )
    return torch.stack(values, dim=-1).permute(2, 1, 0, 3)[None]


def _sample(data, coordinates, *, mode="bilinear"):
    if data.ndim == 3:
        data = data[None]
    sampled = F.grid_sample(
        data.permute(0, 3, 2, 1)[None],
        _normalised_grid(coordinates, data.shape[1:]).to(data.dtype),
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )[0]
    return sampled.permute(0, 3, 2, 1)


def _cubic_spline_coefficients(data):
    """Prefilter voxel values into regular cubic B-spline coefficients."""
    if data.ndim == 3:
        data = data[None]
    array = data.detach().cpu().numpy()
    coefficients = np.stack(
        [
            spline_filter(channel, order=3, output=np.float32, mode="mirror")
            for channel in array
        ]
    )
    return torch.as_tensor(
        coefficients, device=data.device, dtype=data.dtype
    )


def _sample_cubic(coefficients, coordinates):
    """Sample cubic coefficients via the exact eight-trilinear factorisation."""
    if coefficients.ndim == 3:
        coefficients = coefficients[None]

    positions = []
    grouped_weights = []
    for axis in range(3):
        coordinate = coordinates[axis]
        integer = torch.floor(coordinate)
        fraction = coordinate - integer
        one_minus = 1.0 - fraction
        w0 = one_minus.pow(3) / 6.0
        w1 = (3.0 * fraction.pow(3) - 6.0 * fraction.square() + 4.0) / 6.0
        w2 = (
            -3.0 * fraction.pow(3)
            + 3.0 * fraction.square()
            + 3.0 * fraction
            + 1.0
        ) / 6.0
        w3 = fraction.pow(3) / 6.0
        g0 = w0 + w1
        g1 = w2 + w3
        positions.append(
            (
                integer - 1.0 + w1 / g0,
                integer + 1.0 + w3 / g1,
            )
        )
        grouped_weights.append((g0, g1))

    sampled = coefficients.new_zeros(
        (coefficients.shape[0], *coordinates.shape[1:])
    )
    for xo in range(2):
        for yo in range(2):
            for zo in range(2):
                sample_coordinates = torch.stack(
                    (
                        positions[0][xo],
                        positions[1][yo],
                        positions[2][zo],
                    )
                )
                weight = (
                    grouped_weights[0][xo]
                    * grouped_weights[1][yo]
                    * grouped_weights[2][zo]
                )
                sampled = sampled + _sample(
                    coefficients, sample_coordinates
                ) * weight[None]
    return sampled


def _native_pull_coordinates(world_coordinates, image, forward, displacement=None):
    """Map common-world positions through a pull affine into native voxels."""
    transform = torch.as_tensor(
        np.linalg.inv(image.affine) @ np.linalg.inv(forward),
        device=world_coordinates.device,
        dtype=world_coordinates.dtype,
    )
    positions = world_coordinates
    if displacement is not None:
        positions = positions + displacement
    coordinates = torch.einsum(
        "ij,jxyz->ixyz", transform[:3, :3], positions
    )
    return coordinates + transform[:3, 3, None, None, None]


def _affine_coordinates(image, common, forward, *, device):
    common_voxels = _grid(common.shape[:3], device=device, dtype=torch.float64)
    common_affine = torch.as_tensor(common.affine, dtype=torch.float64, device=device)
    input_inverse = torch.as_tensor(
        np.linalg.inv(image.affine), dtype=torch.float64, device=device
    )
    pull = torch.as_tensor(np.linalg.inv(forward), dtype=torch.float64, device=device)
    world = torch.einsum("ij,jxyz->ixyz", common_affine[:3, :3], common_voxels)
    world = world + common_affine[:3, 3, None, None, None]
    input_world = torch.einsum("ij,jxyz->ixyz", pull[:3, :3], world)
    input_world = input_world + pull[:3, 3, None, None, None]
    coordinates = torch.einsum("ij,jxyz->ixyz", input_inverse[:3, :3], input_world)
    return (coordinates + input_inverse[:3, 3, None, None, None]).float()


def _tensor_matrix(channels):
    xx, xy, xz, yy, yz, zz = channels.unbind(0)
    return torch.stack(
        (xx, xy, xz, xy, yy, yz, xz, yz, zz), dim=-1
    ).reshape(*xx.shape, 3, 3)


def _tensor_channels(matrix):
    return torch.stack(
        (
            matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
            matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2],
        )
    )


def _polar_rotation(linear, *, device):
    matrix = torch.as_tensor(linear, dtype=torch.float32, device=device)
    u, _, vh = torch.linalg.svd(matrix)
    # MMORF uses (A A.T)^-1/2 A directly.  Preserve a reflection when A has
    # negative determinant; forcing SO(3) would change FSL radiological tensors.
    return u @ vh


def _rotate_tensor(channels, rotation):
    matrices = _tensor_matrix(channels)
    return _tensor_channels(rotation @ matrices @ rotation.T)


def _mm_to_voxel_axes_rotation(affine, *, device):
    """Return VolumeBSpline::get_mm_to_vox_rotation without voxel scaling."""
    mm_to_voxel = np.linalg.inv(np.asarray(affine, dtype=np.float64))[:3, :3]
    return _polar_rotation(mm_to_voxel, device=device)


def _blur(data, fwhm_mm, voxel_sizes):
    if fwhm_mm <= 0:
        return data
    result = data[None].permute(0, 1, 4, 3, 2)
    normaliser = torch.ones_like(result[:, :1])
    sigma_mm = fwhm_mm / 2.355
    for axis, voxel_size in enumerate(voxel_sizes):
        sigma = sigma_mm / voxel_size
        radius = max(1, int(math.ceil(3 * sigma)))
        points = torch.arange(-radius, radius + 1, device=data.device, dtype=data.dtype)
        kernel = torch.exp(-0.5 * (points / sigma).square())
        kernel /= kernel.sum()
        shape = [1, 1, 1, 1, 1]
        dimension = 4 - axis
        shape[dimension] = kernel.numel()
        padding = [0, 0, 0]
        padding[2 - axis] = radius
        weight = kernel.reshape(shape)
        result = F.conv3d(
            result,
            weight.repeat(data.shape[0], 1, 1, 1, 1),
            padding=tuple(padding),
            groups=data.shape[0],
        )
        normaliser = F.conv3d(normaliser, weight, padding=tuple(padding))
    result = result / normaliser.clamp_min(torch.finfo(result.dtype).eps)
    return result[0].permute(0, 3, 2, 1)


def _resize(data, shape):
    return F.interpolate(
        data.permute(0, 3, 2, 1)[None],
        size=(shape[2], shape[1], shape[0]),
        mode="trilinear",
        align_corners=True,
    )[0].permute(0, 3, 2, 1)


def _robust_normalise(data, mask=None, *, return_scale=False):
    """Scale an MMORF scalar volume so its robust mean is 100.

    This is the two-pass rule in
    ``CostFxnSSDWarpFieldSymmetricMaskedExcluded``: find the global weighted
    mean, then average samples larger than 0.17 times that mean.  It is not a
    percentile/window normalisation.
    """
    if mask is None:
        weights = torch.ones_like(data)
    else:
        weights = mask.to(dtype=data.dtype)
    weighting = weights.sum()
    if not bool(weighting > 0):
        raise ValueError("an MMORF scalar modality has an empty mask")
    global_mean = (data * weights).sum() / weighting
    robust = (data * weights) > (0.17 * global_mean)
    robust_weighting = weights[robust].sum()
    if not bool(robust_weighting > 0):
        raise ValueError("an MMORF scalar modality has no robust-mean samples")
    robust_mean = (data[robust] * weights[robust]).sum() / robust_weighting
    scale = 100.0 / robust_mean.clamp_min(torch.finfo(data.dtype).eps)
    return scale if return_scale else data * scale


def _world_extents(shape, affine):
    corners = np.asarray(
        [
            (x, y, z, 1.0)
            for x in (0, int(shape[0]) - 1)
            for y in (0, int(shape[1]) - 1)
            for z in (0, int(shape[2]) - 1)
        ],
        dtype=np.float64,
    )
    world = corners @ np.asarray(affine, dtype=np.float64).T
    return world[:, :3].min(0), world[:, :3].max(0)


def _control_shape(shape, voxel_sizes, resolution_mm, affine=None):
    # WarpFieldBSpline is an axis-aligned world-mm lattice.  It covers the
    # ordered world extents and adds one coefficient on either side plus the
    # fencepost coefficient (MMORF 0.3.2, WarpFieldBSpline.cu lines 57--69).
    if affine is None:
        extent = np.asarray(
            [(size - 1) * voxel for size, voxel in zip(shape, voxel_sizes)]
        )
    else:
        lower, upper = _world_extents(shape, affine)
        extent = upper - lower
    return tuple(
        max(4, int(math.ceil(float(length) / resolution_mm)) + 3)
        for length in extent
    )


def _cubic_bspline_basis(
    positions,
    *,
    voxel_size,
    resolution_mm,
    control_points,
    extent_min=0.0,
):
    """MMORF cubic B-spline basis for one world-space axis.

    Coefficient zero is centred one knot before the lower world extent.  Rows
    index positions and columns index MMORF coefficients.
    """
    coordinate = (
        positions * float(voxel_size) - float(extent_min)
    ) / float(resolution_mm) + 1.0
    distance = (coordinate[:, None] - torch.arange(
        int(control_points), device=positions.device, dtype=positions.dtype
    )[None]).abs()
    near = (2.0 / 3.0) - distance.square() + 0.5 * distance.pow(3)
    far = (2.0 - distance).clamp_min(0).pow(3) / 6.0
    return torch.where(distance < 1.0, near, torch.where(distance < 2.0, far, 0.0))


def _axis_mapping(linear):
    """Return world axis per voxel axis for a signed-permutation affine."""
    matrix = np.asarray(linear, dtype=np.float64)
    mapping = tuple(int(np.argmax(np.abs(matrix[:, axis]))) for axis in range(3))
    if len(set(mapping)) != 3:
        return None
    kept = np.zeros_like(matrix)
    for voxel_axis, world_axis in enumerate(mapping):
        kept[world_axis, voxel_axis] = matrix[world_axis, voxel_axis]
    tolerance = 1.0e-6 * max(1.0, float(np.abs(matrix).max()))
    return mapping if np.allclose(matrix, kept, atol=tolerance, rtol=0) else None


def _cubic_indices_and_weights(coordinate, control_points):
    integer = torch.floor(coordinate).to(torch.long)
    offsets = torch.arange(-1, 3, device=coordinate.device)
    indices = integer[..., None] + offsets
    distance = (coordinate[..., None] - indices.to(coordinate.dtype)).abs()
    near = (2.0 / 3.0) - distance.square() + 0.5 * distance.pow(3)
    far = (2.0 - distance).clamp_min(0).pow(3) / 6.0
    weights = torch.where(
        distance < 1.0, near, torch.where(distance < 2.0, far, 0.0)
    )
    valid = (indices >= 0) & (indices < int(control_points))
    return indices.clamp(0, int(control_points) - 1), weights * valid


def _expand_control(
    control,
    shape,
    voxel_sizes,
    resolution_mm,
    *,
    positions=None,
    affine=None,
):
    """Evaluate MMORF's axis-aligned world-mm cubic coefficient lattice."""
    if positions is None:
        positions = tuple(
            torch.arange(
                size, device=control.device, dtype=control.dtype
            )
            for size in shape
        )
    if affine is None:
        affine = np.diag((*[float(value) for value in voxel_sizes], 1.0))
    affine = np.asarray(affine, dtype=np.float64)
    lower, _ = _world_extents(shape, affine)
    mapping = _axis_mapping(affine[:3, :3])
    if mapping is not None:
        bases = []
        for voxel_axis, world_axis in enumerate(mapping):
            world_positions = (
                float(affine[world_axis, voxel_axis]) * positions[voxel_axis]
                + float(affine[world_axis, 3])
            )
            bases.append(
                _cubic_bspline_basis(
                    world_positions,
                    voxel_size=1.0,
                    resolution_mm=resolution_mm,
                    control_points=control.shape[1 + world_axis],
                    extent_min=lower[world_axis],
                )
            )
        # Coefficient spatial axes are world x/y/z.  Reorder them into voxel
        # x/y/z before applying the three separable design matrices.
        reordered = control.permute(0, *(1 + axis for axis in mapping))
        bx, by, bz = bases
        field = torch.einsum("xi,cijk->cxjk", bx, reordered)
        field = torch.einsum("yj,cxjk->cxyk", by, field)
        return torch.einsum("zk,cxyk->cxyz", bz, field)

    # Oblique warp spaces are uncommon for MMORF templates but are valid.
    # Evaluate their non-separable world lattice directly using the 4^3 local
    # support, without materialising a dense design matrix.
    voxels = torch.stack(torch.meshgrid(*positions, indexing="ij"))
    linear = torch.as_tensor(
        affine[:3, :3], device=control.device, dtype=control.dtype
    )
    translation = torch.as_tensor(
        affine[:3, 3], device=control.device, dtype=control.dtype
    )
    world = torch.einsum("ij,jxyz->ixyz", linear, voxels)
    world = world + translation[:, None, None, None]
    lower_tensor = torch.as_tensor(
        lower, device=control.device, dtype=control.dtype
    )
    coordinate = (
        (world - lower_tensor[:, None, None, None]) / float(resolution_mm)
        + 1.0
    )
    indices_weights = [
        _cubic_indices_and_weights(coordinate[axis], control.shape[1 + axis])
        for axis in range(3)
    ]
    (ix, wx), (iy, wy), (iz, wz) = indices_weights
    field = control.new_zeros((3, *voxels.shape[1:]))
    for x_offset in range(4):
        for y_offset in range(4):
            for z_offset in range(4):
                values = control[
                    :,
                    ix[..., x_offset],
                    iy[..., y_offset],
                    iz[..., z_offset],
                ]
                weight = (
                    wx[..., x_offset]
                    * wy[..., y_offset]
                    * wz[..., z_offset]
                )
                field = field + values * weight[None]
    return field


def _refinement_mask(factor, *, device, dtype):
    """Return the cardinal cubic refinement mask for an integer scale."""
    factor = int(factor)
    if factor < 2:
        raise ValueError("the B-spline refinement factor must be at least two")
    kernel = torch.ones(factor, device=device, dtype=dtype)
    result = kernel
    for _ in range(3):
        result = F.conv1d(
            result[None, None],
            kernel.flip(0)[None, None],
            padding=factor - 1,
        )[0, 0]
    return result / float(factor**3)


def _refine_control(control, old_resolution_mm, new_resolution_mm, new_shape):
    """Exactly refine a cardinal cubic lattice for MMORF's integer scaling."""
    ratio = float(old_resolution_mm) / float(new_resolution_mm)
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=0, abs_tol=1e-6):
        raise ValueError("MMORF warp resolutions must differ by an integer factor")
    if factor == 1:
        if tuple(control.shape[1:]) != tuple(new_shape):
            raise ValueError("unchanged MMORF resolution changed the control shape")
        return control
    mask = _refinement_mask(factor, device=control.device, dtype=control.dtype)
    radius = 2 * (factor - 1)
    transforms = []
    for old_points, new_points in zip(control.shape[1:], new_shape):
        old_index = torch.arange(old_points, device=control.device)
        new_index = torch.arange(new_points, device=control.device)
        # Both cropped lattices put coefficient 1 at reference position zero.
        centre = factor * old_index - (factor - 1)
        mask_index = new_index[:, None] - centre[None] + radius
        valid = (mask_index >= 0) & (mask_index < mask.numel())
        transform = control.new_zeros((new_points, old_points))
        transform[valid] = mask[mask_index[valid]]
        transforms.append(transform)
    rx, ry, rz = transforms
    refined = torch.einsum("xi,cijk->cxjk", rx, control)
    refined = torch.einsum("yj,cxjk->cxyk", ry, refined)
    return torch.einsum("zk,cxyk->cxyz", rz, refined)


def _sampling_frequency(knot_spacing_mm, smoothing_mm, max_resolution_mm):
    """Match RegistrationCoordinatorMultimodal's modality sampling rule."""
    if smoothing_mm > 0:
        frequency = int(math.floor(2.0 * knot_spacing_mm / smoothing_mm))
    else:
        frequency = 0
    frequency = min(max(frequency, 1), 4)
    if float(frequency) > knot_spacing_mm / max_resolution_mm:
        frequency = int(math.ceil(knot_spacing_mm / max_resolution_mm))
    return max(frequency, 1)


def _robust_world_axes(
    control_shape,
    extent_min,
    knot_spacing_mm,
    sampling_frequency,
    *,
    device,
    dtype,
):
    """Return WarpFieldBSpline::get_robust_sample_positions as 1-D axes."""
    step = float(knot_spacing_mm) / int(sampling_frequency)
    return tuple(
        float(lower)
        + step
        * torch.arange(
            (int(controls) - 3) * int(sampling_frequency) + 1,
            device=device,
            dtype=dtype,
        )
        for controls, lower in zip(control_shape, extent_min)
    )


def _expand_control_world(control, world_axes, knot_spacing_mm, extent_min):
    """Evaluate MMORF coefficients on ordered world x/y/z sample axes."""
    bases = tuple(
        _cubic_bspline_basis(
            axis,
            voxel_size=1.0,
            resolution_mm=knot_spacing_mm,
            control_points=control.shape[spatial_axis + 1],
            extent_min=extent_min[spatial_axis],
        )
        for spatial_axis, axis in enumerate(world_axes)
    )
    bx, by, bz = bases
    field = torch.einsum("xi,cijk->cxjk", bx, control)
    field = torch.einsum("yj,cxjk->cxyk", by, field)
    return torch.einsum("zk,cxyk->cxyz", bz, field)


def _bending_regularisation(
    control,
    knot_spacing_mm,
    sampling_frequency,
    operator=None,
):
    """MMORF Bkk energy and normalization on the full cubic support."""
    frequency = int(sampling_frequency)
    sample_shape = tuple(
        (int(controls) - 3) * frequency + 1 for controls in control.shape[1:]
    )
    if operator is None:
        operator = BendingOperator(
            sample_shape,
            (frequency,) * 3,
            (float(knot_spacing_mm) / frequency,) * 3,
            device=control.device,
            dtype=control.dtype,
        )
    norm = 1.0 / float(
        np.prod([int(value) * frequency for value in control.shape[1:]])
    )
    return norm * operator.energy(control), operator


def _jacobian(field, spacing=(1.0, 1.0, 1.0)):
    # Components and derivative axes must use the same coordinate system.
    derivatives = []
    for component in range(3):
        rows = []
        value = field[component]
        for axis in range(3):
            if value.shape[axis] == 1:
                rows.append(torch.zeros_like(value))
                continue
            left = torch.roll(value, 1, axis)
            right = torch.roll(value, -1, axis)
            derivative = 0.5 * (right - left)
            start = [slice(None)] * 3
            end = [slice(None)] * 3
            start[axis] = 0
            end[axis] = -1
            derivative[tuple(start)] = torch.diff(value, dim=axis)[tuple(start)]
            derivative[tuple(end)] = torch.diff(value, dim=axis)[tuple(end)]
            rows.append(derivative / float(spacing[axis]))
        derivatives.append(torch.stack(rows, dim=-1))
    matrix = torch.stack(derivatives, dim=-2)
    identity = torch.eye(3, dtype=field.dtype, device=field.device)
    return matrix + identity


def _world_jacobian_from_voxel_field(field_world, voxel_to_world):
    """Return I + du_world/dworld for samples laid out in voxel order."""
    identity = torch.eye(3, device=field_world.device, dtype=field_world.dtype)
    derivative_voxel = _jacobian(field_world) - identity
    inverse_linear = torch.as_tensor(
        np.linalg.inv(np.asarray(voxel_to_world, dtype=np.float64)[:3, :3]),
        device=field_world.device,
        dtype=field_world.dtype,
    )
    return identity + torch.einsum(
        "...ij,jk->...ik", derivative_voxel, inverse_linear
    )


def _finite_strain_rotation(jacobian, world_linear=None, iterations=5):
    """Return MMORF's local finite-strain factor (JJ.T)^-1/2 J.

    A Newton polar iteration is used instead of differentiating an SVD.  It is
    stable at the identity deformation, where SVD gradients are undefined, and
    converges quadratically for the diffeomorphic fields accepted by MMORF.
    """
    matrix = jacobian
    if world_linear is not None:
        linear = torch.as_tensor(
            world_linear, device=matrix.device, dtype=matrix.dtype
        )
        inverse = torch.linalg.inv(linear)
        matrix = torch.einsum("ij,...jk,kl->...il", linear, matrix, inverse)
    scale = (
        torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1))
        / math.sqrt(3.0)
    ).clamp_min(torch.finfo(matrix.dtype).eps)
    rotation = matrix / scale[..., None, None]
    for _ in range(int(iterations)):
        inverse_transpose = torch.linalg.inv(rotation).transpose(-2, -1)
        rotation = 0.5 * (rotation + inverse_transpose)
    return rotation


def _rotate_tensor_field(channels, rotation):
    matrices = _tensor_matrix(channels)
    rotated = torch.matmul(rotation, matrices)
    rotated = torch.matmul(rotated, rotation.transpose(-2, -1))
    return _tensor_channels(rotated)


def _symmetric_weight(jacobian):
    """Return MMORF's fixed symmetric volume quadrature weight."""
    return (0.5 * (1.0 + torch.linalg.det(jacobian))).detach()


def _spred_regularisation(jacobian, control_shape, sampling_frequency):
    """Return the analytic SPRED cost approximation used by MMORF 0.3.2.

    The CUDA kernel evaluates 0.25*(det(J)+1) times the sum of the squared
    Frobenius norms of J and inverse(J), minus six.  The coordinator scales its
    sum by 1000/prod(control_shape*sampling_frequency).
    """
    determinant = torch.linalg.det(jacobian)
    inverse = torch.linalg.inv(jacobian)
    per_sample = 0.25 * (determinant + 1.0) * (
        jacobian.square().sum(dim=(-2, -1))
        + inverse.square().sum(dim=(-2, -1))
        - 6.0
    )
    norm = 1000.0 / float(
        np.prod([int(value) * int(sampling_frequency) for value in control_shape])
    )
    return norm * per_sample.clamp_min(1.0e-20).sum(), determinant


def _make_warp(field, reference):
    data = field.permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(data, reference.affine, header)


def _validate_warp(warp, reference):
    warp_image, field = _load_image(warp, "warp", frames=3)
    if field.shape[:3] != reference.shape[:3] or not np.allclose(
        warp_image.affine, reference.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("MMORF warp and reference must use the same grid")
    return warp_image, field


def apply_mmorf_warp(
    image,
    reference,
    warp,
    *,
    affine=None,
    device=None,
    interpolation="linear",
):
    """Apply an MMORF reference-axis millimetre pull field."""
    source = nib.load(os.fspath(image)) if isinstance(image, (str, os.PathLike)) else image
    common = nib.load(os.fspath(reference)) if isinstance(reference, (str, os.PathLike)) else reference
    if not isinstance(source, (nib.Nifti1Image, nib.Nifti2Image)) or not isinstance(
        common, (nib.Nifti1Image, nib.Nifti2Image)
    ):
        raise TypeError("image and reference must be NIfTI paths or images")
    _, field = _validate_warp(warp, common)
    selected_device = configure_device(device)
    forward = _world_forward(affine, source, common)
    base = _affine_coordinates(source, common, forward, device=selected_device)
    reference_axes = _mm_to_voxel_axes_rotation(
        common.affine, device=selected_device
    ).to(torch.float64)
    pull_linear = torch.as_tensor(
        np.linalg.inv(source.affine)[:3, :3] @ np.linalg.inv(forward)[:3, :3],
        dtype=torch.float64,
        device=selected_device,
    )
    field_t = torch.as_tensor(np.moveaxis(field, -1, 0), device=selected_device)
    world_displacement = torch.einsum(
        "ij,jxyz->ixyz", reference_axes.T, field_t.to(torch.float64)
    )
    input_displacement = torch.einsum("ij,jxyz->ixyz", pull_linear, world_displacement)
    coordinates = base + input_displacement.float()
    data = np.asarray(source.dataobj, dtype=np.float32)
    tensor = torch.as_tensor(
        np.moveaxis(data, -1, 0) if data.ndim == 4 else data,
        dtype=torch.float32,
        device=selected_device,
    )
    if interpolation == "cubic":
        sampled = _sample_cubic(_cubic_spline_coefficients(tensor), coordinates)
    else:
        mode = "nearest" if interpolation in ("nearest", "nn") else "bilinear"
        sampled = _sample(tensor, coordinates, mode=mode)
    output = sampled.detach().cpu().numpy()
    if data.ndim == 4:
        output = np.moveaxis(output, 0, -1)
    else:
        output = output[0]
    return image_like(output.astype(np.float32), common)


class TorchMMORF:
    """Register one scalar and one FSL-format tensor pair with a shared warp."""

    def __init__(self, device=None, *, config=None):
        self.device = configure_device(device)
        self.config = MMORFConfig() if config is None else config

    def __call__(
        self,
        moving_scalar,
        reference_scalar,
        moving_tensor,
        reference_tensor,
        *,
        moving_scalar_affine=None,
        moving_tensor_affine=None,
        reference_tensor_affine=None,
    ):
        mov_s, mov_s_data = _load_image(moving_scalar, "moving_scalar")
        ref_s, ref_s_data = _load_image(reference_scalar, "reference_scalar")
        mov_t, mov_t_data = _load_image(moving_tensor, "moving_tensor", frames=6)
        ref_t, ref_t_data = _load_image(reference_tensor, "reference_tensor", frames=6)
        forwards = {
            "reference_scalar": _world_forward(None, ref_s, ref_s),
            "moving_scalar": _world_forward(moving_scalar_affine, mov_s, ref_s),
            "moving_tensor": _world_forward(moving_tensor_affine, mov_t, ref_s),
            "reference_tensor": _world_forward(reference_tensor_affine, ref_t, ref_s),
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        ref_scalar = torch.as_tensor(
            ref_s_data, device=self.device, dtype=torch.float32
        )[None]
        mov_scalar = torch.as_tensor(
            mov_s_data, device=self.device, dtype=torch.float32
        )[None]
        ref_tensor = torch.as_tensor(
            np.moveaxis(ref_t_data, -1, 0),
            device=self.device,
            dtype=torch.float32,
        )
        mov_tensor = torch.as_tensor(
            np.moveaxis(mov_t_data, -1, 0),
            device=self.device,
            dtype=torch.float32,
        )
        ref_rotation = (
            _polar_rotation(forwards["reference_tensor"][:3, :3], device=self.device)
            @ _polar_rotation(ref_t.affine[:3, :3], device=self.device)
        )
        mov_rotation = (
            _polar_rotation(forwards["moving_tensor"][:3, :3], device=self.device)
            @ _polar_rotation(mov_t.affine[:3, :3], device=self.device)
        )
        # VolumeTensor converts FSL diffusivity units to order-one values.
        ref_tensor = 1.0e6 * _rotate_tensor(ref_tensor, ref_rotation)
        mov_tensor = 1.0e6 * _rotate_tensor(mov_tensor, mov_rotation)
        voxel_sizes = tuple(float(value) for value in ref_s.header.get_zooms()[:3])
        world_to_reference_axes = _mm_to_voxel_axes_rotation(
            ref_s.affine, device=self.device
        )
        control = None
        previous_resolution = None
        levels = []
        cfg = self.config
        extent_min, _ = _world_extents(ref_s.shape[:3], ref_s.affine)
        scalar_resolution = min(
            *(abs(float(value)) for value in ref_s.header.get_zooms()[:3]),
            *(abs(float(value)) for value in mov_s.header.get_zooms()[:3]),
        )
        tensor_resolution = min(
            *(abs(float(value)) for value in ref_t.header.get_zooms()[:3]),
            *(abs(float(value)) for value in mov_t.header.get_zooms()[:3]),
        )
        maximum_resolution = min(scalar_resolution, tensor_resolution)
        for level, (resolution, smoothing, penalty, steps) in enumerate(
            zip(
                cfg.warp_resolution_mm,
                cfg.smoothing_mm,
                cfg.regularization,
                cfg.iterations,
            ),
            1,
        ):
            control_shape = _control_shape(
                ref_s.shape[:3], voxel_sizes, resolution, affine=ref_s.affine
            )
            if control is None:
                control = torch.zeros((3, *control_shape), device=self.device)
            elif tuple(control.shape[1:]) != control_shape:
                control = _refine_control(
                    control.detach(),
                    previous_resolution,
                    resolution,
                    control_shape,
                )
            control = control.detach().contiguous().requires_grad_(True)
            scalar_frequency = _sampling_frequency(
                resolution, smoothing, scalar_resolution
            )
            tensor_frequency = _sampling_frequency(
                resolution, smoothing, tensor_resolution
            )
            regulariser_frequency = min(
                max(int(math.floor(resolution / maximum_resolution)), 1), 4
            )
            scalar_axes = _robust_world_axes(
                control_shape,
                extent_min,
                resolution,
                scalar_frequency,
                device=self.device,
                dtype=control.dtype,
            )
            tensor_axes = _robust_world_axes(
                control_shape,
                extent_min,
                resolution,
                tensor_frequency,
                device=self.device,
                dtype=control.dtype,
            )
            regulariser_axes = _robust_world_axes(
                control_shape,
                extent_min,
                resolution,
                regulariser_frequency,
                device=self.device,
                dtype=control.dtype,
            )
            scalar_world = torch.stack(torch.meshgrid(*scalar_axes, indexing="ij"))
            tensor_world = torch.stack(torch.meshgrid(*tensor_axes, indexing="ij"))
            ref_s_smooth = _blur(
                ref_scalar, smoothing, ref_s.header.get_zooms()[:3]
            )
            mov_s_smooth = _blur(
                mov_scalar, smoothing, mov_s.header.get_zooms()[:3]
            )
            ref_t_smooth = _blur(
                ref_tensor, smoothing, ref_t.header.get_zooms()[:3]
            )
            mov_t_smooth = _blur(
                mov_tensor, smoothing, mov_t.header.get_zooms()[:3]
            )
            ref_s_coefficients = _cubic_spline_coefficients(ref_s_smooth)
            mov_s_coefficients = _cubic_spline_coefficients(mov_s_smooth)
            ref_t_coefficients = _cubic_spline_coefficients(ref_t_smooth)
            mov_t_coefficients = _cubic_spline_coefficients(mov_t_smooth)
            ref_s_level = _sample_cubic(
                ref_s_coefficients,
                _native_pull_coordinates(
                    scalar_world, ref_s, forwards["reference_scalar"]
                ),
            )[0]
            mov_s_level = _sample_cubic(
                mov_s_coefficients,
                _native_pull_coordinates(
                    scalar_world, mov_s, forwards["moving_scalar"]
                ),
            )[0]
            ref_t_level = _sample_cubic(
                ref_t_coefficients,
                _native_pull_coordinates(
                    tensor_world, ref_t, forwards["reference_tensor"]
                ),
            )
            scalar_ref_scale = _robust_normalise(ref_s_level, return_scale=True)
            scalar_mov_scale = _robust_normalise(mov_s_level, return_scale=True)
            ref_s_norm = ref_s_level * scalar_ref_scale
            bending_operator = None

            def evaluate_terms():
                nonlocal bending_operator
                scalar_field_world = _expand_control_world(
                    control, scalar_axes, resolution, extent_min
                )
                warped_scalar = _sample_cubic(
                    mov_s_coefficients * scalar_mov_scale,
                    _native_pull_coordinates(
                        scalar_world,
                        mov_s,
                        forwards["moving_scalar"],
                        scalar_field_world,
                    ),
                )[0]
                scalar_step = resolution / scalar_frequency
                scalar_jacobian = _jacobian(
                    scalar_field_world, (scalar_step,) * 3
                )
                # MMORF uses the symmetric determinant as a fixed quadrature
                # weight in data derivatives; it does not differentiate it.
                scalar_weight = _symmetric_weight(scalar_jacobian)
                scalar_cost = (
                    scalar_weight * (warped_scalar - ref_s_norm).square()
                ).mean()

                tensor_field_world = _expand_control_world(
                    control, tensor_axes, resolution, extent_min
                )
                warped_tensor = _sample_cubic(
                    mov_t_coefficients,
                    _native_pull_coordinates(
                        tensor_world,
                        mov_t,
                        forwards["moving_tensor"],
                        tensor_field_world,
                    ),
                )
                tensor_step = resolution / tensor_frequency
                tensor_jacobian = _jacobian(
                    tensor_field_world, (tensor_step,) * 3
                )
                tensor_weight = _symmetric_weight(tensor_jacobian)
                rotation = _finite_strain_rotation(tensor_jacobian)
                rotated_reference_tensor = _rotate_tensor_field(
                    ref_t_level, rotation
                )
                tensor_error = (
                    _tensor_matrix(warped_tensor)
                    - _tensor_matrix(rotated_reference_tensor)
                )
                tensor_cost = (
                    tensor_weight
                    * tensor_error.square().sum(dim=(-2, -1))
                ).mean()

                regulariser_field_world = _expand_control_world(
                    control, regulariser_axes, resolution, extent_min
                )
                regulariser_step = resolution / regulariser_frequency
                regulariser_jacobian = _jacobian(
                    regulariser_field_world, (regulariser_step,) * 3
                )
                determinant = torch.linalg.det(regulariser_jacobian)
                if level == 1:
                    regulariser, bending_operator = _bending_regularisation(
                        control,
                        resolution,
                        regulariser_frequency,
                        bending_operator,
                    )
                    regulariser_name = "bending_bkk"
                else:
                    regulariser, determinant = _spred_regularisation(
                        regulariser_jacobian,
                        control_shape,
                        regulariser_frequency,
                    )
                    # The official Levenberg loop rejects a non-diffeomorphic
                    # candidate and increases damping.  The barrier supplies a
                    # finite line-search objective when that mechanism is absent.
                    regulariser = regulariser + (
                        (1.0e-3 - determinant).clamp_min(0).square().sum()
                        * 1.0e6
                    )
                    regulariser_name = "spred"
                return (
                    scalar_cost,
                    tensor_cost,
                    regulariser,
                    determinant,
                    regulariser_name,
                )

            with torch.no_grad():
                initial_scalar, initial_tensor, _, _, _ = evaluate_terms()
                scalar_cost_scale = (
                    initial_scalar.new_zeros(())
                    if float(initial_scalar) <= 1.0e-8
                    else 50.0 / initial_scalar
                )
                tensor_cost_scale = (
                    initial_tensor.new_zeros(())
                    if float(initial_tensor) <= 1.0e-8
                    else 50.0 / initial_tensor
                )
            optimiser = torch.optim.LBFGS(
                [control],
                lr=cfg.learning_rate,
                max_iter=steps,
                max_eval=max(20, 4 * steps),
                tolerance_grad=1.0e-7,
                tolerance_change=1.0e-9,
                history_size=10,
                line_search_fn="strong_wolfe",
            )
            closure_evaluations = 0

            def closure():
                nonlocal closure_evaluations
                optimiser.zero_grad(set_to_none=True)
                scalar_cost, tensor_cost, regulariser, _, _ = evaluate_terms()
                loss = (
                    cfg.scalar_weight * scalar_cost_scale * scalar_cost
                    + cfg.tensor_weight * tensor_cost_scale * tensor_cost
                    + penalty * regulariser
                )
                loss.backward()
                closure_evaluations += 1
                return loss

            optimiser.step(closure)
            with torch.no_grad():
                (
                    scalar_cost,
                    tensor_cost,
                    regulariser,
                    determinant,
                    regulariser_name,
                ) = evaluate_terms()
                loss = (
                    cfg.scalar_weight * scalar_cost_scale * scalar_cost
                    + cfg.tensor_weight * tensor_cost_scale * tensor_cost
                    + penalty * regulariser
                )
            latest = {
                "total": float(loss.detach()),
                "scalar": float(scalar_cost.detach()),
                "tensor": float(tensor_cost.detach()),
                "scalar_cost_scale": float(scalar_cost_scale),
                "tensor_cost_scale": float(tensor_cost_scale),
                "regulariser": float(regulariser.detach()),
                "regulariser_name": regulariser_name,
                "jacobian_min": float(determinant.detach().min()),
                "jacobian_max": float(determinant.detach().max()),
                "optimizer": "lbfgs_strong_wolfe",
                "closure_evaluations": closure_evaluations,
            }
            levels.append(
                {
                    "level": level,
                    "warp_resolution_mm": resolution,
                    "smoothing_mm": smoothing,
                    "scalar_sampling_frequency": scalar_frequency,
                    "tensor_sampling_frequency": tensor_frequency,
                    "regulariser_sampling_frequency": regulariser_frequency,
                    "scalar_sample_shape": tuple(axis.numel() for axis in scalar_axes),
                    "tensor_sample_shape": tuple(axis.numel() for axis in tensor_axes),
                    "iterations": steps,
                    "control_shape": control_shape,
                    **latest,
                }
            )
            control = control.detach()
            previous_resolution = resolution
        field_world = _expand_control(
            control,
            ref_s.shape[:3],
            voxel_sizes,
            previous_resolution,
            affine=ref_s.affine,
        )
        field = torch.einsum(
            "ij,jxyz->ixyz", world_to_reference_axes, field_world
        )
        warp = _make_warp(field, ref_s)
        dense_jacobian_world = _world_jacobian_from_voxel_field(
            field_world, ref_s.affine
        )
        jacobian = torch.linalg.det(dense_jacobian_world)
        jacobian_image = image_like(
            jacobian.detach().cpu().numpy().astype(np.float32), ref_s
        )
        warped_scalar = apply_mmorf_warp(
            mov_s,
            ref_s,
            warp,
            affine=moving_scalar_affine,
            device=self.device,
            interpolation="cubic",
        )
        warped_tensor = apply_mmorf_warp(
            mov_t,
            ref_s,
            warp,
            affine=moving_tensor_affine,
            device=self.device,
            interpolation="cubic",
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return MMORFResult(
            warp,
            jacobian_image,
            warped_scalar,
            warped_tensor,
            {
                "device": str(self.device),
                "dtype": "float32",
                "tf32": bool(self.device.type == "cuda"),
                "reference_implementation": f"FSL MMORF {MMORF_VERSION}",
                "reference_commit": MMORF_COMMIT,
                "warp_units": MMORF_WARP_UNITS,
                "fsl_affine_contract": True,
                "mmorf_warp_contract": True,
                "mmorf_numerically_equivalent": False,
                "implemented_mmorf_semantics": {
                    "control_lattice": (
                        "axis-aligned world-mm cardinal cubic B-spline lattice"
                    ),
                    "sample_lattice": (
                        "ordered robust world grid with source-derived modality "
                        "and regulariser sampling frequencies"
                    ),
                    "image_sampling": (
                        "native-image world pull with cubic B-spline coefficient "
                        "prefiltering and zero-border 64-neighbour sampling"
                    ),
                    "cost": (
                        "robust scalar normalisation, fixed symmetric 0.5*(1+detJ) "
                        "quadrature weighting and full tensor Frobenius cost"
                    ),
                    "regularizer": (
                        "full-support Bkk bending energy at level one and SPRED "
                        "at later levels"
                    ),
                    "tensor_reorientation": (
                        "header/affine and local nonlinear finite-strain rotations"
                    ),
                },
                "algorithm_difference": {
                    "optimizer": (
                        "limited-memory BFGS with strong-Wolfe line search remains in "
                        "place of MMORF's sparse CUDA full/diagonal "
                        "Gauss-Newton solve, damping and candidate rejection"
                    ),
                    "image_prefilter": (
                        "SciPy mirror-boundary coefficient prefiltering has not "
                        "yet been shown coefficient-identical to the vendored "
                        "Danny Ruijters CUDA prefilter"
                    ),
                    "smoothing": (
                        "separable PyTorch Gaussian smoothing uses a finite "
                        "three-sigma kernel rather than NEWIMAGE's implementation"
                    ),
                    "jacobian_output": (
                        "the saved dense Jacobian uses finite differences rather "
                        "than analytic cubic-spline derivatives"
                    ),
                },
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                ),
                "levels": levels,
            },
        )

    def run(self, *args, output_dir, overwrite=False, **kwargs):
        result = self(*args, **kwargs)
        result.save(output_dir, overwrite=overwrite)
        return result


__all__ = [
    "MMORF_COMMIT",
    "MMORF_VERSION",
    "MMORFConfig",
    "MMORFResult",
    "TorchMMORF",
    "apply_mmorf_warp",
]
