"""PyTorch implementation of the MMORF scalar-plus-tensor registration path.

The public transform follows MMORF 0.3.2: the warp is sampled on the common
reference grid and stores relative displacement in reference-voxel units.
Each input affine is an FSL scaled-mm matrix mapping that input into the common
space.  This convention differs from FNIRT/applywarp displacement fields.
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

from .._dmri import configure_device, image_like
from ..flirt import flirt_to_world_affine


MMORF_VERSION = "0.3.2"
MMORF_COMMIT = "1c1c13b8368f05e1a79a6dafe919d6b61df36bd6"
MMORF_WARP_UNITS = "reference_voxels"


@dataclass(frozen=True)
class MMORFConfig:
    """The five-level MMORF 0.3.2 registration schedule.

    ``warp_resolution_mm``, smoothing, regularisation and iteration counts are
    the defaults documented by MMORF. The PyTorch optimiser is Adam on a
    trilinearly expanded control lattice; it does not reproduce MMORF's CUDA
    LM/MM optimisation, cubic B-spline Hessian, or SPRED regulariser.
    """

    warp_resolution_mm: tuple[float, ...] = (32.0, 32.0, 16.0, 8.0, 4.0)
    smoothing_mm: tuple[float, ...] = (8.0, 8.0, 4.0, 2.0, 1.0)
    regularization: tuple[float, ...] = (4.0e5, 3.7e-1, 3.1e-1, 2.6e-1, 2.2e-1)
    iterations: tuple[int, ...] = (5, 5, 5, 5, 5)
    scalar_weight: float = 1.0
    tensor_weight: float = 1.0
    learning_rate: float = 0.18
    sample_stride: tuple[int, ...] = (8, 8, 4, 2, 1)

    def __post_init__(self):
        count = len(self.warp_resolution_mm)
        if count == 0 or any(
            len(value) != count
            for value in (
                self.smoothing_mm,
                self.regularization,
                self.iterations,
                self.sample_stride,
            )
        ):
            raise ValueError("all MMORF schedules must have the same non-zero length")
        if any(value <= 0 for value in self.warp_resolution_mm):
            raise ValueError("warp resolutions must be positive")
        if any(value < 0 for value in self.smoothing_mm + self.regularization):
            raise ValueError("smoothing and regularisation must be non-negative")
        if any(value < 1 for value in self.iterations + self.sample_stride):
            raise ValueError("iterations and sample strides must be positive")


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


def _resample_affine(data, image, common, forward, *, device):
    tensor = torch.as_tensor(np.moveaxis(data, -1, 0) if data.ndim == 4 else data)
    tensor = tensor.to(device=device, dtype=torch.float32)
    return _sample(tensor, _affine_coordinates(image, common, forward, device=device))


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
    rotation = u @ vh
    if float(torch.linalg.det(rotation)) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _rotate_tensor(channels, rotation):
    matrices = _tensor_matrix(channels)
    return _tensor_channels(rotation @ matrices @ rotation.T)


def _blur(data, fwhm_mm, voxel_sizes):
    if fwhm_mm <= 0:
        return data
    result = data[None].permute(0, 1, 4, 3, 2)
    sigma_mm = fwhm_mm / math.sqrt(8 * math.log(2))
    for axis, voxel_size in enumerate(voxel_sizes):
        sigma = sigma_mm / voxel_size
        radius = max(1, int(math.ceil(3 * sigma)))
        points = torch.arange(-radius, radius + 1, device=data.device, dtype=data.dtype)
        kernel = torch.exp(-0.5 * (points / sigma).square())
        kernel /= kernel.sum()
        shape = [1, 1, 1, 1, 1]
        dimension = 4 - axis
        shape[dimension] = kernel.numel()
        weight = kernel.reshape(shape).repeat(data.shape[0], 1, 1, 1, 1)
        padding = [0, 0, 0]
        padding[2 - axis] = radius
        result = F.conv3d(result, weight, padding=tuple(padding), groups=data.shape[0])
    return result[0].permute(0, 3, 2, 1)


def _resize(data, shape):
    return F.interpolate(
        data.permute(0, 3, 2, 1)[None],
        size=(shape[2], shape[1], shape[0]),
        mode="trilinear",
        align_corners=True,
    )[0].permute(0, 3, 2, 1)


def _robust_normalise(data, mask):
    selected = data[mask]
    if selected.numel() == 0:
        raise ValueError("an MMORF modality has an empty overlap mask")
    low, high = torch.quantile(selected, torch.tensor((0.02, 0.98), device=data.device))
    return (data.clamp(low, high) - low) / (high - low).clamp_min(1e-6)


def _control_shape(shape, voxel_sizes, resolution_mm):
    return tuple(
        max(2, int(math.ceil((size - 1) * voxel / resolution_mm)) + 1)
        for size, voxel in zip(shape, voxel_sizes)
    )


def _expand_control(control, shape):
    return _resize(control, shape)


def _regularisation(control):
    first = []
    second = []
    for axis in range(1, 4):
        one = torch.diff(control, dim=axis)
        first.append(one.square().mean())
        if control.shape[axis] > 2:
            second.append(torch.diff(one, dim=axis).square().mean())
    membrane = torch.stack(first).mean()
    bending = torch.stack(second).mean() if second else membrane
    return membrane, bending


def _jacobian(field):
    # field components and derivatives are both in reference-voxel axes.
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
            rows.append(derivative)
        derivatives.append(torch.stack(rows, dim=-1))
    matrix = torch.stack(derivatives, dim=-2)
    identity = torch.eye(3, dtype=field.dtype, device=field.device)
    return matrix + identity


def _make_warp(field, reference):
    data = field.permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_intent("vector", (), name="MMORF voxel displacement")
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
    """Apply an MMORF voxel-displacement pull field to a scalar/4D image."""
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
    ref_linear = torch.as_tensor(common.affine[:3, :3], dtype=torch.float64, device=selected_device)
    pull_linear = torch.as_tensor(
        np.linalg.inv(source.affine)[:3, :3] @ np.linalg.inv(forward)[:3, :3],
        dtype=torch.float64,
        device=selected_device,
    )
    field_t = torch.as_tensor(np.moveaxis(field, -1, 0), device=selected_device)
    world_displacement = torch.einsum(
        "ij,jxyz->ixyz", ref_linear, field_t.to(torch.float64)
    )
    input_displacement = torch.einsum("ij,jxyz->ixyz", pull_linear, world_displacement)
    coordinates = base + input_displacement.float()
    data = np.asarray(source.dataobj, dtype=np.float32)
    tensor = torch.as_tensor(
        np.moveaxis(data, -1, 0) if data.ndim == 4 else data,
        dtype=torch.float32,
        device=selected_device,
    )
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
            "moving_scalar": _world_forward(moving_scalar_affine, mov_s, ref_s),
            "moving_tensor": _world_forward(moving_tensor_affine, mov_t, ref_s),
            "reference_tensor": _world_forward(reference_tensor_affine, ref_t, ref_s),
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        ref_scalar = torch.as_tensor(ref_s_data, device=self.device)[None]
        mov_scalar = _resample_affine(
            mov_s_data, mov_s, ref_s, forwards["moving_scalar"], device=self.device
        )
        ref_tensor = _resample_affine(
            ref_t_data, ref_t, ref_s, forwards["reference_tensor"], device=self.device
        )
        mov_tensor = _resample_affine(
            mov_t_data, mov_t, ref_s, forwards["moving_tensor"], device=self.device
        )
        ref_rotation = _polar_rotation(forwards["reference_tensor"][:3, :3], device=self.device)
        mov_rotation = _polar_rotation(forwards["moving_tensor"][:3, :3], device=self.device)
        ref_tensor = _rotate_tensor(ref_tensor, ref_rotation)
        mov_tensor = _rotate_tensor(mov_tensor, mov_rotation)
        voxel_sizes = tuple(float(value) for value in ref_s.header.get_zooms()[:3])
        control = None
        levels = []
        cfg = self.config
        for level, (resolution, smoothing, penalty, steps, stride) in enumerate(
            zip(
                cfg.warp_resolution_mm,
                cfg.smoothing_mm,
                cfg.regularization,
                cfg.iterations,
                cfg.sample_stride,
            ),
            1,
        ):
            control_shape = _control_shape(ref_s.shape[:3], voxel_sizes, resolution)
            if control is None:
                control = torch.zeros((3, *control_shape), device=self.device)
            elif tuple(control.shape[1:]) != control_shape:
                control = _resize(control.detach(), control_shape)
            control.requires_grad_(True)
            shape = tuple(max(2, (size - 1) // stride + 1) for size in ref_s.shape[:3])
            ref_s_level = _resize(_blur(ref_scalar, smoothing, voxel_sizes), shape)[0]
            mov_s_level = _resize(_blur(mov_scalar, smoothing, voxel_sizes), shape)[0]
            ref_t_level = _resize(_blur(ref_tensor, smoothing, voxel_sizes), shape)
            mov_t_level = _resize(_blur(mov_tensor, smoothing, voxel_sizes), shape)
            scalar_mask = (ref_s_level != 0) & (mov_s_level != 0)
            tensor_mask = (
                ref_t_level.square().sum(0) > 0
            ) & (mov_t_level.square().sum(0) > 0)
            ref_s_norm = _robust_normalise(ref_s_level, scalar_mask)
            mov_s_norm = _robust_normalise(mov_s_level, scalar_mask)
            tensor_scale = torch.sqrt(ref_t_level.square().sum(0)[tensor_mask]).median()
            tensor_scale = tensor_scale.clamp_min(1e-8)
            base = _grid(shape, device=self.device)
            optimiser = torch.optim.Adam([control], lr=cfg.learning_rate / math.sqrt(level))
            latest = {}
            for _ in range(steps):
                optimiser.zero_grad(set_to_none=True)
                field = _expand_control(control, shape)
                coordinates = base + field / float(stride)
                warped_scalar = _sample(mov_s_norm, coordinates)[0]
                warped_tensor = _sample(mov_t_level, coordinates)
                scalar_loss = (warped_scalar[scalar_mask] - ref_s_norm[scalar_mask]).square().mean()
                tensor_loss = (
                    (warped_tensor[:, tensor_mask] - ref_t_level[:, tensor_mask])
                    .div(tensor_scale)
                    .square()
                    .mean()
                )
                membrane, bending = _regularisation(control)
                regulariser = bending if level == 1 else membrane + bending
                loss = (
                    cfg.scalar_weight * scalar_loss
                    + cfg.tensor_weight * tensor_loss
                    + penalty * regulariser / max(float(np.prod(control_shape)), 1.0)
                )
                loss.backward()
                optimiser.step()
                latest = {
                    "total": float(loss.detach()),
                    "scalar": float(scalar_loss.detach()),
                    "tensor": float(tensor_loss.detach()),
                    "regulariser": float(regulariser.detach()),
                }
            levels.append(
                {
                    "level": level,
                    "warp_resolution_mm": resolution,
                    "smoothing_mm": smoothing,
                    "sample_stride": stride,
                    "iterations": steps,
                    "control_shape": control_shape,
                    **latest,
                }
            )
            control = control.detach()
        field = _expand_control(control, ref_s.shape[:3])
        warp = _make_warp(field, ref_s)
        jacobian = torch.linalg.det(_jacobian(field))
        jacobian_image = image_like(jacobian.detach().cpu().numpy().astype(np.float32), ref_s)
        warped_scalar = apply_mmorf_warp(
            mov_s, ref_s, warp, affine=moving_scalar_affine, device=self.device
        )
        warped_tensor = apply_mmorf_warp(
            mov_t, ref_s, warp, affine=moving_tensor_affine, device=self.device
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
                "algorithm_difference": {
                    "optimizer": (
                        "Adam replaces MMORF's CUDA LM/MM optimizers and "
                        "second-order solver"
                    ),
                    "control_lattice": (
                        "trilinear expansion replaces cubic B-spline "
                        "parameterization"
                    ),
                    "regularizer": (
                        "membrane/bending penalties replace MMORF SPRED "
                        "at later levels"
                    ),
                    "tensor_reorientation": (
                        "affine finite-strain rotation is applied before "
                        "optimization; local nonlinear finite-strain "
                        "reorientation is not implemented"
                    ),
                    "cost": (
                        "one-way normalized scalar SSD and tensor L2 replace "
                        "MMORF's symmetric scalar/tensor objective"
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
