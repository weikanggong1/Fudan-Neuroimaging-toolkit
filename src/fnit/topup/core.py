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
from scipy.ndimage import spline_filter

from ..fnirt.spline import (
    BendingOperator,
    expand_coefficients,
    fit_field_coefficients,
    fsl_control_shape,
    spline_bases,
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
    optimizer_steps_per_iteration: int = 8
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
        if self.optimizer_steps_per_iteration < 1:
            raise ValueError("optimizer_steps_per_iteration must be positive")


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
    return image, data, acquisition, int(axes[0]), zooms


def _average_pool(images, factor):
    if factor == 1:
        return images
    value = images[:, None].permute(0, 1, 4, 3, 2)
    value = F.avg_pool3d(value, kernel_size=factor, stride=factor)
    return value.permute(0, 1, 4, 3, 2)[:, 0]


def _gaussian_blur(images, fwhm_mm, voxel_sizes):
    if fwhm_mm <= 0:
        return images
    output = images[:, None].permute(0, 1, 4, 3, 2)
    sigma_mm = float(fwhm_mm) / math.sqrt(8.0 * math.log(2.0))
    for data_axis, conv_axis in enumerate((4, 3, 2)):
        sigma = sigma_mm / float(voxel_sizes[data_axis])
        radius = int(sigma - 0.001) * 2 + 3
        offsets = torch.arange(
            -radius, radius + 1, device=images.device, dtype=images.dtype
        )
        kernel = torch.exp(-0.5 * (offsets / sigma).square())
        kernel /= kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[conv_axis] = kernel.numel()
        kernel = kernel.reshape(shape)
        padding = [0, 0, 0]
        padding[conv_axis - 2] = radius
        output = F.conv3d(output, kernel, padding=tuple(padding))
    return output.permute(0, 1, 4, 3, 2)[:, 0]


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
    values = images.detach().cpu().numpy()
    coefficients = np.stack(
        [
            spline_filter(
                value,
                order=3,
                output=np.float32,
                mode="grid-wrap",
            )
            for value in values
        ]
    )
    return torch.as_tensor(
        coefficients,
        dtype=images.dtype,
        device=images.device,
    )


def _cubic_weights(fraction):
    one_minus = 1 - fraction
    return torch.stack(
        (
            one_minus.pow(3) / 6,
            (3 * fraction.pow(3) - 6 * fraction.square() + 4) / 6,
            (
                -3 * fraction.pow(3)
                + 3 * fraction.square()
                + 3 * fraction
                + 1
            )
            / 6,
            fraction.pow(3) / 6,
        ),
        dim=1,
    )


def _sample_cubic(coefficients, coordinates, phase_encode_axis):
    """Sample NEWIMAGE-style cubic coefficients with periodic extrapolation."""
    shape = coordinates.shape[1:]
    flat = coordinates.reshape(3, -1)
    valid = torch.ones(flat.shape[1], dtype=torch.bool, device=coefficients.device)
    indices = []
    weights = []
    offsets = torch.arange(4, device=coefficients.device, dtype=torch.long)
    for axis, (coordinate, size) in enumerate(zip(flat, coefficients.shape)):
        if axis != phase_encode_axis:
            valid &= (coordinate >= 0) & (coordinate <= size - 1)
        lower = torch.floor(coordinate)
        fraction = coordinate - lower
        axis_indices = lower.to(torch.long)[:, None] - 1 + offsets[None]
        indices.append(torch.remainder(axis_indices, size))
        weights.append(_cubic_weights(fraction))
    linear = coefficients.reshape(-1)
    sy, sz = coefficients.shape[1:]
    output = coefficients.new_zeros(flat.shape[1])
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                address = (
                    indices[0][:, ix] * (sy * sz)
                    + indices[1][:, iy] * sz
                    + indices[2][:, iz]
                )
                output = output + (
                    linear[address]
                    * weights[0][:, ix]
                    * weights[1][:, iy]
                    * weights[2][:, iz]
                )
    return output.reshape(shape), valid.reshape(shape)


def _field_and_derivative(coefficients, shape, spacing, axis, *, positions=None):
    bases = spline_bases(
        shape,
        spacing,
        (1.0, 1.0, 1.0),
        device=coefficients.device,
        dtype=coefficients.dtype,
        positions=positions,
    )
    field = expand_coefficients(coefficients[None], bases)[0]
    derivatives = [0, 0, 0]
    derivatives[axis] = 1
    derivative_bases = spline_bases(
        shape,
        spacing,
        (1.0, 1.0, 1.0),
        device=coefficients.device,
        dtype=coefficients.dtype,
        derivatives=tuple(derivatives),
        positions=positions,
    )
    derivative = expand_coefficients(coefficients[None], derivative_bases)[0]
    return field, derivative


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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        means = values.mean(axis=(0, 1, 2), dtype=np.float64)
        if np.any(np.abs(means) < np.finfo(np.float32).tiny):
            raise ValueError("an input volume has zero mean intensity")
        global_mean = float(means.mean())
        normalized = values * (100.0 / means)[None, None, None, :]
        images = torch.as_tensor(
            np.moveaxis(normalized, -1, 0).copy(),
            dtype=torch.float32,
            device=self.device,
        )
        phase = torch.as_tensor(acquisition[:, pe_axis], dtype=torch.float32, device=self.device)
        readout = torch.as_tensor(acquisition[:, 3], dtype=torch.float32, device=self.device)
        movement = torch.zeros(
            (images.shape[0], 6), dtype=torch.float32, device=self.device
        )
        # FSL TopupScanManager fixes translation along the PE axis for the
        # opposite-polarity scan. The first scan is the fixed reference.
        movement_indices = tuple(index for index in range(6) if index != pe_axis)
        movement_index = torch.as_tensor(
            movement_indices, dtype=torch.long, device=self.device
        )
        movement_scale = torch.as_tensor(
            tuple(80.0 if index >= 3 else 1.0 for index in movement_indices),
            dtype=torch.float32,
            device=self.device,
        )
        movement_selector = torch.zeros(
            (len(movement_indices), 6), dtype=torch.float32, device=self.device
        )
        movement_selector[
            torch.arange(len(movement_indices), device=self.device), movement_index
        ] = 1
        coefficients = None
        previous_field = None
        level_reports = []

        for level, (warp_mm, factor, fwhm, iterations, regularization) in enumerate(
            zip(
                self.config.warp_resolution_mm,
                self.config.subsampling,
                self.config.fwhm_mm,
                self.config.maximum_iterations,
                self.config.regularization,
            ),
            1,
        ):
            level_images = _average_pool(images, factor)
            level_voxels = tuple(value * factor for value in voxel_sizes)
            level_images = _gaussian_blur(level_images, fwhm, level_voxels)
            interpolation_images = _cubic_spline_coefficients(level_images)
            shape = tuple(int(value) for value in level_images.shape[1:])
            spacing = tuple(
                max(1, int(math.floor(warp_mm / size + 0.5)))
                for size in level_voxels
            )
            if coefficients is None:
                coefficients = torch.zeros(
                    fsl_control_shape(shape, spacing),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                dense = previous_field[None, None].permute(0, 1, 4, 3, 2)
                dense = F.interpolate(
                    dense,
                    size=(shape[2], shape[1], shape[0]),
                    mode="trilinear",
                    align_corners=False,
                )[0, 0].permute(2, 1, 0)
                coefficients = fit_field_coefficients(
                    dense[None], spacing, level_voxels
                )[0]
            coefficients = coefficients.detach().contiguous().requires_grad_(True)
            movement_variable = (
                movement[1, movement_index].detach().clone() * movement_scale
            )
            movement_variable = movement_variable.contiguous().requires_grad_(False)
            bending = BendingOperator(
                shape,
                spacing,
                level_voxels,
                device=self.device,
                dtype=torch.float32,
            )
            grid = _grid(shape, device=self.device, dtype=torch.float32)
            optimizer = torch.optim.LBFGS(
                [coefficients],
                lr=0.75,
                max_iter=int(iterations) * self.config.optimizer_steps_per_iteration,
                history_size=8,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-5,
                tolerance_change=1e-7,
            )
            latest = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                field, derivative = _field_and_derivative(
                    coefficients, shape, spacing, pe_axis
                )
                corrected = []
                masks = []
                for scan_index, image in enumerate(level_images):
                    displacement = field * readout[scan_index] * phase[scan_index] / factor
                    coordinates = grid
                    if scan_index:
                        movement_parameters = (
                            movement_variable / movement_scale
                        ) @ movement_selector
                        coordinates = _rigid_coordinates(
                            coordinates,
                            movement_parameters,
                            level_voxels,
                        )
                    coordinates = coordinates.clone()
                    # FSL general_transform evaluates A^-1*x first and then
                    # adds the displacement in the sampling coordinate frame.
                    coordinates[pe_axis] = coordinates[pe_axis] + displacement
                    sampled, valid = _sample_cubic(
                        interpolation_images[scan_index], coordinates, pe_axis
                    )
                    jacobian = 1 + readout[scan_index] * phase[scan_index] * derivative / factor
                    corrected.append(sampled * jacobian)
                    masks.append(valid & (jacobian > 0.05))
                corrected = torch.stack(corrected)
                mask = torch.stack(masks).all(0)
                mask[:, :, (0, -1)] = False
                non_pe = 1 if pe_axis == 0 else 0
                if non_pe == 0:
                    mask[(0, -1), :, :] = False
                else:
                    mask[:, (0, -1), :] = False
                mean = corrected.mean(0)
                residual = (corrected - mean[None])[:, mask]
                ssd = residual.square().sum() / (mask.sum() * (corrected.shape[0] - 1))
                energy = bending.energy(coefficients[None])
                loss = ssd + ssd.detach() * float(regularization) * energy
                loss.backward()
                latest.update(
                    loss=float(loss.detach()),
                    ssd=float(ssd.detach()),
                    bending=float(energy.detach()),
                    voxels=int(mask.sum()),
                )
                return loss

            optimizer.step(closure)
            if level <= 5:
                coefficients.requires_grad_(False)
                movement_variable.requires_grad_(True)
                optimizer = torch.optim.LBFGS(
                    [movement_variable],
                    lr=0.25,
                    max_iter=int(iterations)
                    * self.config.optimizer_steps_per_iteration,
                    history_size=8,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-5,
                    tolerance_change=1e-7,
                )
                optimizer.step(closure)
            decoded_movement = movement_variable.detach() / movement_scale
            movement = torch.stack(
                (movement[0], decoded_movement @ movement_selector)
            )
            with torch.no_grad():
                previous_field, _ = _field_and_derivative(
                    coefficients.detach(), shape, spacing, pe_axis
                )
            level_reports.append(
                {"level": level, "shape": shape, "knot_spacing": spacing, **latest}
            )

        full_shape = tuple(int(value) for value in images.shape[1:])
        final_spacing = tuple(
            max(1, int(math.floor(self.config.warp_resolution_mm[-1] / size + 0.5)))
            for size in voxel_sizes
        )
        if previous_field.shape != full_shape or tuple(spacing) != final_spacing:
            dense = previous_field[None, None].permute(0, 1, 4, 3, 2)
            dense = F.interpolate(
                dense,
                size=(full_shape[2], full_shape[1], full_shape[0]),
                mode="trilinear",
                align_corners=False,
            )[0, 0].permute(2, 1, 0)
            coefficients = fit_field_coefficients(dense[None], final_spacing, voxel_sizes)[0]
        else:
            coefficients = coefficients.detach()
        full_interpolation_images = _cubic_spline_coefficients(images)
        with torch.no_grad():
            field, derivative = _field_and_derivative(
                coefficients, full_shape, final_spacing, pe_axis
            )
            grid = _grid(full_shape, device=self.device, dtype=torch.float32)
            corrected = []
            jacobians = []
            masks = []
            for index, image in enumerate(images):
                coordinates = grid
                if index:
                    coordinates = _rigid_coordinates(
                        coordinates, movement[index], voxel_sizes
                    )
                coordinates = coordinates.clone()
                coordinates[pe_axis] += field * readout[index] * phase[index]
                sampled, valid = _sample_cubic(
                    full_interpolation_images[index], coordinates, pe_axis
                )
                jacobian = 1 + readout[index] * phase[index] * derivative
                corrected.append(sampled * jacobian)
                jacobians.append(jacobian)
                masks.append(valid & (jacobian > 0.05))
            corrected = torch.stack(corrected)
            common = torch.stack(masks).all(0)
            corrected[:, ~common] = 0
            corrected *= global_mean / 100.0
            jacobians = torch.stack(jacobians)

        field_np = field.cpu().numpy().astype(np.float32)
        corrected_np = np.moveaxis(corrected.cpu().numpy(), 0, -1).astype(np.float32)
        jacobian_np = jacobians.cpu().numpy().astype(np.float32)
        coefficient_np = coefficients.cpu().numpy().astype(np.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else None
        )
        return TOPUPResult(
            field_hz=make_output_image(field_np, reference, intent=FSL_TOPUP_FIELD),
            corrected=make_output_image(corrected_np, reference),
            corrected_mean=make_output_image(corrected_np.mean(axis=3), reference),
            jacobians=tuple(
                make_topup_jacobian_image(value, reference) for value in jacobian_np
            ),
            coefficients=make_topup_coefficient_image(
                coefficient_np, full_shape, voxel_sizes, final_spacing
            ),
            movement_parameters=movement.cpu().numpy().astype(np.float64),
            qc={
                "device": str(self.device),
                "dtype": "float32",
                "tf32": bool(self.device.type == "cuda"),
                "phase_encode_axis": pe_axis,
                "reference_implementation": f"FSL TOPUP {FSL_TOPUP_VERSION}",
                "optimizer": "alternating field and 5-DOF motion L-BFGS",
                "fsl_output_contract": True,
                "fsl_numerically_equivalent": False,
                "bitwise_equivalent": False,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": peak_memory,
                "levels": level_reports,
            },
        )

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
