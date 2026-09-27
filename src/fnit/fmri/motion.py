"""Rigid BOLD motion estimation with batched PyTorch resampling.

The fitted transform is a *pull* mapping from reference voxels to moving
voxels.  Rotations use the input voxel axes and translations are in mm.
``fsl_matrices`` converts the inverse mapping to FLIRT scaled-mm matrices.
"""

from dataclasses import dataclass
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..applywarp.core import _fsl_voxel_matrix
from ..flirt.core import fsl_parameters_from_affine


@dataclass
class MotionResult:
    parameters: np.ndarray
    fsl_matrices: np.ndarray
    corrected: nib.Nifti1Image | None
    reference: nib.Nifti1Image


def _load_image(value):
    image = nib.load(str(value)) if isinstance(value, (str, bytes, os.PathLike)) else value
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError("input and reference must be NIfTI images or paths")
    return image


def _rotation(angles):
    """Return Rz @ Ry @ Rx for one row of radians per frame."""
    x, y, z = angles.unbind(-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    row0 = torch.stack((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx), -1)
    row1 = torch.stack((sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx), -1)
    row2 = torch.stack((-sy, cy * sx, cy * cx), -1)
    return torch.stack((row0, row1, row2), -2)


def _grid(parameters, shape, voxel_sizes, *, device):
    axes = torch.meshgrid(
        *(torch.arange(size, device=device, dtype=torch.float32) for size in shape),
        indexing="ij",
    )
    center = torch.as_tensor([(size - 1) / 2 for size in shape], device=device)
    sizes = torch.as_tensor(voxel_sizes, dtype=torch.float32, device=device)
    points = (torch.stack(axes, -1).reshape(-1, 3) - center) * sizes
    rotated = torch.matmul(points[None], _rotation(parameters[:, :3]).transpose(1, 2))
    source = (rotated + parameters[:, None, 3:]) / sizes + center
    norm = torch.as_tensor([2 / max(size - 1, 1) for size in shape], device=device)
    coords = source * norm - 1
    return coords[..., [2, 1, 0]].reshape(len(parameters), *shape, 3)


def _pyramid(image, shape):
    if tuple(image.shape[-3:]) == tuple(shape):
        return image
    return F.interpolate(image, size=shape, mode="trilinear", align_corners=True)


def _fit_batch(moving, reference, mask, voxel_sizes, *, iterations, device):
    count = moving.shape[0]
    parameters = torch.zeros(count, 6, device=device, dtype=torch.float32)
    levels = ((8.0, iterations[0]), (4.0, iterations[1]), (None, iterations[2]))
    for resolution, steps in levels:
        if not steps:
            continue
        shape = (
            tuple(max(8, int(round((size - 1) * spacing / resolution)) + 1)
                  for size, spacing in zip(reference.shape[-3:], voxel_sizes))
            if resolution is not None else tuple(reference.shape[-3:])
        )
        fixed = _pyramid(reference, shape)
        images = _pyramid(moving, shape)
        weights = _pyramid(mask, shape).clamp(0, 1)
        spacings = tuple((size - 1) * spacing / max(other - 1, 1)
                         for size, other, spacing in zip(reference.shape[-3:], shape, voxel_sizes))
        fixed_values = fixed.reshape(1, -1)
        weight_values = weights.reshape(1, -1)
        weight_sum = weight_values.sum().clamp_min(1)
        fixed_mean = (fixed_values * weight_values).sum(-1, keepdim=True) / weight_sum
        fixed_centered = (fixed_values - fixed_mean) * weight_values.sqrt()
        fixed_norm = fixed_centered.square().sum(-1).sqrt().clamp_min(1e-6)
        # One unit of the optimised rotation variable represents 0.02 radians.
        # Translation variables are mm, giving similar gradient scales.
        variable = torch.cat((parameters[:, :3] / 0.02, parameters[:, 3:]), -1).detach()
        variable.requires_grad_(True)
        optimizer = torch.optim.Adam([variable], lr=0.12 if resolution else 0.06)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            pose = torch.cat((variable[:, :3] * 0.02, variable[:, 3:]), -1)
            sampled = F.grid_sample(
                images, _grid(pose, shape, spacings, device=device),
                mode="bilinear", padding_mode="border", align_corners=True,
            ).reshape(count, -1)
            mean = (sampled * weight_values).sum(-1, keepdim=True) / weight_sum
            moving_centered = (sampled - mean) * weight_values.sqrt()
            corr = (moving_centered * fixed_centered).sum(-1) / (
                moving_centered.square().sum(-1).sqrt().clamp_min(1e-6) * fixed_norm
            )
            (1 - corr).sum().backward()
            optimizer.step()
        parameters = torch.cat((variable[:, :3] * 0.02, variable[:, 3:]), -1).detach()
    return parameters


def _fsl_matrices(parameters, input_image, reference_image):
    input_fsl = _fsl_voxel_matrix(input_image)
    reference_fsl = _fsl_voxel_matrix(reference_image)
    sizes = np.asarray(input_image.header.get_zooms()[:3], dtype=np.float64)
    center = (np.asarray(reference_image.shape[:3], dtype=np.float64) - 1) / 2
    voxel_mm = np.diag([*sizes, 1.0])
    output = np.empty((len(parameters), 4, 4), dtype=np.float64)
    angles = torch.as_tensor(parameters[:, :3], dtype=torch.float64)
    rotations = _rotation(angles).numpy()
    for index, (pose, rotation) in enumerate(zip(parameters, rotations)):
        pull = np.eye(4)
        pull[:3, :3] = rotation
        pull[:3, 3] = sizes * center + pose[3:] - rotation @ (sizes * center)
        pull_voxel = np.linalg.inv(voxel_mm) @ pull @ voxel_mm
        output[index] = reference_fsl @ np.linalg.inv(pull_voxel) @ np.linalg.inv(input_fsl)
    return output


def estimate_motion(
    input_bold,
    reference,
    *,
    mask=None,
    device=None,
    batch_size=16,
    iterations=(35, 25, 15),
    resample=False,
):
    """Fit each BOLD frame to a same-grid reference and optionally resample.

    Parameters are returned as ``[rx, ry, rz, tx, ty, tz]`` in radians and mm.
    The FLIRT matrices map each input frame to the reference in scaled-mm axes.
    """
    input_image = _load_image(input_bold)
    reference_image = _load_image(reference)
    if input_image.ndim != 4 or reference_image.ndim != 3:
        raise ValueError("input_bold must be 4D and reference must be 3D")
    if input_image.shape[:3] != reference_image.shape or not np.allclose(
        input_image.affine, reference_image.affine, atol=1e-4
    ):
        raise ValueError("BOLD and reference must have the same voxel grid")
    if batch_size < 1 or len(iterations) != 3 or any(value < 0 for value in iterations):
        raise ValueError("batch_size must be positive and iterations must have three nonnegative counts")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    reference_data = np.asarray(reference_image.dataobj, dtype=np.float32)
    if mask is None:
        positive = reference_data[reference_data > 0]
        threshold = np.percentile(positive, 35) if positive.size else 0
        mask_data = (reference_data > threshold).astype(np.float32)
    else:
        mask_image = _load_image(mask)
        if mask_image.shape != reference_image.shape or not np.allclose(
            mask_image.affine, reference_image.affine, atol=1e-4
        ):
            raise ValueError("mask must use the reference voxel grid")
        mask_data = (np.asarray(mask_image.dataobj) > 0).astype(np.float32)
    fixed = torch.as_tensor(reference_data, device=device)[None, None]
    weights = torch.as_tensor(mask_data, device=device)[None, None]
    data = np.asarray(input_image.dataobj, dtype=np.float32)
    parameters = []
    corrected = np.empty(input_image.shape, dtype=np.float32) if resample else None
    spacings = tuple(float(value) for value in input_image.header.get_zooms()[:3])
    for start in range(0, input_image.shape[3], batch_size):
        stop = min(start + batch_size, input_image.shape[3])
        block = data[..., start:stop]
        moving = torch.as_tensor(np.moveaxis(block, -1, 0).copy(), device=device)[:, None]
        fitted = _fit_batch(moving, fixed, weights, spacings, iterations=iterations, device=device)
        parameters.append(fitted.cpu().numpy())
        if corrected is not None:
            with torch.no_grad():
                sampled = F.grid_sample(
                    moving, _grid(fitted, reference_image.shape, spacings, device=device),
                    mode="bilinear", padding_mode="border", align_corners=True,
                )[:, 0]
                corrected[..., start:stop] = np.moveaxis(sampled.cpu().numpy(), 0, -1)
    parameters = np.concatenate(parameters)
    corrected_image = (
        nib.Nifti1Image(corrected, reference_image.affine, input_image.header.copy())
        if corrected is not None else None
    )
    return MotionResult(parameters, _fsl_matrices(parameters, input_image, reference_image),
                        corrected_image, reference_image)


def matrices_to_mcflirt_parameters(matrices, reference):
    """Convert FLIRT matrices to MCFLIRT rotation/radian and translation/mm rows.

    Translations use the reference image intensity-weighted centre, matching
    MCFLIRT's ``-plots`` convention. This was checked against same-run FSL
    matrices and `.par` on the real 490-frame UKB example.
    """
    image = _load_image(reference)
    if image.ndim != 3:
        raise ValueError("reference must be 3D")
    data = np.asarray(image.dataobj, dtype=np.float64)
    total = data.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("reference must have a positive finite intensity sum")
    indices = np.meshgrid(*(np.arange(size) for size in image.shape), indexing="ij")
    centre_voxel = np.asarray([(axis * data).sum() / total for axis in indices])
    centre = (_fsl_voxel_matrix(image) @ np.r_[centre_voxel, 1])[:3]
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("matrices must be a stack of 4x4 FLIRT transforms")
    return np.stack([fsl_parameters_from_affine(matrix, centre)[:6]
                     for matrix in matrices])
