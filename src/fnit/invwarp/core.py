"""Invert a FSL pull warp on a specified output grid."""

import numpy as np
import torch

from ..applywarp.core import (FSL_FNIRT_DISPLACEMENT_FIELD, _fsl_voxel_matrix,
                              _load_nifti, _spatial_grid)
from ..convertwarp.core import _PullField, _device, _field_image, WarpFieldResult


class TorchInvWarp:
    """Invert a smooth FSL dense or cubic-coefficient warp by fixed-point search."""

    def __init__(self, device="cpu"):
        self.device = _device(device)

    @torch.inference_mode()
    def __call__(self, reference, warp, *, warp_convention="auto",
                 output_convention="relative", iterations=30, tolerance_mm=0.01):
        if output_convention not in ("relative", "absolute"):
            raise ValueError("output_convention must be relative or absolute")
        if iterations < 1 or tolerance_mm <= 0:
            raise ValueError("iterations and tolerance_mm must be positive")
        reference = _load_nifti(reference, "reference")
        if len(reference.shape) != 3:
            raise ValueError("reference must be a 3D NIfTI image")
        field = _PullField(warp, self.device, warp_convention)
        axes = torch.meshgrid(
            *(torch.linspace(0, size - 1, min(size, 5), device=self.device,
                              dtype=torch.float64) for size in field.shape), indexing="ij")
        grid = torch.stack(axes).reshape(3, -1)
        scaled = torch.as_tensor(field.scaled, dtype=torch.float64, device=self.device)
        sample_mm = (scaled[:3, :3] @ grid + scaled[:3, 3:4]).reshape(
            3, *(min(size, 5) for size in field.shape))
        mapped, _ = field.sample(sample_mm)
        x = np.column_stack((sample_mm.reshape(3, -1).T.cpu().numpy(),
                             np.ones(sample_mm.numel() // 3)))
        y = mapped.reshape(3, -1).T.cpu().numpy()
        fit, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = fit[:3].T
        affine[:3, 3] = fit[3]
        if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-8:
            raise ValueError("warp has a singular global affine component")
        inverse = torch.as_tensor(np.linalg.inv(affine), dtype=torch.float64,
                                  device=self.device)
        target = _spatial_grid(reference.shape, _fsl_voxel_matrix(reference),
                               self.device).reshape(3, *reference.shape)
        flat = target.reshape(3, -1)
        estimate = (inverse[:3, :3] @ flat
                    + inverse[:3, 3:4]).reshape(target.shape)
        for _ in range(iterations):
            mapped, _ = field.sample(estimate)
            correction = inverse[:3, :3] @ (target - mapped).reshape(3, -1)
            estimate = estimate + correction.reshape(target.shape)
            if float(correction.abs().max()) < tolerance_mm:
                break
        mapped, valid = field.sample(estimate)
        error = (mapped - target).square().sum(dim=0).sqrt()
        output = estimate - target if output_convention == "relative" else estimate
        image = _field_image(reference, output, intent_code=(
            FSL_FNIRT_DISPLACEMENT_FIELD if output_convention == "relative" else 0))
        return WarpFieldResult(image, float(valid.float().mean().cpu()), {
            "device": str(self.device), "output_convention": output_convention,
            "input_convention": field.convention,
            "iterations": iterations, "tolerance_mm": tolerance_mm,
            "median_residual_mm_in_field": float(error[valid].median().cpu()) if valid.any() else None,
            "matrix_coordinates": "FSL scaled-mm",
        })

    def run(self, reference, warp, output, **kwargs):
        return self(reference, warp, **kwargs).save(output)


def invwarp(reference, warp, **kwargs):
    device = kwargs.pop("device", "cpu")
    return TorchInvWarp(device)(reference, warp, **kwargs)
