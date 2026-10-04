"""Invert a FSL pull warp on a specified output grid."""

from numbers import Integral, Real

import numpy as np
import torch

from ..applywarp.core import (FSL_FNIRT_DISPLACEMENT_FIELD, _fsl_voxel_matrix,
                              _load_nifti, _spatial_grid)
from ..convertwarp.core import _PullField, _device, _field_image, WarpFieldResult


def _correction_max_abs(correction):
    """Reduce the CPU stopping scalar without a full absolute-value scratch."""
    if (correction.device.type == "cpu" and not correction.requires_grad
            and correction.dtype in (torch.float32, torch.float64)
            and not torch._C._are_functorch_transforms_active()
            and torch.autograd.forward_ad.unpack_dual(correction).tangent is None):
        minimum, maximum = torch.aminmax(correction)
        value = torch.maximum(minimum.abs(), maximum.abs())
        if not bool(torch.isnan(value)):
            return value
    # Keep exceptional NaN payloads, CUDA, gradients and functional transforms
    # on the original operator path.
    return correction.abs().max()


class TorchInvWarp:
    """Invert a smooth FSL dense or cubic-coefficient warp by fixed-point search."""

    def __init__(self, device="cpu"):
        self.device = _device(device)

    @torch.inference_mode()
    def __call__(self, reference, warp, *, warp_convention="auto",
                 output_convention="relative", iterations=30, tolerance_mm=0.01):
        if output_convention not in ("relative", "absolute"):
            raise ValueError("output_convention must be relative or absolute")
        if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if (isinstance(tolerance_mm, bool) or not isinstance(tolerance_mm, Real)
                or not np.isfinite(tolerance_mm) or tolerance_mm <= 0):
            raise ValueError("tolerance_mm must be finite and positive")
        reference = _load_nifti(reference, "reference")
        if len(reference.shape) != 3:
            raise ValueError("reference must be a 3D NIfTI image")
        field = _PullField(warp, self.device, warp_convention)
        # This field is fixed throughout one inversion. CPU grid_sample uses
        # channels-last storage; prepare it once instead of copying each call.
        prepared_source = (field.values[None].contiguous(memory_format=torch.channels_last_3d)
                           if self.device.type == "cpu" else None)
        axes = torch.meshgrid(
            *(torch.linspace(0, size - 1, min(size, 5), device=self.device,
                              dtype=torch.float64) for size in field.shape), indexing="ij")
        grid = torch.stack(axes).reshape(3, -1)
        scaled = torch.as_tensor(field.scaled, dtype=torch.float64, device=self.device)
        sample_mm = (scaled[:3, :3] @ grid + scaled[:3, 3:4]).reshape(
            3, *(min(size, 5) for size in field.shape))
        if self.device.type == "cpu":
            mapped, _ = field.sample(sample_mm, prepared_source=prepared_source, calculate_valid=False)
        else:
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
        # CPU scratch avoids two full FP64 allocations per iteration.  Keep
        # CUDA's existing operations and allocation behaviour unchanged.
        if self.device.type == "cpu":
            residual_buffer = torch.empty_like(flat)
            correction_buffer = torch.empty_like(flat)
        converged = False
        for iteration in range(iterations):
            if self.device.type == "cpu":
                mapped, _ = field.sample(estimate, prepared_source=prepared_source, calculate_valid=False)
            else:
                mapped, _ = field.sample(estimate)
            if self.device.type == "cpu":
                torch.sub(flat, mapped.reshape(3, -1), out=residual_buffer)
                torch.mm(inverse[:3, :3], residual_buffer, out=correction_buffer)
                correction = correction_buffer
                estimate.add_(correction.reshape(target.shape))
            else:
                correction = inverse[:3, :3] @ (target - mapped).reshape(3, -1)
                estimate = estimate + correction.reshape(target.shape)
            if float(_correction_max_abs(correction)) < tolerance_mm:
                converged = True
                break
        if self.device.type == "cpu":
            mapped, valid = field.sample(estimate, prepared_source=prepared_source)
        else:
            mapped, valid = field.sample(estimate)
        error = (mapped - target).square().sum(dim=0).sqrt()
        output = estimate - target if output_convention == "relative" else estimate
        image = _field_image(reference, output, intent_code=(
            FSL_FNIRT_DISPLACEMENT_FIELD if output_convention == "relative" else 0))
        return WarpFieldResult(image, float(valid.float().mean().cpu()), {
            "device": str(self.device), "output_convention": output_convention,
            "input_convention": field.convention,
            "iterations": iterations, "tolerance_mm": tolerance_mm,
            "iterations_used": iteration + 1, "converged": converged,
            "median_residual_mm_in_field": float(error[valid].median().cpu()) if valid.any() else None,
            "matrix_coordinates": "FSL scaled-mm",
        })

    def run(self, reference, warp, output, **kwargs):
        return self(reference, warp, **kwargs).save(output)


def invwarp(reference, warp, **kwargs):
    device = kwargs.pop("device", "cpu")
    return TorchInvWarp(device)(reference, warp, **kwargs)
