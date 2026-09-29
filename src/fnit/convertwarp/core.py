"""Compose FSL scaled-mm affine and nonlinear pull transforms."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..applywarp.core import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    _expand_cubic_coefficients,
    _fsl_voxel_matrix,
    _load_dense_warp,
    _load_nifti,
    _matrix,
    _sample_linear,
    _spatial_grid,
)


def _device(value):
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


class _PullField:
    def __init__(self, warp, device, warp_convention):
        image = _load_nifti(warp, "warp")
        self.shape = tuple(int(size) for size in image.shape[:3])
        intent = int(image.header["intent_code"])
        self.embedded_inverse = None
        if intent == FSL_CUBIC_SPLINE_COEFFICIENTS:
            self.values, self.shape, scaled, embedded = _expand_cubic_coefficients(image, device)
            self.embedded_inverse = torch.as_tensor(
                np.linalg.inv(embedded), dtype=torch.float64, device=device)
            self.convention = "relative"
        else:
            image, values, scaled, self.convention, _ = _load_dense_warp(
                image, warp_convention)
            self.values = torch.as_tensor(
                np.moveaxis(values, -1, 0).copy(), device=device)
        self.scaled = scaled
        self.scaled_inverse = torch.as_tensor(
            np.linalg.inv(scaled), dtype=torch.float64, device=device)

    def sample(self, query):
        flat = query.reshape(3, -1)
        voxels = (self.scaled_inverse[:3, :3] @ flat
                  + self.scaled_inverse[:3, 3:4]).reshape(query.shape)
        values, valid = _sample_linear(self.values, voxels)
        if self.embedded_inverse is not None:
            base = (self.embedded_inverse[:3, :3] @ flat
                    + self.embedded_inverse[:3, 3:4]).reshape(query.shape)
            return base + values.to(torch.float64), valid
        if self.convention == "absolute":
            return values.to(torch.float64), valid
        return query + values.to(torch.float64), valid


def _field_image(reference, coordinates, *, intent_code=0):
    reference = _load_nifti(reference, "reference")
    if len(reference.shape) != 3:
        raise ValueError("reference must be a 3D NIfTI image")
    data = np.moveaxis(coordinates.detach().cpu().numpy(), 0, -1).astype(np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = type(reference)(data, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    image.set_qform(qform, int(qcode))
    image.set_sform(sform, int(scode))
    image.header["intent_code"] = intent_code
    return image


@dataclass(frozen=True)
class WarpFieldResult:
    image: nib.spatialimages.SpatialImage
    valid_fraction: float
    qc: dict

    def save(self, output):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(self.image, str(path))
        return self


class TorchConvertWarp:
    """FSL ``convertwarp --warp1 --premat`` subset; output is a pull field."""

    def __init__(self, device="cpu"):
        self.device = _device(device)

    @torch.inference_mode()
    def __call__(self, reference, warp1, *, premat=None, postmat=None,
                 warp_convention="auto", output_convention="relative"):
        if output_convention not in ("relative", "absolute"):
            raise ValueError("output_convention must be relative or absolute")
        reference = _load_nifti(reference, "reference")
        if len(reference.shape) != 3:
            raise ValueError("reference must be a 3D NIfTI image")
        field = _PullField(warp1, self.device, warp_convention)
        reference_mm = _spatial_grid(reference.shape, _fsl_voxel_matrix(reference),
                                     self.device).reshape(3, *reference.shape)
        post_inverse = torch.as_tensor(
            np.linalg.inv(_matrix(postmat, "postmat")), dtype=torch.float64,
            device=self.device)
        flat = reference_mm.reshape(3, -1)
        query = (post_inverse[:3, :3] @ flat
                 + post_inverse[:3, 3:4]).reshape(reference_mm.shape)
        source, valid = field.sample(query)
        pre_inverse = torch.as_tensor(
            np.linalg.inv(_matrix(premat, "premat")), dtype=torch.float64,
            device=self.device)
        flat = source.reshape(3, -1)
        source = (pre_inverse[:3, :3] @ flat
                  + pre_inverse[:3, 3:4]).reshape(reference_mm.shape)
        output = source - reference_mm if output_convention == "relative" else source
        image = _field_image(reference, output)
        return WarpFieldResult(image, float(valid.float().mean().cpu()), {
            "device": str(self.device), "output_convention": output_convention,
            "input_convention": field.convention,
            "transform_order": "premat then warp1 then postmat",
            "matrix_coordinates": "FSL scaled-mm",
        })

    def run(self, reference, warp1, output, **kwargs):
        return self(reference, warp1, **kwargs).save(output)

    @torch.inference_mode()
    def from_mmorf(self, reference, source, mmorf_warp, *, affine,
                   output_convention="relative"):
        """Convert an FNIT MMORF pull field and FA FLIRT matrix to an FSL field."""
        if output_convention not in ("relative", "absolute"):
            raise ValueError("output_convention must be relative or absolute")
        from ..mmorf.core import _mm_to_voxel_axes_rotation, _world_forward

        reference = _load_nifti(reference, "reference")
        source = _load_nifti(source, "source")
        mmorf_warp = _load_nifti(mmorf_warp, "mmorf_warp")
        if len(reference.shape) != 3 or len(source.shape) != 3:
            raise ValueError("reference and source must be 3D NIfTI images")
        if mmorf_warp.shape != (*reference.shape, 3) or not np.allclose(
                mmorf_warp.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError("MMORF warp must share the reference grid")
        world_forward = _world_forward(_matrix(affine, "affine"), source, reference)
        source_scaled_pull = torch.as_tensor(
            _fsl_voxel_matrix(source) @ np.linalg.inv(source.affine)
            @ np.linalg.inv(world_forward), dtype=torch.float64,
            device=self.device)
        reference_world = _spatial_grid(reference.shape, reference.affine,
                                        self.device).reshape(3, *reference.shape)
        field = torch.as_tensor(
            np.moveaxis(np.asarray(mmorf_warp.dataobj, dtype=np.float32), -1, 0).copy(),
            device=self.device)
        rotation = _mm_to_voxel_axes_rotation(reference.affine, device=self.device)
        displacement = rotation.T.to(torch.float64) @ field.reshape(3, -1).to(torch.float64)
        query = reference_world.reshape(3, -1) + displacement
        source_scaled = (source_scaled_pull[:3, :3] @ query
                         + source_scaled_pull[:3, 3:4]).reshape(reference_world.shape)
        reference_scaled = _spatial_grid(
            reference.shape, _fsl_voxel_matrix(reference), self.device
        ).reshape(reference_world.shape)
        output = (source_scaled - reference_scaled if output_convention == "relative"
                  else source_scaled)
        return WarpFieldResult(_field_image(reference, output), 1.0, {
            "device": str(self.device), "output_convention": output_convention,
            "input_convention": "MMORF reference-axis mm",
            "transform_order": "FA FLIRT affine then MMORF nonlinear",
            "matrix_coordinates": "FSL scaled-mm",
        })

    def run_mmorf(self, reference, source, mmorf_warp, output, *, affine,
                  output_convention="relative"):
        return self.from_mmorf(
            reference, source, mmorf_warp, affine=affine,
            output_convention=output_convention).save(output)


def convertwarp(reference, warp1, **kwargs):
    device = kwargs.pop("device", "cpu")
    return TorchConvertWarp(device)(reference, warp1, **kwargs)
